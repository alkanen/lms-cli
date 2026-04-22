"""
bash — run an arbitrary shell command on the client computer.

Permission is required for every call unless the user has granted a
session-scoped exact or pattern-based allow for the specific command.
The tool is disabled by default because it gives the model unrestricted
access to the client machine.
"""

from __future__ import annotations

import fnmatch
import logging
import re
import shlex
import subprocess
from typing import TYPE_CHECKING, Any

from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_OUTPUT_CHARS = 1024
_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

_CAPTURE_MODES = ("stdout", "stderr", "interleaved", "separate")


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """Return *(text, truncated)* where *text* is at most *max_chars* characters."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


if TYPE_CHECKING:
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.workspace import Workspace


def _normalize(command: str) -> str:
    """Tokenise *command* and rejoin canonically, preserving token boundaries.

    Uses ``shlex.join`` so that tokens containing spaces are re-quoted, making
    ``rm "file with spaces.txt"`` and ``rm file with spaces.txt`` produce
    distinct keys rather than collapsing to the same string.
    """
    try:
        return shlex.join(shlex.split(command))
    except ValueError:
        return command.strip()


class BashTool(Tool):
    NAME = "bash"
    DESCRIPTION = "Run an arbitrary shell command on the client computer."
    PERMISSION_REQUIRED = True
    DISABLED_BY_DEFAULT = True

    def __init__(
        self,
        workspace: Workspace,
        permission_manager: PermissionManager,
        permission_required: bool,
        name: str,
        description: str,
    ) -> None:
        super().__init__(
            workspace, permission_manager, permission_required, name, description
        )
        self._exact_grants: set[str] = set()
        self._pattern_grants: list[str] = []

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------

    def reset_session_state(self) -> None:
        self._exact_grants.clear()
        self._pattern_grants.clear()
        logger.debug("bash: session grants cleared")

    # ------------------------------------------------------------------
    # Permission helpers
    # ------------------------------------------------------------------

    def request_permission(self, action: str, **kwargs: Any) -> tuple[bool, str]:
        if not self.permission_required:
            return True, ""
        cmd = kwargs.get("command", "")
        if cmd:
            normalized = _normalize(cmd)
            if normalized in self._exact_grants:
                logger.debug("bash: exact grant matched — skipping prompt")
                return True, ""
            for pattern in self._pattern_grants:
                if fnmatch.fnmatch(normalized, pattern):
                    logger.debug(
                        "bash: pattern grant %r matched — skipping prompt", pattern
                    )
                    return True, ""
        return super().request_permission(action, **kwargs)

    def extra_permission_options(self, **kwargs: Any) -> list[str]:
        """Return extra permission options for *command*.

        Normal case (2+ tokens): ``["always", "always: <exe> <leading_args> *"]``.
        Single-token command:    ``["always", "always: <exe> *"]``.
        Unparseable / empty:     ``["always"]``.

        Including ``"always"`` here intercepts the universal always-choice before
        PermissionManager records it as a tool-wide grant, so that ``on_permission_granted``
        can store a command-specific exact grant instead.
        """
        cmd = kwargs.get("command", "")
        if not cmd:
            # Always intercept "always" so the universal choice never creates a
            # tool-wide PermissionManager grant for bash.
            return ["always"]
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            return ["always"]
        if not tokens:
            return ["always"]
        if len(tokens) == 1:
            return ["always", f"always: {tokens[0]} *"]
        leading = shlex.join(tokens[:-1])
        return ["always", f"always: {leading} *"]

    def on_permission_granted(self, choice: str, **kwargs: Any) -> None:
        cmd = kwargs.get("command", "")
        if not cmd:
            return
        normalized = _normalize(cmd)
        if not normalized:
            return
        if choice == "always":
            self._exact_grants.add(normalized)
            logger.info("bash: exact grant stored for %r", normalized)
        elif choice.startswith("always: ") and choice.endswith(" *"):
            pattern = choice[len("always: ") :]
            if pattern not in self._pattern_grants:
                self._pattern_grants.append(pattern)
                logger.info("bash: pattern grant stored: %r", pattern)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def execute_log(self, **kwargs: Any) -> str | None:
        cmd: str = kwargs.get("command", "")
        if not cmd:
            return "<empty command>"
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            return "<unparseable command>"
        if not tokens:
            return "<empty command>"
        # Strip leading KEY=value tokens so env var values are never written to
        # logs.  Phase 3 will add proper env var support to execute() itself.
        exe_idx = next(
            (i for i, t in enumerate(tokens) if not _ENV_VAR_RE.match(t)), len(tokens)
        )
        if exe_idx >= len(tokens):
            return "<empty command>"
        summary = shlex.join(tokens[exe_idx:])
        return summary if len(summary) <= 60 else f"{summary[:57]}..."

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=(
                "Run a shell command on the client computer and return its output. "
                "The command is parsed via shlex and executed directly."
            ),
            arguments=[
                ToolArgument(
                    name="command",
                    description="The shell command to run (e.g. 'ls -la ./src').",
                    argument_type="string",
                    required=True,
                ),
                ToolArgument(
                    name="capture",
                    description=(
                        "Which output stream(s) to capture. "
                        "'stdout' (default) captures stdout only. "
                        "'stderr' captures stderr only. "
                        "'interleaved' merges stderr into stdout. "
                        "'separate' returns stdout and stderr as separate fields."
                    ),
                    argument_type="string",
                    required=False,
                    enum=list(_CAPTURE_MODES),
                ),
                ToolArgument(
                    name="max_output_chars",
                    description=(
                        "Maximum number of characters to return from captured output "
                        f"(default {_DEFAULT_MAX_OUTPUT_CHARS}). Output beyond this "
                        "limit is truncated."
                    ),
                    argument_type="integer",
                    required=False,
                    minimum=1,
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(  # type: ignore[override]
        self,
        *,
        command: str,
        capture: str = "stdout",
        max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
    ) -> dict:
        if capture not in _CAPTURE_MODES:
            return self._err(
                "invalid_arguments",
                f"Invalid capture mode {capture!r}. Must be one of: "
                + ", ".join(_CAPTURE_MODES),
                400,
            )
        if max_output_chars < 1:
            return self._err(
                "invalid_arguments",
                f"max_output_chars must be >= 1, got {max_output_chars}.",
                400,
            )
        try:
            args = shlex.split(command)
        except ValueError as exc:
            logger.debug("bash: shlex parse failed: %s", exc)
            return self._err("invalid_command", f"Failed to parse command: {exc}", 400)
        if not args:
            return self._err("invalid_command", "Command is empty.", 400)
        if _ENV_VAR_RE.match(args[0]):
            logger.debug("bash: rejected env-var prefix in args[0]=%r", args[0])
            return self._err(
                "invalid_command",
                "Environment variable prefixes (e.g. KEY=val) are not supported "
                "until Phase 3. Run the command without the prefix or set the "
                "variable in a prior step.",
                400,
            )
        logger.debug("bash: running %r (%d arg(s))", args[0], len(args) - 1)

        # Build stream kwargs for subprocess based on capture mode.
        if capture == "interleaved":
            stream_kwargs: dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
            }
        elif capture == "stderr":
            stream_kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE}
        elif capture == "separate":
            stream_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
        else:
            # stdout only — stderr is captured solely for error reporting on non-zero exit
            stream_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}

        try:
            proc = subprocess.run(
                args,
                **stream_kwargs,
                text=True,
                timeout=_DEFAULT_TIMEOUT,
                cwd=self._workspace.root,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.debug("bash: executable not found: %r", args[0])
            return self._err("execution_error", f"Command not found: {args[0]}", 400)
        except subprocess.TimeoutExpired:
            logger.warning("bash: %r timed out after %ds", args[0], _DEFAULT_TIMEOUT)
            return self._err(
                "execution_error",
                f"Command timed out after {_DEFAULT_TIMEOUT} seconds.",
                408,
            )
        except Exception as exc:
            logger.exception("bash: unexpected error running %r", args[0])
            return self._err("execution_error", str(exc), 500)

        if proc.returncode != 0:
            logger.debug("bash: %r exited with status %d", args[0], proc.returncode)
            message = f"Command exited with status {proc.returncode}."
            # interleaved merges stderr into stdout; proc.stderr is None in that mode
            raw_error = (proc.stdout if capture == "interleaved" else proc.stderr) or ""
            if raw_error:
                error_output, _ = _truncate(raw_error.strip(), max_output_chars)
                message = f"{message} {error_output}"
            return self._err("execution_error", message, 400)

        if capture == "separate":
            stdout_text, stdout_truncated = _truncate(
                proc.stdout or "", max_output_chars
            )
            stderr_text, stderr_truncated = _truncate(
                proc.stderr or "", max_output_chars
            )
            data: dict[str, Any] = {"stdout": stdout_text, "stderr": stderr_text}
            if stdout_truncated or stderr_truncated:
                data["warning"] = f"Output truncated at {max_output_chars} characters"
        else:
            raw = (proc.stderr or "") if capture == "stderr" else (proc.stdout or "")
            logger.debug("bash: %r succeeded, output=%d chars", args[0], len(raw))
            text, truncated = _truncate(raw, max_output_chars)
            data = {"output": text}
            if truncated:
                data["warning"] = f"Output truncated at {max_output_chars} characters"

        return self._ok(data)
