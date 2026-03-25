"""
completer.py — prompt_toolkit Completer for the ai-cli REPL.

Provides completions for:
- Slash command names and their subcommands / flags
- Tool names for ``/tools info|enable|disable|allow|disallow``
- File paths for ``@path`` references (workspace-relative, ``../``, and
  absolute ``/`` paths are all supported)
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

if TYPE_CHECKING:
    from ai_cli.core.tool_registry import ToolRegistry
    from ai_cli.core.workspace import Workspace


# Matches a partial @ reference at the very end of the text before the cursor.
# Group 1: optional '!' (bypass-ignore flag)
# Group 2: the partial path being typed (may be empty)
_AT_PARTIAL_RE = re.compile(r"@(!?)([^\s,;:!?()\[\]{}'\"<>]*)$")

_TOOLS_SUBCOMMANDS = ["allow", "disable", "disallow", "enable", "info", "list"]
_TOOLS_NAME_SUBCMDS = frozenset({"allow", "disable", "disallow", "enable", "info"})
_TOOLS_FLAG_SUBCMDS = frozenset({"allow", "disable", "disallow", "enable"})
_SESSION_SUBCOMMANDS = ["name"]

# Default cap on file-path completions to keep the UI responsive.
# Overridable via ``repl_behavior.completion_max_results`` in config.
DEFAULT_MAX_PATH_COMPLETIONS = 200


class REPLCompleter(Completer):
    """Tab completer for the ai-cli REPL.

    Parameters
    ----------
    slash_commands:
        Top-level slash command names *without* the leading ``/``
        (e.g. ``["help", "exit", "tools", ...]``).
    tool_registry:
        Live registry used to enumerate tool names for ``/tools`` subcommands.
        Queried lazily on each completion request.
    workspace:
        Workspace used to resolve and filter file-path completions for
        ``@path`` references.
    max_path_completions:
        Maximum number of ``@path`` completions returned per keystroke.
        Defaults to :data:`DEFAULT_MAX_PATH_COMPLETIONS`.
    """

    def __init__(
        self,
        slash_commands: list[str],
        tool_registry: ToolRegistry | None = None,
        workspace: Workspace | None = None,
        max_path_completions: int = DEFAULT_MAX_PATH_COMPLETIONS,
    ) -> None:
        if max_path_completions < 1:
            raise ValueError(
                f"max_path_completions must be >= 1, got {max_path_completions}"
            )
        self._slash_commands = sorted(slash_commands)
        self._tool_registry = tool_registry
        self._workspace = workspace
        self._max_path_completions = max_path_completions

    # ------------------------------------------------------------------
    # Completer API
    # ------------------------------------------------------------------

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor

        # Check for a partial @ reference at the cursor position first, so
        # that e.g. "@/foo" doesn't accidentally trigger slash completion.
        at_m = _AT_PARTIAL_RE.search(text)
        if at_m:
            bypass_ignore = bool(at_m.group(1))
            yield from self._complete_path(at_m.group(2), bypass_ignore=bypass_ignore)
            return

        stripped = text.lstrip()
        if stripped.startswith("/"):
            yield from self._complete_slash(stripped)

    # ------------------------------------------------------------------
    # Slash command completion
    # ------------------------------------------------------------------

    def _complete_slash(self, text: str) -> Iterable[Completion]:
        """Yield completions for *text* which starts with ``/``."""
        parts = text.split()

        # Still typing the top-level command word (no trailing space yet).
        if len(parts) == 1 and not text.endswith(" "):
            prefix = parts[0][1:].lower()  # strip "/" and normalise case
            for cmd in self._slash_commands:
                if cmd.startswith(prefix):
                    yield Completion(
                        "/" + cmd,
                        start_position=-len(parts[0]),
                        display="/" + cmd,
                    )
            return

        if not parts:
            return

        cmd = parts[0][1:].lower()
        if cmd == "tools":
            yield from self._complete_tools(parts, text)
        elif cmd == "session":
            yield from self._complete_session(parts, text)
        elif cmd == "rounds":
            yield from self._complete_rounds(parts, text)

    def _complete_tools(self, parts: list[str], text: str) -> Iterable[Completion]:
        trailing = text.endswith(" ")

        # Completing the subcommand.
        if len(parts) == 1 or (len(parts) == 2 and not trailing):
            prefix = (parts[1] if len(parts) > 1 else "").lower()
            raw_len = len(parts[1]) if len(parts) > 1 else 0
            for sub in _TOOLS_SUBCOMMANDS:
                if sub.startswith(prefix):
                    yield Completion(sub, start_position=-raw_len)
            return

        subcmd = parts[1].lower()
        if subcmd not in _TOOLS_NAME_SUBCMDS:
            return

        after = parts[2:]  # tokens after "/tools <subcmd>"

        if subcmd in _TOOLS_FLAG_SUBCMDS:
            # /tools enable|disable|allow|disallow [--session] <name>
            if not after or (len(after) == 1 and not trailing):
                prefix = after[0] if after else ""
                yield from self._flag_then_tool_names(prefix)
                return
            if after[0] == "--session" and (
                len(after) == 1 or (len(after) == 2 and not trailing)
            ):
                # --session already present; next token is the tool name.
                prefix = after[1] if len(after) > 1 else ""
                for name in self._tool_names():
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))
        else:
            # /tools info <name>
            if not after or (len(after) == 1 and not trailing):
                prefix = after[0] if after else ""
                for name in self._tool_names():
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))

    def _flag_then_tool_names(self, prefix: str) -> Iterable[Completion]:
        """Yield ``--session`` then all matching tool names for *prefix*."""
        if "--session".startswith(prefix):
            yield Completion("--session", start_position=-len(prefix))
        for name in self._tool_names():
            if name.startswith(prefix):
                yield Completion(name, start_position=-len(prefix))

    def _complete_session(self, parts: list[str], text: str) -> Iterable[Completion]:
        trailing = text.endswith(" ")
        if len(parts) == 1 or (len(parts) == 2 and not trailing):
            prefix = (parts[1] if len(parts) > 1 else "").lower()
            raw_len = len(parts[1]) if len(parts) > 1 else 0
            for sub in _SESSION_SUBCOMMANDS:
                if sub.startswith(prefix):
                    yield Completion(sub, start_position=-raw_len)

    def _complete_rounds(self, parts: list[str], text: str) -> Iterable[Completion]:
        trailing = text.endswith(" ")
        # Only offer --session (the numeric argument is free-form).
        if len(parts) == 1 or (len(parts) == 2 and not trailing):
            prefix = parts[1] if len(parts) > 1 else ""
            if "--session".startswith(prefix):
                yield Completion("--session", start_position=-len(prefix))

    # ------------------------------------------------------------------
    # @ file-path completion
    # ------------------------------------------------------------------

    def _complete_path(
        self, partial: str, bypass_ignore: bool = False
    ) -> Iterable[Completion]:
        """Yield file-path completions for a partial ``@``-reference.

        ``..`` and absolute paths (``/``) are allowed — Python's pathlib
        treats a leading ``/`` as absolute when joined, so
        ``root / "/etc"`` resolves to ``/etc``.  Ignore rules are only
        applied for paths inside the workspace root; paths that escape it
        (or when *bypass_ignore* is ``True``) are listed as-is.
        """
        if self._workspace is None:
            return

        root = self._workspace.root

        # Split into a confirmed directory prefix and the stem being typed.
        if "/" in partial:
            dir_part, stem = partial.rsplit("/", 1)
        else:
            dir_part = ""
            stem = partial

        # pathlib: root / "/absolute" == Path("/absolute"); root / ".." goes up.
        if dir_part:
            try:
                base_dir = (root / dir_part).resolve()
            except OSError:
                return
        else:
            base_dir = root

        if not base_dir.is_dir():
            return

        within_workspace = base_dir.is_relative_to(root)

        count = 0
        try:
            with os.scandir(base_dir) as it:
                for entry in it:
                    if not entry.name.startswith(stem):
                        continue

                    try:
                        is_dir = entry.is_dir(follow_symlinks=True)
                    except OSError:
                        continue
                    abs_path = base_dir / entry.name

                    if not bypass_ignore and within_workspace:
                        try:
                            if self._workspace.is_ignored(abs_path, is_dir=is_dir):
                                continue
                        except OSError:
                            continue

                    rel = f"{dir_part}/{entry.name}" if dir_part else entry.name
                    suffix = "/" if is_dir else ""
                    yield Completion(
                        rel + suffix,
                        start_position=-len(partial),
                        display=entry.name + suffix,
                    )
                    count += 1
                    if count >= self._max_path_completions:
                        break
        except OSError:
            return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tool_names(self) -> list[str]:
        if self._tool_registry is None:
            return []
        return sorted(t["name"] for t in self._tool_registry.all_tools_info())
