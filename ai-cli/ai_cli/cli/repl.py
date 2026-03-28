"""
repl.py — Main REPL loop.

Reads user input via prompt_toolkit, routes slash commands and plain text,
drives the LLM streaming loop (including the agentic tool-call cycle), and
coordinates Session, ToolRegistry, LLMClient, and Display.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import os
import re
import select
import shlex
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from PIL import Image as _PILImage
from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal as _pt_run_in_terminal
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent

from ai_cli.cli.completer import DEFAULT_MAX_PATH_COMPLETIONS, REPLCompleter
from ai_cli.core.llm_client import LLMError
from ai_cli.core.session_manager import SessionError
from ai_cli.core.workspace import _DOT_AI_CLI, get_global_dir

try:
    import termios
    import tty as _tty

    _HAS_TTY = True
except ImportError:  # Windows
    _HAS_TTY = False

if TYPE_CHECKING:
    from ai_cli.cli.display import Display
    from ai_cli.core.config_manager import ConfigManager
    from ai_cli.core.llm_client import LLMClient
    from ai_cli.core.session_manager import Session
    from ai_cli.core.tool_registry import ToolRegistry
    from ai_cli.core.workspace import Workspace

logger = logging.getLogger(__name__)

# Default pixel limit for images attached via @path (width × height).
# Equivalent to one full-HD video frame (1920×1080).
_DEFAULT_MAX_PIXELS: int = 1920 * 1080

# Default maximum number of consecutive tool-call rounds per user turn.
# Overridable via config key ``max_tool_rounds``, ``--max-tool-rounds`` CLI
# flag, or the ``/rounds`` slash command.
_DEFAULT_MAX_TOOL_ROUNDS = 10

# MIME types for recognised image extensions.  Files with these extensions are
# base64-encoded and sent as image_url content blocks instead of text.
_IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Matches @path and @!path references in user input.
# Excludes characters that commonly appear as trailing punctuation in prose
# (commas, brackets, quotes, etc.) so that e.g. "@foo.py," resolves "foo.py".
_AT_RE = re.compile(r"@(!?)([^\s,;:!?()\[\]{}'\"<>]+)")

# Slash commands shown by /help, in display order.
_SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show this help message"),
    ("/exit", "Exit ai-cli"),
    ("/clear", "Clear the current conversation history"),
    ("/verbose", "Toggle verbose mode (show full tool args and results)"),
    ("/markdown", "Toggle Markdown rendering of LLM output"),
    (
        "/compact [instructions]",
        "Summarise the conversation; optional: guide the summary",
    ),
    ("/tools", "List currently enabled tools"),
    ("/tools list", "List all registered tools with enabled/allowed/tier status"),
    ("/tools info <name>", "Show details for a single tool"),
    ("/tools enable|disable [--session] <name>", "Enable or disable a tool"),
    (
        "/tools allow|disallow [--session] <name>",
        "Allow or disallow a tool (hard gate)",
    ),
    ("/session", "Show information about the current session"),
    ("/session name <name>", "Set a display name for this session"),
    ("/history", "Browse the full conversation history in a scrollable view"),
    (
        "/rounds [--session] <N>",
        "Set the maximum tool-call rounds per turn (omit --session to persist)",
    ),
]


def _build_keyboard_shortcuts(*, enable_suspend: bool) -> list[tuple[str, str]]:
    """Return the keyboard-shortcut rows for ``/help``.

    Ctrl+Z is only listed when process suspension is both supported on this
    platform (Unix/tty required) and enabled in config.
    """
    shortcuts: list[tuple[str, str]] = [
        ("Ctrl+C / Esc", "Abort the current response"),
    ]
    if enable_suspend and _HAS_TTY and sys.stdin.isatty():
        shortcuts.append(("Ctrl+Z", "Suspend to background"))
    shortcuts += [
        ("Ctrl+L", "Clear the screen"),
        ("Ctrl+G", "Open current input in $EDITOR"),
    ]
    return shortcuts


# Unique top-level command names (no leading "/", no arguments) derived from
# _SLASH_COMMANDS.  Used to populate the tab completer.
_SLASH_COMMAND_NAMES: list[str] = list(
    dict.fromkeys(cmd.lstrip("/").split()[0] for cmd, _ in _SLASH_COMMANDS)
)


class _AbortMonitor:
    """Background thread that signals an event on ESC or Ctrl+C.

    Puts stdin into cbreak mode (individual keypresses without Enter) using a
    50 ms ``select`` timeout so the ``_stop`` flag is polled quickly without
    burning CPU.  Terminal settings are always restored in ``finally``.

    Only active on Unix when stdin is a real tty.  The monitor is no-op on
    Windows or when stdin is a pipe (e.g. during tests).
    """

    def __init__(self, abort_event: threading.Event) -> None:
        self._abort_event = abort_event
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background watcher thread."""
        self._stop.clear()
        if _HAS_TTY and sys.stdin.isatty():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Signal the watcher to stop and wait briefly for it to exit.

        Joining with a short timeout ensures the terminal settings are
        restored before the caller returns to the next PromptSession prompt.
        Without this, the monitor's ``finally`` block could run after
        prompt_toolkit has already set up its own terminal mode and silently
        clobber it.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.1)

    def pause(self) -> None:
        """Stop the monitor synchronously so stdin is free for an interactive prompt.

        Calls :meth:`stop` and joins the thread so the caller can be certain
        the monitor is not mid-read when the prompt acquires stdin.  The
        terminal is restored to its original settings by :meth:`stop`, so
        ``pt_prompt`` / ``input()`` each configure their own mode.
        """
        self.stop()

    def resume(self) -> None:
        """Restart the monitor after the interactive prompt returns."""
        self.start()

    def _run(self) -> None:
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except termios.error:
            return
        try:
            _tty.setcbreak(fd)
            while not self._stop.is_set():
                try:
                    ready, _, _ = select.select([fd], [], [], 0.05)
                except (InterruptedError, OSError):
                    break
                if ready:
                    ch = os.read(fd, 1)
                    if ch == b"\x03":  # Ctrl+C — always abort immediately.
                        self._abort_event.set()
                        return
                    if ch == b"\x1b":
                        # ESC may be a lone keypress (intended abort) or the
                        # start of an escape sequence (e.g. arrow keys send
                        # ESC [ A).  Peek briefly: if more bytes arrive within
                        # 20 ms it's a sequence — consume them and continue
                        # watching.  If nothing follows, treat it as a lone
                        # ESC and abort.
                        try:
                            seq_ready, _, _ = select.select([fd], [], [], 0.02)
                        except (InterruptedError, OSError):
                            seq_ready = []
                        if not seq_ready:
                            self._abort_event.set()
                            return
                        # Drain up to 8 bytes of the escape sequence.
                        with contextlib.suppress(OSError):
                            os.read(fd, 8)
                        continue
                    # Any other byte is consumed and lost.  This is an
                    # inherent limitation of raw-reading stdin: there is no
                    # portable POSIX way to "push back" bytes to the terminal
                    # input queue (TIOCSTI is deprecated in Linux ≥ 6.2).
                    # In practice users rarely type during LLM streaming, so
                    # the impact is minor.
        finally:
            with contextlib.suppress(termios.error):
                termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _make_key_bindings() -> KeyBindings:
    """Return prompt_toolkit key bindings added to every REPL PromptSession.

    Ctrl+L — clear the terminal screen.
    Ctrl+G — open the current prompt buffer in ``$VISUAL`` / ``$EDITOR``.
    """
    kb = KeyBindings()

    @kb.add("c-l")
    def _clear_screen(event: KeyPressEvent) -> None:
        event.app.renderer.clear()

    @kb.add("c-g")
    def _open_editor(event: KeyPressEvent) -> None:
        buf = event.app.current_buffer
        current_text = buf.text
        editor_str = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
        try:
            editor_parts = shlex.split(editor_str)
        except ValueError:
            editor_parts = [editor_str]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(current_text)
            fname = f.name

        def _run_editor() -> None:
            try:
                subprocess.call([*editor_parts, fname])
                with open(fname, encoding="utf-8") as fh:
                    new_text = fh.read()
                # Strip the trailing newline editors typically append.
                if new_text.endswith("\n"):
                    new_text = new_text[:-1]
                buf.set_document(Document(new_text, len(new_text)))
            except OSError as exc:
                print(f"Could not open editor: {exc}", file=sys.stderr)
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(fname)

        _pt_run_in_terminal(_run_editor)

    return kb


class REPL:
    """
    Main interaction loop.

    Parameters
    ----------
    session:
        The active conversation session.
    tool_registry:
        Registry of available tools; used to get definitions and execute calls.
    llm_client:
        Backend used to stream LLM responses.
    display:
        All user-facing output and interactive prompts.
    workspace:
        Used to resolve and read files referenced via ``@path`` syntax.
    config:
        Layered configuration manager.  Used to read ``max_tool_rounds`` at
        startup and to persist runtime changes (e.g. ``/rounds``) to the
        project config file.  When ``None``, built-in defaults are used and
        persistence is still attempted via the workspace path.
    """

    def __init__(
        self,
        session: Session,
        tool_registry: ToolRegistry,
        llm_client: LLMClient,
        display: Display,
        workspace: Workspace,
        config: ConfigManager | None = None,
    ) -> None:
        self._session = session
        self._tool_registry = tool_registry
        self._llm = llm_client
        self._display = display
        self._workspace = workspace
        self._config = config
        # Maximum tool-call rounds per user turn — readable from config and
        # overridable at runtime via /rounds.
        self._max_tool_rounds: int = _DEFAULT_MAX_TOOL_ROUNDS
        if config is not None:
            raw = config.get("max_tool_rounds", _DEFAULT_MAX_TOOL_ROUNDS)
            parsed: int | None = None
            if isinstance(raw, int) and not isinstance(raw, bool):
                parsed = raw
            elif isinstance(raw, str):
                try:
                    parsed = int(raw)
                except ValueError:
                    parsed = None
            if isinstance(parsed, int) and parsed >= 1:
                self._max_tool_rounds = parsed
            elif raw != _DEFAULT_MAX_TOOL_ROUNDS:
                logger.warning(
                    "Invalid max_tool_rounds value %r in config; using default %d.",
                    raw,
                    _DEFAULT_MAX_TOOL_ROUNDS,
                )
        # Schemas injected by tool_manager.enable for the next API call only.
        # Maps tool name → schema so we can both inject the schema into the
        # tools list AND pass allow_transient=True when executing that tool.
        # Populated during tool execution, consumed and cleared at the next send.
        self._pending_transients: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, *, _prompt_session: PromptSession | None = None) -> None:
        """
        Start the REPL loop.

        Reads input until EOF (Ctrl+D) or ``/exit``.  KeyboardInterrupt
        (Ctrl+C) cancels the current line and re-prompts.

        Parameters
        ----------
        _prompt_session:
            Injected ``PromptSession`` for testing.  When ``None`` (the
            default), a session backed by ``~/.ai-cli/history`` is created.
        """
        if _prompt_session is None:
            history_path = get_global_dir() / "history"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            repl_cfg = self._config.get("repl_behavior", {}) if self._config else {}
            repl_cfg = repl_cfg if isinstance(repl_cfg, dict) else {}
            cwt = bool(repl_cfg.get("complete_while_typing", False))
            suspend = bool(repl_cfg.get("enable_suspend", True))
            raw_max = repl_cfg.get(
                "completion_max_results", DEFAULT_MAX_PATH_COMPLETIONS
            )
            try:
                max_completions = int(raw_max)
                if max_completions < 1:
                    raise ValueError
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid repl_behavior.completion_max_results value %r; "
                    "using default %d.",
                    raw_max,
                    DEFAULT_MAX_PATH_COMPLETIONS,
                )
                max_completions = DEFAULT_MAX_PATH_COMPLETIONS
            _prompt_session = PromptSession(
                history=FileHistory(str(history_path)),
                completer=REPLCompleter(
                    slash_commands=_SLASH_COMMAND_NAMES,
                    tool_registry=self._tool_registry,
                    workspace=self._workspace,
                    max_path_completions=max_completions,
                ),
                complete_while_typing=cwt,
                enable_suspend=suspend,
                key_bindings=_make_key_bindings(),
                **self._display.prompt_session_kwargs(),
            )

        while True:
            try:
                raw = _prompt_session.prompt("> ")
            except KeyboardInterrupt:
                self._display.show_status(
                    "Interrupted. Type /exit or press Ctrl+D to quit."
                )
                continue
            except EOFError:
                break

            raw = raw.strip()
            if not raw:
                continue

            self._handle_input(raw)

    # ------------------------------------------------------------------
    # Input routing
    # ------------------------------------------------------------------

    def _handle_input(self, raw: str) -> None:
        if raw.startswith("/"):
            self._handle_slash_command(raw[1:].strip())
        else:
            user_input = self._preprocess_at_references(raw)
            if user_input is None:
                return  # @ reference failed; error already shown, don't send
            self._send_to_llm(user_input)

    def _handle_slash_command(self, command: str) -> None:
        cmd = command.split()[0].lower() if command.strip() else ""

        if cmd == "help":
            repl_cfg = self._config.get("repl_behavior", {}) if self._config else {}
            repl_cfg = repl_cfg if isinstance(repl_cfg, dict) else {}
            suspend = bool(repl_cfg.get("enable_suspend", True))
            self._display.show_help(
                _SLASH_COMMANDS
                + [("", "")]
                + _build_keyboard_shortcuts(enable_suspend=suspend)
            )

        elif cmd == "exit":
            raise SystemExit(0)

        elif cmd == "clear":
            try:
                self._session.clear()
            except SessionError as exc:
                self._display.show_error(f"Could not clear history: {exc}")
                return
            self._display.show_status("Conversation history cleared.")

        elif cmd == "verbose":
            self._display.toggle_verbose()
            state = "on" if self._display.verbose else "off"
            self._display.show_status(f"Verbose mode {state}.")

        elif cmd == "markdown":
            self._display.toggle_markdown()
            state = "on" if self._display.markdown_enabled else "off"
            self._display.show_status(f"Markdown rendering {state}.")

        elif cmd == "compact":
            instructions = command[len(cmd) :].strip()
            self._display.show_status("Compacting conversation history…")
            try:
                self._session.compact(instructions=instructions)
                self._display.show_status("Compaction complete.")
            except SessionError as exc:
                self._display.show_error(f"Compaction failed: {exc}")

        elif cmd == "tools":
            remainder = command[len(cmd) :].strip()
            self._handle_tools_subcommand(remainder)

        elif cmd == "session":
            remainder = command[len(cmd) :].strip()
            self._handle_session_subcommand(remainder)

        elif cmd == "history":
            try:
                messages = self._session.get_messages()
            except SessionError as exc:
                self._display.show_error(f"Could not load history: {exc}")
                return
            self._display.show_history(messages)

        elif cmd == "rounds":
            self._handle_rounds_subcommand(command[len(cmd) :].strip())

        elif cmd == "":
            self._display.show_error(
                "No command provided. Type /help for a list of commands."
            )

        else:
            self._display.show_error(
                f"Unknown command: /{cmd}. Type /help for a list of commands."
            )

    # ------------------------------------------------------------------
    # /tools subcommand handler
    # ------------------------------------------------------------------

    def _handle_tools_subcommand(self, remainder: str) -> None:
        """Dispatch /tools [subcommand] [args]."""
        parts = remainder.split()
        sub = parts[0].lower() if parts else ""

        if not sub:
            self._display.show_tool_list(self._tool_registry.all_enabled())
            return

        if sub == "list":
            self._display.show_tool_list_all(self._tool_registry.all_tools_info())
            return

        if sub == "info":
            if len(parts) < 2:
                self._display.show_error("Usage: /tools info <name>")
                return
            name = parts[1]
            info = self._tool_registry.tool_info(name)
            if info is None:
                self._display.show_error(f"Unknown tool: '{name}'")
                return
            self._display.show_tool_info(info)
            return

        if sub in ("enable", "disable", "allow", "disallow"):
            rest = parts[1:]
            session_flag = "--session" in rest
            if session_flag:
                rest = [p for p in rest if p != "--session"]
            if not rest:
                self._display.show_error(f"Usage: /tools {sub} [--session] <name>")
                return
            name = rest[0]
            if self._tool_registry.get(name) is None:
                self._display.show_error(f"Unknown tool: '{name}'")
                return
            if sub == "enable":
                if session_flag:
                    self._tool_registry.enable_session(name)
                else:
                    self._tool_registry.enable(name)
            elif sub == "disable":
                if session_flag:
                    self._tool_registry.disable_session(name)
                else:
                    self._tool_registry.disable(name)
            elif sub == "allow":
                if session_flag:
                    self._tool_registry.allow_session(name)
                else:
                    self._tool_registry.allow(name)
            elif sub == "disallow":
                if session_flag:
                    self._tool_registry.disallow_session(name)
                else:
                    self._tool_registry.disallow(name)
            _past = {
                "enable": "enabled",
                "disable": "disabled",
                "allow": "allowed",
                "disallow": "disallowed",
            }
            scope = "this session" if session_flag else "persistently"
            self._display.show_status(f"Tool '{name}': {_past[sub]} {scope}.")
            return

        self._display.show_error(
            f"Unknown /tools subcommand: '{sub}'. "
            "Try /tools, /tools list, /tools info <name>, "
            "or /tools enable|disable|allow|disallow [--session] <name>."
        )

    # ------------------------------------------------------------------
    # /session subcommand handler
    # ------------------------------------------------------------------

    def _handle_session_subcommand(self, remainder: str) -> None:
        """Dispatch /session [subcommand] [args]."""
        parts = remainder.split()
        sub = parts[0].lower() if parts else ""

        if not sub:
            self._display.show_session_info(self._session)
            return

        if sub == "name":
            if len(parts) < 2:
                self._display.show_error("Usage: /session name <new-name>")
                return
            new_name = " ".join(parts[1:])
            try:
                self._session.set_name(new_name)
                self._display.show_status(f"Session name set to '{new_name}'.")
            except SessionError as exc:
                self._display.show_error(f"Could not set session name: {exc}")
            return

        self._display.show_error(
            f"Unknown /session subcommand: '{sub}'. "
            "Try /session or /session name <name>."
        )

    # ------------------------------------------------------------------
    # /rounds subcommand handler
    # ------------------------------------------------------------------

    def _handle_rounds_subcommand(self, remainder: str) -> None:
        """Dispatch /rounds [--session] <N>."""
        parts = remainder.split()
        session_flag = "--session" in parts
        parts = [p for p in parts if p != "--session"]

        if not parts:
            self._display.show_error("Usage: /rounds [--session] <N>")
            return

        try:
            value = int(parts[0])
        except ValueError:
            self._display.show_error(
                f"Invalid value '{parts[0]}': must be a positive integer."
            )
            return

        if value < 1:
            self._display.show_error("max_tool_rounds must be at least 1.")
            return

        self._max_tool_rounds = value
        if not session_flag:
            if self._persist_setting("max_tool_rounds", value):
                self._display.show_status(
                    f"Max tool rounds set to {value} (persisted to project config)."
                )
            else:
                self._display.show_error(
                    f"Max tool rounds set to {value} for this session, "
                    "but could not persist to project config (see logs)."
                )
        else:
            self._display.show_status(
                f"Max tool rounds set to {value} for this session."
            )

    # ------------------------------------------------------------------
    # Config persistence helper
    # ------------------------------------------------------------------

    def _persist_setting(self, key: str, value: object) -> bool:
        """Write a top-level key to the project .ai-cli/config.yaml.

        Returns ``True`` on success, ``False`` if the write failed (the error
        is logged as a warning in that case).
        """
        config_path = self._workspace.root / _DOT_AI_CLI / "config.yaml"
        try:
            if config_path.is_file():
                data: dict = (
                    yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                )
                if not isinstance(data, dict):
                    data = {}
            else:
                data = {}
            data[key] = value
            config_path.write_text(
                yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            logger.warning("Could not persist setting '%s': %s", key, exc)
            return False
        return True

    # ------------------------------------------------------------------
    # @ file reference expansion
    # ------------------------------------------------------------------

    def _preprocess_at_references(self, text: str) -> str | list[dict] | None:
        """
        Replace ``@path`` and ``@!path`` tokens with file content.

        Text files: the token is replaced inline with a ``[file: …]`` block;
        the return value remains a plain ``str``.

        Image files (``.png``, ``.jpg``/``.jpeg``, ``.gif``, ``.webp``): the
        token is removed from the text and the file is base64-encoded into an
        ``image_url`` content block.  When at least one image is present the
        method returns a ``list[dict]`` (OpenAI ``chat/completions`` content
        block array) in the original interleaved order rather than a ``str``.

        Paths are resolved relative to the workspace root.  Absolute paths
        (``@/etc/hosts``) and parent-traversal paths (``@../secret.txt``) are
        intentionally allowed — the user decides what to share with the model.
        Access errors (``PermissionError``, missing files) show a user-facing
        error and return ``None`` to signal that the send should be aborted.

        ``@path`` respects workspace ignore rules for files inside the root.
        ``@!path`` and any path that escapes the root bypass ignore rules.
        Images always bypass ignore rules.

        Image dimensions are checked against ``max_pixels_per_image`` (config,
        default ``1920 × 1080``).  Oversized images produce an error and return
        ``None``.

        Returns ``None`` if any ``@`` reference fails — the caller should
        discard the message and return control to the user.
        """
        max_pixels: int = _DEFAULT_MAX_PIXELS
        if self._config is not None:
            raw_max = self._config.get("max_pixels_per_image", _DEFAULT_MAX_PIXELS)
            if isinstance(raw_max, bool):
                logger.warning(
                    "Boolean 'max_pixels_per_image' config value %r is invalid; "
                    "using default %d.",
                    raw_max,
                    _DEFAULT_MAX_PIXELS,
                )
            else:
                try:
                    parsed_max = int(raw_max)
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid 'max_pixels_per_image' config value %r; using default %d.",
                        raw_max,
                        _DEFAULT_MAX_PIXELS,
                    )
                else:
                    if parsed_max <= 0:
                        logger.warning(
                            "Non-positive 'max_pixels_per_image' value %d; using default %d.",
                            parsed_max,
                            _DEFAULT_MAX_PIXELS,
                        )
                    else:
                        max_pixels = parsed_max

        # We scan matches manually so we can interleave text segments and
        # image blocks in their original order.
        result_blocks: list[dict] = []  # built only when images are present
        current_text_parts: list[str] = []  # text accumulated since last image
        full_text_parts: list[str] = []  # all text (for the no-image path)
        any_images = False
        any_errors = False
        last_end = 0

        def _flush_text() -> None:
            """Emit accumulated text as a content block (if non-empty)."""
            if not current_text_parts:
                return
            combined = "".join(current_text_parts)
            current_text_parts.clear()
            if combined.strip():
                result_blocks.append({"type": "text", "text": combined})

        def _handle_match(match: re.Match[str]) -> str:
            """Process one @ref token; return the text replacement."""
            nonlocal any_images, any_errors
            bypass_ignore = bool(match.group(1))
            path = match.group(2)
            mime = _IMAGE_MIME_TYPES.get(Path(path).suffix.lower())

            # Resolve the path permissively.  Python's pathlib treats a leading
            # "/" as absolute, so ``workspace.root / "/etc/hosts"`` resolves to
            # ``/etc/hosts``.  ``..`` traversal is normalised by ``.resolve()``.
            try:
                abs_path = (self._workspace.root / path).resolve()
            except OSError as exc:
                self._display.show_error(f"@{path}: {exc.strerror or str(exc)}")
                any_errors = True
                return str(match.group(0))

            if mime is not None:
                # Images bypass ignore rules; read bytes, check size, encode.
                try:
                    raw_bytes = abs_path.read_bytes()
                except OSError as exc:
                    self._display.show_error(f"@{path}: {exc.strerror or str(exc)}")
                    any_errors = True
                    return str(match.group(0))

                try:
                    with _PILImage.open(io.BytesIO(raw_bytes)) as img:
                        w, h = img.size
                        if w * h > max_pixels:
                            self._display.show_error(
                                f"@{path}: image too large "
                                f"({w}×{h} = {w * h:,} pixels; "
                                f"limit is {max_pixels:,}). "
                                "Set max_pixels_per_image in config to raise the limit."
                            )
                            any_errors = True
                            return str(match.group(0))
                except Exception as exc:
                    logger.warning(
                        "Could not open image %s to check dimensions: %s", path, exc
                    )
                    self._display.show_error(
                        f"@{path}: could not open image file — {exc}"
                    )
                    any_errors = True
                    return str(match.group(0))

                b64 = base64.b64encode(raw_bytes).decode("ascii")
                _flush_text()
                result_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": "auto",
                        },
                    }
                )
                any_images = True
                return ""  # remove token from surrounding text

            # Text file: apply ignore rules only for within-workspace paths
            # unless the user explicitly bypasses them with @!.
            within_workspace = abs_path.is_relative_to(self._workspace.root)
            if (
                not bypass_ignore
                and within_workspace
                and self._workspace.is_ignored(abs_path)
            ):
                self._display.show_error(
                    f"@{path}: file not found or excluded by ignore rules"
                )
                any_errors = True
                return str(match.group(0))

            try:
                content: str = abs_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                reason = exc.strerror if isinstance(exc, OSError) else str(exc)
                self._display.show_error(f"@{path}: {reason}")
                any_errors = True
                return str(match.group(0))
            return f"[file: {path}]\n{content}\n[/file]"

        for match in _AT_RE.finditer(text):
            prefix = text[last_end : match.start()]
            if prefix:
                current_text_parts.append(prefix)
                full_text_parts.append(prefix)

            replacement = _handle_match(match)

            if replacement:
                current_text_parts.append(replacement)
                full_text_parts.append(replacement)

            last_end = match.end()

        # Trailing text after the last match
        tail = text[last_end:]
        if tail:
            current_text_parts.append(tail)
            full_text_parts.append(tail)

        if any_errors:
            return None

        if not any_images:
            return "".join(full_text_parts)

        # Flush any remaining text, then return the interleaved block list.
        _flush_text()
        return result_blocks

    # ------------------------------------------------------------------
    # LLM streaming and agentic tool loop
    # ------------------------------------------------------------------

    def _send_to_llm(self, user_input: str | list[dict]) -> None:
        try:
            if isinstance(user_input, list):
                self._session.add_raw_message({"role": "user", "content": user_input})
            else:
                self._session.add_message("user", user_input)
        except SessionError as exc:
            self._display.show_error(f"Could not save message: {exc}")
            return

        # Single abort event for the entire multi-round exchange.  The
        # _AbortMonitor background thread sets it on ESC or Ctrl+C; the main
        # thread sets it on KeyboardInterrupt.
        abort = threading.Event()
        monitor = _AbortMonitor(abort)

        # Wrap the permission prompt so the monitor yields stdin while the
        # user is responding.  Without this, _AbortMonitor races with
        # pt_prompt for the same fd: it may consume the CPR response
        # (ESC[...R) as a lone ESC and spuriously set the abort flag, or
        # swallow the user's keypress entirely.
        pm = self._tool_registry.permission_manager
        _orig_prompt_fn = pm.prompt_fn

        def _paused_prompt_fn(question: str, extras: list[str]) -> tuple[str, str]:
            monitor.pause()
            try:
                return _orig_prompt_fn(question, extras)
            finally:
                monitor.resume()

        try:
            pm.prompt_fn = _paused_prompt_fn
            monitor.start()
            try:
                self._send_rounds(user_input, abort)
            finally:
                monitor.stop()
        finally:
            pm.prompt_fn = _orig_prompt_fn

    def _send_rounds(
        self, user_input: str | list[dict], abort: threading.Event
    ) -> None:
        """Inner loop that drives multi-round tool calls.  Separated so that
        the abort monitor's ``try/finally`` in ``_send_to_llm`` stays clean."""
        for _ in range(self._max_tool_rounds):
            if abort.is_set():
                self._display.show_status("Aborted.")
                return

            tool_calls: list[dict] = []
            text_parts: list[str] = []

            try:
                messages = self._session.get_messages()
            except SessionError as exc:
                self._display.show_error(f"Could not read conversation history: {exc}")
                return

            # Consume transients injected by tool_manager.enable in the previous
            # round, then clear so they don't persist beyond this round.
            active_transients = dict(self._pending_transients)
            self._pending_transients.clear()

            self._display.begin_assistant_turn()
            stream = None
            try:
                # Build the tools list, de-duplicating by name so that a
                # transient schema for an already-enabled tool doesn't appear
                # twice (some LLM APIs reject duplicate tool names).
                # Transient schemas take precedence over the enabled definitions.
                # Use .get() defensively — malformed definitions are skipped.
                tools_by_name: dict[str, dict] = {}
                for defn in self._tool_registry.definitions():
                    func = defn.get("function")
                    fname = func.get("name") if isinstance(func, dict) else None
                    if fname:
                        tools_by_name[fname] = defn
                    else:
                        logger.warning(
                            "Skipping tool schema with missing name: %r", defn
                        )
                tools_by_name.update(active_transients)
                stream = self._llm.send(
                    messages,
                    tools=list(tools_by_name.values()),
                )
                for chunk in stream:
                    # NOTE: abort is only checked *between* chunks.  If the
                    # iterator is blocked waiting for the server (e.g. no
                    # tokens have arrived yet), ESC/Ctrl+C won't interrupt
                    # the blocking read until the next chunk arrives.  A
                    # complete fix requires running the stream in a worker
                    # thread or plumbing a cancel token into LLMClient.send().
                    if abort.is_set():
                        break
                    if chunk["type"] == "text":
                        self._display.stream_text(chunk["delta"])
                        text_parts.append(chunk["delta"])
                    elif chunk["type"] == "reasoning":
                        self._display.stream_reasoning(chunk["delta"])
                    elif chunk["type"] == "tool_call":
                        tool_calls.append(chunk)
                    elif chunk["type"] == "done":
                        usage = chunk.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens")
                        if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                            self._session.record_usage(prompt_tokens)
                        context_window = self._llm.get_model_metadata().get(
                            "context_window", 0
                        )
                        self._display.update_usage(usage, context_window)
            except KeyboardInterrupt:
                abort.set()
            except LLMError as exc:
                self._display.show_error(f"LLM error: {exc}")
                return
            finally:
                # Explicitly close the stream so the underlying HTTP connection
                # is released immediately rather than waiting for GC.
                if stream is not None:
                    with contextlib.suppress(Exception):
                        stream.close()
                self._display.end_assistant_turn()

            if abort.is_set():
                self._display.show_status("Aborted.")
                return

            full_text = "".join(text_parts)

            if tool_calls:
                # Persist the assistant turn as a proper tool-call message so the
                # LLM can associate each tool result with its originating request.
                assistant_msg: dict = {
                    "role": "assistant",
                    "content": full_text or None,
                    "tool_calls": [
                        {
                            "id": call["call_id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in tool_calls
                    ],
                }
                try:
                    self._session.add_raw_message(assistant_msg)
                except SessionError as exc:
                    self._display.show_error(f"Could not save assistant message: {exc}")
                    return
            elif full_text:
                try:
                    self._session.add_message("assistant", full_text)
                except SessionError as exc:
                    self._display.show_error(f"Could not save assistant message: {exc}")
                    return

            if not tool_calls:
                break

            for i, call in enumerate(tool_calls):
                if abort.is_set():
                    # The assistant message (with tool_calls) was already saved.
                    # Inject stub results for every unexecuted call so the
                    # session history stays valid — the API requires a role:tool
                    # result for every call_id in the preceding assistant turn.
                    for pending in tool_calls[i:]:
                        try:
                            self._session.add_raw_message(
                                {
                                    "role": "tool",
                                    "tool_call_id": pending["call_id"],
                                    "content": json.dumps(
                                        {
                                            "status": "error",
                                            "error": "aborted",
                                            "message": "Aborted by user.",
                                            "code": 499,
                                        }
                                    ),
                                }
                            )
                        except SessionError as exc:
                            logger.error(
                                "Failed to inject abort stub for call_id=%r: %s",
                                pending["call_id"],
                                exc,
                            )
                    self._display.show_status("Aborted.")
                    return
                self._display.show_tool_call(call["name"], call["arguments"])
                # Transiently-enabled tools must bypass the registry's enabled
                # check — they were injected into the LLM's tools list for this
                # round specifically, so allow_transient=True lets them execute.
                allow_transient = call["name"] in active_transients
                result = self._tool_registry.execute(
                    call["name"], call["arguments"], allow_transient=allow_transient
                )
                # Disallowed tools: show a user-facing hint but replace the
                # result with a generic unknown-tool error before it reaches
                # the LLM — the agent must not learn that the tool exists.
                if result.get("error") == "tool_disallowed":
                    self._display.show_error(
                        f"Tool '{call['name']}' is not available in the current "
                        f"configuration. Use '/tools allow {call['name']}' to add it to the list of available tools."
                    )
                    result = {
                        "status": "error",
                        "error": "unknown_tool",
                        "message": f"No tool named '{call['name']}'.",
                        "code": 404,
                    }
                # Only tool_manager may inject transient schemas; any other tool
                # returning this key is ignored.  Each schema is also validated
                # against the registry so only known tools can be transiently
                # enabled — arbitrary schemas cannot bypass the enable gate.
                # Pop the key regardless so it never enters conversation history.
                # Guard against malformed result shapes from user-defined tools.
                data = result.get("data")
                if not isinstance(data, dict):
                    data = None
                if call["name"] == "tool_manager" and result.get("status") == "success":
                    schemas = (
                        data.pop("transient_schemas", None)
                        if data is not None
                        else None
                    )
                    if isinstance(schemas, list):
                        for schema in schemas:
                            if not isinstance(schema, dict):
                                continue
                            func = schema.get("function")
                            if not isinstance(func, dict):
                                continue
                            name = func.get("name")
                            if name and self._tool_registry.get(name) is not None:
                                self._pending_transients[name] = schema
                elif data is not None:
                    data.pop("transient_schemas", None)
                display_str: str | None = None
                tool_obj = self._tool_registry.get(call["name"])
                if tool_obj is not None:
                    with contextlib.suppress(Exception):
                        display_str = tool_obj.format_display(
                            args=call["arguments"], result=result
                        )
                self._display.show_tool_result(call["name"], result, display_str)
                try:
                    self._session.add_raw_message(
                        {
                            "role": "tool",
                            "tool_call_id": call["call_id"],
                            "content": json.dumps(result, default=str),
                        }
                    )
                except SessionError as exc:
                    self._display.show_error(f"Could not save tool result: {exc}")
                    return
        else:
            logger.warning(
                "Tool call limit (%d rounds) reached for session; stopping.",
                self._max_tool_rounds,
            )
            self._display.show_error(
                f"Tool call limit ({self._max_tool_rounds} rounds) reached. Stopping."
            )

        self._check_compaction()

    # ------------------------------------------------------------------
    # Auto-compaction
    # ------------------------------------------------------------------

    def _check_compaction(self) -> None:
        try:
            if self._session.should_compact():
                self._display.show_status("Context window nearing limit — compacting…")
                self._session.compact()
                self._display.show_status("Compaction complete.")
        except SessionError as exc:
            self._display.show_error(f"Compaction failed: {exc}")
