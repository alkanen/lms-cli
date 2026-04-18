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
import logging
import math
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

from ai_cli.cli.completer import (
    DEFAULT_MAX_PATH_COMPLETIONS,
    REPLCompleter,
    _tokenize_command,
)
from ai_cli.core.agent import Agent, AgentSpec
from ai_cli.core.session_manager import SessionError
from ai_cli.core.skill_registry import SkillRegistry
from ai_cli.core.workspace import _DOT_AI_CLI, get_global_dir

try:
    import termios
    import tty as _tty

    _HAS_TTY = True
except ImportError:  # Windows
    _HAS_TTY = False

if TYPE_CHECKING:
    from ai_cli.cli.display import Display
    from ai_cli.core.agent_registry import AgentRegistry
    from ai_cli.core.config_manager import ConfigManager
    from ai_cli.core.llm_client import LLMClient
    from ai_cli.core.mcp_manager import MCPManager
    from ai_cli.core.session_manager import Session
    from ai_cli.core.task_manager import TaskManager
    from ai_cli.core.task_orchestrator import TaskOrchestrator
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

# Past-tense labels for /tools and /mcp enable/disable/allow/disallow status
# messages. Hoisted to module scope so tool- and server-level branches cannot
# drift apart if wording changes later.
_PERMISSION_PAST_TENSE: dict[str, str] = {
    "enable": "enabled",
    "disable": "disabled",
    "allow": "allowed",
    "disallow": "disallowed",
}

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
    (
        "/index [path] [--file <path>] [--label <name>] [--full] [--remove]",
        "Index files for semantic search; [path] adds a root, --file indexes a single file",
    ),
    ("/mcp", "List configured MCP servers and their connection status"),
    ("/mcp list", "List all MCP servers with status and tool count"),
    ("/mcp info <server>", "Show tools exposed by an MCP server"),
    (
        "/mcp enable|disable [--persist] <server> [<tool>]",
        "Enable or disable an MCP server or individual tool",
    ),
    (
        "/mcp allow|disallow [--persist] <server> [<tool>]",
        "Allow or disallow an MCP server or individual tool (hard gate)",
    ),
    ("/skills", "List loaded skill names"),
    ("/skills list", "List loaded skills with descriptions"),
    ("/skills info <name>", "Show full details for a loaded skill"),
    ("/skills reload", "Rescan skills and refresh runtime skill mappings"),
    ("/agents", "List configured agent types"),
    (
        "/plan [goal] [--autonomous]",
        "Start or resume plan → execute → review loop (--autonomous skips plan review checkpoint)",
    ),
    ("/tasks", "List unfinished root tasks"),
    ("/tasks list [<path>]", "List all tasks (or children of <path>) with detail"),
    ("/tasks tree [<path>] [--depth <n>]", "Show task tree"),
    ("/tasks info <path>", "Show full detail for a task"),
    ("/tasks add [<path>]", "Wizard: create a task (or subtask under <path>)"),
    ("/tasks edit <path>", "Wizard: edit a task"),
    ("/tasks delete [<path>]", "Delete task (or all tasks+goal if no path)"),
    (
        '/tasks note obsolete <path> <index> [--reason "<text>"]',
        "Mark an active note obsolete by index",
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


def _skill_aliases_for_registry(
    registry: SkillRegistry,
) -> tuple[dict[str, str], list[str]]:
    """Return direct slash aliases for skills plus any collision warnings."""
    aliases: dict[str, str] = {}
    warnings: list[str] = []
    reserved = set(_SLASH_COMMAND_NAMES)
    for name in sorted(registry.names()):
        if name in reserved:
            warnings.append(
                f"Skill '{name}': slash alias '/{name}' conflicts with an existing command and was skipped."
            )
            continue
        aliases[name] = name
    return aliases, warnings


def _levenshtein_distance(left: str, right: str) -> int:
    """Compute Levenshtein edit distance between two short command names."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


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
        agent_registry: AgentRegistry | None = None,
        task_manager: TaskManager | None = None,
        mcp_manager: MCPManager | None = None,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self._session = session
        self._tool_registry = tool_registry
        self._llm = llm_client
        self._display = display
        self._workspace = workspace
        self._config = config
        self._agent_registry = agent_registry
        self._task_manager = task_manager
        self._mcp_manager = mcp_manager
        self._skill_registry = (
            skill_registry if skill_registry is not None else SkillRegistry({})
        )
        self._skill_aliases, self._skill_alias_warnings = _skill_aliases_for_registry(
            self._skill_registry
        )
        # Orchestrator instance — created on first /plan and reused across calls.
        self._orchestrator: TaskOrchestrator | None = None
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
        # Coordinator agent wrapping the send/tool-call loop.
        coordinator_spec = AgentSpec(
            name="coordinator",
            system_message="",  # not used — system message is in the session
            tools=[],  # not used — registry already built
            model="",  # not used — llm_client already configured
            max_response_tokens=0,  # not used
            max_tool_rounds=self._max_tool_rounds,
        )
        self._main_agent = Agent(
            spec=coordinator_spec,
            session=self._session,
            llm_client=self._llm,
            tool_registry=self._tool_registry,
            display=self._display,
        )

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
                    task_manager=self._task_manager,
                    mcp_manager=self._mcp_manager,
                    skill_registry_getter=lambda: self._skill_registry,
                    skill_aliases_getter=lambda: self._skill_aliases,
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

        skill_name = self._skill_aliases.get(cmd)
        if skill_name is not None:
            remainder = command[len(cmd) :].strip()
            self._invoke_skill_alias(skill_name, remainder)
            return

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

        elif cmd == "index":
            self._handle_index_command(command[len(cmd) :].strip())

        elif cmd == "mcp":
            remainder = command[len(cmd) :].strip()
            self._handle_mcp_subcommand(remainder)

        elif cmd == "skills":
            remainder = command[len(cmd) :].strip()
            self._handle_skills_subcommand(remainder)

        elif cmd == "agents":
            self._handle_agents_command()

        elif cmd == "plan":
            remainder = command[len(cmd) :].strip()
            self._handle_plan_command(remainder)

        elif cmd == "tasks":
            remainder = command[len(cmd) :].strip()
            self._handle_tasks_subcommand(remainder)

        elif cmd == "":
            self._display.show_error(
                "No command provided. Type /help for a list of commands."
            )

        else:
            suggestions = self._suggest_slash_commands(cmd)
            suggestion_text = ""
            if suggestions:
                if len(suggestions) == 1:
                    suggestion_text = f" Did you mean /{suggestions[0]}?"
                else:
                    joined = ", ".join(f"/{name}" for name in suggestions)
                    suggestion_text = f" Did you mean one of: {joined}?"
            self._display.show_error(
                f"Unknown command: /{cmd}.{suggestion_text} Type /help for a list of commands."
            )

    def _invoke_skill_alias(self, skill_name: str, remainder: str) -> None:
        """Inject a user message that instructs the model to load and apply a skill."""
        user_request: str | list[dict] = remainder
        if remainder:
            processed = self._preprocess_at_references(remainder)
            if processed is None:
                return
            user_request = processed

        injected = (
            f"Use the skill named '{skill_name}' for this request. "
            f"Call the skills tool with name='{skill_name}' before responding, then follow that skill's instructions."
        )
        if isinstance(user_request, list):
            blocks = [{"type": "text", "text": injected}]  # regular user message
            if user_request:
                blocks[0]["text"] += "\n\nUser request:"
                blocks.extend(user_request)
            self._send_to_llm(blocks)
            return

        if user_request:
            injected += f"\n\nUser request:\n{user_request}"
        self._send_to_llm(injected)

    def _suggest_slash_commands(self, cmd: str) -> list[str]:
        """Return the closest slash commands/aliases under the configured threshold."""
        if not cmd:
            return []
        candidates = sorted(set(_SLASH_COMMAND_NAMES) | set(self._skill_aliases))
        threshold = max(2, math.ceil(len(cmd) * 0.2))
        matches: list[tuple[int, str]] = []
        for candidate in candidates:
            distance = _levenshtein_distance(cmd, candidate)
            if distance <= threshold:
                matches.append((distance, candidate))
        if not matches:
            return []
        matches.sort(key=lambda item: (item[0], item[1]))
        best_distance = matches[0][0]
        return [
            candidate for distance, candidate in matches if distance == best_distance
        ][:3]

    # ------------------------------------------------------------------
    # /skills subcommand handler
    # ------------------------------------------------------------------

    def _handle_skills_subcommand(self, remainder: str) -> None:
        """Dispatch /skills [subcommand] [args]."""
        try:
            parts = shlex.split(remainder)
        except ValueError as exc:
            self._display.show_error(f"Could not parse /skills arguments: {exc}")
            return

        sub = parts[0].lower() if parts else ""

        if not sub:
            self._display.show_skills_simple(self._skill_rows())
            return

        if sub == "list":
            if len(parts) != 1:
                self._display.show_error("Usage: /skills list")
                return
            self._display.show_skills_list(self._skill_rows())
            return

        if sub == "info":
            if len(parts) != 2:
                self._display.show_error("Usage: /skills info <name>")
                return
            skill = self._skill_registry.get(parts[1])
            if skill is None:
                self._display.show_error(f"Unknown skill: '{parts[1]}'")
                return
            self._display.show_skill_info(
                {
                    "name": skill.name,
                    "description": skill.description,
                    "instructions": skill.instructions,
                }
            )
            return

        if sub == "reload":
            if len(parts) != 1:
                self._display.show_error("Usage: /skills reload")
                return
            self._reload_skills()
            return

        self._display.show_error(
            f"Unknown /skills subcommand: '{sub}'. "
            "Try /skills, /skills list, /skills info <name>, or /skills reload."
        )

    def _skill_rows(self) -> list[dict]:
        rows = []
        for name, spec in sorted(self._skill_registry.items()):
            rows.append({"name": name, "description": spec.description})
        return rows

    def _reload_skills(self) -> None:
        try:
            skills = SkillRegistry.load(
                self._workspace.root, global_dir=get_global_dir()
            )
            aliases, alias_warnings = _skill_aliases_for_registry(skills)
            new_skills_tool = None
            if skills.has_skills:
                from ai_cli.tools.skills import SkillsTool

                new_skills_tool = SkillsTool(
                    skills,
                    self._workspace,
                    self._tool_registry.permission_manager,
                )
        except Exception as exc:
            self._display.show_error(f"Failed to reload skills: {exc}")
            return

        old_skill_registry = self._skill_registry
        old_skill_aliases = self._skill_aliases
        old_skill_alias_warnings = self._skill_alias_warnings
        old_skills_tool = self._tool_registry.get("skills")
        read_file_tool = self._tool_registry.get("read_file")
        set_skill_registry = None
        if read_file_tool is not None:
            set_skill_registry = getattr(read_file_tool, "set_skill_registry", None)

        try:
            self._skill_registry = skills
            self._skill_aliases = aliases
            self._skill_alias_warnings = alias_warnings
            if callable(set_skill_registry):
                set_skill_registry(skills if skills.has_skills else None)
            self._tool_registry.unregister("skills")
            if new_skills_tool is not None:
                self._tool_registry.register_instance(new_skills_tool)
        except Exception as exc:
            self._skill_registry = old_skill_registry
            self._skill_aliases = old_skill_aliases
            self._skill_alias_warnings = old_skill_alias_warnings
            if callable(set_skill_registry):
                with contextlib.suppress(Exception):
                    set_skill_registry(
                        old_skill_registry if old_skill_registry.has_skills else None
                    )
            with contextlib.suppress(Exception):
                self._tool_registry.unregister("skills")
                if old_skills_tool is not None:
                    self._tool_registry.register_instance(old_skills_tool)
            self._display.show_error(f"Failed to reload skills: {exc}")
            return

        for warning in skills.warnings:
            self._display.show_status(f"Warning: {warning}")
        for warning in alias_warnings:
            self._display.show_status(f"Warning: {warning}")
        self._display.show_status(f"Skills reloaded: {len(skills)} skill(s) loaded.")

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
            scope = "this session" if session_flag else "persistently"
            self._display.show_status(
                f"Tool '{name}': {_PERMISSION_PAST_TENSE[sub]} {scope}."
            )
            return

        self._display.show_error(
            f"Unknown /tools subcommand: '{sub}'. "
            "Try /tools, /tools list, /tools info <name>, "
            "or /tools enable|disable|allow|disallow [--session] <name>."
        )

    # ------------------------------------------------------------------
    # /mcp subcommand handler
    # ------------------------------------------------------------------

    def _handle_mcp_subcommand(self, remainder: str) -> None:
        """Dispatch /mcp [subcommand] [args]."""
        if self._mcp_manager is None:
            self._display.show_status("No MCP servers configured.")
            return

        parts = remainder.split()
        sub = parts[0].lower() if parts else ""

        if not sub or sub == "list":
            statuses = self._mcp_manager.status()
            if not statuses:
                self._display.show_status("No MCP servers configured.")
                return
            lines = ["MCP Servers"]
            for s in statuses:
                if s.connected:
                    lines.append(f"  {s.name:<20}  connected    {s.tool_count} tool(s)")
                else:
                    lines.append(f"  {s.name:<20}  ERROR: {s.error}")
            self._display.show_status("\n".join(lines))
            return

        if sub == "info":
            if len(parts) < 2:
                self._display.show_error("Usage: /mcp info <server>")
                return
            server_name = parts[1]
            if server_name not in self._mcp_manager.server_names():
                self._display.show_error(f"Unknown MCP server: '{server_name}'")
                return
            tools = self._mcp_manager.get_server_tools(server_name)
            if not tools:
                self._display.show_status(f"{server_name}: no tools discovered.")
                return
            lines = [f"{server_name}"]
            for t in tools:
                ns_name = f"{server_name}__{t}"
                tool_obj = self._tool_registry.get(ns_name)
                desc = tool_obj.description if tool_obj else ""
                lines.append(f"  {ns_name:<45}  {desc}")
            self._display.show_status("\n".join(lines))
            return

        if sub in ("enable", "disable", "allow", "disallow"):
            rest = parts[1:]
            persist = bool(rest and rest[0] == "--persist")
            if persist:
                rest = rest[1:]
            if not rest:
                self._display.show_error(
                    f"Usage: /mcp {sub} [--persist] <server> [<tool>]"
                )
                return
            server_name = rest[0]
            if server_name not in self._mcp_manager.server_names():
                self._display.show_error(f"Unknown MCP server: '{server_name}'")
                return
            tool_name = rest[1] if len(rest) > 1 else None

            from ai_cli.core.mcp_manager import MCPError

            try:
                if tool_name is not None:
                    known = self._mcp_manager.get_server_tools(server_name)
                    if tool_name not in known:
                        self._display.show_error(
                            f"Unknown tool '{tool_name}' for server '{server_name}'"
                        )
                        return
                    getattr(self._mcp_manager, f"{sub}_tool")(
                        server_name, tool_name, persist=persist
                    )
                    scope = "persistently" if persist else "this session"
                    self._display.show_status(
                        f"MCP tool '{server_name}__{tool_name}': "
                        f"{_PERMISSION_PAST_TENSE[sub]} {scope}."
                    )
                else:
                    getattr(self._mcp_manager, f"{sub}_server")(
                        server_name, persist=persist
                    )
                    scope = "persistently" if persist else "this session"
                    self._display.show_status(
                        f"MCP server '{server_name}': all tools "
                        f"{_PERMISSION_PAST_TENSE[sub]} {scope}."
                    )
            except MCPError as exc:
                self._display.show_error(f"MCP error: {exc}")
            return

        self._display.show_error(
            f"Unknown /mcp subcommand: '{sub}'. "
            "Try /mcp list, /mcp info <server>, "
            "or /mcp enable|disable|allow|disallow [--persist] <server> [<tool>]."
        )

    # ------------------------------------------------------------------
    # /agents command handler
    # ------------------------------------------------------------------

    def _handle_agents_command(self) -> None:
        """Display configured agent types."""
        if self._agent_registry is None or not self._agent_registry.has_agents:
            self._display.show_status("No agent types configured.")
            return
        rows = []
        for name, spec in sorted(self._agent_registry.specs.items()):
            rows.append(
                {
                    "name": name,
                    "model": spec.model,
                    "persistence": spec.persistence,
                    "tools": ", ".join(spec.tools) or "(none)",
                    "max_tool_rounds": spec.max_tool_rounds,
                }
            )
        self._display.show_agents(rows)

    # ------------------------------------------------------------------
    # /plan command handler
    # ------------------------------------------------------------------

    def _handle_plan_command(self, remainder: str) -> None:
        """Start or resume the plan → execute → review loop.

        ``/plan "goal"``             — sets goal, plans, then pauses for review.
        ``/plan``                    — resumes using the stored goal.
        ``/plan --autonomous``       — skip plan checkpoint; run unattended.
        ``/plan "goal" --autonomous``— set goal and run unattended.

        Each step prints a one-line status summarising what the planner /
        executor / reviewer just did (e.g. ``Step 0: planning — created 4
        task(s)``).  If the loop exits early with a "planner could not make
        progress" message, the planner agent likely lacks the tools it needs
        to decompose tasks — check its ``tools:`` list in ``config.yaml``
        (``tasks_create``, ``tasks_update``, etc.).

        For a full record of every planning round (prompt, result status,
        snapshot diff) raise the orchestrator's log level by adding the
        following to your ``config.yaml``::

            logging:
              level: INFO
              modules:
                ai_cli.core.task_orchestrator: DEBUG

        Records are written as JSONL to ``<session_dir>/session.log``.
        """
        from ai_cli.core.task_manager import TaskStorageError, TaskValidationError
        from ai_cli.core.task_orchestrator import TaskOrchestrator

        if self._task_manager is None:
            self._display.show_error("Task manager is not available in this session.")
            return
        if self._agent_registry is None or not self._agent_registry.has_agents:
            self._display.show_error(
                "No agents configured. Add an 'agents:' section to config.yaml."
            )
            return
        if not self._agent_registry.has("executor"):
            self._display.show_error(
                "No 'executor' agent configured. /plan requires at least an executor."
            )
            return
        if not self._agent_registry.has("planner"):
            self._display.show_error(
                "No 'planner' agent configured. /plan requires both an executor and a planner."
            )
            return

        try:
            try:
                tokens = shlex.split(remainder)
            except ValueError as exc:
                self._display.show_error(f"Invalid /plan arguments: {exc}")
                return

            autonomous = False
            goal_tokens: list[str] = []
            for token in tokens:
                if token == "--autonomous":
                    autonomous = True
                elif token.startswith("--"):
                    self._display.show_error(
                        f"Unknown flag {token!r}. "
                        'Usage: /plan ["<goal>"] [--autonomous]'
                    )
                    return
                else:
                    goal_tokens.append(token)

            goal: str | None = " ".join(goal_tokens) or None
            if goal is None:
                goal = self._task_manager.get_goal()
                if not goal:
                    self._display.show_error('No goal set. Provide one: /plan "<goal>"')
                    return

            if self._orchestrator is None:
                if self._config is None:
                    self._display.show_error(
                        "Configuration is not available in this session. "
                        "/plan requires a loaded config."
                    )
                    return
                self._orchestrator = TaskOrchestrator(
                    self._task_manager,
                    self._agent_registry,
                    self._display,
                    workspace=self._workspace,
                    config=self._config,
                    coordinator_llm=self._llm,
                    global_tool_registry=self._tool_registry,
                )

            self._orchestrator.run(goal, autonomous=autonomous)
        except TaskStorageError as exc:
            self._display.show_error(f"Task storage error: {exc}")
        except TaskValidationError as exc:
            self._display.show_error(f"Task validation error: {exc}")
        except KeyError as exc:
            self._display.show_error(f"Agent not found: {exc}")

    # ------------------------------------------------------------------
    # /tasks subcommand handler
    # ------------------------------------------------------------------

    def _handle_tasks_subcommand(self, remainder: str) -> None:
        """Dispatch /tasks [subcommand] [args]."""
        from ai_cli.core.task_manager import TaskStorageError

        if self._task_manager is None:
            self._display.show_error("Task manager is not available in this session.")
            return

        try:
            self._run_tasks_dispatch(remainder)
        except TaskStorageError as exc:
            self._display.show_error(f"Task storage error: {exc}")

    def _run_tasks_dispatch(self, remainder: str) -> None:
        """Inner /tasks dispatcher; separated so TaskStorageError can be caught at one site."""
        from ai_cli.core.task_manager import TaskNotFoundError, TaskValidationError

        assert self._task_manager is not None  # guaranteed by _handle_tasks_subcommand

        try:
            parts = shlex.split(remainder)
        except ValueError as exc:
            self._display.show_error(f"Could not parse /tasks arguments: {exc}")
            return
        sub = parts[0].lower() if parts else ""

        # bare /tasks — simple list of unfinished root tasks
        if not sub:
            try:
                details = self._task_manager.list_task_details(parent_id=None)
                unfinished = [d for d in details if d["status"] != "done"]
                nodes = [self._task_to_node(d) for d in unfinished]
                goal = self._task_manager.get_goal()
            except (TaskNotFoundError, TaskValidationError) as exc:
                self._display.show_error(str(exc))
                return
            if goal:
                self._display.show_status(f"Goal: {goal}")
            self._display.show_tasks_simple(nodes)
            return

        # Known subcommands — reject anything else as a usage error.
        _KNOWN_SUBS = {
            "list",
            "tree",
            "info",
            "add",
            "edit",
            "delete",
            "close",
            "open",
            "note",
        }
        if sub not in _KNOWN_SUBS:
            self._display.show_error(
                f"Unknown /tasks subcommand: '{sub}'. "
                "Try /tasks, /tasks list, /tasks tree, /tasks info <path>, "
                "/tasks add, /tasks edit <path>, /tasks delete [<path>], "
                '/tasks note obsolete <path> <index> [--reason "<text>"].'
            )
            return

        args = parts[1:]  # everything after the subcommand

        if sub == "list":
            # /tasks list [<path>]
            if len(args) > 1:
                self._display.show_error("Usage: /tasks list [<path>]")
                return
            try:
                path = self._normalize_optional_task_path(args[0]) if args else None
                detail_map = self._task_manager.get_all_task_details_map()
                if path:
                    parent = self._find_in_detail_map(path, detail_map)
                    parent_id = parent["id"]
                else:
                    parent_id = None
                details = [
                    d for d in detail_map.values() if d.get("parent_id") == parent_id
                ]
                nodes = [self._task_to_node(d) for d in details]
            except (TaskNotFoundError, TaskValidationError) as exc:
                self._display.show_error(str(exc))
                return
            self._display.show_tasks_list(nodes)
            return

        if sub == "tree":
            # /tasks tree [<path>] [--depth <n>]
            depth = self._get_tree_depth()
            # Parse --depth override
            clean_args = []
            i = 0
            while i < len(args):
                if args[i] == "--depth":
                    if i + 1 >= len(args):
                        self._display.show_error(
                            "Usage: /tasks tree [<path>] [--depth <n>]"
                        )
                        return
                    try:
                        parsed_depth = int(args[i + 1])
                        if parsed_depth < 1:
                            raise ValueError
                        depth = parsed_depth
                    except ValueError:
                        self._display.show_error(
                            f"Invalid --depth value: {args[i + 1]!r}"
                            " (must be a positive integer)"
                        )
                        return
                    i += 2
                else:
                    clean_args.append(args[i])
                    i += 1
            if len(clean_args) > 1:
                self._display.show_error("Usage: /tasks tree [<path>] [--depth <n>]")
                return
            try:
                path = (
                    self._normalize_optional_task_path(clean_args[0])
                    if clean_args
                    else None
                )
                detail_map = self._task_manager.get_all_task_details_map()
                if path:
                    parent = self._find_in_detail_map(path, detail_map)
                    nodes = [self._build_task_tree(parent["id"], detail_map, depth, 1)]
                else:
                    nodes = [
                        self._build_task_tree(t["id"], detail_map, depth, 1)
                        for t in detail_map.values()
                        if t.get("parent_id") is None
                    ]
            except (TaskNotFoundError, TaskValidationError) as exc:
                self._display.show_error(str(exc))
                return
            goal = self._task_manager.get_goal()
            if goal:
                self._display.show_status(f"Goal: {goal}")
            self._display.show_tasks_tree(nodes, depth)
            return

        if sub == "info":
            if len(args) != 1:
                self._display.show_error("Usage: /tasks info <path>")
                return
            try:
                path = self._normalize_task_path_arg(args[0])
                task = self._task_manager.find_by_path(path)
            except (TaskNotFoundError, TaskValidationError) as exc:
                self._display.show_error(str(exc))
                return
            self._display.show_task_info(task)
            return

        if sub == "add":
            if len(args) > 1:
                self._display.show_error("Usage: /tasks add [<path>]")
                return
            try:
                path = self._normalize_optional_task_path(args[0]) if args else None
                parent_task = self._task_manager.find_by_path(path) if path else None
            except (TaskNotFoundError, TaskValidationError) as exc:
                self._display.show_error(str(exc))
                return
            fields = self._tasks_add_wizard(parent_task)
            if fields is None:
                return
            try:
                task = self._task_manager.create_task(
                    name=fields["name"],
                    definition_of_done=fields["definition_of_done"],
                    description=fields.get("description", ""),
                    parent_id=parent_task["id"] if parent_task else None,
                    priority=fields.get("priority", "medium"),
                )
                self._display.show_status(f"Task '{task['name']}' created.")
            except (TaskNotFoundError, TaskValidationError) as exc:
                self._display.show_error(str(exc))
            return

        if sub == "edit":
            if len(args) != 1:
                self._display.show_error("Usage: /tasks edit <path>")
                return
            try:
                path = self._normalize_task_path_arg(args[0])
                task = self._task_manager.find_by_path(path)
            except (TaskNotFoundError, TaskValidationError) as exc:
                self._display.show_error(str(exc))
                return
            fields = self._tasks_edit_wizard(task)
            if fields is None:
                return
            if not fields:
                self._display.show_status("No changes made.")
                return
            try:
                updated = self._task_manager.update_task(task["id"], **fields)
                self._display.show_status(f"Task '{updated['name']}' updated.")
            except (TaskNotFoundError, TaskValidationError) as exc:
                self._display.show_error(str(exc))
            return

        if sub == "delete":
            if len(args) > 1:
                self._display.show_error(
                    "Usage: /tasks delete [<path>]  (too many arguments)"
                )
                return
            try:
                path = self._normalize_optional_task_path(args[0]) if args else None
                if path:
                    task = self._task_manager.find_by_path(path)
                    confirm_msg = (
                        f"Delete task '{task['name']}' and all its subtasks? [y/N] "
                    )
                else:
                    confirm_msg = "Delete ALL tasks and the goal (full reset)? [y/N] "
            except (TaskNotFoundError, TaskValidationError) as exc:
                self._display.show_error(str(exc))
                return
            try:
                from prompt_toolkit import prompt as _pt_prompt

                answer = _pt_prompt(confirm_msg).strip().lower()
            except (EOFError, KeyboardInterrupt):
                self._display.show_status("Cancelled.")
                return
            if answer not in ("y", "yes"):
                self._display.show_status("Cancelled.")
                return
            try:
                if path:
                    self._task_manager.delete_task(task["id"])
                    self._display.show_status(f"Task '{task['name']}' deleted.")
                else:
                    self._task_manager.clear()
                    self._display.show_status("All tasks and goal cleared.")
            except (TaskNotFoundError, TaskValidationError) as exc:
                self._display.show_error(str(exc))
            return

        if sub == "close":
            if len(args) != 1:
                self._display.show_error(
                    "Usage: /tasks close <path>  (requires exactly one argument)"
                )
                return
            try:
                path = self._normalize_task_path_arg(args[0])
                task = self._task_manager.find_by_path(path)
                updated = self._task_manager.close_task(task["id"])
                self._display.show_status(
                    f"Task '{updated['name']}' and all its subtasks closed."
                )
            except (TaskNotFoundError, TaskValidationError) as exc:
                self._display.show_error(str(exc))
            return

        if sub == "open":
            if len(args) != 1:
                self._display.show_error(
                    "Usage: /tasks open <path>  (requires exactly one argument)"
                )
                return
            try:
                path = self._normalize_task_path_arg(args[0])
                task = self._task_manager.find_by_path(path)
                updated = self._task_manager.open_task(task["id"])
                self._display.show_status(f"Task '{updated['name']}' re-opened.")
            except (TaskNotFoundError, TaskValidationError) as exc:
                self._display.show_error(str(exc))
            return

        if sub == "note":
            if not args or args[0].lower() != "obsolete":
                self._display.show_error(
                    'Usage: /tasks note obsolete <path> <index> [--reason "<text>"]'
                )
                return

            note_args = args[1:]
            if len(note_args) < 2:
                self._display.show_error(
                    'Usage: /tasks note obsolete <path> <index> [--reason "<text>"]'
                )
                return

            path_arg = note_args[0]
            index_arg = note_args[1]

            reason = ""
            extras = note_args[2:]
            if extras:
                if len(extras) == 2 and extras[0] == "--reason":
                    reason = extras[1]
                else:
                    self._display.show_error(
                        'Usage: /tasks note obsolete <path> <index> [--reason "<text>"]'
                    )
                    return

            try:
                path = self._normalize_task_path_arg(path_arg)
            except TaskValidationError as exc:
                self._display.show_error(str(exc))
                return

            try:
                note_index = int(index_arg)
            except ValueError:
                self._display.show_error("'index' must be an integer.")
                return

            try:
                task = self._task_manager.find_by_path(path)
                updated = self._task_manager.obsolete_note(
                    task["id"], note_index, reason=reason
                )
                self._display.show_status(
                    f"Marked note {note_index} obsolete for task '{updated['name']}'."
                )
            except (TaskNotFoundError, TaskValidationError) as exc:
                self._display.show_error(str(exc))
            return

    def _get_tree_depth(self) -> int:
        """Return the configured tasks.tree_depth (default 3)."""
        default = 3
        if self._config is None:
            return default
        tasks_cfg = self._config.get("tasks")
        if not isinstance(tasks_cfg, dict):
            return default
        raw = tasks_cfg.get("tree_depth", default)
        try:
            depth = int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid tasks.tree_depth value %r; using default %d.", raw, default
            )
            return default
        if depth < 1:
            logger.warning(
                "Invalid tasks.tree_depth value %r; using default %d.", raw, default
            )
            return default
        return depth

    def _task_to_node(self, full: dict) -> dict:
        """Build a display node from a ``task_detail`` dict."""
        subtasks = full.get("subtasks", [])
        return {
            "id": full["id"],
            "name": full["name"],
            "status": full["status"],
            "priority": full["priority"],
            "description": full.get("description", ""),
            "subtask_count": len(subtasks),
            "done_subtask_count": sum(1 for s in subtasks if s["status"] == "done"),
        }

    def _build_task_tree(
        self, task_id: str, detail_map: dict, max_depth: int, current: int
    ) -> dict:
        """Recursively build a tree node dict for display from a pre-loaded detail map.

        *detail_map* is ``{task_id: task_detail}`` returned by
        ``TaskManager.get_all_task_details_map()`` — avoids N+1 disk reads.
        """
        full = detail_map.get(task_id, {})
        subtasks = full.get("subtasks", [])
        done_count = sum(1 for s in subtasks if s["status"] == "done")
        node: dict = {
            "id": task_id,
            "name": full.get("name", "?"),
            "status": full.get("status", "?"),
            "priority": full.get("priority", "?"),
            "description": full.get("description", ""),
            "subtask_count": len(subtasks),
            "done_subtask_count": done_count,
            "children": None,
        }
        if current < max_depth and subtasks:
            node["children"] = [
                self._build_task_tree(s["id"], detail_map, max_depth, current + 1)
                for s in subtasks
            ]
        elif not subtasks:
            node["children"] = []
        # else: children remains None → depth limit reached
        return node

    @staticmethod
    def _find_in_detail_map(path: str, detail_map: dict) -> dict:
        """Resolve a dot-separated name path against an already-loaded detail_map.

        Raises :exc:`TaskValidationError` for invalid segments and
        :exc:`TaskNotFoundError` if any segment is not found.
        Avoids a second ``_load()`` when the caller already has the full map.
        """
        from ai_cli.core.task_manager import (
            TaskNotFoundError,
            TaskValidationError,
            normalize_task_path,
        )

        segments = normalize_task_path(path).split(".")
        for seg in segments:
            if not seg or re.fullmatch(r"[A-Za-z0-9_]+", seg) is None:
                raise TaskValidationError(
                    f"Invalid path segment {seg!r}: must match ^[A-Za-z0-9_]+$."
                )
        current_parent_id: str | None = None
        found: dict | None = None
        for seg in segments:
            found = None
            for task in detail_map.values():
                if (
                    task.get("parent_id") == current_parent_id
                    and task.get("name") == seg
                ):
                    found = task
                    break
            if found is None:
                raise TaskNotFoundError(f"Task not found at path segment {seg!r}")
            current_parent_id = found["id"]
        assert found is not None
        return found

    @staticmethod
    def _normalize_task_path_arg(path: str) -> str:
        from ai_cli.core.task_manager import normalize_task_path

        return normalize_task_path(path)

    @staticmethod
    def _normalize_optional_task_path(path: str) -> str | None:
        from ai_cli.core.task_manager import normalize_task_path

        if not path.strip():
            return None
        return normalize_task_path(path)

    def _tasks_add_wizard(self, parent_task: dict | None) -> dict | None:
        """Prompt for new task fields.  Returns field dict or None if cancelled."""
        from prompt_toolkit import prompt as _pt_prompt

        scope = f"under '{parent_task['name']}'" if parent_task else "at root"
        self._display.show_status(f"Creating task {scope}. Press Ctrl+C to cancel.")

        try:
            name = _pt_prompt("  Name (required): ").strip()
            if not name:
                self._display.show_status("Cancelled (name is required).")
                return None

            description = _pt_prompt("  Description (optional): ").strip()

            dod = ""
            while len(dod.strip()) < 5:
                dod = _pt_prompt(
                    "  Definition of done (required, min 5 non-whitespace chars): "
                ).strip()
                if len(dod.strip()) < 5:
                    self._display.show_error(
                        "Definition of done must be at least 5 non-whitespace characters."
                    )

            raw_prio = (
                _pt_prompt("  Priority [low/medium/high] (default: medium): ")
                .strip()
                .lower()
            )
            priority = raw_prio if raw_prio in ("low", "medium", "high") else "medium"

        except (EOFError, KeyboardInterrupt):
            self._display.show_status("Cancelled.")
            return None

        return {
            "name": name,
            "description": description,
            "definition_of_done": dod,
            "priority": priority,
        }

    def _tasks_edit_wizard(self, task: dict) -> dict | None:
        """Prompt for updated task fields.  Returns changed fields or None if cancelled.

        Enter ``~`` at any optional field to clear it to its empty/default value.
        Leave blank to keep the current value.  Ctrl+C cancels the whole wizard.
        """
        from prompt_toolkit import prompt as _pt_prompt

        _CLEAR = object()  # sentinel: user typed "~" to clear the field

        self._display.show_status(
            f"Editing '{task['name']}'. "
            "Leave blank to keep current value, ~ to clear, Ctrl+C to cancel."
        )

        def _ask(label: str, current: str, clearable: bool = False) -> object:
            """Prompt with current value shown.

            Returns:
              - *current* if input is blank (keep unchanged)
              - ``_CLEAR`` sentinel if input is ``~`` and *clearable* is True
              - the new string value otherwise
              - ``None`` if the user cancelled (EOFError / KeyboardInterrupt)
            """
            hint = " (~ to clear)" if clearable else ""
            try:
                raw = _pt_prompt(f"  {label}{hint} [{current!r}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if clearable and raw == "~":
                return _CLEAR
            return raw if raw else current

        try:
            name = _ask("Name", task.get("name", ""))
            if name is None:
                self._display.show_status("Cancelled.")
                return None

            description = _ask(
                "Description", task.get("description", ""), clearable=True
            )
            if description is None:
                self._display.show_status("Cancelled.")
                return None

            dod = _ask("Definition of done", task.get("definition_of_done", ""))
            if dod is None:
                self._display.show_status("Cancelled.")
                return None

            raw_prio = _ask(
                "Priority [low/medium/high]", task.get("priority", "medium")
            )
            if raw_prio is None:
                self._display.show_status("Cancelled.")
                return None
            assert isinstance(raw_prio, str)
            priority = (
                raw_prio.lower()
                if raw_prio.lower() in ("low", "medium", "high")
                else task.get("priority", "medium")
            )

            next_action = _ask(
                "Next action", task.get("next_action", ""), clearable=True
            )
            if next_action is None:
                self._display.show_status("Cancelled.")
                return None

            raw_blockers = _ask(
                "Blockers (comma-separated)",
                ", ".join(task.get("blockers", [])),
                clearable=True,
            )
            if raw_blockers is None:
                self._display.show_status("Cancelled.")
                return None

        except (EOFError, KeyboardInterrupt):
            self._display.show_status("Cancelled.")
            return None

        # Resolve clearable fields.
        description_val: str = "" if description is _CLEAR else str(description)
        next_action_val: str = "" if next_action is _CLEAR else str(next_action)
        if raw_blockers is _CLEAR:
            blockers: list[str] = []
        else:
            blockers = [b.strip() for b in str(raw_blockers).split(",") if b.strip()]

        # Collect only changed fields.
        fields: dict = {}
        if name != task.get("name", ""):
            fields["name"] = str(name)
        if description_val != task.get("description", ""):
            fields["description"] = description_val
        if dod != task.get("definition_of_done", ""):
            fields["definition_of_done"] = str(dod)
        if priority != task.get("priority", "medium"):
            fields["priority"] = priority
        if next_action_val != task.get("next_action", ""):
            fields["next_action"] = next_action_val
        if blockers != task.get("blockers", []):
            fields["blockers"] = blockers

        return fields

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
        self._main_agent.spec.max_tool_rounds = value
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
    # /index subcommand handler
    # ------------------------------------------------------------------

    def _handle_index_command(self, remainder: str) -> None:
        """Dispatch /index [path] [--file <path>] [--label <name>] [--full] [--remove]."""
        import asyncio
        import time

        ei = self._workspace.embedding_index
        if ei is None:
            self._display.show_error(
                "Embedding index is not enabled. "
                "Set 'embeddings.enabled: true' in config."
            )
            return

        # Parse arguments using the same tokenizer as the completer so that
        # backslash-escaped spaces (and quoted strings) in paths are handled.
        # Appending a space ensures the final token is treated as complete.
        args, _ = _tokenize_command(remainder + " ") if remainder.strip() else ([], "")
        path: str | None = None
        label: str | None = None
        single_file: str | None = None
        full = False
        remove = False

        i = 0
        while i < len(args):
            if args[i] == "--full":
                full = True
            elif args[i] == "--remove":
                remove = True
            elif args[i] == "--label":
                if i + 1 >= len(args):
                    self._display.show_error("/index --label requires a label value.")
                    return
                label = args[i + 1]
                i += 1
            elif args[i] == "--file":
                if i + 1 >= len(args):
                    self._display.show_error("/index --file requires a file path.")
                    return
                single_file = args[i + 1]
                i += 1
            elif args[i].startswith("--"):
                self._display.show_error(f"Unknown /index flag: {args[i]!r}")
                return
            else:
                if path is not None:
                    self._display.show_error(
                        "/index accepts at most one PATH argument."
                    )
                    return
                path = args[i]
            i += 1

        if path and single_file:
            self._display.show_error(
                "/index: positional PATH and --file are mutually exclusive."
            )
            return

        if remove and single_file:
            self._display.show_error("/index --remove cannot be used with --file.")
            return

        if remove and not path:
            self._display.show_error("/index --remove requires a path argument.")
            return

        if remove and path:
            _p = Path(path)
            resolved = (_p if _p.is_absolute() else self._workspace.root / _p).resolve()
            try:
                ei.remove_root(resolved)
            except ValueError as exc:
                self._display.show_error(str(exc))
                return
            self._display.show_status(f"Removed root: {resolved}")
            return

        # --file: index a single file directly without root scanning.
        if single_file:
            _sf = Path(single_file)
            resolved_file = (
                _sf if _sf.is_absolute() else self._workspace.root / _sf
            ).resolve()
            if not resolved_file.is_file():
                self._display.show_error(f"Not a regular file: {resolved_file}")
                return
            self._handle_index_single_file(ei, resolved_file, full=full)
            return

        roots_to_index: list[Path] | None = None
        if path:
            _p2 = Path(path)
            resolved = (
                _p2 if _p2.is_absolute() else self._workspace.root / _p2
            ).resolve()
            if not resolved.is_dir():
                self._display.show_error(
                    f"Positional path must be a directory. "
                    f"For single files, use `/index --file {resolved}`."
                )
                return
            root_paths = {r.path for r in ei.roots}
            if resolved not in root_paths and resolved != self._workspace.root:
                try:
                    ei.add_root(resolved, label=label)
                except ValueError as exc:
                    self._display.show_error(str(exc))
                    return
                self._display.show_status(f"Added root: {resolved}")
            elif resolved in root_paths and label:
                try:
                    ei.update_root_label(resolved, label)
                except ValueError as exc:
                    self._display.show_error(str(exc))
                    return
                self._display.show_status(f"Updated label for root: {resolved}")
            roots_to_index = [resolved]

        # ------------------------------------------------------------------
        # Run indexing in a background thread so the main thread stays
        # responsive to Ctrl+C (SIGINT) and Ctrl+Z (SIGTSTP).
        # ------------------------------------------------------------------
        import signal

        cancelled = threading.Event()
        stats_holder: list = [None]
        error_holder: list = [None]
        # Shared progress state written by the worker, read by the main thread.
        progress: dict = {"current": 0, "total": 0, "file": ""}

        def _on_progress(current: int, total: int, file_path: str) -> None:
            progress["current"] = current
            progress["total"] = total
            progress["file"] = file_path

        # Shared chunk-level progress (reset to 0 at start of each file).
        chunk_progress: dict = {"done": 0, "total": 0}

        def _on_chunk_progress(done: int, total: int) -> None:
            chunk_progress["done"] = done
            chunk_progress["total"] = total

        async def _index_and_close() -> object:
            try:
                return await ei.index(
                    roots=roots_to_index,
                    incremental=not full,
                    cancelled=cancelled,
                    on_progress=_on_progress,
                    on_chunk_progress=_on_chunk_progress,
                )
            finally:
                # Close the async HTTP client so the next run gets a fresh one
                # bound to its own event loop.
                await ei.aclose()

        def _run_worker() -> None:
            try:
                stats_holder[0] = asyncio.run(_index_and_close())
            except Exception as exc:  # noqa: BLE001
                error_holder[0] = exc

        worker = threading.Thread(target=_run_worker, daemon=True)
        worker.start()

        # Install a custom SIGINT handler that sets the cancellation flag
        # without raising KeyboardInterrupt, keeping the main thread
        # unblocked so worker.join() runs to completion uninterrupted.
        # A second Ctrl+C restores normal behaviour so the user can still
        # force-exit if the worker is stuck on a slow network call.
        old_sigint = signal.getsignal(signal.SIGINT)

        def _cancel_handler(signum: int, frame: object) -> None:
            cancelled.set()
            # Restore original handler: second Ctrl+C raises KeyboardInterrupt.
            signal.signal(signal.SIGINT, old_sigint)

        signal.signal(signal.SIGINT, _cancel_handler)

        try:
            from tqdm import tqdm as _tqdm

            _has_tqdm = True
        except ImportError:
            _has_tqdm = False

        t0 = time.monotonic()
        try:
            if _has_tqdm:
                with (
                    _tqdm(
                        total=0,
                        unit="file",
                        desc="Scanning",
                        dynamic_ncols=True,
                        position=0,
                        leave=True,
                    ) as pbar_files,
                    _tqdm(
                        total=0,
                        unit="chunk",
                        desc="Chunks",
                        dynamic_ncols=True,
                        position=1,
                        leave=False,
                    ) as pbar_chunks,
                ):
                    last_total = 0
                    last_current = 0
                    last_chunk_done = 0
                    last_chunk_total = 0
                    while worker.is_alive():
                        cur = progress["current"]
                        tot = progress["total"]
                        c_done = chunk_progress["done"]
                        c_total = chunk_progress["total"]

                        # File-level bar.
                        if tot != last_total:
                            pbar_files.reset(total=tot)
                            pbar_files.set_description("Indexing")
                            last_total = tot
                            last_current = 0

                        if cur > last_current:
                            pbar_files.update(cur - last_current)
                            last_current = cur
                            fname = Path(progress["file"]).name
                            if fname:
                                pbar_files.set_postfix_str(fname, refresh=False)

                        # Chunk-level bar: reset when a new file begins.
                        if c_total != last_chunk_total or c_done < last_chunk_done:
                            pbar_chunks.reset(total=c_total if c_total > 0 else None)
                            pbar_chunks.update(c_done)
                            last_chunk_total = c_total
                            last_chunk_done = c_done
                        elif c_done > last_chunk_done:
                            pbar_chunks.update(c_done - last_chunk_done)
                            last_chunk_done = c_done

                        worker.join(timeout=0.1)

                    # Final updates to 100 %.
                    if last_total > 0 and last_current < last_total:
                        pbar_files.update(last_total - last_current)
                    c_done = chunk_progress["done"]
                    if last_chunk_total > 0 and c_done > last_chunk_done:
                        pbar_chunks.update(c_done - last_chunk_done)
            else:
                worker.join()
        except KeyboardInterrupt:
            # Second Ctrl+C (original handler restored after first).
            cancelled.set()
            worker.join()
            self._display.show_status("Indexing cancelled.")
            return
        finally:
            signal.signal(signal.SIGINT, old_sigint)

        if cancelled.is_set():
            self._display.show_status("Indexing cancelled.")
            return

        if error_holder[0] is not None:
            self._display.show_error(f"Indexing failed: {error_holder[0]}")
            return

        stats = stats_holder[0]
        elapsed = time.monotonic() - t0
        self._display.show_status(
            f"Index complete in {elapsed:.1f}s: "
            f"{stats.files_indexed} indexed, {stats.files_skipped} skipped, "
            f"{stats.files_deleted} deleted, {stats.chunks_added} chunks"
        )

    # ------------------------------------------------------------------
    # /index --file helper
    # ------------------------------------------------------------------

    def _handle_index_single_file(
        self,
        ei: object,
        file_path: Path,
        *,
        full: bool = False,
    ) -> None:
        """Index a single file, showing chunk-level progress and diagnostics."""
        import asyncio
        import signal
        import time

        chunk_progress: dict = {"done": 0, "total": 0}
        stats_holder: list = [None]
        error_holder: list = [None]

        def _on_chunk_progress(done: int, total: int) -> None:
            chunk_progress["done"] = done
            chunk_progress["total"] = total

        cancelled = threading.Event()

        async def _run() -> object:
            try:
                return await ei.index_file(  # type: ignore[attr-defined]
                    file_path,
                    incremental=not full,
                    on_chunk_progress=_on_chunk_progress,
                    cancelled=cancelled,
                )
            finally:
                await ei.aclose()  # type: ignore[attr-defined]

        def _worker() -> None:
            try:
                stats_holder[0] = asyncio.run(_run())
            except Exception as exc:  # noqa: BLE001
                error_holder[0] = exc

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        old_sigint = signal.getsignal(signal.SIGINT)

        def _cancel(signum: int, frame: object) -> None:
            cancelled.set()
            signal.signal(signal.SIGINT, old_sigint)

        signal.signal(signal.SIGINT, _cancel)

        t0 = time.monotonic()
        try:
            try:
                from tqdm import tqdm as _tqdm

                _has_tqdm = True
            except ImportError:
                _has_tqdm = False

            if _has_tqdm:
                with _tqdm(
                    total=0,
                    unit="chunk",
                    desc=file_path.name,
                    dynamic_ncols=True,
                    position=0,
                    leave=True,
                ) as pbar:
                    last_chunk_done = 0
                    last_chunk_total = 0
                    while worker.is_alive():
                        c_done = chunk_progress["done"]
                        c_total = chunk_progress["total"]
                        if c_total != last_chunk_total or c_done < last_chunk_done:
                            pbar.reset(total=c_total if c_total > 0 else None)
                            pbar.update(c_done)
                            last_chunk_total = c_total
                            last_chunk_done = c_done
                        elif c_done > last_chunk_done:
                            pbar.update(c_done - last_chunk_done)
                            last_chunk_done = c_done
                        worker.join(timeout=0.1)
                    c_done = chunk_progress["done"]
                    if last_chunk_total > 0 and c_done > last_chunk_done:
                        pbar.update(c_done - last_chunk_done)
            else:
                worker.join()
        except KeyboardInterrupt:
            cancelled.set()
            worker.join()
            self._display.show_status("Indexing cancelled.")
            return
        finally:
            signal.signal(signal.SIGINT, old_sigint)

        if cancelled.is_set():
            self._display.show_status("Indexing cancelled.")
            return

        if error_holder[0] is not None:
            self._display.show_error(f"Indexing failed: {error_holder[0]}")
            return

        stats = stats_holder[0]
        elapsed = time.monotonic() - t0
        if stats is not None and stats.files_skipped:
            self._display.show_status(
                f"{file_path.name}: unchanged — skipped ({elapsed:.1f}s)"
            )
        elif stats is not None and stats.files_indexed:
            self._display.show_status(
                f"{file_path.name}: indexed {stats.chunks_added} chunks in {elapsed:.1f}s"
            )
        else:
            self._display.show_status(
                f"{file_path.name}: no chunks produced — check logs for details"
                f" (elapsed {elapsed:.1f}s)"
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
                result = self._main_agent.run(user_input, abort=abort)
            finally:
                monitor.stop()
        finally:
            pm.prompt_fn = _orig_prompt_fn

        if result.status == "error":
            return
        if result.status == "tool_limit":
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
