"""
display.py — Abstract Display interface and PlainDisplay implementation.

All user-facing output and interactive prompts are routed through a Display
instance.  The REPL holds one Display and calls its methods; it never writes
to stdout directly.  Swapping the display backend (plain vs rich) requires
only a different object passed at startup.
"""

from __future__ import annotations

import json
import logging
import pydoc
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from prompt_toolkit import prompt as pt_prompt

if TYPE_CHECKING:
    from ai_cli.core.config_manager import ConfigManager
    from ai_cli.core.session_manager import Session, SessionMeta
    from ai_cli.tools.base import Tool

logger = logging.getLogger(__name__)


class Display(ABC):
    """
    Abstract base class for all display backends.

    The boolean flags ``verbose`` and ``markdown_enabled`` are stored here and
    flipped by the concrete ``toggle_*`` methods so subclasses do not need to
    re-implement them.  All output and interaction methods are abstract.
    """

    def __init__(self, *, verbose: bool = False, markdown_enabled: bool = True) -> None:
        self._verbose = verbose
        self._markdown_enabled = markdown_enabled

    # ------------------------------------------------------------------
    # Mode flags
    # ------------------------------------------------------------------

    @property
    def verbose(self) -> bool:
        """True when verbose mode is active (full tool args/results shown)."""
        return self._verbose

    def toggle_verbose(self) -> None:
        """Switch between summary and verbose mode."""
        self._verbose = not self._verbose

    @property
    def markdown_enabled(self) -> bool:
        """True when LLM text should be rendered as Markdown rather than raw."""
        return self._markdown_enabled

    def toggle_markdown(self) -> None:
        """Switch between Markdown rendering and raw text output."""
        self._markdown_enabled = not self._markdown_enabled

    # ------------------------------------------------------------------
    # Streaming assistant output
    # ------------------------------------------------------------------

    @abstractmethod
    def begin_assistant_turn(self) -> None:
        """Called once before the first text delta from the LLM arrives."""

    @abstractmethod
    def stream_text(self, delta: str) -> None:
        """Called for each text chunk as it arrives from the LLM."""

    def stream_reasoning(self, delta: str) -> None:  # noqa: B027
        """
        Called for each reasoning/thinking chunk as it arrives from the LLM.

        Reasoning content comes from ``reasoning_content`` delta fields (o1/o3)
        or ``<think>…</think>`` tags when ``extract_think_tags`` is enabled.

        The default implementation is a no-op.  Subclasses may override to
        display reasoning content (e.g. in verbose mode or as a dim preview).
        """

    @abstractmethod
    def end_assistant_turn(self) -> None:
        """Called once after the final chunk.  Flush/finalise any buffered output."""

    def update_usage(self, usage: dict, context_window: int) -> None:  # noqa: B027
        """
        Called once per LLM turn with the token usage returned in the ``done``
        chunk.

        *usage* has keys ``prompt_tokens``, ``completion_tokens``,
        ``total_tokens``.  *context_window* is the model's total context size.

        The default implementation is a no-op.  Display backends that show a
        status bar (e.g. RichDisplay) override this to update the token counter.
        """

    # ------------------------------------------------------------------
    # Tool activity
    # ------------------------------------------------------------------

    @abstractmethod
    def show_tool_call(self, name: str, args: dict) -> None:
        """
        Notify the user that a tool is about to run.

        Summary mode: one compact line.
        Verbose mode: name + full pretty-printed args.
        """

    @abstractmethod
    def show_tool_result(
        self, name: str, result: dict, display_str: str | None = None
    ) -> None:
        """
        Show the outcome of a tool call.

        Summary mode: silent (the LLM incorporates the result in its reply).
        Verbose mode: *display_str* if provided (may contain ANSI codes),
        otherwise the full pretty-printed *result* dict.
        """

    # ------------------------------------------------------------------
    # Status and errors
    # ------------------------------------------------------------------

    @abstractmethod
    def show_status(self, message: str) -> None:
        """Informational, non-error message (compaction notice, session saved, …)."""

    @abstractmethod
    def show_error(self, message: str) -> None:
        """User-visible error.  Does not raise — caller decides whether to abort."""

    # ------------------------------------------------------------------
    # Slash-command output
    # ------------------------------------------------------------------

    @abstractmethod
    def show_help(self, commands: list[tuple[str, str]]) -> None:
        """
        Render the slash-command help table.

        *commands* is a list of ``(command, description)`` pairs in display order.
        """

    @abstractmethod
    def show_tool_list(self, tools: list[Tool]) -> None:
        """Render the list of currently enabled tools and their descriptions."""

    @abstractmethod
    def show_session_info(self, session: Session) -> None:
        """Render metadata for the current session (id, start time, message count, name)."""

    @abstractmethod
    def show_tool_list_all(self, tools_info: list[dict]) -> None:
        """
        Render all registered tools with their enabled/allowed/permission status.

        *tools_info* is a list of dicts as returned by
        :meth:`~ai_cli.core.tool_registry.ToolRegistry.all_tools_info`.
        """

    @abstractmethod
    def show_tool_info(self, tool_info: dict) -> None:
        """
        Render detailed information for a single tool.

        *tool_info* is a dict as returned by
        :meth:`~ai_cli.core.tool_registry.ToolRegistry.tool_info`.
        """

    @abstractmethod
    def show_history(self, messages: list[dict]) -> None:
        """
        Render the full conversation history in a scrollable view.

        *messages* is the list returned by ``Session.get_messages()``.
        Each entry is a dict with at minimum ``role`` and ``content`` keys;
        ``content`` may be a plain string or a list of content blocks.
        """

    # ------------------------------------------------------------------
    # Interactive prompts
    # ------------------------------------------------------------------

    @abstractmethod
    def show_permission_prompt(
        self,
        question: str,
        extra_options: list[str],
    ) -> tuple[str, str]:
        """
        Render a permission prompt and return the user's decision.

        Universal choices (yes / no / always / custom) are always shown first.
        *extra_options* are tool-specific strings appended below (e.g.
        ``'file:./src/foo.py'``, ``'dir:./src/'``).

        Returns ``(choice, user_text)`` where:

        * ``choice``    — ``'yes'``, ``'no'``, ``'always'``, ``'custom'``, or a
                          verbatim string from *extra_options*.
        * ``user_text`` — the user's free-text message when ``choice == 'custom'``,
                          empty string otherwise.
        """

    @abstractmethod
    def show_session_list(self, sessions: list[SessionMeta]) -> SessionMeta | None:
        """
        Render a list of resumable sessions and return the user's choice.

        Returns the chosen :class:`~ai_cli.core.session_manager.SessionMeta`,
        or ``None`` if the user declines or the list is empty.
        """


# ---------------------------------------------------------------------------
# PlainDisplay — prompt_toolkit for input, print() for output
# ---------------------------------------------------------------------------

_UNIVERSAL_OPTIONS: list[tuple[str, str, str]] = [
    ("y", "yes", "Allow once"),
    ("n", "no", "Deny"),
    ("a", "always", "Allow always for this session"),
    ("c", "custom", "Deny with a message"),
]


class PlainDisplay(Display):
    """
    Simple display backend using ``print()`` for output and ``prompt_toolkit``
    for interactive input.

    Suitable for initial development and smoke-testing.  ``markdown_enabled``
    has no visual effect here — output is always raw text.
    """

    def __init__(self, *, verbose: bool = False, markdown_enabled: bool = True) -> None:
        super().__init__(verbose=verbose, markdown_enabled=markdown_enabled)
        self._reasoning_started = False

    # ------------------------------------------------------------------
    # Streaming assistant output
    # ------------------------------------------------------------------

    def begin_assistant_turn(self) -> None:
        self._reasoning_started = False  # reset per-turn reasoning prefix state

    def stream_text(self, delta: str) -> None:
        if self._verbose and self._reasoning_started:
            # Reasoning was streamed without a trailing newline; end that line
            # then print the closing marker on its own line before any text.
            print()
            print("[/thinking]")
            self._reasoning_started = False
        print(delta, end="", flush=True)

    def stream_reasoning(self, delta: str) -> None:
        if not self._verbose:
            return
        if not self._reasoning_started:
            print("[thinking] ", end="", flush=True)
            self._reasoning_started = True
        print(delta, end="", flush=True)

    def end_assistant_turn(self) -> None:
        if self._verbose and self._reasoning_started:
            # Reasoning was the only output — close the block before the turn ends.
            self._reasoning_started = False
            print()
            print("[/thinking]")
        print()  # move to a fresh line after the response

    # ------------------------------------------------------------------
    # Tool activity
    # ------------------------------------------------------------------

    def show_tool_call(self, name: str, args: dict) -> None:
        if self._verbose:
            print(f"[tool] {name}")
            print(json.dumps(args, indent=2))
        else:
            summary = ", ".join(f"{k}={v!r}" for k, v in args.items())
            print(f"▶ {name}({summary})")

    def show_tool_result(
        self, name: str, result: dict, display_str: str | None = None
    ) -> None:
        if self._verbose:
            print(f"[result:{name}]")
            if display_str is not None:
                print(display_str)
            else:
                print(json.dumps(result, indent=2))
        # silent in summary mode

    # ------------------------------------------------------------------
    # Status and errors
    # ------------------------------------------------------------------

    def show_status(self, message: str) -> None:
        print(f"# {message}")

    def show_error(self, message: str) -> None:
        print(f"✗ {message}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Slash-command output
    # ------------------------------------------------------------------

    def show_help(self, commands: list[tuple[str, str]]) -> None:
        print("\nAvailable commands:")
        width = max((len(cmd) for cmd, _ in commands), default=0)
        for cmd, description in commands:
            print(f"  {cmd:<{width}}  {description}")

    def show_tool_list(self, tools: list[Tool]) -> None:
        if not tools:
            print("No tools currently enabled.")
            return
        print("\nEnabled tools:")
        for tool in tools:
            print(f"  {tool.name:<20}  {tool.description}")

    def show_session_info(self, session: Session) -> None:
        meta = session.get_meta()
        print(f"\nSession:   {session.session_id}")
        if meta.get("name"):
            print(f"Name:      {meta['name']}")
        started = meta.get("started_at", "unknown")
        print(f"Started:   {started}")
        print(f"Messages:  {meta.get('message_count', 0)}")

    def show_tool_list_all(self, tools_info: list[dict]) -> None:
        if not tools_info:
            print("No tools registered.")
            return
        print("\nAll tools:")
        for info in tools_info:
            if not info.get("allowed", True):
                status = "disallowed"
            elif info.get("enabled", False):
                status = "enabled"
            else:
                status = "disabled"
            if info.get("permission_required", False):
                status += ", perm"
            tier = info.get("tier", "")
            desc = info.get("description", "")[:50]
            print(f"  {info['name']:<20}  [{status:<18}]  {tier:<10}  {desc}")

    def show_tool_info(self, tool_info: dict) -> None:
        name = tool_info.get("name", "")
        print(f"\nTool: {name}")
        print(f"  Description:  {tool_info.get('description', '')}")
        print(f"  Tier:         {tool_info.get('tier', 'unknown')}")
        if not tool_info.get("allowed", True):
            print("  Status:       disallowed")
        elif tool_info.get("enabled", False):
            print("  Status:       enabled")
        else:
            print("  Status:       disabled")
        perm = "required" if tool_info.get("permission_required") else "not required"
        print(f"  Permission:   {perm}")
        params = tool_info.get("parameters", {})
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        required = params.get("required", []) if isinstance(params, dict) else []
        if props:
            print("  Parameters:")
            for pname, pdef in props.items():
                req = " (required)" if pname in required else ""
                ptype = pdef.get("type", "") if isinstance(pdef, dict) else ""
                pdesc = pdef.get("description", "") if isinstance(pdef, dict) else ""
                print(f"    {pname}: {ptype}{req} — {pdesc}")

    def show_history(self, messages: list[dict]) -> None:
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if isinstance(text, str):
                            text_parts.append(text)
                content = " ".join(text_parts)
            if not isinstance(content, str):
                content = ""
            lines.append(f"[{role}] {content}")
            lines.append("")
        pydoc.pager("\n".join(lines))

    # ------------------------------------------------------------------
    # Interactive prompts
    # ------------------------------------------------------------------

    def show_permission_prompt(
        self,
        question: str,
        extra_options: list[str],
    ) -> tuple[str, str]:
        print(f"\n{question}")
        for key, _, label in _UNIVERSAL_OPTIONS:
            print(f"  [{key}] {label}")
        for i, opt in enumerate(extra_options):
            print(f"  [{i}] {opt}")

        while True:
            try:
                raw = pt_prompt("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return ("no", "")
            for key, choice, _ in _UNIVERSAL_OPTIONS:
                if raw in (key, choice):
                    if choice == "custom":
                        try:
                            user_text = pt_prompt("Message: ").strip()
                        except (EOFError, KeyboardInterrupt):
                            return ("no", "")
                        return ("custom", user_text)
                    return (choice, "")
            try:
                idx = int(raw)
                if 0 <= idx < len(extra_options):
                    return (extra_options[idx], "")
            except ValueError:
                pass
            print("Invalid choice, please try again.")

    def show_session_list(self, sessions: list[SessionMeta]) -> SessionMeta | None:
        if not sessions:
            return None

        print("\nResumable sessions:")
        for i, s in enumerate(sessions):
            ts = s.started_at.strftime("%Y-%m-%d %H:%M UTC")
            preview = s.first_user_message[:60] or "(no messages)"
            print(f"  [{i}] {ts}  {s.session_id}  {preview}  ({s.message_count} msgs)")
        print("  [q] Start a new session")

        while True:
            try:
                raw = pt_prompt("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if raw in ("q", ""):
                return None
            try:
                idx = int(raw)
                if 0 <= idx < len(sessions):
                    return sessions[idx]
            except ValueError:
                pass
            print("Invalid choice, please try again.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_display(config: ConfigManager, *, verbose: bool = False) -> Display:
    """
    Instantiate and return the configured :class:`Display` backend.

    Reads ``display_backend`` (default ``'plain'``) and ``display_markdown``
    (default ``True``) from *config* via ``config.get()``.  An unknown backend
    name logs a warning and falls back to ``PlainDisplay``.
    """
    backend: str = config.get("display_backend", "plain")
    markdown_enabled: bool = config.get("display_markdown", True)

    if backend == "plain":
        return PlainDisplay(verbose=verbose, markdown_enabled=markdown_enabled)

    if backend == "rich":
        logger.warning(
            "RichDisplay is not yet implemented; falling back to PlainDisplay."
        )
        return PlainDisplay(verbose=verbose, markdown_enabled=markdown_enabled)

    logger.warning("Unknown display_backend %r; falling back to PlainDisplay.", backend)
    return PlainDisplay(verbose=verbose, markdown_enabled=markdown_enabled)
