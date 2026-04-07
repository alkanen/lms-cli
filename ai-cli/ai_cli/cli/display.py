"""
display.py — Abstract Display interface, PlainDisplay, and RichDisplay.

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
from collections.abc import Callable, Generator
from datetime import datetime
from typing import TYPE_CHECKING

from prompt_toolkit import prompt as pt_prompt
from rich.console import Console, ConsoleOptions, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.segment import Segment
from rich.spinner import Spinner
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

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

    def reset(self) -> None:  # noqa: B027
        """Reset per-run display state for agent reuse.

        Called on session-persistent sub-agents before each new delegation.
        The default implementation is a no-op; :class:`SubAgentDisplay`
        overrides this to clear its captured-text buffer.
        """

    def prompt_session_kwargs(self) -> dict:
        """
        Return extra kwargs to pass to the REPL's ``PromptSession`` constructor.

        ``PlainDisplay`` returns an empty dict (no toolbar).  ``RichDisplay``
        returns ``{"bottom_toolbar": …, "refresh_interval": 1}`` so the REPL's
        ``PromptSession`` shows a live context/timer bar.
        """
        return {}

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

        *display_str* is an optional ANSI-capable string returned by
        ``tool.format_display(args, result)``.  Whether it is shown in summary
        mode is left to each backend — backends may choose to always render it,
        show it only in verbose mode, or remain silent.  The default contract
        is: summary mode silent, verbose mode shows *display_str* (if provided)
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

    def show_agents(self, rows: list[dict]) -> None:  # noqa: B027
        """
        Render configured agent types.

        *rows* is a list of dicts with keys ``name``, ``model``,
        ``persistence``, ``tools``, and ``max_tool_rounds``.

        The default implementation is a no-op.  Concrete display backends
        override this to render a table.
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

    def show_agents(self, rows: list[dict]) -> None:
        if not rows:
            print("No agent types configured.")
            return
        print("\nConfigured agent types:")
        for row in rows:
            tools_str = row.get("tools", "(none)")
            print(
                f"  {row['name']:<20}  model={row['model']}  "
                f"persistence={row['persistence']}  "
                f"max_rounds={row['max_tool_rounds']}  tools={tools_str}"
            )

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
# RichDisplay helpers
# ---------------------------------------------------------------------------


class _LeftBorderRenderable:
    """
    Wraps any Rich renderable with a coloured vertical bar on the left edge.

    Produces output like::

        │ first line
        │ second line

    without adding blank top/bottom borders (unlike Panel with a custom Box).
    """

    def __init__(self, renderable: RenderableType, style: str = "bold") -> None:
        self._renderable = renderable
        self._style = style

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> Generator[Segment, None, None]:
        bar_style = Style.parse(self._style)
        inner_opts = options.update(width=max(1, options.max_width - 2))
        for line in console.render_lines(self._renderable, inner_opts):
            yield Segment("│ ", bar_style)
            yield from line
            yield Segment("\n")


# ---------------------------------------------------------------------------
# _LiveRenderable
# ---------------------------------------------------------------------------


class _LiveRenderable:
    """Renderable that re-evaluates its content on every Rich Live refresh.

    Rather than capturing a static snapshot at the moment a chunk arrives,
    this wrapper calls *build_fn* inside ``__rich_console__``.  Because Rich's
    ``Live`` calls ``__rich_console__`` on its renderable for every periodic
    refresh tick (``refresh_per_second``), the toolbar timer and spinner
    advance smoothly even during long pauses between streaming chunks.
    """

    def __init__(self, build_fn: Callable[[], RenderableType]) -> None:
        self._build_fn = build_fn

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> Generator[Segment, None, None]:
        yield from console.render(self._build_fn(), options)


# ---------------------------------------------------------------------------
# RichDisplay
# ---------------------------------------------------------------------------


class RichDisplay(Display):
    """
    Rich-backed display using scrolling output + prompt_toolkit input.

    Uses ``rich.live.Live(transient=True)`` while the LLM is streaming so
    raw text updates appear in place.  At ``end_assistant_turn()`` the live
    area is erased and the fully formatted turn is printed permanently.

    A bottom toolbar (token counter + timers + last status) is wired into the
    REPL's ``PromptSession`` via :meth:`prompt_session_kwargs`.
    """

    def __init__(self, *, verbose: bool = False, markdown_enabled: bool = True) -> None:
        super().__init__(verbose=verbose, markdown_enabled=markdown_enabled)
        self._console = Console(highlight=False, markup=False)
        self._stderr_console = Console(highlight=False, markup=False, stderr=True)
        # Streaming state
        self._live: Live | None = None
        self._text_acc: str = ""
        self._reasoning_acc: str = ""
        self._thinking_spinner: Spinner | None = None
        # Timing
        self._turn_start_time: datetime | None = None
        self._last_turn_duration: float | None = None
        # Usage / toolbar
        self._prompt_tokens: int = 0
        self._context_window: int = 0
        self._last_status: str = ""

    # ------------------------------------------------------------------
    # Bottom toolbar (injected into PromptSession)
    # ------------------------------------------------------------------

    def prompt_session_kwargs(self) -> dict:
        return {"bottom_toolbar": self._build_toolbar, "refresh_interval": 1}

    def _build_toolbar(self) -> str:
        now = datetime.now()
        parts: list[str] = []

        if self._context_window > 0:
            ratio = min(self._prompt_tokens / self._context_window, 1.0)
            filled = int(ratio * 20)
            bar = "█" * filled + "░" * (20 - filled)
            pct = int(ratio * 100)
            parts.append(f"[ctx: {bar} {pct}%]")

        if self._turn_start_time is not None:
            # Turn is active — show a live elapsed timer.
            elapsed = (now - self._turn_start_time).total_seconds()
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            parts.append(f"⏱ {mins:02d}:{secs:02d}")
        elif self._last_turn_duration is not None:
            # Turn finished — show the fixed duration of the last completed turn.
            mins = int(self._last_turn_duration // 60)
            secs = int(self._last_turn_duration % 60)
            parts.append(f"⏱ {mins:02d}:{secs:02d}")

        if self._last_status:
            parts.append(self._last_status)

        return "  ".join(parts)

    # ------------------------------------------------------------------
    # Streaming assistant output
    # ------------------------------------------------------------------

    def begin_assistant_turn(self) -> None:
        self._text_acc = ""
        self._reasoning_acc = ""
        self._turn_start_time = datetime.now()
        self._thinking_spinner = Spinner("dots", " Thinking…")
        self._console.print(Rule("Assistant", style="bold cyan", align="left"))
        if self._live is not None:
            self._live.stop()
        # _LiveRenderable re-evaluates _build_live_renderable() on every tick
        # so the toolbar timer advances even between streaming chunks.
        self._live = Live(
            _LiveRenderable(self._build_live_renderable),
            transient=True,
            console=self._console,
            refresh_per_second=10,
        )
        self._live.start()

    def stream_text(self, delta: str) -> None:
        self._text_acc += delta

    def stream_reasoning(self, delta: str) -> None:
        self._reasoning_acc += delta

    def _build_live_renderable(self) -> RenderableType:
        parts: list[RenderableType] = []

        # Show context bar + elapsed timer above the streaming text so the
        # user can see how long the model has been thinking.
        toolbar = self._build_toolbar()
        if toolbar:
            parts.append(Text(toolbar, style="dim"))

        if not self._text_acc and not self._reasoning_acc:
            # Waiting for the first token — show an animated spinner.
            if self._thinking_spinner is not None:
                parts.append(self._thinking_spinner)
        else:
            if self._reasoning_acc:
                preview = self._reasoning_acc[-200:]
                parts.append(Text(f"⟨thinking…⟩ {preview}", style="dim italic"))
            if self._text_acc:
                parts.append(Text(self._text_acc))

        if not parts:
            return Text("")
        return Group(*parts)

    def end_assistant_turn(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self._turn_start_time is not None:
            self._last_turn_duration = (
                datetime.now() - self._turn_start_time
            ).total_seconds()
            self._turn_start_time = None

        reasoning_text = self._reasoning_acc
        response_text = self._text_acc

        if self._verbose and reasoning_text:
            self._console.print(Rule("Reasoning", style="dim", align="left"))
            self._console.print(
                _LeftBorderRenderable(
                    Text(reasoning_text, style="dim italic"), style="dim"
                )
            )

        if response_text:
            if self._verbose and reasoning_text:
                self._console.print(Rule("Response", style="dim cyan", align="left"))
            content: RenderableType = (
                Markdown(response_text)
                if self._markdown_enabled
                else Text(response_text)
            )
            self._console.print(_LeftBorderRenderable(content, style="bold cyan"))

    def update_usage(self, usage: dict, context_window: int) -> None:
        self._prompt_tokens = usage.get("prompt_tokens", 0)
        self._context_window = context_window

    # ------------------------------------------------------------------
    # Tool activity
    # ------------------------------------------------------------------

    def show_tool_call(self, name: str, args: dict) -> None:
        if self._verbose:
            self._console.print(
                Rule(f"Tool: {name}", style="bold yellow", align="left")
            )
            if args:
                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_column("key", style="bold yellow")
                table.add_column("value")
                for k, v in args.items():
                    val_str = str(v)
                    if len(val_str) > 80:
                        val_str = val_str[:77] + "…"
                    table.add_row(k, val_str)
                self._console.print(_LeftBorderRenderable(table, style="bold yellow"))
        else:
            summary = ", ".join(f"{k}={v!r}" for k, v in args.items())
            self._console.print(f"▶ {name}({summary})")

    def show_tool_result(
        self, name: str, result: dict, display_str: str | None = None
    ) -> None:
        if self._verbose:
            if display_str is not None:
                self._console.print(
                    _LeftBorderRenderable(
                        Text.from_ansi(display_str), style="bold yellow"
                    )
                )
            else:
                json_str = json.dumps(result, indent=2, default=str)
                self._console.print(
                    _LeftBorderRenderable(
                        Syntax(json_str, "json", theme="monokai"), style="bold yellow"
                    )
                )
        # summary mode: silent

    # ------------------------------------------------------------------
    # Status and errors
    # ------------------------------------------------------------------

    def show_status(self, message: str) -> None:
        self._last_status = message
        self._console.print(f"  {message}", style="green")

    def show_error(self, message: str) -> None:
        self._last_status = f"✗ {message}"
        self._stderr_console.print(f"  ✗ {message}", style="bold red")

    # ------------------------------------------------------------------
    # Slash-command output
    # ------------------------------------------------------------------

    def show_help(self, commands: list[tuple[str, str]]) -> None:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("cmd", style="bold cyan")
        table.add_column("desc")
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        self._console.print("\nAvailable commands:")
        self._console.print(table)

    def show_tool_list(self, tools: list[Tool]) -> None:
        if not tools:
            self._console.print("No tools currently enabled.")
            return
        table = Table(show_header=True)
        table.add_column("Tool", style="bold")
        table.add_column("Description")
        for tool in tools:
            table.add_row(tool.name, tool.description)
        self._console.print("\nEnabled tools:")
        self._console.print(table)

    def show_session_info(self, session: Session) -> None:
        meta = session.get_meta()
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("key", style="bold")
        table.add_column("value")
        table.add_row("Session", session.session_id)
        if meta.get("name"):
            table.add_row("Name", str(meta["name"]))
        table.add_row("Started", str(meta.get("started_at", "unknown")))
        table.add_row("Messages", str(meta.get("message_count", 0)))
        self._console.print(table)

    def show_tool_list_all(self, tools_info: list[dict]) -> None:
        if not tools_info:
            self._console.print("No tools registered.")
            return
        table = Table(show_header=True)
        table.add_column("Name", style="bold")
        table.add_column("Status")
        table.add_column("Tier")
        table.add_column("Description")
        for info in tools_info:
            if not info.get("allowed", True):
                status = "disallowed"
                status_style = "dim red"
            elif info.get("enabled", False):
                status = "enabled"
                status_style = "green"
            else:
                status = "disabled"
                status_style = "dim"
            if info.get("permission_required", False):
                status += ", perm"
            desc = info.get("description", "")[:50]
            table.add_row(
                info["name"],
                Text(status, style=status_style),
                info.get("tier", ""),
                desc,
            )
        self._console.print("\nAll tools:")
        self._console.print(table)

    def show_agents(self, rows: list[dict]) -> None:
        if not rows:
            self._console.print("No agent types configured.")
            return
        table = Table(show_header=True)
        table.add_column("Name", style="bold")
        table.add_column("Model")
        table.add_column("Persistence")
        table.add_column("Max Rounds", justify="right")
        table.add_column("Tools")
        for row in rows:
            table.add_row(
                row["name"],
                row["model"],
                row["persistence"],
                str(row["max_tool_rounds"]),
                row["tools"],
            )
        self._console.print("\nConfigured agent types:")
        self._console.print(table)

    def show_tool_info(self, tool_info: dict) -> None:
        name = tool_info.get("name", "")
        self._console.print(f"\nTool: {name}", style="bold")
        if not tool_info.get("allowed", True):
            status = "disallowed"
        elif tool_info.get("enabled", False):
            status = "enabled"
        else:
            status = "disabled"
        perm = "required" if tool_info.get("permission_required") else "not required"
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("key", style="bold")
        table.add_column("value")
        table.add_row("Description", tool_info.get("description", ""))
        table.add_row("Tier", tool_info.get("tier", "unknown"))
        table.add_row("Status", status)
        table.add_row("Permission", perm)
        self._console.print(table)
        params = tool_info.get("parameters", {})
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        required = params.get("required", []) if isinstance(params, dict) else []
        if props:
            self._console.print("  Parameters:", style="bold")
            for pname, pdef in props.items():
                req = " (required)" if pname in required else ""
                ptype = pdef.get("type", "") if isinstance(pdef, dict) else ""
                pdesc = pdef.get("description", "") if isinstance(pdef, dict) else ""
                self._console.print(f"    {pname}: {ptype}{req} — {pdesc}")

    def show_history(self, messages: list[dict]) -> None:
        with self._console.pager(styles=True):
            for msg in messages:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if role == "user":
                    rule_style = "bold green"
                    border_style = "bold green"
                elif role == "assistant":
                    rule_style = "bold cyan"
                    border_style = "bold cyan"
                else:
                    rule_style = "bold yellow"
                    border_style = "bold yellow"
                self._console.print(
                    Rule(role.capitalize(), style=rule_style, align="left")
                )
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
                if content:
                    renderable: RenderableType
                    if role == "assistant" and self._markdown_enabled:
                        renderable = Markdown(content)
                    else:
                        renderable = Text(content)
                    self._console.print(
                        _LeftBorderRenderable(renderable, style=border_style)
                    )

    # ------------------------------------------------------------------
    # Interactive prompts
    # ------------------------------------------------------------------

    def show_permission_prompt(
        self,
        question: str,
        extra_options: list[str],
    ) -> tuple[str, str]:
        self._console.print(f"\n{question}", style="bold")
        for key, _, label in _UNIVERSAL_OPTIONS:
            self._console.print(f"  [{key}] {label}")
        for i, opt in enumerate(extra_options):
            self._console.print(f"  [{i}] {opt}")

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
            self._console.print("Invalid choice, please try again.", style="dim red")

    def show_session_list(self, sessions: list[SessionMeta]) -> SessionMeta | None:
        if not sessions:
            return None
        table = Table(show_header=True)
        table.add_column("#", style="bold")
        table.add_column("Started")
        table.add_column("Session ID")
        table.add_column("Preview")
        table.add_column("Msgs", justify="right")
        for i, s in enumerate(sessions):
            ts = s.started_at.strftime("%Y-%m-%d %H:%M UTC")
            preview = s.first_user_message[:60] or "(no messages)"
            table.add_row(str(i), ts, s.session_id, preview, str(s.message_count))
        self._console.print("\nResumable sessions:")
        self._console.print(table)
        self._console.print("  [q] Start a new session")

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
            self._console.print("Invalid choice, please try again.", style="dim red")


# ---------------------------------------------------------------------------
# SubAgentDisplay — captures output in a buffer for non-interactive agents
# ---------------------------------------------------------------------------


class SubAgentDisplay(Display):
    """Display backend for sub-agents.

    Instead of writing to a terminal, all streamed text is captured in an
    internal buffer.  Tool calls and status messages are routed to the
    logger.  Permission prompts are denied by default — sub-agents should
    not require interactive approval at runtime.
    """

    def __init__(self, *, verbose: bool = False, markdown_enabled: bool = True) -> None:
        super().__init__(verbose=verbose, markdown_enabled=markdown_enabled)
        self._buffer: list[str] = []
        self._last_usage: dict = {}

    # -- Public API --------------------------------------------------------

    @property
    def captured_text(self) -> str:
        """Return all streamed text as a single string."""
        return "".join(self._buffer)

    def reset(self) -> None:
        """Clear the buffer for reuse (session-persistent agents)."""
        self._buffer.clear()
        self._last_usage = {}

    # -- Streaming assistant output ----------------------------------------

    def begin_assistant_turn(self) -> None:
        pass

    def stream_text(self, delta: str) -> None:
        self._buffer.append(delta)

    def stream_reasoning(self, delta: str) -> None:
        logger.debug("Sub-agent reasoning: %s", delta)

    def end_assistant_turn(self) -> None:
        pass

    def update_usage(self, usage: dict, context_window: int) -> None:
        self._last_usage = usage

    # -- Tool activity -----------------------------------------------------

    def show_tool_call(self, name: str, args: dict) -> None:
        if self._verbose:
            logger.debug("Sub-agent tool call: %s (keys: %s)", name, list(args))
        else:
            logger.debug("Sub-agent tool call: %s", name)

    def show_tool_result(
        self, name: str, result: dict, display_str: str | None = None
    ) -> None:
        logger.debug("Sub-agent tool result: %s → %s", name, result.get("status"))

    # -- Status and errors -------------------------------------------------

    def show_status(self, message: str) -> None:
        logger.info("Sub-agent status: %s", message)

    def show_error(self, message: str) -> None:
        logger.warning("Sub-agent error: %s", message)

    # -- Slash-command output (no-ops for sub-agents) ----------------------

    def show_help(self, commands: list[tuple[str, str]]) -> None:
        pass

    def show_tool_list(self, tools: list[Tool]) -> None:
        pass

    def show_session_info(self, session: Session) -> None:
        pass

    def show_tool_list_all(self, tools_info: list[dict]) -> None:
        pass

    def show_tool_info(self, tool_info: dict) -> None:
        pass

    def show_history(self, messages: list[dict]) -> None:
        pass

    def show_agents(self, rows: list[dict]) -> None:
        pass

    # -- Interactive prompts -----------------------------------------------

    def show_permission_prompt(
        self,
        question: str,
        extra_options: list[str],
    ) -> tuple[str, str]:
        logger.warning(
            "Sub-agent permission prompt denied (non-interactive): %s", question
        )
        return ("no", "")

    def show_session_list(self, sessions: list[SessionMeta]) -> SessionMeta | None:
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_display(config: ConfigManager, *, verbose: bool = False) -> Display:
    """
    Instantiate and return the configured :class:`Display` backend.

    Reads ``display_backend`` (default ``'rich'``) and ``display_markdown``
    (default ``True``) from *config* via ``config.get()``.  An unknown backend
    name logs a warning and falls back to ``PlainDisplay``.
    """
    backend: str = config.get("display_backend", "rich")
    markdown_enabled: bool = config.get("display_markdown", True)

    if backend == "plain":
        return PlainDisplay(verbose=verbose, markdown_enabled=markdown_enabled)

    if backend == "rich":
        return RichDisplay(verbose=verbose, markdown_enabled=markdown_enabled)

    logger.warning("Unknown display_backend %r; falling back to PlainDisplay.", backend)
    return PlainDisplay(verbose=verbose, markdown_enabled=markdown_enabled)
