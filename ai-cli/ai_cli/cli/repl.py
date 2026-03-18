"""
repl.py — Main REPL loop.

Reads user input via prompt_toolkit, routes slash commands and plain text,
drives the LLM streaming loop (including the agentic tool-call cycle), and
coordinates Session, ToolRegistry, LLMClient, and Display.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from ai_cli.core.llm_client import LLMError
from ai_cli.core.session_manager import SessionError
from ai_cli.core.workspace import WorkspaceError, get_global_dir

if TYPE_CHECKING:
    from ai_cli.cli.display import Display
    from ai_cli.core.llm_client import LLMClient
    from ai_cli.core.session_manager import Session
    from ai_cli.core.tool_registry import ToolRegistry
    from ai_cli.core.workspace import Workspace

logger = logging.getLogger(__name__)

# Maximum number of consecutive tool-call rounds per user turn.
_MAX_TOOL_ROUNDS = 10

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
    ("/compact", "Summarise the conversation to free up context window space"),
    ("/tools", "List currently enabled tools"),
    ("/session", "Show information about the current session"),
]


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
    """

    def __init__(
        self,
        session: Session,
        tool_registry: ToolRegistry,
        llm_client: LLMClient,
        display: Display,
        workspace: Workspace,
    ) -> None:
        self._session = session
        self._tool_registry = tool_registry
        self._llm = llm_client
        self._display = display
        self._workspace = workspace
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
            _prompt_session = PromptSession(history=FileHistory(str(history_path)))

        while True:
            try:
                raw = _prompt_session.prompt("> ")
            except KeyboardInterrupt:
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
            self._send_to_llm(user_input)

    def _handle_slash_command(self, command: str) -> None:
        cmd = command.split()[0].lower() if command.strip() else ""

        if cmd == "help":
            self._display.show_help(_SLASH_COMMANDS)

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
            self._display.show_status("Compacting conversation history…")
            try:
                self._session.compact()
                self._display.show_status("Compaction complete.")
            except SessionError as exc:
                self._display.show_error(f"Compaction failed: {exc}")

        elif cmd == "tools":
            self._display.show_tool_list(self._tool_registry.all_enabled())

        elif cmd == "session":
            self._display.show_session_info(self._session)

        elif cmd == "":
            self._display.show_error(
                "No command provided. Type /help for a list of commands."
            )

        else:
            self._display.show_error(
                f"Unknown command: /{cmd}. Type /help for a list of commands."
            )

    # ------------------------------------------------------------------
    # @ file reference expansion
    # ------------------------------------------------------------------

    def _preprocess_at_references(self, text: str) -> str:
        """
        Replace ``@path`` and ``@!path`` tokens with the file's content.

        ``@path`` respects ignore rules; ``@!path`` bypasses them.
        On any error the token is left in place and an error is shown.
        """

        def replace(match: re.Match[str]) -> str:
            bypass_ignore = bool(match.group(1))
            path = match.group(2)
            try:
                if bypass_ignore:
                    resolved = self._workspace.resolve(path)
                    content: str = resolved.read_text(encoding="utf-8")
                else:
                    if not self._workspace.file_exists(path):
                        self._display.show_error(
                            f"@{path}: file not found or excluded by ignore rules"
                        )
                        return str(match.group(0))
                    content = self._workspace.read_file(path)
            except (WorkspaceError, OSError, UnicodeDecodeError) as exc:
                self._display.show_error(f"@{path}: {exc}")
                return str(match.group(0))
            return f"[file: {path}]\n{content}\n[/file]"

        return _AT_RE.sub(replace, text)

    # ------------------------------------------------------------------
    # LLM streaming and agentic tool loop
    # ------------------------------------------------------------------

    def _send_to_llm(self, user_input: str) -> None:
        try:
            self._session.add_message("user", user_input)
        except SessionError as exc:
            self._display.show_error(f"Could not save message: {exc}")
            return

        for _ in range(_MAX_TOOL_ROUNDS):
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
                for chunk in self._llm.send(
                    messages,
                    tools=list(tools_by_name.values()),
                ):
                    if chunk["type"] == "text":
                        self._display.stream_text(chunk["delta"])
                        text_parts.append(chunk["delta"])
                    elif chunk["type"] == "tool_call":
                        tool_calls.append(chunk)
                    elif chunk["type"] == "done":
                        prompt_tokens = chunk.get("usage", {}).get("prompt_tokens")
                        if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                            self._session.record_usage(prompt_tokens)
            except LLMError as exc:
                self._display.end_assistant_turn()
                self._display.show_error(f"LLM error: {exc}")
                return
            self._display.end_assistant_turn()

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

            for call in tool_calls:
                self._display.show_tool_call(call["name"], call["arguments"])
                # Transiently-enabled tools must bypass the registry's enabled
                # check — they were injected into the LLM's tools list for this
                # round specifically, so allow_transient=True lets them execute.
                allow_transient = call["name"] in active_transients
                result = self._tool_registry.execute(
                    call["name"], call["arguments"], allow_transient=allow_transient
                )
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
                self._display.show_tool_result(call["name"], result)
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
                _MAX_TOOL_ROUNDS,
            )
            self._display.show_error(
                f"Tool call limit ({_MAX_TOOL_ROUNDS} rounds) reached. Stopping."
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
