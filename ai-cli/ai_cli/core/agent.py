"""
Agent — declarative specs, result types, and the runtime loop.

``AgentSpec`` describes an agent type (parsed from config); ``AgentResult``
carries the outcome of a single ``Agent.run()`` invocation; ``Agent`` drives
the send → stream → tool-call → repeat loop.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ai_cli.core.llm_client import LLMError
from ai_cli.core.session_manager import SessionError

if TYPE_CHECKING:
    from ai_cli.cli.display import Display
    from ai_cli.core.llm_client import LLMClient
    from ai_cli.core.session_manager import Session
    from ai_cli.core.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class BackendConfig:
    """Connection details for an LLM backend.

    When ``None`` on an ``AgentSpec``, the agent inherits the
    coordinator's backend.  Authentication follows the same contract as
    ``ConfigManager``: config stores the environment variable name
    (``api_key_env``), and the actual key is resolved from the environment
    at agent instantiation time.
    """

    base_url: str
    api_key_env: str | None = None


@dataclass
class AgentSpec:
    """Declarative description of an agent type, parsed from config."""

    name: str
    system_message: str
    tools: list[str]
    model: str
    max_response_tokens: int = 4096
    persistence: Literal["ephemeral", "session"] = "ephemeral"
    backend: BackendConfig | None = None
    tool_permission_overrides: dict[str, bool] = field(default_factory=dict)
    max_tool_rounds: int = 10
    context_limit_threshold: float = 0.90


@dataclass
class AgentResult:
    """Returned by ``Agent.run()`` when the send/tool/repeat loop ends."""

    text: str
    status: Literal["ok", "context_limit", "tool_limit", "error"]
    partial: bool = False
    error_message: str = ""


class Agent:
    """Runtime agent that drives the send → tool-call → repeat loop.

    The body of :meth:`run` is extracted from what was previously
    ``REPL._send_rounds``.  It depends only on ``Session``, ``LLMClient``,
    ``ToolRegistry``, and ``Display`` — no REPL-specific state.
    """

    def __init__(
        self,
        spec: AgentSpec,
        session: Session,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        display: Display,
    ) -> None:
        self.spec = spec
        self._session = session
        self._llm = llm_client
        self._tool_registry = tool_registry
        self._display = display
        self._pending_transients: dict[str, dict] = {}

    def run(
        self,
        prompt: str | list[dict],
        *,
        abort: threading.Event | None = None,
    ) -> AgentResult:
        """Drive the send → tool-call → repeat loop for one prompt.

        Adds *prompt* as a user message to the session, then enters the
        send/tool-call loop.  Returns an ``AgentResult`` when the LLM
        issues end_turn, the tool-round limit is hit, or the abort event
        is set.
        """
        try:
            if isinstance(prompt, list):
                self._session.add_raw_message({"role": "user", "content": prompt})
            else:
                self._session.add_message("user", prompt)
        except SessionError as exc:
            self._display.show_error(f"Could not save message: {exc}")
            return AgentResult(
                text="",
                status="error",
                partial=False,
                error_message=str(exc),
            )

        all_text_parts: list[str] = []

        for _ in range(self.spec.max_tool_rounds):
            if abort is not None and abort.is_set():
                self._display.show_status("Aborted.")
                return AgentResult(
                    text="".join(all_text_parts), status="ok", partial=True
                )

            tool_calls: list[dict] = []
            text_parts: list[str] = []

            try:
                messages = self._session.get_messages()
            except SessionError as exc:
                self._display.show_error(f"Could not read conversation history: {exc}")
                return AgentResult(
                    text="".join(all_text_parts),
                    status="error",
                    partial=True,
                    error_message=str(exc),
                )

            # Consume transients injected by tool_manager.enable in the
            # previous round, then clear so they don't persist beyond this
            # round.
            active_transients = dict(self._pending_transients)
            self._pending_transients.clear()

            self._display.begin_assistant_turn()
            stream = None
            try:
                # Build the tools list, de-duplicating by name so that a
                # transient schema for an already-enabled tool doesn't appear
                # twice (some LLM APIs reject duplicate tool names).
                # Transient schemas take precedence over the enabled
                # definitions.
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
                    if abort is not None and abort.is_set():
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
                if abort is not None:
                    abort.set()
                else:
                    raise
            except LLMError as exc:
                self._display.show_error(f"LLM error: {exc}")
                return AgentResult(
                    text="".join(all_text_parts),
                    status="error",
                    partial=True,
                    error_message=str(exc),
                )
            finally:
                if stream is not None:
                    with contextlib.suppress(Exception):
                        stream.close()
                self._display.end_assistant_turn()

            full_text = "".join(text_parts)
            all_text_parts.extend(text_parts)

            if abort is not None and abort.is_set():
                self._display.show_status("Aborted.")
                return AgentResult(
                    text="".join(all_text_parts), status="ok", partial=True
                )

            if tool_calls:
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
                    return AgentResult(
                        text="".join(all_text_parts),
                        status="error",
                        partial=True,
                        error_message=str(exc),
                    )
            elif full_text:
                try:
                    self._session.add_message("assistant", full_text)
                except SessionError as exc:
                    self._display.show_error(f"Could not save assistant message: {exc}")
                    return AgentResult(
                        text="".join(all_text_parts),
                        status="error",
                        partial=True,
                        error_message=str(exc),
                    )

            if not tool_calls:
                break

            for i, call in enumerate(tool_calls):
                if abort is not None and abort.is_set():
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
                    return AgentResult(
                        text="".join(all_text_parts),
                        status="ok",
                        partial=True,
                    )
                self._display.show_tool_call(call["name"], call["arguments"])
                allow_transient = call["name"] in active_transients
                result = self._tool_registry.execute(
                    call["name"],
                    call["arguments"],
                    allow_transient=allow_transient,
                )
                if result.get("error") == "tool_disallowed":
                    self._display.show_error(
                        f"Tool '{call['name']}' is not available in the "
                        f"current configuration. Use '/tools allow "
                        f"{call['name']}' to add it to the list of "
                        f"available tools."
                    )
                    result = {
                        "status": "error",
                        "error": "unknown_tool",
                        "message": f"No tool named '{call['name']}'.",
                        "code": 404,
                    }
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
                    return AgentResult(
                        text="".join(all_text_parts),
                        status="error",
                        partial=True,
                        error_message=str(exc),
                    )
        else:
            logger.warning(
                "Tool call limit (%d rounds) reached; stopping.",
                self.spec.max_tool_rounds,
            )
            return AgentResult(
                text="".join(all_text_parts),
                status="tool_limit",
                partial=True,
            )

        return AgentResult(text="".join(all_text_parts), status="ok")
