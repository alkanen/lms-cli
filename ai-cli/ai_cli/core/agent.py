"""
Agent — declarative specs, result types, and the runtime loop.

``AgentSpec`` describes an agent type (parsed from config); ``AgentResult``
carries the outcome of a single ``Agent.run()`` invocation; ``Agent`` drives
the send → stream → tool-call → repeat loop.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ai_cli.core.llm_client import LLMError
from ai_cli.core.session_manager import SessionError, SessionProtocol

if TYPE_CHECKING:
    from ai_cli.cli.display import Display
    from ai_cli.core.config_manager import ConfigManager
    from ai_cli.core.llm_client import LLMClient
    from ai_cli.core.tool_registry import ToolRegistry
    from ai_cli.core.workspace import Workspace

logger = logging.getLogger(__name__)

_INFO_TEXT_PREVIEW_CHARS = 120
_DEBUG_TEXT_PREVIEW_CHARS = 800


def _preview_text(text: str, limit: int) -> str:
    """Return a truncated preview of *text* capped at *limit* chars."""
    if len(text) <= limit:
        return text
    return text[:limit]


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
    # Optional per-agent context-window override.  When set, this takes
    # precedence over the LLM client's reported context_window so that
    # agents running different models (or even different servers) can have
    # accurate threshold checks without needing a separate backend entry.
    context_window: int | None = None
    # Optional allow-list for the `skills` tool. None means unrestricted
    # (all loaded skills available). An empty list means no skills allowed.
    skills: list[str] | None = None


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
        session: SessionProtocol,
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
        # Tracks the role of the last message successfully persisted to the
        # session so that _close_open_tool_cycle() can decide whether to inject
        # a synthetic assistant message without an O(n) get_messages() read.
        self._last_persisted_role: str = ""

    def reset(self) -> None:
        """Reset per-run state for agent reuse.

        Called by :class:`~ai_cli.core.agent_registry.AgentRegistry` on
        session-persistent agents before each new delegation so the captured
        display output and any pending transient schemas from the previous run
        do not bleed into the next result.
        """
        logger.debug(
            "Agent '%s': resetting runtime state (pending_transients=%d, last_role=%r)",
            self.spec.name,
            len(self._pending_transients),
            self._last_persisted_role,
        )
        self._display.reset()
        self._pending_transients.clear()
        self._last_persisted_role = ""

    def _close_open_tool_cycle(self, reason: str) -> None:
        """Inject a synthetic assistant message if the session ends with role=tool.

        Tool-response messages must always be followed by an assistant message
        before the next user turn.  Call this before any early return that does
        not otherwise write a closing assistant message, so the session is left
        in a state the LLM will accept on the next prompt.

        Uses ``_last_persisted_role`` as a fast-path guard.  When the tracked
        role is "assistant" the cycle is already closed and no read is needed.
        When it is "tool" we injected the tool messages ourselves and know a
        close is required without a read.  Only when the tracked role is
        ambiguous (empty string or "user") do we fall back to ``get_messages()``
        to handle tool cycles left open by prior runs.
        """
        if self._last_persisted_role == "assistant":
            logger.debug(
                "Agent '%s': tool cycle already closed; no synthetic assistant message needed",
                self.spec.name,
            )
            return
        if self._last_persisted_role != "tool":
            # Ambiguous: may be the start of a new run on a session-persistent
            # agent whose previous run ended with a tool message.  Check the
            # actual session to be safe.
            try:
                messages = self._session.get_messages()
            except SessionError as exc:
                logger.warning(
                    "Agent '%s': failed to inspect session while checking for open tool cycle: %s",
                    self.spec.name,
                    exc,
                )
                return
            if not (messages and messages[-1].get("role") == "tool"):
                logger.debug(
                    "Agent '%s': session does not end with tool role; no repair needed",
                    self.spec.name,
                )
                return
        logger.info(
            "Agent '%s': closing open tool cycle with synthetic assistant message (%s)",
            self.spec.name,
            reason,
        )
        try:
            self._session.add_message("assistant", reason)
            self._last_persisted_role = "assistant"
        except SessionError as exc:
            logger.error(
                "Agent '%s': failed to write closing assistant message: %s",
                self.spec.name,
                exc,
            )

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
        prompt_kind = "blocks" if isinstance(prompt, list) else "text"
        prompt_size = len(prompt) if isinstance(prompt, list) else len(prompt)
        logger.info(
            "Agent '%s': run started (prompt_kind=%s, prompt_size=%d, max_tool_rounds=%d, persistence=%s)",
            self.spec.name,
            prompt_kind,
            prompt_size,
            self.spec.max_tool_rounds,
            self.spec.persistence,
        )
        # Close any open tool cycle from a prior aborted/crashed run *before*
        # appending the new user message, so the session never reaches
        # [..., tool, user].  Force the shared helper to consult the actual
        # session state by clearing the sentinel — _last_persisted_role may not
        # reflect external inconsistencies (crashes, other writers).
        self._last_persisted_role = ""  # sentinel: forces get_messages() fallback
        self._close_open_tool_cycle("[Prior run ended without closing tool cycle.]")
        # Always re-verify the actual session state after the close attempt so
        # _last_persisted_role is fresh (not a stale cached value), and to detect
        # a failed close (add_message raised SessionError inside the helper).
        try:
            _prior_msgs = self._session.get_messages()
            self._last_persisted_role = (
                _prior_msgs[-1].get("role", "") if _prior_msgs else ""
            )
        except SessionError as exc:
            logger.warning(
                "Agent '%s': could not verify session state after attempting to close prior tool cycle: %s",
                self.spec.name,
                exc,
            )
            self._display.show_error(
                f"Could not verify session state after closing prior tool cycle: {exc}"
            )
            return AgentResult(
                text="",
                status="error",
                partial=False,
                error_message=str(exc),
            )
        if self._last_persisted_role == "tool":
            msg = "Could not close prior tool cycle; refusing to append user message."
            logger.warning("Agent '%s': %s", self.spec.name, msg)
            self._display.show_error(msg)
            return AgentResult(
                text="",
                status="error",
                partial=False,
                error_message=msg,
            )

        try:
            if isinstance(prompt, list):
                self._session.add_raw_message({"role": "user", "content": prompt})
            else:
                self._session.add_message("user", prompt)
            self._last_persisted_role = "user"
            logger.debug(
                "Agent '%s': persisted user prompt (prompt_kind=%s)",
                self.spec.name,
                prompt_kind,
            )
        except SessionError as exc:
            logger.warning(
                "Agent '%s': failed to persist user prompt: %s",
                self.spec.name,
                exc,
            )
            self._display.show_error(f"Could not save message: {exc}")
            return AgentResult(
                text="",
                status="error",
                partial=False,
                error_message=str(exc),
            )

        all_text_parts: list[str] = []
        context_limit_hit = False

        for round_index in range(self.spec.max_tool_rounds):
            logger.debug(
                "Agent '%s': starting round %d/%d",
                self.spec.name,
                round_index + 1,
                self.spec.max_tool_rounds,
            )
            if abort is not None and abort.is_set():
                logger.info(
                    "Agent '%s': abort detected before round %d",
                    self.spec.name,
                    round_index + 1,
                )
                self._close_open_tool_cycle("[Aborted by user.]")
                self._display.show_status("Aborted.")
                return AgentResult(
                    text="".join(all_text_parts), status="ok", partial=True
                )

            tool_calls: list[dict] = []
            text_parts: list[str] = []

            try:
                messages = self._session.get_messages()
            except SessionError as exc:
                logger.warning(
                    "Agent '%s': failed to read conversation history in round %d: %s",
                    self.spec.name,
                    round_index + 1,
                    exc,
                )
                self._display.show_error(f"Could not read conversation history: {exc}")
                self._close_open_tool_cycle("[Session read error.]")
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
            logger.debug(
                "Agent '%s': round %d using %d messages and %d transient tool schemas",
                self.spec.name,
                round_index + 1,
                len(messages),
                len(active_transients),
            )

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
                logger.debug(
                    "Agent '%s': sending round %d to LLM with %d tools",
                    self.spec.name,
                    round_index + 1,
                    len(tools_by_name),
                )
                stream = self._llm.send(
                    messages,
                    tools=list(tools_by_name.values()),
                )
                for chunk in stream:
                    if abort is not None and abort.is_set():
                        logger.info(
                            "Agent '%s': abort detected while streaming round %d",
                            self.spec.name,
                            round_index + 1,
                        )
                        break
                    if chunk["type"] == "text":
                        self._display.stream_text(chunk["delta"])
                        text_parts.append(chunk["delta"])
                    elif chunk["type"] == "reasoning":
                        self._display.stream_reasoning(chunk["delta"])
                    elif chunk["type"] == "tool_call":
                        logger.info(
                            "Agent '%s': LLM requested tool '%s' in round %d",
                            self.spec.name,
                            chunk.get("name"),
                            round_index + 1,
                        )
                        tool_calls.append(chunk)
                    elif chunk["type"] == "done":
                        usage = chunk.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens")
                        if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                            self._session.record_usage(prompt_tokens)
                        # Prefer the per-agent override so agents running
                        # a different model (or backend) than the
                        # coordinator get an accurate threshold check.
                        context_window = (
                            self.spec.context_window
                            if self.spec.context_window is not None
                            else self._llm.get_model_metadata().get("context_window", 0)
                        )
                        logger.debug(
                            "Agent '%s': round %d completed stream stop_reason=%r usage=%r context_window=%r",
                            self.spec.name,
                            round_index + 1,
                            chunk.get("stop_reason"),
                            usage,
                            context_window,
                        )
                        self._display.update_usage(usage, context_window)
                        # Context overflow check — break out of the stream loop
                        # so the assistant message is persisted before returning.
                        if (
                            isinstance(context_window, int)
                            and context_window > 0
                            and isinstance(prompt_tokens, int)
                            and prompt_tokens / context_window
                            >= self.spec.context_limit_threshold
                        ):
                            context_limit_msg = (
                                f"Context limit reached "
                                f"({prompt_tokens}/{context_window} tokens)."
                            )
                            logger.warning(
                                "Agent '%s': context limit reached in round %d (%d/%d >= %.2f)",
                                self.spec.name,
                                round_index + 1,
                                prompt_tokens,
                                context_window,
                                self.spec.context_limit_threshold,
                            )
                            self._display.show_status(context_limit_msg)
                            context_limit_hit = True
                            break
                    else:
                        logger.warning(
                            "Agent '%s': ignoring unknown stream chunk type %r in round %d",
                            self.spec.name,
                            chunk.get("type"),
                            round_index + 1,
                        )
            except KeyboardInterrupt:
                logger.warning(
                    "Agent '%s': keyboard interrupt received in round %d",
                    self.spec.name,
                    round_index + 1,
                )
                if abort is not None:
                    abort.set()
                else:
                    raise
            except LLMError as exc:
                logger.warning(
                    "Agent '%s': LLM error in round %d: %s",
                    self.spec.name,
                    round_index + 1,
                    exc,
                )
                self._close_open_tool_cycle("[LLM error.]")
                self._display.show_error(f"LLM error: {exc}")
                return AgentResult(
                    text="".join(all_text_parts),
                    status="error",
                    partial=True,
                    error_message=str(exc),
                )
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception as exc:
                        logger.warning(
                            "Agent '%s': failed to close LLM stream in round %d: %s",
                            self.spec.name,
                            round_index + 1,
                            exc,
                        )
                self._display.end_assistant_turn()

            full_text = "".join(text_parts)
            all_text_parts.extend(text_parts)
            logger.debug(
                "Agent '%s': round %d produced text_chars=%d tool_calls=%d",
                self.spec.name,
                round_index + 1,
                len(full_text),
                len(tool_calls),
            )

            if abort is not None and abort.is_set():
                logger.info(
                    "Agent '%s': abort detected after round %d stream processing",
                    self.spec.name,
                    round_index + 1,
                )
                self._close_open_tool_cycle("[Aborted by user.]")
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
                    self._last_persisted_role = "assistant"
                    logger.debug(
                        "Agent '%s': persisted assistant tool-call message with %d calls",
                        self.spec.name,
                        len(tool_calls),
                    )
                except SessionError as exc:
                    logger.warning(
                        "Agent '%s': failed to persist assistant tool-call message: %s",
                        self.spec.name,
                        exc,
                    )
                    self._display.show_error(f"Could not save assistant message: {exc}")
                    self._close_open_tool_cycle("[Session write error.]")
                    return AgentResult(
                        text="".join(all_text_parts),
                        status="error",
                        partial=True,
                        error_message=str(exc),
                    )
            elif full_text:
                try:
                    self._session.add_message("assistant", full_text)
                    self._last_persisted_role = "assistant"
                    logger.debug(
                        "Agent '%s': persisted assistant text message (%d chars)",
                        self.spec.name,
                        len(full_text),
                    )
                except SessionError as exc:
                    logger.warning(
                        "Agent '%s': failed to persist assistant text message: %s",
                        self.spec.name,
                        exc,
                    )
                    self._display.show_error(f"Could not save assistant message: {exc}")
                    self._close_open_tool_cycle("[Session write error.]")
                    return AgentResult(
                        text="".join(all_text_parts),
                        status="error",
                        partial=True,
                        error_message=str(exc),
                    )

            if context_limit_hit:
                # Stub responses for any tool_calls the LLM emitted in this
                # turn so the session history stays consistent — dangling
                # tool_calls without responses will confuse the next LLM call.
                for pending in tool_calls:
                    try:
                        self._session.add_raw_message(
                            {
                                "role": "tool",
                                "tool_call_id": pending["call_id"],
                                "content": json.dumps(
                                    {
                                        "status": "error",
                                        "error": "context_limit",
                                        "message": "Context limit reached; tool not executed.",
                                        "code": 503,
                                    }
                                ),
                            }
                        )
                        self._last_persisted_role = "tool"
                    except SessionError as exc:
                        logger.error(
                            "Agent '%s': failed to inject context-limit stub for call_id=%r: %s",
                            self.spec.name,
                            pending["call_id"],
                            exc,
                        )
                self._close_open_tool_cycle("[Stopped: context limit reached.]")
                _cl_text = "".join(all_text_parts)
                logger.info(
                    "Agent '%s' finished: status=context_limit error=%r text_len=%d text_preview=%r",
                    self.spec.name,
                    context_limit_msg,
                    len(_cl_text),
                    _preview_text(_cl_text, _INFO_TEXT_PREVIEW_CHARS),
                )
                logger.debug(
                    "Agent '%s' context_limit text_debug_preview=%r",
                    self.spec.name,
                    _preview_text(_cl_text, _DEBUG_TEXT_PREVIEW_CHARS),
                )
                return AgentResult(
                    text=_cl_text,
                    status="context_limit",
                    partial=True,
                    error_message=context_limit_msg,
                )

            if not tool_calls:
                break

            for i, call in enumerate(tool_calls):
                if abort is not None and abort.is_set():
                    logger.info(
                        "Agent '%s': abort detected before executing tool index %d in round %d",
                        self.spec.name,
                        i,
                        round_index + 1,
                    )
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
                            self._last_persisted_role = "tool"
                        except SessionError as exc:
                            logger.error(
                                "Agent '%s': failed to inject abort stub for call_id=%r: %s",
                                self.spec.name,
                                pending["call_id"],
                                exc,
                            )
                    self._close_open_tool_cycle("[Aborted by user.]")
                    self._display.show_status("Aborted.")
                    return AgentResult(
                        text="".join(all_text_parts),
                        status="ok",
                        partial=True,
                    )
                self._display.show_tool_call(call["name"], call["arguments"])
                allow_transient = call["name"] in active_transients
                logger.info(
                    "Agent '%s': executing tool '%s' (call_id=%s, transient=%s)",
                    self.spec.name,
                    call["name"],
                    call.get("call_id"),
                    allow_transient,
                )
                result = self._tool_registry.execute(
                    call["name"],
                    call["arguments"],
                    allow_transient=allow_transient,
                )
                if result.get("error") == "tool_disallowed":
                    logger.warning(
                        "Agent '%s': tool '%s' was disallowed by registry; returning unknown_tool to model",
                        self.spec.name,
                        call["name"],
                    )
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
                                logger.warning(
                                    "Agent '%s': ignoring transient schema with invalid type %r",
                                    self.spec.name,
                                    type(schema).__name__,
                                )
                                continue
                            func = schema.get("function")
                            if not isinstance(func, dict):
                                logger.warning(
                                    "Agent '%s': ignoring transient schema without function mapping: %r",
                                    self.spec.name,
                                    schema,
                                )
                                continue
                            name = func.get("name")
                            if name and self._tool_registry.get(name) is not None:
                                self._pending_transients[name] = schema
                                logger.info(
                                    "Agent '%s': enabled transient tool schema '%s' for next round",
                                    self.spec.name,
                                    name,
                                )
                            else:
                                logger.warning(
                                    "Agent '%s': transient schema '%s' ignored because no matching tool is registered",
                                    self.spec.name,
                                    name,
                                )
                elif data is not None:
                    data.pop("transient_schemas", None)
                display_str: str | None = None
                tool_obj = self._tool_registry.get(call["name"])
                if tool_obj is not None:
                    try:
                        display_str = tool_obj.format_display(
                            args=call["arguments"], result=result
                        )
                    except Exception as exc:
                        logger.warning(
                            "Agent '%s': tool '%s' display formatting failed: %s",
                            self.spec.name,
                            call["name"],
                            exc,
                        )
                self._display.show_tool_result(call["name"], result, display_str)
                logger.debug(
                    "Agent '%s': tool '%s' completed with status=%s",
                    self.spec.name,
                    call["name"],
                    result.get("status"),
                )
                try:
                    self._session.add_raw_message(
                        {
                            "role": "tool",
                            "tool_call_id": call["call_id"],
                            "content": json.dumps(result, default=str),
                        }
                    )
                    self._last_persisted_role = "tool"
                except SessionError as exc:
                    logger.warning(
                        "Agent '%s': failed to persist tool result for '%s' (call_id=%s): %s",
                        self.spec.name,
                        call["name"],
                        call.get("call_id"),
                        exc,
                    )
                    self._display.show_error(f"Could not save tool result: {exc}")
                    return AgentResult(
                        text="".join(all_text_parts),
                        status="error",
                        partial=True,
                        error_message=str(exc),
                    )
        else:
            logger.warning(
                "Agent '%s': tool call limit (%d rounds) reached; stopping.",
                self.spec.name,
                self.spec.max_tool_rounds,
            )
            # The final round always ends with saved tool-response messages.
            # Inject a synthetic assistant turn to close the open tool cycle
            # so the next user prompt does not produce an invalid role sequence.
            self._close_open_tool_cycle("[Stopped: tool call limit reached.]")
            _tl_text = "".join(all_text_parts)
            logger.info(
                "Agent '%s' finished: status=tool_limit text_len=%d text_preview=%r",
                self.spec.name,
                len(_tl_text),
                _preview_text(_tl_text, _INFO_TEXT_PREVIEW_CHARS),
            )
            logger.debug(
                "Agent '%s' tool_limit text_debug_preview=%r",
                self.spec.name,
                _preview_text(_tl_text, _DEBUG_TEXT_PREVIEW_CHARS),
            )
            return AgentResult(
                text=_tl_text,
                status="tool_limit",
                partial=True,
            )

        final_text = "".join(all_text_parts)
        logger.info(
            "Agent '%s' finished: status=ok text_len=%d text_preview=%r",
            self.spec.name,
            len(final_text),
            _preview_text(final_text, _INFO_TEXT_PREVIEW_CHARS),
        )
        logger.debug(
            "Agent '%s' ok text_debug_preview=%r",
            self.spec.name,
            _preview_text(final_text, _DEBUG_TEXT_PREVIEW_CHARS),
        )
        return AgentResult(text=final_text, status="ok")


def build_agent_tool_registry(
    spec: AgentSpec,
    workspace: Workspace,
    config: ConfigManager,
    display: Display,
    global_tool_registry: ToolRegistry,
) -> ToolRegistry:
    """Build a scoped ``ToolRegistry`` and ``PermissionManager`` for a sub-agent.

    Each sub-agent gets its own ``PermissionManager`` so that "always allow"
    grants are scoped to that agent and its ``Display``.  Sharing the
    coordinator's ``PermissionManager`` would leak grants between agents and
    potentially bypass ``SubAgentDisplay``'s default-deny behaviour.

    Only the tools listed in ``spec.tools`` are registered.  Unknown tool names
    are logged and skipped.  ``spec.tool_permission_overrides`` are applied
    in-memory to the freshly registered instances (no config writes).
    """
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.skill_registry import SkillRegistry
    from ai_cli.core.tool_registry import ToolRegistry
    from ai_cli.tools.skills import SkillsTool

    logger.info(
        "Agent '%s': building scoped tool registry from %d declared tools",
        spec.name,
        len(spec.tools),
    )
    permission_manager = PermissionManager(prompt_fn=display.show_permission_prompt)
    registry = ToolRegistry(workspace, config, permission_manager)

    # De-duplicate while preserving declaration order so duplicate entries in
    # spec.tools don't cause redundant registrations or noisy override warnings.
    registered: list[str] = []
    for name in dict.fromkeys(spec.tools):
        # Sub-agents cannot recursively invoke other agents: CallAgentTool and
        # CallAgentsParallelTool have non-standard constructors that the generic
        # register() path cannot satisfy, so skip them silently.
        if name in ("call_agent", "call_agents_parallel"):
            logger.debug(
                "Agent '%s': skipping recursive tool '%s' during scoped registry build",
                spec.name,
                name,
            )
            continue
        tool_instance = global_tool_registry.get(name)
        if tool_instance is None:
            logger.warning("Agent '%s': unknown tool '%s' — skipped", spec.name, name)
            continue
        tool_cls = type(tool_instance)
        try:
            if isinstance(tool_instance, SkillsTool):
                source_registry = getattr(tool_instance, "_skills", None)
                if not isinstance(source_registry, SkillRegistry):
                    logger.warning(
                        "Agent '%s': cannot clone skills tool (missing source SkillRegistry) — skipped",
                        spec.name,
                    )
                    continue

                if spec.skills is None:
                    scoped_skills = SkillRegistry(source_registry.skills)
                else:
                    scoped_specs: dict = {}
                    for configured in spec.skills:
                        matched = source_registry.get(configured)
                        if matched is None:
                            logger.warning(
                                "Agent '%s': configured skill '%s' is not loaded — skipped",
                                spec.name,
                                configured,
                            )
                            continue
                        scoped_specs[matched.name] = matched
                    scoped_skills = SkillRegistry(scoped_specs)

                registry.register_instance(
                    SkillsTool(scoped_skills, workspace, permission_manager)
                )
            elif getattr(tool_cls, "REGISTER_VIA_INSTANCE", False):
                # Tools with REGISTER_VIA_INSTANCE = True have non-standard
                # constructors (e.g. they require a TaskManager or other
                # context injected at construction time) and cannot be re-
                # instantiated via the standard registry.register() path.
                # Build a per-registry instance to avoid leaking mutable
                # state (set_registry backrefs, permission overrides, etc.)
                # across agents via shared objects.
                task_manager = getattr(tool_instance, "_tm", None)
                if task_manager is None:
                    logger.warning(
                        "Agent '%s': cannot clone instance-only tool '%s' "
                        "(missing _tm task manager reference) — skipped",
                        spec.name,
                        name,
                    )
                    continue
                instance_tool_cls: Any = tool_cls
                registry.register_instance(
                    instance_tool_cls(task_manager, workspace, permission_manager)
                )
            else:
                registry.register(tool_cls)
        except Exception as exc:
            logger.warning(
                "Agent '%s': failed to register tool '%s' — skipped (%s: %s)",
                spec.name,
                name,
                type(exc).__name__,
                exc,
            )
            continue
        registered.append(name)
        logger.debug("Agent '%s': registered scoped tool '%s'", spec.name, name)

    # Apply project config so that user settings (allowed, permission_required,
    # disabled) from config.yaml are honoured as starting defaults.
    registry.apply_config()
    logger.debug(
        "Agent '%s': applied project tool configuration to scoped registry",
        spec.name,
    )

    # The agent spec is the authority on which tools are enabled: explicitly
    # re-enable every registered tool so the config's `disabled` flag cannot
    # silently remove a tool the spec declared.  `allowed` and
    # `permission_required` from config are intentionally preserved — they
    # control whether the tool requires a permission prompt, not whether the
    # tool is accessible at all to this agent.
    # Note: runtime `/tools allow|disallow|enable|disable` changes made after
    # sub-agent creation are NOT reflected here (sub-agents snapshot config at
    # build time).  See VULN-010 for details.
    for name in registered:
        registry.enable_session(name)
        logger.debug(
            "Agent '%s': enabled scoped tool '%s' for session", spec.name, name
        )

    # Apply per-agent permission overrides last so they win over config.
    for name, override in spec.tool_permission_overrides.items():
        instance = registry.get(name)
        if instance is not None:
            instance.permission_required = override
            logger.info(
                "Agent '%s': applied permission override for tool '%s' -> %s",
                spec.name,
                name,
                override,
            )
        else:
            logger.warning(
                "Agent '%s': permission override for unknown/unregistered tool '%s' ignored",
                spec.name,
                name,
            )

    logger.info(
        "Agent '%s': scoped tool registry ready with %d registered tools and %d permission overrides",
        spec.name,
        len(registered),
        len(spec.tool_permission_overrides),
    )

    return registry
