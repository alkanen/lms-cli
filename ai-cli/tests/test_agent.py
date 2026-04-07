"""Tests for ai_cli.core.agent data structures and Agent.run()."""

import dataclasses
import threading
from unittest.mock import MagicMock

from ai_cli.core.agent import Agent, AgentResult, AgentSpec, BackendConfig
from ai_cli.core.llm_client import LLMError
from ai_cli.core.session_manager import SessionError


class TestBackendConfig:
    def test_defaults(self):
        bc = BackendConfig(base_url="http://localhost:11434/v1")
        assert bc.base_url == "http://localhost:11434/v1"
        assert bc.api_key_env is None

    def test_custom_api_key_env(self):
        bc = BackendConfig(base_url="http://example.com", api_key_env="MY_API_KEY")
        assert bc.api_key_env == "MY_API_KEY"


class TestAgentSpec:
    def test_required_fields(self):
        spec = AgentSpec(
            name="test",
            system_message="You are a test agent.",
            tools=["read_file"],
            model="llama3.2:3b",
        )
        assert spec.name == "test"
        assert spec.system_message == "You are a test agent."
        assert spec.tools == ["read_file"]
        assert spec.model == "llama3.2:3b"

    def test_defaults(self):
        spec = AgentSpec(name="t", system_message="m", tools=[], model="m")
        assert spec.max_response_tokens == 4096
        assert spec.persistence == "ephemeral"
        assert spec.backend is None
        assert spec.tool_permission_overrides == {}
        assert spec.max_tool_rounds == 10
        assert spec.context_limit_threshold == 0.90

    def test_custom_values(self):
        backend = BackendConfig(
            base_url="http://localhost:11435/v1", api_key_env="OLLAMA_KEY"
        )
        spec = AgentSpec(
            name="coder",
            system_message="Write code.",
            tools=["read_file", "write_file"],
            model="qwen2.5-coder:14b",
            max_response_tokens=8192,
            persistence="session",
            backend=backend,
            tool_permission_overrides={"write_file": False},
            max_tool_rounds=20,
            context_limit_threshold=0.85,
        )
        assert spec.max_response_tokens == 8192
        assert spec.persistence == "session"
        assert spec.backend is backend
        assert spec.tool_permission_overrides == {"write_file": False}
        assert spec.max_tool_rounds == 20
        assert spec.context_limit_threshold == 0.85

    def test_tool_permission_overrides_independent_instances(self):
        """Default dict should not be shared between instances."""
        a = AgentSpec(name="a", system_message="m", tools=[], model="m")
        b = AgentSpec(name="b", system_message="m", tools=[], model="m")
        a.tool_permission_overrides["x"] = True
        assert "x" not in b.tool_permission_overrides


class TestAgentResult:
    def test_ok_result(self):
        r = AgentResult(text="Done.", status="ok")
        assert r.text == "Done."
        assert r.status == "ok"
        assert r.partial is False
        assert r.error_message == ""

    def test_context_limit(self):
        r = AgentResult(text="Partial output.", status="context_limit", partial=True)
        assert r.partial is True

    def test_error_result(self):
        r = AgentResult(
            text="",
            status="error",
            partial=True,
            error_message="Connection refused",
        )
        assert r.status == "error"
        assert r.error_message == "Connection refused"

    def test_tool_limit(self):
        r = AgentResult(text="halfway", status="tool_limit", partial=True)
        assert r.status == "tool_limit"
        assert r.partial is True


# ---------------------------------------------------------------------------
# Helpers for Agent.run() tests
# ---------------------------------------------------------------------------

_COORD_SPEC = AgentSpec(
    name="test",
    system_message="",
    tools=[],
    model="",
    max_tool_rounds=10,
)


def _make_agent(
    spec=None, session=None, llm=None, tool_registry=None, display=None
) -> Agent:
    if session is None:
        session = MagicMock()
        session.get_messages.return_value = []
        session.should_compact.return_value = False
    if display is None:
        display = MagicMock()
    if tool_registry is None:
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
    if llm is None:
        llm = MagicMock()
        llm.send.return_value = iter([])
        llm.get_model_metadata.return_value = {}
    return Agent(
        spec=spec or dataclasses.replace(_COORD_SPEC),
        session=session,
        llm_client=llm,
        tool_registry=tool_registry,
        display=display,
    )


def _text_chunks(*parts):
    """Return chunk list for a simple text-only response."""
    chunks = [{"type": "text", "delta": p} for p in parts]
    chunks.append({"type": "done", "stop_reason": "stop", "usage": {}})
    return chunks


# ---------------------------------------------------------------------------
# Agent.run() tests
# ---------------------------------------------------------------------------


class TestAgentRun:
    def test_text_only_response(self):
        llm = MagicMock()
        llm.send.return_value = iter(_text_chunks("Hello", " world"))
        llm.get_model_metadata.return_value = {}
        agent = _make_agent(llm=llm)

        result = agent.run("Hi")

        assert result.status == "ok"
        assert result.text == "Hello world"
        assert result.partial is False

    def test_text_streamed_to_display(self):
        llm = MagicMock()
        llm.send.return_value = iter(_text_chunks("A", "B"))
        llm.get_model_metadata.return_value = {}
        display = MagicMock()
        agent = _make_agent(llm=llm, display=display)

        agent.run("go")

        calls = [c[0][0] for c in display.stream_text.call_args_list]
        assert calls == ["A", "B"]

    def test_tool_call_and_followup(self):
        """Tool call in round 1, text response in round 2."""
        session = MagicMock()
        session.get_messages.return_value = []

        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {
            "status": "success",
            "data": {"content": "file data"},
        }
        tool_registry.get.return_value = None

        llm = MagicMock()
        llm.get_model_metadata.return_value = {}
        # Round 1: tool call
        # Round 2: text response
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": "read_file",
                        "call_id": "c1",
                        "arguments": {"path": "foo.py"},
                    },
                    {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                ]
            ),
            iter(_text_chunks("Here is the file.")),
        ]

        agent = _make_agent(session=session, llm=llm, tool_registry=tool_registry)
        result = agent.run("Read foo.py")

        assert result.status == "ok"
        assert result.text == "Here is the file."
        assert llm.send.call_count == 2
        tool_registry.execute.assert_called_once_with(
            "read_file", {"path": "foo.py"}, allow_transient=False
        )

    def test_abort_before_first_round(self):
        abort = threading.Event()
        abort.set()
        agent = _make_agent()

        result = agent.run("go", abort=abort)

        assert result.partial is True
        assert result.text == ""

    def test_abort_mid_streaming(self):
        abort = threading.Event()

        def _chunks():
            yield {"type": "text", "delta": "Hello"}
            abort.set()
            yield {"type": "text", "delta": " world"}
            yield {"type": "done", "stop_reason": "stop", "usage": {}}

        llm = MagicMock()
        llm.send.return_value = _chunks()
        llm.get_model_metadata.return_value = {}
        display = MagicMock()
        agent = _make_agent(llm=llm, display=display)

        result = agent.run("Hi", abort=abort)

        assert result.partial is True
        status_calls = [c[0][0] for c in display.show_status.call_args_list]
        assert any("Aborted" in s for s in status_calls)

    def test_abort_mid_tool_execution(self):
        """Abort set after first tool call — remaining calls get stub results."""
        abort = threading.Event()
        session = MagicMock()
        session.get_messages.return_value = []
        raw_messages: list[dict] = []
        session.add_raw_message.side_effect = lambda m: raw_messages.append(m)

        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.get.return_value = None

        def _execute(name, args, **kwargs):
            abort.set()  # abort after first tool
            return {"status": "success", "data": {}}

        tool_registry.execute.side_effect = _execute

        llm = MagicMock()
        llm.get_model_metadata.return_value = {}
        llm.send.return_value = iter(
            [
                {
                    "type": "tool_call",
                    "name": "write_file",
                    "call_id": "call-A",
                    "arguments": {},
                },
                {
                    "type": "tool_call",
                    "name": "write_file",
                    "call_id": "call-B",
                    "arguments": {},
                },
                {"type": "done", "stop_reason": "tool_calls", "usage": {}},
            ]
        )

        agent = _make_agent(session=session, llm=llm, tool_registry=tool_registry)
        result = agent.run("write two files", abort=abort)

        assert result.partial is True

        # call-B should get an abort stub
        tool_msgs = [m for m in raw_messages if m.get("role") == "tool"]
        stub_ids = {m["tool_call_id"] for m in tool_msgs}
        assert "call-B" in stub_ids

    def test_tool_round_limit(self):
        spec = AgentSpec(
            name="limited",
            system_message="",
            tools=[],
            model="",
            max_tool_rounds=2,
        )
        session = MagicMock()
        session.get_messages.return_value = []

        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {"status": "success", "data": {}}
        tool_registry.get.return_value = None

        llm = MagicMock()
        llm.get_model_metadata.return_value = {}
        # Both rounds return a tool call — exceeds limit of 2.
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": "read_file",
                        "call_id": f"c{i}",
                        "arguments": {},
                    },
                    {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                ]
            )
            for i in range(3)  # 3 available but only 2 rounds allowed
        ]

        agent = _make_agent(
            spec=spec, session=session, llm=llm, tool_registry=tool_registry
        )
        result = agent.run("loop forever")

        assert result.status == "tool_limit"
        assert result.partial is True
        assert llm.send.call_count == 2

    def test_llm_error_returns_error_result(self):
        llm = MagicMock()
        llm.send.side_effect = LLMError("Connection refused")
        display = MagicMock()
        agent = _make_agent(llm=llm, display=display)

        result = agent.run("go")

        assert result.status == "error"
        assert "Connection refused" in result.error_message
        display.show_error.assert_called_once()

    def test_session_error_on_get_messages(self):
        session = MagicMock()
        session.get_messages.side_effect = SessionError("corrupt")
        display = MagicMock()
        agent = _make_agent(session=session, display=display)

        result = agent.run("go")

        assert result.status == "error"
        assert "corrupt" in result.error_message

    def test_none_abort_is_safe(self):
        """abort=None (the default) must not crash."""
        llm = MagicMock()
        llm.send.return_value = iter(_text_chunks("ok"))
        llm.get_model_metadata.return_value = {}
        agent = _make_agent(llm=llm)

        result = agent.run("go")

        assert result.status == "ok"
        assert result.text == "ok"

    def test_begin_end_turn_bracketing(self):
        llm = MagicMock()
        llm.send.return_value = iter(_text_chunks("x"))
        llm.get_model_metadata.return_value = {}
        display = MagicMock()
        agent = _make_agent(llm=llm, display=display)

        agent.run("go")

        display.begin_assistant_turn.assert_called_once()
        display.end_assistant_turn.assert_called_once()

    def test_usage_recorded(self):
        llm = MagicMock()
        llm.send.return_value = iter(
            [
                {"type": "text", "delta": "hi"},
                {
                    "type": "done",
                    "stop_reason": "stop",
                    "usage": {"prompt_tokens": 42, "completion_tokens": 10},
                },
            ]
        )
        llm.get_model_metadata.return_value = {"context_window": 4096}
        session = MagicMock()
        session.get_messages.return_value = []
        display = MagicMock()
        agent = _make_agent(llm=llm, session=session, display=display)

        agent.run("go")

        session.record_usage.assert_called_once_with(42)
        display.update_usage.assert_called_once()

    def test_disallowed_tool_replaced_with_unknown(self):
        """tool_disallowed result is replaced with unknown_tool for the LLM."""
        session = MagicMock()
        session.get_messages.return_value = []
        raw_messages: list[dict] = []
        session.add_raw_message.side_effect = lambda m: raw_messages.append(m)

        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {
            "status": "error",
            "error": "tool_disallowed",
        }
        tool_registry.get.return_value = None

        llm = MagicMock()
        llm.get_model_metadata.return_value = {}
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": "bad_tool",
                        "call_id": "c1",
                        "arguments": {},
                    },
                    {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                ]
            ),
            iter(_text_chunks("ok")),
        ]

        agent = _make_agent(session=session, llm=llm, tool_registry=tool_registry)
        result = agent.run("try bad tool")

        # The tool result sent to the session should be the sanitised version.
        import json

        tool_result_msgs = [m for m in raw_messages if m.get("role") == "tool"]
        assert len(tool_result_msgs) == 1
        content = json.loads(tool_result_msgs[0]["content"])
        assert content["error"] == "unknown_tool"
        assert result.status == "ok"

    def test_prompt_added_to_session_as_string(self):
        session = MagicMock()
        session.get_messages.return_value = []
        llm = MagicMock()
        llm.send.return_value = iter(_text_chunks("hi"))
        llm.get_model_metadata.return_value = {}
        agent = _make_agent(session=session, llm=llm)

        agent.run("Hello there")

        session.add_message.assert_any_call("user", "Hello there")

    def test_prompt_added_to_session_as_content_blocks(self):
        session = MagicMock()
        session.get_messages.return_value = []
        llm = MagicMock()
        llm.send.return_value = iter(_text_chunks("hi"))
        llm.get_model_metadata.return_value = {}
        agent = _make_agent(session=session, llm=llm)

        blocks = [{"type": "text", "text": "Hello"}]
        agent.run(blocks)

        session.add_raw_message.assert_any_call({"role": "user", "content": blocks})

    def test_prompt_session_error_returns_error(self):
        session = MagicMock()
        session.add_message.side_effect = SessionError("disk full")
        display = MagicMock()
        agent = _make_agent(session=session, display=display)

        result = agent.run("go")

        assert result.status == "error"
        assert "disk full" in result.error_message

    def test_abort_mid_stream_includes_current_round_text(self):
        """Text streamed before abort in the current round is in the result."""
        abort = threading.Event()

        def _chunks():
            yield {"type": "text", "delta": "partial"}
            abort.set()
            yield {"type": "done", "stop_reason": "stop", "usage": {}}

        llm = MagicMock()
        llm.send.return_value = _chunks()
        llm.get_model_metadata.return_value = {}
        agent = _make_agent(llm=llm)

        result = agent.run("go", abort=abort)

        assert "partial" in result.text

    def test_keyboard_interrupt_reraises_without_abort(self):
        """KeyboardInterrupt re-raises when abort is None."""
        llm = MagicMock()
        llm.send.side_effect = KeyboardInterrupt()
        agent = _make_agent(llm=llm)

        import pytest as _pt

        with _pt.raises(KeyboardInterrupt):
            agent.run("go")


# ---------------------------------------------------------------------------
# Context overflow detection
# ---------------------------------------------------------------------------


class TestContextOverflow:
    def _make_done_chunk(self, prompt_tokens: int) -> dict:
        return {
            "type": "done",
            "stop_reason": "stop",
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 10},
        }

    def test_context_limit_triggered_at_threshold(self):
        """91% of context window → status='context_limit', partial=True."""
        context_window = 1000
        prompt_tokens = 910  # 91% >= 90% threshold

        spec = dataclasses.replace(_COORD_SPEC, context_limit_threshold=0.90)
        llm = MagicMock()
        llm.send.return_value = iter(
            [
                {"type": "text", "delta": "partial text"},
                self._make_done_chunk(prompt_tokens),
            ]
        )
        llm.get_model_metadata.return_value = {"context_window": context_window}
        display = MagicMock()
        session = MagicMock()
        session.get_messages.return_value = []
        agent = _make_agent(spec=spec, llm=llm, display=display, session=session)

        result = agent.run("go")

        assert result.status == "context_limit"
        assert result.partial is True
        assert "partial text" in result.text
        assert "Context limit" in result.error_message
        status_calls = [c[0][0] for c in display.show_status.call_args_list]
        assert any("Context limit" in s for s in status_calls)
        # Assistant message must be persisted even when context limit fires.
        session.add_message.assert_called_with("assistant", "partial text")

    def test_context_limit_not_triggered_below_threshold(self):
        """89% of context window → normal 'ok' return."""
        context_window = 1000
        prompt_tokens = 890  # 89% < 90% threshold

        spec = dataclasses.replace(_COORD_SPEC, context_limit_threshold=0.90)
        llm = MagicMock()
        llm.send.return_value = iter(
            [
                {"type": "text", "delta": "full response"},
                self._make_done_chunk(prompt_tokens),
            ]
        )
        llm.get_model_metadata.return_value = {"context_window": context_window}
        agent = _make_agent(spec=spec, llm=llm)

        result = agent.run("go")

        assert result.status == "ok"
        assert result.partial is False
        assert result.text == "full response"

    def test_context_limit_not_triggered_when_context_window_zero(self):
        """context_window == 0 → never triggers, avoids division by zero."""
        spec = dataclasses.replace(_COORD_SPEC, context_limit_threshold=0.90)
        llm = MagicMock()
        llm.send.return_value = iter(
            [
                {"type": "text", "delta": "response"},
                self._make_done_chunk(99999),
            ]
        )
        llm.get_model_metadata.return_value = {"context_window": 0}
        agent = _make_agent(spec=spec, llm=llm)

        result = agent.run("go")

        assert result.status == "ok"
        assert result.partial is False

    def test_context_limit_stubs_pending_tool_calls(self):
        """When context limit fires mid-turn, dangling tool_calls get stub responses."""
        context_window = 1000
        prompt_tokens = 910  # 91% >= 90%

        spec = dataclasses.replace(_COORD_SPEC, context_limit_threshold=0.90)
        llm = MagicMock()
        llm.send.return_value = iter(
            [
                {
                    "type": "tool_call",
                    "call_id": "call-abc",
                    "name": "read_file",
                    "arguments": {"path": "foo.py"},
                },
                self._make_done_chunk(prompt_tokens),
            ]
        )
        llm.get_model_metadata.return_value = {"context_window": context_window}
        session = MagicMock()
        session.get_messages.return_value = []
        agent = _make_agent(spec=spec, llm=llm, session=session)

        result = agent.run("go")

        assert result.status == "context_limit"
        # A stub tool response must have been written for the dangling tool call.
        stub_ids = [
            (c.kwargs.get("message") or (c.args[0] if c.args else {})).get(
                "tool_call_id"
            )
            for c in session.add_raw_message.call_args_list
            if (c.kwargs.get("message") or (c.args[0] if c.args else {})).get("role")
            == "tool"
        ]
        assert "call-abc" in stub_ids
