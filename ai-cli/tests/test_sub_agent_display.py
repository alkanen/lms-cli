"""Tests for ai_cli.cli.display.SubAgentDisplay."""

import logging
import threading
from unittest.mock import MagicMock

import pytest

from ai_cli.cli.display import SubAgentDisplay


@pytest.fixture()
def display() -> SubAgentDisplay:
    return SubAgentDisplay()


class TestStreamAndCapture:
    def test_empty_by_default(self, display):
        assert display.captured_text == ""

    def test_stream_text_accumulates(self, display):
        display.stream_text("Hello")
        display.stream_text(", ")
        display.stream_text("world!")
        assert display.captured_text == "Hello, world!"

    def test_reset_clears_buffer(self, display):
        display.stream_text("data")
        display.reset()
        assert display.captured_text == ""

    def test_reset_clears_usage(self, display):
        display.update_usage({"prompt_tokens": 100}, 4096)
        display.reset()
        assert display._last_usage == {}


class TestUsage:
    def test_update_usage_stores_dict(self, display):
        usage = {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700}
        display.update_usage(usage, 8192)
        assert display._last_usage == usage

    def test_update_usage_overwrites(self, display):
        display.update_usage({"prompt_tokens": 1}, 100)
        display.update_usage({"prompt_tokens": 2}, 200)
        assert display._last_usage == {"prompt_tokens": 2}


class TestPermissionPrompt:
    def test_returns_no(self, display):
        result = display.show_permission_prompt("Allow write?", ["file:foo.py"])
        assert result == ("no", "")

    def test_logs_warning(self, display, caplog):
        with caplog.at_level(logging.WARNING):
            display.show_permission_prompt("Allow write?", [])
        assert "Sub-agent permission prompt denied" in caplog.text
        assert "Allow write?" in caplog.text


class TestLogging:
    def test_show_error_logs_warning(self, display, caplog):
        with caplog.at_level(logging.WARNING):
            display.show_error("something broke")
        assert "Sub-agent error: something broke" in caplog.text

    def test_show_status_logs_info(self, display, caplog):
        with caplog.at_level(logging.INFO):
            display.show_status("compacting")
        assert "Sub-agent status: compacting" in caplog.text

    def test_stream_reasoning_logs_debug(self, display, caplog):
        with caplog.at_level(logging.DEBUG):
            display.stream_reasoning("thinking hard")
        assert "Sub-agent reasoning: thinking hard" in caplog.text

    def test_show_tool_call_logs_debug(self, display, caplog):
        with caplog.at_level(logging.DEBUG):
            display.show_tool_call("read_file", {"path": "foo.py"})
        assert "Sub-agent tool call: read_file" in caplog.text
        # Non-verbose: args values must NOT appear in the log.
        assert "foo.py" not in caplog.text

    def test_show_tool_call_verbose_logs_keys(self, caplog):
        d = SubAgentDisplay(verbose=True)
        with caplog.at_level(logging.DEBUG):
            d.show_tool_call("write_file", {"path": "x.py", "content": "secret"})
        assert "Sub-agent tool call: write_file" in caplog.text
        assert "path" in caplog.text
        # Verbose logs keys but never raw values.
        assert "secret" not in caplog.text

    def test_show_tool_result_logs_debug(self, display, caplog):
        with caplog.at_level(logging.DEBUG):
            display.show_tool_result("read_file", {"status": "success"}, None)
        assert "Sub-agent tool result: read_file" in caplog.text


class TestSessionList:
    def test_returns_none(self, display):
        assert display.show_session_list([]) is None


class TestNoOpMethods:
    """All no-op methods must be callable without raising."""

    @pytest.mark.parametrize(
        "method, args",
        [
            ("begin_assistant_turn", ()),
            ("end_assistant_turn", ()),
            ("show_help", ([("cmd", "desc")],)),
            ("show_tool_list", ([],)),
            ("show_session_info", (None,)),
            ("show_tool_list_all", ([],)),
            ("show_tool_info", ({},)),
            ("show_history", ([],)),
        ],
    )
    def test_noop_callable(self, display, method, args):
        getattr(display, method)(*args)


class TestParentForwarding:
    def test_forwards_streaming_to_parent_when_parent_is_verbose(self):
        parent = MagicMock()
        parent.verbose = True
        parent.markdown_enabled = True
        d = SubAgentDisplay(
            verbose=False,
            parent_display=parent,
            agent_name="planner",
        )

        d.begin_assistant_turn()
        d.stream_reasoning("thinking")
        d.stream_text("done")
        d.end_assistant_turn()

        parent.begin_assistant_turn.assert_called_once_with(title="Agent (planner)")
        parent.stream_reasoning.assert_called_once_with("thinking")
        parent.stream_text.assert_called_once_with("done")
        parent.end_assistant_turn.assert_called_once_with()

    def test_forwards_custom_title_to_parent_when_verbose(self):
        parent = MagicMock()
        parent.verbose = True
        parent.markdown_enabled = True
        d = SubAgentDisplay(
            verbose=False,
            parent_display=parent,
            agent_name="planner",
        )

        d.begin_assistant_turn(title="Reviewer")

        parent.begin_assistant_turn.assert_called_once_with(title="Reviewer (planner)")

    def test_does_not_forward_when_parent_is_not_verbose(self):
        parent = MagicMock()
        parent.verbose = False
        parent.markdown_enabled = True
        d = SubAgentDisplay(
            verbose=False,
            parent_display=parent,
            agent_name="explorer",
        )

        d.begin_assistant_turn()
        d.stream_text("hidden")
        d.stream_reasoning("hidden-thinking")
        d.end_assistant_turn()

        parent.begin_assistant_turn.assert_not_called()
        parent.stream_text.assert_not_called()
        parent.stream_reasoning.assert_not_called()
        parent.end_assistant_turn.assert_not_called()
        assert d.captured_text == "hidden"

    def test_serializes_forwarded_streaming_for_shared_parent(self):
        parent = MagicMock()
        parent.verbose = True
        parent.markdown_enabled = True

        first_begin_entered = threading.Event()
        release_first_begin = threading.Event()
        second_begin_attempted = threading.Event()
        second_begin_entered_parent = threading.Event()
        second_begin_returned = threading.Event()
        begin_count = {"n": 0}

        def _begin_side_effect(*, title):
            begin_count["n"] += 1
            if begin_count["n"] == 1:
                first_begin_entered.set()
                assert release_first_begin.wait(timeout=1.0)
            elif begin_count["n"] == 2:
                second_begin_entered_parent.set()

        parent.begin_assistant_turn.side_effect = _begin_side_effect

        d1 = SubAgentDisplay(
            verbose=False,
            parent_display=parent,
            agent_name="planner",
        )
        d2 = SubAgentDisplay(
            verbose=False,
            parent_display=parent,
            agent_name="reviewer",
        )

        def _run_first() -> None:
            d1.begin_assistant_turn()
            d1.stream_text("a")
            d1.end_assistant_turn()

        def _run_second() -> None:
            assert first_begin_entered.wait(timeout=1.0)
            second_begin_attempted.set()
            d2.begin_assistant_turn()
            second_begin_returned.set()
            d2.end_assistant_turn()

        t1 = threading.Thread(target=_run_first)
        t2 = threading.Thread(target=_run_second)
        t1.start()
        t2.start()

        # While the first forwarded turn is active, the second must block.
        assert first_begin_entered.wait(timeout=1.0)
        assert second_begin_attempted.wait(timeout=1.0)
        assert not second_begin_entered_parent.wait(timeout=0.2)

        release_first_begin.set()
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)
        assert not t1.is_alive()
        assert not t2.is_alive()
        assert second_begin_entered_parent.is_set()
        assert second_begin_returned.is_set()


class TestConstructorDefaults:
    def test_inherits_display_defaults(self):
        d = SubAgentDisplay()
        assert d.verbose is False
        assert d.markdown_enabled is True

    def test_custom_flags(self):
        d = SubAgentDisplay(verbose=True, markdown_enabled=False)
        assert d.verbose is True
        assert d.markdown_enabled is False
