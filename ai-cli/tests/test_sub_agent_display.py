"""Tests for ai_cli.cli.display.SubAgentDisplay."""

import logging

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


class TestConstructorDefaults:
    def test_inherits_display_defaults(self):
        d = SubAgentDisplay()
        assert d.verbose is False
        assert d.markdown_enabled is True

    def test_custom_flags(self):
        d = SubAgentDisplay(verbose=True, markdown_enabled=False)
        assert d.verbose is True
        assert d.markdown_enabled is False
