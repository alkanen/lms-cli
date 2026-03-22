"""Tests for ai_cli.cli.display — Display ABC, PlainDisplay, and factory."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from ai_cli.cli.display import _UNIVERSAL_OPTIONS, Display, PlainDisplay, create_display
from ai_cli.core.session_manager import SessionMeta

_PAGER_PATCH = "ai_cli.cli.display.pydoc.pager"

_PATCH = "ai_cli.cli.display.pt_prompt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_meta(
    session_id: str = "20240101T000000-aabbccdd",
    first_user_message: str = "Hello there",
    message_count: int = 3,
) -> SessionMeta:
    return SessionMeta(
        session_id=session_id,
        workspace_path=MagicMock(),
        started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        message_count=message_count,
        name=None,
        first_user_message=first_user_message,
        last_message_role="assistant",
        last_message_preview="Sure!",
    )


def _plain(verbose: bool = False) -> PlainDisplay:
    return PlainDisplay(verbose=verbose)


# ---------------------------------------------------------------------------
# Display ABC — mode flags (tested via PlainDisplay as the concrete class)
# ---------------------------------------------------------------------------


class TestDisplayFlags:
    def test_verbose_default_false(self):
        d = _plain()
        assert d.verbose is False

    def test_toggle_verbose(self):
        d = _plain()
        d.toggle_verbose()
        assert d.verbose is True
        d.toggle_verbose()
        assert d.verbose is False

    def test_verbose_init_true(self):
        d = PlainDisplay(verbose=True)
        assert d.verbose is True

    def test_markdown_enabled_default_true(self):
        d = _plain()
        assert d.markdown_enabled is True

    def test_toggle_markdown(self):
        d = _plain()
        d.toggle_markdown()
        assert d.markdown_enabled is False
        d.toggle_markdown()
        assert d.markdown_enabled is True

    def test_markdown_enabled_init_false(self):
        d = PlainDisplay(markdown_enabled=False)
        assert d.markdown_enabled is False

    def test_display_is_abstract(self):
        with pytest.raises(TypeError):
            Display()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# PlainDisplay — streaming
# ---------------------------------------------------------------------------


class TestPlainDisplayStreaming:
    def test_stream_text_prints_without_newline(self, capsys):
        d = _plain()
        d.stream_text("hello")
        d.stream_text(" world")
        out = capsys.readouterr().out
        assert out == "hello world"

    def test_end_assistant_turn_adds_newline(self, capsys):
        d = _plain()
        d.stream_text("hi")
        d.end_assistant_turn()
        out = capsys.readouterr().out
        assert out.endswith("\n")

    def test_begin_assistant_turn_produces_no_output(self, capsys):
        d = _plain()
        d.begin_assistant_turn()
        assert capsys.readouterr().out == ""

    def test_full_turn_sequence(self, capsys):
        d = _plain()
        d.begin_assistant_turn()
        d.stream_text("Hello")
        d.stream_text(", world")
        d.end_assistant_turn()
        out = capsys.readouterr().out
        assert out == "Hello, world\n"


# ---------------------------------------------------------------------------
# PlainDisplay — tool activity
# ---------------------------------------------------------------------------


class TestPlainDisplayTools:
    def test_show_tool_call_summary_mode(self, capsys):
        d = _plain(verbose=False)
        d.show_tool_call("read_file", {"path": "src/foo.py"})
        out = capsys.readouterr().out
        assert "read_file" in out
        assert "src/foo.py" in out
        assert out.startswith("▶")

    def test_show_tool_call_verbose_mode(self, capsys):
        d = _plain(verbose=True)
        d.show_tool_call("read_file", {"path": "src/foo.py"})
        out = capsys.readouterr().out
        assert "read_file" in out
        assert '"path"' in out
        assert '"src/foo.py"' in out

    def test_show_tool_result_silent_in_summary_mode(self, capsys):
        d = _plain(verbose=False)
        d.show_tool_result("read_file", {"status": "success"})
        assert capsys.readouterr().out == ""

    def test_show_tool_result_shown_in_verbose_mode(self, capsys):
        d = _plain(verbose=True)
        d.show_tool_result("read_file", {"status": "success"})
        out = capsys.readouterr().out
        assert "read_file" in out
        assert "success" in out


# ---------------------------------------------------------------------------
# PlainDisplay — status and errors
# ---------------------------------------------------------------------------


class TestPlainDisplayStatusError:
    def test_show_status(self, capsys):
        d = _plain()
        d.show_status("Session compacted.")
        out = capsys.readouterr().out
        assert "Session compacted." in out

    def test_show_error(self, capsys):
        d = _plain()
        d.show_error("Something went wrong.")
        err = capsys.readouterr().err
        assert "Something went wrong." in err
        assert "✗" in err


# ---------------------------------------------------------------------------
# PlainDisplay — slash-command output
# ---------------------------------------------------------------------------


class TestPlainDisplayHelp:
    def test_show_help_prints_all_commands(self, capsys):
        d = _plain()
        d.show_help([("/help", "Show help"), ("/exit", "Quit")])
        out = capsys.readouterr().out
        assert "/help" in out
        assert "Show help" in out
        assert "/exit" in out
        assert "Quit" in out

    def test_show_help_empty_list(self, capsys):
        d = _plain()
        d.show_help([])
        # Should not raise; some header line is fine
        capsys.readouterr()


class TestPlainDisplayToolList:
    def _make_tool(self, name: str, description: str) -> MagicMock:
        t = MagicMock()
        t.name = name
        t.description = description
        return t

    def test_show_tool_list_prints_tools(self, capsys):
        d = _plain()
        d.show_tool_list([self._make_tool("read_file", "Read a file")])
        out = capsys.readouterr().out
        assert "read_file" in out
        assert "Read a file" in out

    def test_show_tool_list_empty(self, capsys):
        d = _plain()
        d.show_tool_list([])
        assert "No tools" in capsys.readouterr().out


class TestPlainDisplaySessionInfo:
    def test_show_session_info_prints_id_and_count(self, capsys):
        session = MagicMock()
        session.session_id = "20240101T000000-aabbccdd"
        session.get_meta.return_value = {
            "started_at": "2024-01-01T12:00:00+00:00",
            "message_count": 5,
            "name": None,
        }
        d = _plain()
        d.show_session_info(session)
        out = capsys.readouterr().out
        assert "20240101T000000-aabbccdd" in out
        assert "5" in out

    def test_show_session_info_prints_name_when_set(self, capsys):
        session = MagicMock()
        session.session_id = "20240101T000000-aabbccdd"
        session.get_meta.return_value = {
            "started_at": "2024-01-01T12:00:00+00:00",
            "message_count": 2,
            "name": "My project chat",
        }
        d = _plain()
        d.show_session_info(session)
        assert "My project chat" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# PlainDisplay — permission prompt
# ---------------------------------------------------------------------------


class TestPlainDisplayPermissionPrompt:
    def _prompt(self, inputs: list[str], extra: list[str] | None = None):
        d = _plain()
        with patch(_PATCH, side_effect=inputs):
            return d.show_permission_prompt("Allow read_file?", extra or [])

    def test_yes_short(self):
        choice, text = self._prompt(["y"])
        assert choice == "yes"
        assert text == ""

    def test_yes_full_word(self):
        choice, text = self._prompt(["yes"])
        assert choice == "yes"

    def test_no_short(self):
        choice, _ = self._prompt(["n"])
        assert choice == "no"

    def test_always(self):
        choice, _ = self._prompt(["a"])
        assert choice == "always"

    def test_custom_collects_message(self):
        choice, text = self._prompt(["c", "Please don't do that"])
        assert choice == "custom"
        assert text == "Please don't do that"

    def test_extra_option_by_index(self):
        choice, text = self._prompt(["0"], extra=["file:./src/foo.py"])
        assert choice == "file:./src/foo.py"
        assert text == ""

    def test_invalid_then_valid(self):
        choice, _ = self._prompt(["x", "??", "n"])
        assert choice == "no"

    def test_question_is_printed(self, capsys):
        d = _plain()
        with patch(_PATCH, side_effect=["y"]):
            d.show_permission_prompt("Allow this?", [])
        assert "Allow this?" in capsys.readouterr().out

    def test_extra_options_are_printed(self, capsys):
        d = _plain()
        with patch(_PATCH, side_effect=["0"]):
            d.show_permission_prompt("Allow?", ["file:./src/foo.py"])
        assert "file:./src/foo.py" in capsys.readouterr().out

    def test_eof_returns_no(self):
        choice, text = self._prompt([EOFError()])
        assert choice == "no"
        assert text == ""

    def test_keyboard_interrupt_returns_no(self):
        choice, text = self._prompt([KeyboardInterrupt()])
        assert choice == "no"
        assert text == ""

    def test_eof_during_custom_message_returns_no(self):
        # 'c' accepted, then EOF on the message prompt
        choice, text = self._prompt(["c", EOFError()])
        assert choice == "no"
        assert text == ""

    def test_universal_options_are_printed(self, capsys):
        d = _plain()
        with patch(_PATCH, side_effect=["y"]):
            d.show_permission_prompt("Allow?", [])
        out = capsys.readouterr().out
        for _, _, label in _UNIVERSAL_OPTIONS:
            assert label in out


# ---------------------------------------------------------------------------
# PlainDisplay — session list
# ---------------------------------------------------------------------------


class TestPlainDisplaySessionList:
    def test_empty_list_returns_none(self):
        d = _plain()
        result = d.show_session_list([])
        assert result is None

    def test_pick_by_index(self):
        s = _make_session_meta()
        d = _plain()
        with patch(_PATCH, side_effect=["0"]):
            result = d.show_session_list([s])
        assert result is s

    def test_quit_returns_none(self):
        s = _make_session_meta()
        d = _plain()
        with patch(_PATCH, side_effect=["q"]):
            result = d.show_session_list([s])
        assert result is None

    def test_empty_input_returns_none(self):
        s = _make_session_meta()
        d = _plain()
        with patch(_PATCH, side_effect=[""]):
            result = d.show_session_list([s])
        assert result is None

    def test_invalid_then_valid(self):
        s = _make_session_meta()
        d = _plain()
        with patch(_PATCH, side_effect=["99", "oops", "0"]):
            result = d.show_session_list([s])
        assert result is s

    def test_eof_returns_none(self):
        s = _make_session_meta()
        d = _plain()
        with patch(_PATCH, side_effect=[EOFError()]):
            assert d.show_session_list([s]) is None

    def test_keyboard_interrupt_returns_none(self):
        s = _make_session_meta()
        d = _plain()
        with patch(_PATCH, side_effect=[KeyboardInterrupt()]):
            assert d.show_session_list([s]) is None

    def test_session_info_printed(self, capsys):
        s = _make_session_meta(first_user_message="Tell me about Python")
        d = _plain()
        with patch(_PATCH, side_effect=["q"]):
            d.show_session_list([s])
        out = capsys.readouterr().out
        assert s.session_id in out
        assert "Tell me about Python" in out


# ---------------------------------------------------------------------------
# PlainDisplay — show_tool_list_all
# ---------------------------------------------------------------------------


class TestPlainDisplayToolListAll:
    def _info(
        self,
        name: str,
        *,
        enabled: bool = True,
        allowed: bool = True,
        permission_required: bool = False,
        tier: str = "bundled",
    ) -> dict:
        return {
            "name": name,
            "description": f"The {name} tool.",
            "enabled": enabled,
            "allowed": allowed,
            "permission_required": permission_required,
            "tier": tier,
        }

    def test_prints_tool_names(self, capsys):
        d = _plain()
        d.show_tool_list_all([self._info("read_file"), self._info("write_file")])
        out = capsys.readouterr().out
        assert "read_file" in out
        assert "write_file" in out

    def test_enabled_status_shown(self, capsys):
        d = _plain()
        d.show_tool_list_all([self._info("echo", enabled=True)])
        assert "enabled" in capsys.readouterr().out

    def test_disabled_status_shown(self, capsys):
        d = _plain()
        d.show_tool_list_all([self._info("echo", enabled=False)])
        assert "disabled" in capsys.readouterr().out

    def test_disallowed_status_shown(self, capsys):
        d = _plain()
        d.show_tool_list_all([self._info("echo", allowed=False)])
        assert "disallowed" in capsys.readouterr().out

    def test_empty_list_shows_no_tools_message(self, capsys):
        d = _plain()
        d.show_tool_list_all([])
        assert "No tools" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# PlainDisplay — show_tool_info
# ---------------------------------------------------------------------------


class TestPlainDisplayToolInfo:
    def _info(
        self,
        *,
        enabled: bool = True,
        allowed: bool = True,
        permission_required: bool = False,
    ) -> dict:
        return {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "enabled": enabled,
            "allowed": allowed,
            "permission_required": permission_required,
            "tier": "bundled",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                },
                "required": ["path"],
            },
        }

    def test_prints_name(self, capsys):
        d = _plain()
        d.show_tool_info(self._info())
        assert "read_file" in capsys.readouterr().out

    def test_prints_description(self, capsys):
        d = _plain()
        d.show_tool_info(self._info())
        assert "Read a file" in capsys.readouterr().out

    def test_enabled_status(self, capsys):
        d = _plain()
        d.show_tool_info(self._info(enabled=True))
        assert "enabled" in capsys.readouterr().out

    def test_disallowed_status(self, capsys):
        d = _plain()
        d.show_tool_info(self._info(allowed=False))
        assert "disallowed" in capsys.readouterr().out

    def test_disabled_status(self, capsys):
        d = _plain()
        d.show_tool_info(self._info(enabled=False))
        assert "disabled" in capsys.readouterr().out

    def test_prints_parameter_name(self, capsys):
        d = _plain()
        d.show_tool_info(self._info())
        assert "path" in capsys.readouterr().out

    def test_marks_required_parameter(self, capsys):
        d = _plain()
        d.show_tool_info(self._info())
        assert "path: string (required)" in capsys.readouterr().out

    def test_permission_not_required(self, capsys):
        d = _plain()
        d.show_tool_info(self._info(permission_required=False))
        assert "not required" in capsys.readouterr().out

    def test_permission_required(self, capsys):
        d = _plain()
        d.show_tool_info(self._info(permission_required=True))
        # "required" appears either in "required" or "not required"
        out = capsys.readouterr().out
        assert "required" in out
        assert "not required" not in out


# ---------------------------------------------------------------------------
# PlainDisplay — stream_reasoning
# ---------------------------------------------------------------------------


class TestPlainDisplayReasoning:
    def test_reasoning_silent_in_summary_mode(self, capsys):
        d = _plain(verbose=False)
        d.begin_assistant_turn()
        d.stream_reasoning("thinking...")
        assert capsys.readouterr().out == ""

    def test_reasoning_shown_in_verbose_mode(self, capsys):
        d = _plain(verbose=True)
        d.begin_assistant_turn()
        d.stream_reasoning("step 1")
        out = capsys.readouterr().out
        assert "step 1" in out

    def test_reasoning_prefix_on_first_call(self, capsys):
        d = _plain(verbose=True)
        d.begin_assistant_turn()
        d.stream_reasoning("step 1")
        out = capsys.readouterr().out
        assert "[thinking]" in out

    def test_reasoning_no_prefix_on_second_call(self, capsys):
        d = _plain(verbose=True)
        d.begin_assistant_turn()
        d.stream_reasoning("step 1")
        capsys.readouterr()  # clear
        d.stream_reasoning("step 2")
        out = capsys.readouterr().out
        assert "[thinking]" not in out
        assert "step 2" in out

    def test_reasoning_flag_reset_between_turns(self, capsys):
        d = _plain(verbose=True)
        d.begin_assistant_turn()
        d.stream_reasoning("turn 1 reasoning")
        capsys.readouterr()
        d.begin_assistant_turn()  # new turn resets the flag
        d.stream_reasoning("turn 2 reasoning")
        out = capsys.readouterr().out
        assert "[thinking]" in out  # prefix shown again for the new turn

    def test_reasoning_closed_before_text(self, capsys):
        d = _plain(verbose=True)
        d.begin_assistant_turn()
        d.stream_reasoning("inner step")
        d.stream_text("answer")
        out = capsys.readouterr().out
        assert "[/thinking]" in out
        assert out.index("[/thinking]") < out.index("answer")

    def test_reasoning_closed_at_end_of_turn_when_no_text(self, capsys):
        d = _plain(verbose=True)
        d.begin_assistant_turn()
        d.stream_reasoning("inner step")
        d.end_assistant_turn()
        out = capsys.readouterr().out
        assert "[/thinking]" in out

    def test_reasoning_not_closed_twice(self, capsys):
        # If stream_text closes reasoning, end_assistant_turn must not close again.
        d = _plain(verbose=True)
        d.begin_assistant_turn()
        d.stream_reasoning("r")
        d.stream_text("t")
        d.end_assistant_turn()
        out = capsys.readouterr().out
        assert out.count("[/thinking]") == 1

    def test_reasoning_close_not_shown_in_summary_mode(self, capsys):
        d = _plain(verbose=False)
        d.begin_assistant_turn()
        d.stream_reasoning("inner")
        d.stream_text("answer")
        d.end_assistant_turn()
        out = capsys.readouterr().out
        assert "[/thinking]" not in out
        assert "answer" in out


# ---------------------------------------------------------------------------
# PlainDisplay — update_usage
# ---------------------------------------------------------------------------


class TestPlainDisplayUpdateUsage:
    def test_update_usage_does_not_raise(self):
        d = _plain()
        d.update_usage({"prompt_tokens": 100, "completion_tokens": 50}, 128000)


# ---------------------------------------------------------------------------
# PlainDisplay — show_history
# ---------------------------------------------------------------------------


class TestPlainDisplayHistory:
    def test_show_history_calls_pager(self):
        d = _plain()
        messages = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi!"},
        ]
        with patch(_PAGER_PATCH) as mock_pager:
            d.show_history(messages)
        mock_pager.assert_called_once()

    def test_show_history_includes_role_and_content(self):
        d = _plain()
        messages = [{"role": "user", "content": "Hello"}]
        with patch(_PAGER_PATCH) as mock_pager:
            d.show_history(messages)
        text = mock_pager.call_args[0][0]
        assert "user" in text
        assert "Hello" in text

    def test_show_history_handles_block_content(self):
        d = _plain()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "block text here"}],
            }
        ]
        with patch(_PAGER_PATCH) as mock_pager:
            d.show_history(messages)
        text = mock_pager.call_args[0][0]
        assert "block text here" in text

    def test_show_history_non_string_text_block_skipped(self):
        # A block with text=None or text=<non-str> must not raise TypeError in join()
        d = _plain()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": None},
                    {"type": "text", "text": 42},
                    {"type": "text", "text": "valid"},
                ],
            }
        ]
        with patch(_PAGER_PATCH) as mock_pager:
            d.show_history(messages)
        text = mock_pager.call_args[0][0]
        assert "valid" in text

    def test_show_history_none_content_does_not_render_literally(self):
        # assistant tool-call messages have content=None; must not appear as "None"
        d = _plain()
        messages = [{"role": "assistant", "content": None}]
        with patch(_PAGER_PATCH) as mock_pager:
            d.show_history(messages)
        text = mock_pager.call_args[0][0]
        assert "None" not in text

    def test_show_history_empty_list(self):
        d = _plain()
        with patch(_PAGER_PATCH) as mock_pager:
            d.show_history([])
        mock_pager.assert_called_once()


# ---------------------------------------------------------------------------
# PlainDisplay — show_tool_result with display_str
# ---------------------------------------------------------------------------


class TestPlainDisplayToolResultDisplayStr:
    def test_display_str_shown_in_verbose_mode(self, capsys):
        d = _plain(verbose=True)
        d.show_tool_result("read_file", {"status": "success"}, display_str="custom")
        out = capsys.readouterr().out
        assert "custom" in out

    def test_display_str_overrides_json_in_verbose_mode(self, capsys):
        d = _plain(verbose=True)
        d.show_tool_result(
            "read_file", {"status": "success"}, display_str="custom display"
        )
        out = capsys.readouterr().out
        assert "custom display" in out
        assert '{"status"' not in out  # JSON not shown when display_str provided

    def test_display_str_silent_in_summary_mode(self, capsys):
        d = _plain(verbose=False)
        d.show_tool_result("read_file", {"status": "success"}, display_str="custom")
        assert capsys.readouterr().out == ""

    def test_none_display_str_falls_back_to_json(self, capsys):
        d = _plain(verbose=True)
        d.show_tool_result("read_file", {"status": "success"}, display_str=None)
        out = capsys.readouterr().out
        assert "success" in out


# ---------------------------------------------------------------------------
# create_display factory
# ---------------------------------------------------------------------------


class TestCreateDisplay:
    def _config(self, backend: str = "plain", markdown: bool = True) -> MagicMock:
        cfg = MagicMock()
        cfg.get.side_effect = lambda key, default=None: {
            "display_backend": backend,
            "display_markdown": markdown,
        }.get(key, default)
        return cfg

    def test_plain_backend(self):
        d = create_display(self._config("plain"))
        assert isinstance(d, PlainDisplay)

    def test_rich_falls_back_to_plain(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            d = create_display(self._config("rich"))
        assert isinstance(d, PlainDisplay)
        assert "not yet implemented" in caplog.text.lower()

    def test_unknown_backend_falls_back_to_plain(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            d = create_display(self._config("curses"))
        assert isinstance(d, PlainDisplay)
        assert "unknown" in caplog.text.lower()

    def test_verbose_flag_passed_through(self):
        d = create_display(self._config(), verbose=True)
        assert d.verbose is True

    def test_markdown_flag_passed_through(self):
        d = create_display(self._config(markdown=False))
        assert d.markdown_enabled is False
