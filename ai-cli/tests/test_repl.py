"""Tests for ai_cli.cli.repl and Session.clear()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_cli.cli.repl import _MAX_TOOL_ROUNDS, _SLASH_COMMANDS, REPL
from ai_cli.core.llm_client import LLMError
from ai_cli.core.session_manager import Session, SessionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(chunks=None):
    llm = MagicMock()
    if chunks is None:
        chunks = [
            {"type": "text", "delta": "Hello!"},
            {"type": "done", "stop_reason": "stop", "usage": {}},
        ]
    llm.send.return_value = iter(chunks)
    return llm


def _make_repl(
    session=None,
    tool_registry=None,
    llm=None,
    display=None,
    workspace=None,
) -> REPL:
    return REPL(
        session=session or MagicMock(),
        tool_registry=tool_registry or MagicMock(),
        llm_client=llm or _make_llm(),
        display=display or MagicMock(),
        workspace=workspace or MagicMock(),
    )


def _make_prompt_session(*inputs):
    """Return a mock PromptSession that yields *inputs* then raises EOFError."""
    pt = MagicMock()
    pt.prompt.side_effect = [*inputs, EOFError()]
    return pt


# ---------------------------------------------------------------------------
# Session.clear()
# ---------------------------------------------------------------------------


class TestSessionClear:
    def _make_session(self, tmp_path: Path) -> Session:
        session_dir = tmp_path / "s1"
        session_dir.mkdir()
        return Session("s1", session_dir, MagicMock())

    def test_clear_removes_current_history(self, tmp_path):
        s = self._make_session(tmp_path)
        s._write_meta({"message_count": 0})
        s.add_message("user", "Hello")
        assert s._current_path.exists()
        s.clear()
        assert not s._current_path.exists()

    def test_clear_preserves_full_history(self, tmp_path):
        s = self._make_session(tmp_path)
        s._write_meta({"message_count": 0})
        s.add_message("user", "Hello")
        assert s._full_path.exists()
        s.clear()
        assert s._full_path.exists()

    def test_clear_resets_message_count(self, tmp_path):
        s = self._make_session(tmp_path)
        s._write_meta({"message_count": 0})
        s.add_message("user", "Hello")
        s.add_message("assistant", "Hi")
        s.clear()
        assert s._read_meta()["message_count"] == 0

    def test_clear_resets_last_message_fields(self, tmp_path):
        s = self._make_session(tmp_path)
        s._write_meta({"message_count": 0})
        s.add_message("user", "Hello")
        s.clear()
        meta = s._read_meta()
        assert meta["last_message_role"] == ""
        assert meta["last_message_preview"] == ""

    def test_clear_is_idempotent(self, tmp_path):
        s = self._make_session(tmp_path)
        s._write_meta({"message_count": 0})
        s.clear()
        s.clear()  # should not raise

    def test_clear_raises_session_error_on_oserror(self, tmp_path):
        s = self._make_session(tmp_path)
        s._write_meta({"message_count": 0})
        s.add_message("user", "Hello")
        # Replace current_path with a directory so unlink raises
        s._current_path.unlink()
        s._current_path.mkdir()
        with pytest.raises(SessionError, match="Could not clear"):
            s.clear()


# ---------------------------------------------------------------------------
# REPL.run() — basic loop behaviour
# ---------------------------------------------------------------------------


class TestREPLRun:
    def test_eof_exits_loop(self):
        repl = _make_repl()
        pt = _make_prompt_session()  # immediately EOFError
        repl.run(_prompt_session=pt)  # should return normally

    def test_keyboard_interrupt_re_prompts(self):
        repl = _make_repl()
        pt = _make_prompt_session(KeyboardInterrupt())
        repl.run(_prompt_session=pt)
        assert pt.prompt.call_count == 2  # once for interrupt, once for EOF

    def test_empty_input_is_ignored(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        pt = _make_prompt_session("   ")
        repl.run(_prompt_session=pt)
        display.begin_assistant_turn.assert_not_called()

    def test_slash_command_dispatched(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        pt = _make_prompt_session("/help")
        repl.run(_prompt_session=pt)
        display.show_help.assert_called_once_with(_SLASH_COMMANDS)

    def test_plain_input_sent_to_llm(self):
        llm = _make_llm()
        session = MagicMock()
        session.should_compact.return_value = False
        repl = _make_repl(session=session, llm=llm)
        pt = _make_prompt_session("Hello there")
        repl.run(_prompt_session=pt)
        session.add_message.assert_any_call("user", "Hello there")


# ---------------------------------------------------------------------------
# REPL._handle_slash_command()
# ---------------------------------------------------------------------------


class TestREPLSlashCommands:
    def test_help(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._handle_slash_command("help")
        display.show_help.assert_called_once_with(_SLASH_COMMANDS)

    def test_exit_raises_system_exit(self):
        repl = _make_repl()
        with pytest.raises(SystemExit):
            repl._handle_slash_command("exit")

    def test_clear_calls_session_clear(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("clear")
        session.clear.assert_called_once()
        display.show_status.assert_called_once()

    def test_clear_shows_error_on_session_error(self):
        session = MagicMock()
        session.clear.side_effect = SessionError("disk full")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("clear")
        display.show_error.assert_called_once()
        display.show_status.assert_not_called()

    def test_verbose_toggles_and_reports(self):
        display = MagicMock()
        display.verbose = False
        repl = _make_repl(display=display)
        repl._handle_slash_command("verbose")
        display.toggle_verbose.assert_called_once()
        display.show_status.assert_called_once()

    def test_verbose_status_reflects_new_state(self):
        display = MagicMock()
        display.verbose = False  # known starting state

        def _toggle():
            display.verbose = not display.verbose

        display.toggle_verbose.side_effect = _toggle
        repl = _make_repl(display=display)
        repl._handle_slash_command("verbose")
        assert display.verbose is True
        msg = display.show_status.call_args[0][0]
        assert "on" in msg

    def test_markdown_toggles_and_reports(self):
        display = MagicMock()
        display.markdown_enabled = True
        repl = _make_repl(display=display)
        repl._handle_slash_command("markdown")
        display.toggle_markdown.assert_called_once()
        display.show_status.assert_called_once()

    def test_compact_calls_session_compact(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("compact")
        session.compact.assert_called_once()

    def test_compact_shows_error_on_failure(self):
        session = MagicMock()
        session.compact.side_effect = SessionError("LLM down")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("compact")
        display.show_error.assert_called_once()

    def test_tools_calls_show_tool_list(self):
        tool_registry = MagicMock()
        tool_registry.all_enabled.return_value = []
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools")
        display.show_tool_list.assert_called_once_with([])

    def test_session_calls_show_session_info(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session")
        display.show_session_info.assert_called_once_with(session)

    def test_unknown_command_shows_error(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._handle_slash_command("foobar")
        display.show_error.assert_called_once()
        msg = display.show_error.call_args[0][0]
        assert "foobar" in msg

    def test_empty_command_shows_error(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._handle_slash_command("")
        display.show_error.assert_called_once()

    def test_command_is_case_insensitive(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._handle_slash_command("HELP")
        display.show_help.assert_called_once()


# ---------------------------------------------------------------------------
# REPL._send_to_llm()
# ---------------------------------------------------------------------------


class TestREPLSendToLLM:
    def test_basic_text_response(self):
        session = MagicMock()
        session.should_compact.return_value = False
        display = MagicMock()
        llm = _make_llm(
            [
                {"type": "text", "delta": "Hi!"},
                {"type": "done", "stop_reason": "stop", "usage": {}},
            ]
        )
        repl = _make_repl(session=session, llm=llm, display=display)
        repl._send_to_llm("Hello")
        display.begin_assistant_turn.assert_called_once()
        display.stream_text.assert_called_once_with("Hi!")
        display.end_assistant_turn.assert_called_once()
        session.add_message.assert_any_call("user", "Hello")
        session.add_message.assert_any_call("assistant", "Hi!")

    def test_llm_error_shows_error_and_returns(self):
        session = MagicMock()
        display = MagicMock()
        llm = MagicMock()
        llm.send.side_effect = LLMError("timeout")
        repl = _make_repl(session=session, llm=llm, display=display)
        repl._send_to_llm("Hello")
        display.show_error.assert_called_once()
        assert "timeout" in display.show_error.call_args[0][0]
        # assistant message must NOT be saved
        for c in session.add_message.call_args_list:
            assert c[0][0] != "assistant"

    def test_session_error_on_user_message_shows_error(self):
        session = MagicMock()
        session.add_message.side_effect = SessionError("disk full")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._send_to_llm("Hello")
        display.show_error.assert_called_once()

    def test_tool_call_executed_and_result_saved(self):
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {"status": "success", "data": {}}

        # First LLM call returns a tool_call; second returns text only
        llm = MagicMock()
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": "read_file",
                        "call_id": "1",
                        "arguments": {"path": "foo.py"},
                    },
                    {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                ]
            ),
            iter(
                [
                    {"type": "text", "delta": "Done."},
                    {"type": "done", "stop_reason": "stop", "usage": {}},
                ]
            ),
        ]

        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )
        repl._send_to_llm("Read foo.py")

        tool_registry.execute.assert_called_once_with(
            "read_file", {"path": "foo.py"}, allow_transient=False
        )
        display.show_tool_call.assert_called_once_with("read_file", {"path": "foo.py"})
        display.show_tool_result.assert_called_once()
        # assistant tool-call message and tool result saved via add_raw_message
        raw_calls = session.add_raw_message.call_args_list
        assert any(
            c[0][0]["role"] == "assistant" and "tool_calls" in c[0][0]
            for c in raw_calls
        )
        assert any(
            c[0][0]["role"] == "tool" and "tool_call_id" in c[0][0] for c in raw_calls
        )

    def test_show_tool_call_before_execute(self):
        """show_tool_call must be called before execute()."""
        call_order = []
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()
        display.show_tool_call.side_effect = lambda *a, **kw: call_order.append("show")
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.side_effect = lambda *a, **kw: (
            call_order.append("exec") or {"status": "success", "data": {}}
        )

        llm = MagicMock()
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": "read_file",
                        "call_id": "1",
                        "arguments": {},
                    },
                    {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                ]
            ),
            iter(
                [
                    {"type": "text", "delta": "Done."},
                    {"type": "done", "stop_reason": "stop", "usage": {}},
                ]
            ),
        ]

        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )
        repl._send_to_llm("go")
        assert call_order == ["show", "exec"]

    def test_tool_call_depth_limit(self):
        session = MagicMock()
        session.get_messages.return_value = []
        session.should_compact.return_value = False
        display = MagicMock()
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {"status": "success", "data": {}}

        # LLM always returns a tool_call — never stops
        llm = MagicMock()
        llm.send.return_value = iter(
            [
                {
                    "type": "tool_call",
                    "name": "read_file",
                    "call_id": "1",
                    "arguments": {},
                },
                {"type": "done", "stop_reason": "tool_calls", "usage": {}},
            ]
        )

        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )

        with patch.object(
            llm,
            "send",
            side_effect=[
                iter(
                    [
                        {
                            "type": "tool_call",
                            "name": "read_file",
                            "call_id": str(i),
                            "arguments": {},
                        },
                        {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                    ]
                )
                for i in range(_MAX_TOOL_ROUNDS)
            ],
        ):
            repl._send_to_llm("loop forever")

        display.show_error.assert_called_once()
        assert str(_MAX_TOOL_ROUNDS) in display.show_error.call_args[0][0]

    def test_empty_text_response_not_saved(self):
        session = MagicMock()
        session.should_compact.return_value = False
        display = MagicMock()
        llm = _make_llm([{"type": "done", "stop_reason": "stop", "usage": {}}])
        repl = _make_repl(session=session, llm=llm, display=display)
        repl._send_to_llm("Hello")
        saved_roles = [c[0][0] for c in session.add_message.call_args_list]
        assert "assistant" not in saved_roles

    def _tool_call_round(self, tool_name, result_data):
        """Helper: LLM returns one tool_call then stops; tool returns result_data."""
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {
            "status": "success",
            "data": result_data,
        }
        llm = MagicMock()
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": tool_name,
                        "call_id": "1",
                        "arguments": {},
                    },
                    {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                ]
            ),
            iter([{"type": "done", "stop_reason": "stop", "usage": {}}]),
        ]
        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )
        return repl, tool_registry

    def test_transient_schemas_only_accepted_from_tool_manager(self):
        # A non-tool_manager tool returning transient_schemas must be ignored.
        schema = {"type": "function", "function": {"name": "read_file"}}
        repl, tool_registry = self._tool_call_round(
            "malicious_tool", {"transient_schemas": [schema]}
        )
        tool_registry.get.return_value = MagicMock()  # simulate registered tool
        repl._send_to_llm("exploit")
        assert repl._pending_transients == {}

    def test_transient_schemas_rejected_for_unregistered_names(self):
        # Even from tool_manager, schemas for unknown tools must not be accepted.
        schema = {"type": "function", "function": {"name": "ghost_tool"}}
        repl, tool_registry = self._tool_call_round(
            "tool_manager", {"transient_schemas": [schema]}
        )
        tool_registry.get.return_value = None  # "ghost_tool" not in registry
        repl._send_to_llm("enable ghost")
        assert repl._pending_transients == {}

    def test_transient_schemas_accepted_from_tool_manager_for_known_tool(self):
        # tool_manager returning a schema for a registered tool name is accepted
        # and injected into the *next* LLM call's tools list.
        schema = {"type": "function", "function": {"name": "read_file"}}
        repl, tool_registry = self._tool_call_round(
            "tool_manager", {"transient_schemas": [schema]}
        )
        tool_registry.get.return_value = MagicMock()  # "read_file" is registered
        repl._send_to_llm("enable read_file")
        # The second llm.send call (round 2) receives the transient schema.
        llm = repl._llm
        second_tools = llm.send.call_args_list[1][1]["tools"]
        assert any(t["function"]["name"] == "read_file" for t in second_tools)

    def test_tools_list_deduplicates_by_name(self):
        # If a transient schema and an already-enabled definition share a name,
        # only one schema should be sent to the LLM (transient takes precedence).
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()

        enabled_schema = {
            "type": "function",
            "function": {"name": "read_file", "description": "enabled version"},
        }
        transient_schema = {
            "type": "function",
            "function": {"name": "read_file", "description": "transient version"},
        }

        tool_registry = MagicMock()
        tool_registry.definitions.return_value = [enabled_schema]
        tool_registry.execute.return_value = {"status": "success", "data": {}}
        tool_registry.get.return_value = MagicMock()

        llm = MagicMock()
        llm.send.return_value = iter(
            [{"type": "done", "stop_reason": "stop", "usage": {}}]
        )

        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )
        repl._pending_transients = {"read_file": transient_schema}
        repl._send_to_llm("go")

        sent_tools = llm.send.call_args[1]["tools"]
        read_file_schemas = [
            t for t in sent_tools if t["function"]["name"] == "read_file"
        ]
        assert len(read_file_schemas) == 1
        assert read_file_schemas[0]["function"]["description"] == "transient version"


# ---------------------------------------------------------------------------
# REPL._preprocess_at_references()
# ---------------------------------------------------------------------------


class TestREPLAtReferences:
    def test_no_references_unchanged(self):
        repl = _make_repl()
        assert repl._preprocess_at_references("Hello world") == "Hello world"

    def test_at_reference_replaced_with_content(self):
        workspace = MagicMock()
        workspace.file_exists.return_value = True
        workspace.read_file.return_value = "line1\nline2\n"
        repl = _make_repl(workspace=workspace)
        result = repl._preprocess_at_references("Check @src/foo.py please")
        assert "[file: src/foo.py]" in result
        assert "line1" in result
        assert "[/file]" in result

    def test_at_reference_missing_file_left_in_place(self):
        workspace = MagicMock()
        workspace.file_exists.return_value = False
        display = MagicMock()
        repl = _make_repl(workspace=workspace, display=display)
        result = repl._preprocess_at_references("Check @missing.py please")
        assert "@missing.py" in result
        display.show_error.assert_called_once()

    def test_at_bang_bypasses_ignore(self):
        workspace = MagicMock()
        resolved = MagicMock()
        resolved.read_text.return_value = "secret\n"
        workspace.resolve.return_value = resolved
        repl = _make_repl(workspace=workspace)
        result = repl._preprocess_at_references("See @!secret.key")
        assert "secret" in result
        workspace.file_exists.assert_not_called()

    def test_workspace_error_leaves_token_in_place(self):
        from ai_cli.core.workspace import WorkspaceError

        workspace = MagicMock()
        workspace.file_exists.side_effect = WorkspaceError("outside workspace")
        display = MagicMock()
        repl = _make_repl(workspace=workspace, display=display)
        result = repl._preprocess_at_references("@../escape.py")
        assert "@../escape.py" in result
        display.show_error.assert_called_once()

    def test_multiple_references_all_replaced(self):
        workspace = MagicMock()
        workspace.file_exists.return_value = True
        workspace.read_file.side_effect = ["content_a\n", "content_b\n"]
        repl = _make_repl(workspace=workspace)
        result = repl._preprocess_at_references("@a.py and @b.py")
        assert "content_a" in result
        assert "content_b" in result


# ---------------------------------------------------------------------------
# REPL._check_compaction()
# ---------------------------------------------------------------------------


class TestREPLCompaction:
    def test_no_compaction_when_not_needed(self):
        session = MagicMock()
        session.should_compact.return_value = False
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._check_compaction()
        session.compact.assert_not_called()
        display.show_status.assert_not_called()

    def test_compaction_triggered_when_needed(self):
        session = MagicMock()
        session.should_compact.return_value = True
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._check_compaction()
        session.compact.assert_called_once()
        assert display.show_status.call_count == 2  # before and after

    def test_compaction_error_shown(self):
        session = MagicMock()
        session.should_compact.return_value = True
        session.compact.side_effect = SessionError("LLM unavailable")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._check_compaction()
        display.show_error.assert_called_once()
