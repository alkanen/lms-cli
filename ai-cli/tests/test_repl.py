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

    def test_disallowed_tool_shows_user_hint_and_sends_unknown_tool_to_llm(self):
        # When the registry returns tool_disallowed, the REPL must:
        # 1. Show a user-facing hint via show_error (with correct wording).
        # 2. Replace the result with unknown_tool before it reaches the LLM.
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {
            "status": "error",
            "error": "tool_disallowed",
            "message": "Tool 'secret' is not available.",
            "code": 403,
        }

        llm = MagicMock()
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": "secret",
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
        repl._send_to_llm("use secret")

        # User sees the hint with the correct wording.
        display.show_error.assert_called_once()
        hint = display.show_error.call_args[0][0]
        assert "secret" in hint
        assert "allow" in hint
        assert "list of available tools" in hint

        # The message saved for the LLM must contain unknown_tool, not tool_disallowed.
        tool_result_msgs = [
            c[0][0]
            for c in session.add_raw_message.call_args_list
            if c[0][0].get("role") == "tool"
        ]
        assert len(tool_result_msgs) == 1
        import json as _json

        content = _json.loads(tool_result_msgs[0]["content"])
        assert content.get("error") == "unknown_tool"
        assert content.get("error") != "tool_disallowed"

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


# ---------------------------------------------------------------------------
# /compact with instructions
# ---------------------------------------------------------------------------


class TestCompactSubcommand:
    def test_compact_no_instructions_passes_empty_string(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("compact")
        session.compact.assert_called_once_with(instructions="")

    def test_compact_with_instructions_passes_them(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("compact Summarise the key decisions only")
        session.compact.assert_called_once_with(
            instructions="Summarise the key decisions only"
        )

    def test_compact_error_shows_error(self):
        session = MagicMock()
        session.compact.side_effect = SessionError("LLM down")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("compact focus on bugs")
        display.show_error.assert_called_once()


# ---------------------------------------------------------------------------
# /tools subcommands
# ---------------------------------------------------------------------------


class TestToolsSubcommand:
    def _reg(self, all_enabled=None, all_tools_info=None, tool_info_val=None):
        tr = MagicMock()
        tr.all_enabled.return_value = all_enabled if all_enabled is not None else []
        tr.all_tools_info.return_value = (
            all_tools_info if all_tools_info is not None else []
        )
        tr.tool_info.return_value = tool_info_val
        return tr

    def test_tools_no_subcommand_shows_enabled_list(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools")
        display.show_tool_list.assert_called_once_with([])

    def test_tools_list_shows_all_tools(self):
        info_list = [{"name": "echo", "enabled": True, "allowed": True}]
        tool_registry = self._reg(all_tools_info=info_list)
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools list")
        display.show_tool_list_all.assert_called_once_with(info_list)

    def test_tools_info_known_tool(self):
        info = {"name": "echo", "description": "Echoes."}
        tool_registry = self._reg(tool_info_val=info)
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools info echo")
        tool_registry.tool_info.assert_called_once_with("echo")
        display.show_tool_info.assert_called_once_with(info)

    def test_tools_info_unknown_tool_shows_error(self):
        tool_registry = self._reg(tool_info_val=None)
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools info ghost")
        display.show_error.assert_called_once()

    def test_tools_info_missing_name_shows_error(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools info")
        display.show_error.assert_called_once()

    def test_tools_enable_calls_enable(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools enable read_file")
        tool_registry.enable.assert_called_once_with("read_file")
        display.show_status.assert_called_once()

    def test_tools_disable_calls_disable(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools disable write_file")
        tool_registry.disable.assert_called_once_with("write_file")

    def test_tools_enable_session_calls_enable_session(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools enable --session read_file")
        tool_registry.enable_session.assert_called_once_with("read_file")
        tool_registry.enable.assert_not_called()

    def test_tools_disable_session_calls_disable_session(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools disable --session write_file")
        tool_registry.disable_session.assert_called_once_with("write_file")

    def test_tools_allow_calls_allow(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools allow bash")
        tool_registry.allow.assert_called_once_with("bash")

    def test_tools_disallow_calls_disallow(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools disallow bash")
        tool_registry.disallow.assert_called_once_with("bash")

    def test_tools_allow_session(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools allow --session bash")
        tool_registry.allow_session.assert_called_once_with("bash")
        tool_registry.allow.assert_not_called()

    def test_tools_disallow_session(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools disallow --session bash")
        tool_registry.disallow_session.assert_called_once_with("bash")
        tool_registry.disallow.assert_not_called()

    def test_tools_enable_missing_name_shows_error(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools enable")
        display.show_error.assert_called_once()

    def test_tools_enable_unknown_tool_shows_error(self):
        tool_registry = self._reg()
        tool_registry.get.return_value = None  # unknown tool
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools enable ghost_tool")
        display.show_error.assert_called_once()
        tool_registry.enable.assert_not_called()

    def test_tools_unknown_subcommand_shows_error(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools frobnicate")
        display.show_error.assert_called_once()
        assert "frobnicate" in display.show_error.call_args[0][0]

    def test_status_message_uses_correct_past_tense(self):
        for sub, expected_past in [
            ("enable", "enabled"),
            ("disable", "disabled"),
            ("allow", "allowed"),
            ("disallow", "disallowed"),
        ]:
            tool_registry = self._reg()
            display = MagicMock()
            repl = _make_repl(tool_registry=tool_registry, display=display)
            repl._handle_slash_command(f"tools {sub} read_file")
            msg = display.show_status.call_args[0][0]
            assert expected_past in msg, (
                f"{sub!r} → expected {expected_past!r} in {msg!r}"
            )

    def test_status_message_mentions_scope_persistent(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools enable read_file")
        msg = display.show_status.call_args[0][0]
        assert "persistently" in msg

    def test_status_message_mentions_scope_session(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools enable --session read_file")
        msg = display.show_status.call_args[0][0]
        assert "session" in msg


# ---------------------------------------------------------------------------
# /session subcommands
# ---------------------------------------------------------------------------


class TestSessionSubcommand:
    def test_session_no_subcommand_shows_info(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session")
        display.show_session_info.assert_called_once_with(session)

    def test_session_name_calls_set_name(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session name my-chat")
        session.set_name.assert_called_once_with("my-chat")
        display.show_status.assert_called_once()

    def test_session_name_with_spaces(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session name bug fix session")
        session.set_name.assert_called_once_with("bug fix session")

    def test_session_name_missing_name_shows_error(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session name")
        display.show_error.assert_called_once()
        session.set_name.assert_not_called()

    def test_session_name_error_shows_error(self):
        session = MagicMock()
        session.set_name.side_effect = SessionError("disk full")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session name my-name")
        display.show_error.assert_called_once()

    def test_session_unknown_subcommand_shows_error(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session frobnicate")
        display.show_error.assert_called_once()
        assert "frobnicate" in display.show_error.call_args[0][0]


# ---------------------------------------------------------------------------
# /history command
# ---------------------------------------------------------------------------


class TestHistoryCommand:
    def test_history_calls_show_history(self):
        session = MagicMock()
        messages = [{"role": "user", "content": "hi"}]
        session.get_messages.return_value = messages
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("history")
        display.show_history.assert_called_once_with(messages)

    def test_history_session_error_shows_error(self):
        session = MagicMock()
        session.get_messages.side_effect = SessionError("disk full")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("history")
        display.show_error.assert_called_once()
        display.show_history.assert_not_called()


# ---------------------------------------------------------------------------
# REPL reasoning chunk routing and update_usage
# ---------------------------------------------------------------------------


class TestREPLReasoningAndUsage:
    def test_reasoning_chunk_routed_to_stream_reasoning(self):
        session = MagicMock()
        session.should_compact.return_value = False
        display = MagicMock()
        llm = _make_llm(
            [
                {"type": "reasoning", "delta": "thinking..."},
                {"type": "text", "delta": "answer"},
                {"type": "done", "stop_reason": "stop", "usage": {}},
            ]
        )
        repl = _make_repl(session=session, llm=llm, display=display)
        repl._send_to_llm("hello")
        display.stream_reasoning.assert_called_once_with("thinking...")

    def test_text_chunk_not_routed_to_stream_reasoning(self):
        session = MagicMock()
        session.should_compact.return_value = False
        display = MagicMock()
        llm = _make_llm(
            [
                {"type": "text", "delta": "answer"},
                {"type": "done", "stop_reason": "stop", "usage": {}},
            ]
        )
        repl = _make_repl(session=session, llm=llm, display=display)
        repl._send_to_llm("hello")
        display.stream_reasoning.assert_not_called()

    def test_update_usage_called_on_done_chunk(self):
        session = MagicMock()
        session.should_compact.return_value = False
        display = MagicMock()
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        llm = _make_llm(
            [
                {"type": "text", "delta": "Hi"},
                {"type": "done", "stop_reason": "stop", "usage": usage},
            ]
        )
        repl = _make_repl(session=session, llm=llm, display=display)
        repl._send_to_llm("hello")
        display.update_usage.assert_called_once()
        call_usage = display.update_usage.call_args[0][0]
        assert call_usage == usage


# ---------------------------------------------------------------------------
# REPL format_display plumbing
# ---------------------------------------------------------------------------


class TestREPLFormatDisplay:
    def _make_tool_call_round(self, display_str):
        """LLM calls read_file once, tool returns success; tool.format_display returns display_str."""
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()
        tool_obj = MagicMock()
        tool_obj.format_display.return_value = display_str
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {"status": "success", "data": {}}
        tool_registry.get.return_value = tool_obj

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
            iter([{"type": "done", "stop_reason": "stop", "usage": {}}]),
        ]
        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )
        return repl, display, tool_obj

    def test_format_display_called_with_args_and_result(self):
        repl, display, tool_obj = self._make_tool_call_round("formatted")
        repl._send_to_llm("read foo.py")
        tool_obj.format_display.assert_called_once_with(
            args={"path": "foo.py"}, result={"status": "success", "data": {}}
        )

    def test_format_display_result_passed_to_show_tool_result(self):
        repl, display, _ = self._make_tool_call_round("my custom string")
        repl._send_to_llm("read foo.py")
        display.show_tool_result.assert_called_once_with(
            "read_file", {"status": "success", "data": {}}, "my custom string"
        )

    def test_format_display_none_passed_when_returns_none(self):
        repl, display, _ = self._make_tool_call_round(None)
        repl._send_to_llm("read foo.py")
        display.show_tool_result.assert_called_once_with(
            "read_file", {"status": "success", "data": {}}, None
        )

    def test_format_display_exception_does_not_propagate(self):
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()
        tool_obj = MagicMock()
        tool_obj.format_display.side_effect = RuntimeError("crash")
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {"status": "success", "data": {}}
        tool_registry.get.return_value = tool_obj

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
            iter([{"type": "done", "stop_reason": "stop", "usage": {}}]),
        ]
        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )
        repl._send_to_llm("go")  # should not raise
        display.show_tool_result.assert_called_once_with(
            "read_file", {"status": "success", "data": {}}, None
        )
