"""Tests for ai_cli.core.session_manager."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_cli.core.llm_client import LLMError
from ai_cli.core.session_manager import (
    Session,
    SessionError,
    SessionManager,
    SessionMeta,
    _generate_session_id,
    _parse_session_meta,
    _truncate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(
    text_response: str = "Summary text.",
    count_tokens: int = 100,
    context_window: int = 10000,
) -> MagicMock:
    """Return a mock LLMClient."""
    llm = MagicMock()
    llm.count_tokens.return_value = count_tokens
    llm.get_model_metadata.return_value = {
        "model": "gpt-4o",
        "context_window": context_window,
        "max_response_tokens": 4096,
    }
    llm.send.return_value = iter(
        [
            {"type": "text", "delta": text_response},
            {"type": "done", "stop_reason": "stop", "usage": {}},
        ]
    )
    return llm


def _make_session(tmp_path: Path, llm: MagicMock | None = None) -> Session:
    """Return a Session rooted at *tmp_path*."""
    if llm is None:
        llm = _make_llm()
    session_dir = tmp_path / "s1"
    session_dir.mkdir()
    return Session("s1", session_dir, llm)


def _make_workspace(root: Path) -> MagicMock:
    ws = MagicMock()
    ws.root = root
    return ws


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_string_unchanged(self):
        assert _truncate("hello") == "hello"

    def test_exact_length_unchanged(self):
        text = "x" * 120
        assert _truncate(text) == text

    def test_long_string_truncated(self):
        text = "x" * 200
        result = _truncate(text)
        assert len(result) == 120
        assert result.endswith("…")

    def test_empty_string(self):
        assert _truncate("") == ""


# ---------------------------------------------------------------------------
# _generate_session_id
# ---------------------------------------------------------------------------


class TestGenerateSessionId:
    def test_returns_string(self):
        assert isinstance(_generate_session_id(), str)

    def test_contains_timestamp_prefix(self):
        sid = _generate_session_id()
        # Format: YYYYMMDDTHHMMSS-<8hex>
        assert len(sid) == 15 + 1 + 8  # "20240101T120000-a1b2c3d4"
        assert sid[8] == "T"
        assert sid[15] == "-"

    def test_unique_ids(self):
        ids = {_generate_session_id() for _ in range(50)}
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# _parse_session_meta
# ---------------------------------------------------------------------------


class TestParseSessionMeta:
    def test_parses_valid_data(self):
        data = {
            "session_id": "abc",
            "workspace_path": "/home/user/proj",
            "started_at": "2024-01-01T12:00:00+00:00",
            "message_count": 5,
            "name": "My session",
            "first_user_message": "Hello",
            "last_message_role": "assistant",
            "last_message_preview": "Sure!",
        }
        meta = _parse_session_meta(data)
        assert meta.session_id == "abc"
        assert meta.workspace_path == Path("/home/user/proj")
        assert meta.message_count == 5
        assert meta.name == "My session"
        assert meta.started_at.year == 2024

    def test_bad_timestamp_falls_back(self):
        meta = _parse_session_meta({"started_at": "not-a-date"})
        assert meta.started_at == datetime.min.replace(tzinfo=timezone.utc)

    def test_missing_fields_use_defaults(self):
        meta = _parse_session_meta({})
        assert meta.session_id == ""
        assert meta.message_count == 0
        assert meta.name is None

    def test_null_workspace_path_becomes_empty_path(self):
        meta = _parse_session_meta({"workspace_path": None})
        assert meta.workspace_path == Path("")

    def test_null_session_id_becomes_empty_string(self):
        # dict.get(key, default) returns None (not the default) when key is null
        # in YAML, so we must coerce explicitly.
        meta = _parse_session_meta({"session_id": None})
        assert meta.session_id == ""
        assert isinstance(meta.session_id, str)

    def test_null_string_fields_become_empty_strings(self):
        meta = _parse_session_meta(
            {
                "first_user_message": None,
                "last_message_role": None,
                "last_message_preview": None,
            }
        )
        assert meta.first_user_message == ""
        assert meta.last_message_role == ""
        assert meta.last_message_preview == ""

    def test_null_name_stays_none(self):
        meta = _parse_session_meta({"name": None})
        assert meta.name is None

    def test_non_string_name_coerced_to_str(self):
        meta = _parse_session_meta({"name": 42})
        assert meta.name == "42"

    def test_non_numeric_message_count_falls_back_to_zero(self):
        meta = _parse_session_meta({"message_count": "oops"})
        assert meta.message_count == 0

    def test_null_message_count_falls_back_to_zero(self):
        meta = _parse_session_meta({"message_count": None})
        assert meta.message_count == 0

    def test_naive_datetime_gets_utc_tzinfo(self):
        # Timestamps without offset (e.g. written by older code) must become UTC-aware
        meta = _parse_session_meta({"started_at": "2024-06-15T10:30:00"})
        assert meta.started_at.tzinfo is not None
        assert meta.started_at == datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_aware_datetime_preserved(self):
        meta = _parse_session_meta({"started_at": "2024-06-15T10:30:00+00:00"})
        assert meta.started_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Session — add_message / get_messages
# ---------------------------------------------------------------------------


class TestSessionMessages:
    def test_get_messages_empty_when_no_file(self, tmp_path):
        session = _make_session(tmp_path)
        assert session.get_messages() == []

    def test_add_message_appended_to_current(self, tmp_path):
        session = _make_session(tmp_path)
        session.add_message("user", "Hello")
        messages = session.get_messages()
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}

    def test_add_multiple_messages_ordered(self, tmp_path):
        session = _make_session(tmp_path)
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there")
        messages = session.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_add_message_also_writes_full_history(self, tmp_path):
        session = _make_session(tmp_path)
        session.add_message("user", "Hello")
        full_path = session.session_dir / "history_full.jsonl"
        assert full_path.exists()
        entry = json.loads(full_path.read_text())
        assert entry["role"] == "user"
        assert "timestamp" in entry

    def test_add_message_raises_on_invalid_role(self, tmp_path):
        session = _make_session(tmp_path)
        with pytest.raises(SessionError, match="Invalid role"):
            session.add_message("robot", "Hello")

    def test_add_message_raises_on_non_string_role(self, tmp_path):
        session = _make_session(tmp_path)
        with pytest.raises(SessionError, match="must be strings"):
            session.add_message(42, "Hello")  # type: ignore[arg-type]

    def test_add_message_raises_on_non_string_content(self, tmp_path):
        session = _make_session(tmp_path)
        with pytest.raises(SessionError, match="must be strings"):
            session.add_message("user", None)  # type: ignore[arg-type]

    def test_add_message_accepts_all_valid_roles(self, tmp_path):
        session = _make_session(tmp_path)
        session._write_meta({"message_count": 0})
        for role in ("system", "user", "assistant", "tool"):
            session.add_message(role, f"message from {role}")
        assert len(session.get_messages()) == 4

    def test_add_message_raises_session_error_on_write_failure(self, tmp_path):
        session = _make_session(tmp_path)
        # Point both history paths into a non-existent directory → OSError on open
        ghost = tmp_path / "ghost"
        session._current_path = ghost / "history_current.jsonl"
        session._full_path = ghost / "history_full.jsonl"
        with pytest.raises(SessionError, match="Could not append"):
            session.add_message("user", "Hello")

    def test_add_message_does_not_update_meta_on_write_failure(self, tmp_path):
        session = _make_session(tmp_path)
        session._write_meta({"message_count": 0})
        # current_path is valid; full_path points into a ghost dir → second write fails
        ghost = tmp_path / "ghost"
        session._full_path = ghost / "history_full.jsonl"
        with pytest.raises(SessionError):
            session.add_message("user", "Hello")
        # metadata must not have been updated
        assert session._read_meta()["message_count"] == 0

    def test_add_message_rolls_back_current_on_full_write_failure(self, tmp_path):
        session = _make_session(tmp_path)
        session._write_meta({"message_count": 0})
        # Write a first message successfully so current has some content.
        session.add_message("user", "First")
        size_before = session._current_path.stat().st_size
        # Now break full_path so the second message's second write fails.
        ghost = tmp_path / "ghost"
        session._full_path = ghost / "history_full.jsonl"
        with pytest.raises(SessionError):
            session.add_message("user", "Second")
        # current must be rolled back to its pre-attempt size.
        assert session._current_path.stat().st_size == size_before

    def test_add_message_rolls_back_history_on_metadata_failure(self, tmp_path):
        session = _make_session(tmp_path)
        session._write_meta({"message_count": 0})
        session.add_message("user", "First")
        size_current_before = session._current_path.stat().st_size
        size_full_before = session._full_path.stat().st_size
        # Point meta_path to a non-existent directory so _write_meta raises SessionError.
        session._meta_path = tmp_path / "ghost_dir" / "metadata.yaml"
        with pytest.raises(SessionError):
            session.add_message("user", "Second")
        # Both history files must be rolled back
        assert session._current_path.stat().st_size == size_current_before
        assert session._full_path.stat().st_size == size_full_before

    def test_get_messages_raises_session_error_on_unicode_error(self, tmp_path):
        session = _make_session(tmp_path)
        session._current_path.write_bytes(b'{"role":"user","content":"ok"}\n\xff\xfe')
        with pytest.raises(SessionError, match="Could not read"):
            session.get_messages()

    def test_get_messages_raises_session_error_on_oserror(self, tmp_path):
        session = _make_session(tmp_path)
        # Replace the file with a directory — opening a dir as a file raises OSError
        session._current_path.mkdir(parents=True, exist_ok=True)
        with pytest.raises(SessionError, match="Could not read"):
            session.get_messages()

    def test_get_messages_skips_malformed_lines(self, tmp_path, caplog):
        session = _make_session(tmp_path)
        current = session.session_dir / "history_current.jsonl"
        current.write_text('{"role": "user", "content": "ok"}\nnot-json\n')
        with caplog.at_level(logging.WARNING):
            messages = session.get_messages()
        assert len(messages) == 1
        assert "malformed" in caplog.text.lower()

    def test_get_messages_skips_entries_missing_role_or_content(self, tmp_path, caplog):
        session = _make_session(tmp_path)
        current = session.session_dir / "history_current.jsonl"
        current.write_text(
            '{"role": "user", "content": "ok"}\n'
            '{"role": "user"}\n'  # missing content
            '{"content": "hi"}\n'  # missing role
            '{"role": 42, "content": "hi"}\n'  # non-string role
        )
        with caplog.at_level(logging.WARNING):
            messages = session.get_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "ok"

    def test_get_messages_skips_non_dict_json(self, tmp_path, caplog):
        # Valid JSON that is not an object (array, string, number) must not crash.
        session = _make_session(tmp_path)
        current = session.session_dir / "history_current.jsonl"
        current.write_text(
            '{"role": "user", "content": "ok"}\n'
            '[{"role": "user", "content": "list"}]\n'  # array, not object
            '"just a string"\n'  # string
            "42\n"  # number
        )
        with caplog.at_level(logging.WARNING):
            messages = session.get_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "ok"
        assert caplog.text  # at least one warning emitted

    def test_get_messages_strips_extra_keys(self, tmp_path):
        session = _make_session(tmp_path)
        current = session.session_dir / "history_current.jsonl"
        current.write_text(
            '{"role": "user", "content": "hi", "timestamp": "2024-01-01", "extra": 42}\n'
        )
        messages = session.get_messages()
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "hi"}
        assert "timestamp" not in messages[0]
        assert "extra" not in messages[0]


# ---------------------------------------------------------------------------
# Session — metadata
# ---------------------------------------------------------------------------


class TestSessionMetadata:
    def test_add_message_updates_message_count(self, tmp_path):
        session = _make_session(tmp_path)
        # Write initial metadata (as SessionManager.new() would)
        session._write_meta({"message_count": 0})
        session.add_message("user", "Hi")
        meta = session._read_meta()
        assert meta["message_count"] == 1

    def test_add_message_updates_last_message_fields(self, tmp_path):
        session = _make_session(tmp_path)
        session._write_meta({"message_count": 0})
        session.add_message("assistant", "Response here")
        meta = session._read_meta()
        assert meta["last_message_role"] == "assistant"
        assert meta["last_message_preview"] == "Response here"

    def test_first_user_message_captured_once(self, tmp_path):
        session = _make_session(tmp_path)
        session._write_meta({"message_count": 0, "first_user_message": ""})
        session.add_message("user", "First")
        session.add_message("assistant", "Reply")
        session.add_message("user", "Second")
        meta = session._read_meta()
        assert meta["first_user_message"] == "First"

    def test_get_meta_returns_metadata_dict(self, tmp_path):
        session = _make_session(tmp_path)
        session._write_meta({"message_count": 7, "name": "test"})
        meta = session.get_meta()
        assert meta["message_count"] == 7
        assert meta["name"] == "test"

    def test_set_name_persists(self, tmp_path):
        session = _make_session(tmp_path)
        session._write_meta({})
        session.set_name("My Session")
        assert session._read_meta()["name"] == "My Session"

    def test_add_message_tolerates_corrupt_message_count(self, tmp_path, caplog):
        session = _make_session(tmp_path)
        session._write_meta({"message_count": "bad", "first_user_message": ""})
        with caplog.at_level(logging.WARNING):
            session.add_message("user", "Hello")
        assert session._read_meta()["message_count"] == 1
        assert "non-numeric" in caplog.text.lower()

    def test_write_meta_is_atomic_on_oserror(self, tmp_path):
        session = _make_session(tmp_path)
        session._write_meta({"message_count": 0})
        # Make the tmp path unwritable by pointing meta_path into a non-existent dir
        session._meta_path = tmp_path / "s1" / "nonexistent" / "metadata.yaml"
        with pytest.raises(SessionError, match="Could not write"):
            session._write_meta({"message_count": 1})


# ---------------------------------------------------------------------------
# Session — token_usage / should_compact
# ---------------------------------------------------------------------------


class TestSessionReadMeta:
    def test_read_meta_raises_session_error_on_corrupt_yaml(self, tmp_path):
        session = _make_session(tmp_path)
        session._meta_path.write_text("key: [unclosed", encoding="utf-8")
        with pytest.raises(SessionError, match="parse"):
            session._read_meta()

    def test_read_meta_returns_empty_dict_when_no_file(self, tmp_path):
        session = _make_session(tmp_path)
        assert session._read_meta() == {}

    def test_read_meta_returns_empty_dict_for_empty_file(self, tmp_path):
        session = _make_session(tmp_path)
        session._meta_path.write_text("", encoding="utf-8")
        assert session._read_meta() == {}

    def test_read_meta_raises_when_not_a_mapping(self, tmp_path):
        session = _make_session(tmp_path)
        session._meta_path.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(SessionError, match="not a YAML mapping"):
            session._read_meta()


class TestSessionTokens:
    def test_token_usage_delegates_to_llm(self, tmp_path):
        llm = _make_llm(count_tokens=500, context_window=8000)
        session = _make_session(tmp_path, llm)
        session.add_message("user", "Hello")
        used, window = session.token_usage()
        assert used == 500
        assert window == 8000

    def test_should_compact_false_below_threshold(self, tmp_path):
        llm = _make_llm(count_tokens=800, context_window=10000)
        session = _make_session(tmp_path, llm)
        assert session.should_compact() is False

    def test_should_compact_true_above_threshold(self, tmp_path):
        # (8600 + 500) = 9100 > 10000 * 0.9 = 9000
        llm = _make_llm(count_tokens=8600, context_window=10000)
        session = _make_session(tmp_path, llm)
        assert session.should_compact() is True

    def test_should_compact_false_at_threshold(self, tmp_path):
        # (8500 + 500) = 9000 is NOT > 9000
        llm = _make_llm(count_tokens=8500, context_window=10000)
        session = _make_session(tmp_path, llm)
        assert session.should_compact() is False

    def test_should_compact_accounts_for_overhead(self, tmp_path):
        # Without overhead: 8800 < 9000 → would be False.
        # With overhead: (8800 + 500) = 9300 > 9000 → True.
        llm = _make_llm(count_tokens=8800, context_window=10000)
        session = _make_session(tmp_path, llm)
        assert session.should_compact() is True

    def test_record_usage_overrides_estimator(self, tmp_path):
        # Estimator says 500; API says 7000 → token_usage returns API value.
        llm = _make_llm(count_tokens=500, context_window=8000)
        session = _make_session(tmp_path, llm)
        session.record_usage(7000)
        used, window = session.token_usage()
        assert used == 7000
        assert window == 8000
        # count_tokens should not have been called.
        llm.count_tokens.assert_not_called()

    def test_record_usage_zero_is_stored(self, tmp_path):
        # Zero is a valid API-reported value and must override the estimator.
        llm = _make_llm(count_tokens=500, context_window=8000)
        session = _make_session(tmp_path, llm)
        session.record_usage(0)
        used, _ = session.token_usage()
        assert used == 0
        llm.count_tokens.assert_not_called()

    def test_clear_resets_token_cache(self, tmp_path):
        # After clear(), token_usage should fall back to the estimator again.
        llm = _make_llm(count_tokens=100, context_window=8000)
        session = _make_session(tmp_path, llm)
        session.record_usage(7000)
        session.clear()
        used, _ = session.token_usage()
        assert used == 100  # estimator called, not cached value
        llm.count_tokens.assert_called_once()


# ---------------------------------------------------------------------------
# Session — compact
# ---------------------------------------------------------------------------


class TestSessionCompact:
    def test_compact_no_op_on_empty_history(self, tmp_path):
        session = _make_session(tmp_path)
        session.compact()  # should not raise, should not call LLM
        session._llm.send.assert_not_called()

    def test_compact_rewrites_current_history(self, tmp_path):
        llm = _make_llm(text_response="Concise summary.")
        session = _make_session(tmp_path, llm)
        session._write_meta({"message_count": 2})
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi")

        # Reset send mock (add_message doesn't call it, but be explicit)
        llm.send.return_value = iter(
            [
                {"type": "text", "delta": "Concise summary."},
                {"type": "done", "stop_reason": "stop", "usage": {}},
            ]
        )
        session.compact()

        messages = session.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "Concise summary." in messages[0]["content"]

    def test_compact_preserves_full_history(self, tmp_path):
        llm = _make_llm(text_response="Summary.")
        session = _make_session(tmp_path, llm)
        session._write_meta({"message_count": 0})
        session.add_message("user", "Hello")

        llm.send.return_value = iter(
            [
                {"type": "text", "delta": "Summary."},
                {"type": "done", "stop_reason": "stop", "usage": {}},
            ]
        )
        session.compact()

        full_lines = (
            (session.session_dir / "history_full.jsonl").read_text().splitlines()
        )
        assert len(full_lines) == 1  # original message still there

    def test_compact_raises_on_empty_summary(self, tmp_path):
        llm = _make_llm(text_response="")
        session = _make_session(tmp_path, llm)
        session._write_meta({"message_count": 0})
        session.add_message("user", "Hello")

        llm.send.return_value = iter(
            [{"type": "done", "stop_reason": "stop", "usage": {}}]
        )
        with pytest.raises(SessionError, match="empty"):
            session.compact()

    def test_compact_wraps_llm_error_as_session_error(self, tmp_path):
        llm = _make_llm()
        session = _make_session(tmp_path, llm)
        session._write_meta({"message_count": 0})
        session.add_message("user", "Hello")

        llm.send.side_effect = LLMError("connection refused")
        with pytest.raises(SessionError, match="LLM call failed"):
            session.compact()

    def test_compact_restores_history_on_metadata_failure(self, tmp_path):
        llm = _make_llm(text_response="Summary.")
        session = _make_session(tmp_path, llm)
        session._write_meta({"message_count": 0})
        session.add_message("user", "Hello")
        old_history = session._current_path.read_bytes()

        llm.send.return_value = iter(
            [
                {"type": "text", "delta": "Summary."},
                {"type": "done", "stop_reason": "stop", "usage": {}},
            ]
        )
        # Point meta_path to a non-existent directory so _write_meta raises SessionError.
        session._meta_path = tmp_path / "ghost_dir" / "metadata.yaml"

        with pytest.raises(SessionError):
            session.compact()

        # History must be restored to pre-compaction content
        assert session._current_path.read_bytes() == old_history

    def test_compact_updates_metadata(self, tmp_path):
        llm = _make_llm(text_response="Summary text.")
        session = _make_session(tmp_path, llm)
        session._write_meta({"message_count": 3})
        session.add_message("user", "Hello")

        llm.send.return_value = iter(
            [
                {"type": "text", "delta": "Summary text."},
                {"type": "done", "stop_reason": "stop", "usage": {}},
            ]
        )
        session.compact()

        meta = session._read_meta()
        assert meta["message_count"] == 1
        assert meta["last_message_role"] == "system"

    def test_compact_sends_conversation_messages_directly(self, tmp_path):
        llm = _make_llm(text_response="Summary.")
        session = _make_session(tmp_path, llm)
        session._write_meta({"message_count": 0})
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there")

        llm.send.return_value = iter(
            [
                {"type": "text", "delta": "Summary."},
                {"type": "done", "stop_reason": "stop", "usage": {}},
            ]
        )
        session.compact()

        call_messages = llm.send.call_args[0][0]
        roles = [m["role"] for m in call_messages]
        # system prompt first, then real messages, then summarize instruction
        assert roles[0] == "system"
        assert {"role": "user", "content": "Hello"} in call_messages
        assert {"role": "assistant", "content": "Hi there"} in call_messages
        assert call_messages[-1]["role"] == "user"  # summarize instruction is last

    def test_compact_passes_instructions_to_llm(self, tmp_path):
        llm = _make_llm(text_response="Summary.")
        session = _make_session(tmp_path, llm)
        session._write_meta({"message_count": 0})
        session.add_message("user", "Hello")

        llm.send.return_value = iter(
            [
                {"type": "text", "delta": "Summary."},
                {"type": "done", "stop_reason": "stop", "usage": {}},
            ]
        )
        session.compact(instructions="focus on code")

        # Instructions appear in the final summarize message (last in the list)
        call_messages = llm.send.call_args[0][0]
        assert "focus on code" in call_messages[-1]["content"]


# ---------------------------------------------------------------------------
# SessionManager — new / load / list / most_recent
# ---------------------------------------------------------------------------


class TestSessionManagerInit:
    def test_init_raises_session_error_when_sessions_dir_is_file(self, tmp_path):
        ws = _make_workspace(tmp_path / "proj")
        sessions_path = tmp_path / "sessions"
        sessions_path.write_text("not a directory")
        with pytest.raises(SessionError, match="Could not create sessions directory"):
            SessionManager(ws, _make_llm(), sessions_path)


class TestSessionManagerNew:
    def test_new_creates_session_dir(self, tmp_path):
        ws = _make_workspace(tmp_path / "proj")
        sm = SessionManager(ws, _make_llm(), tmp_path / "sessions")
        session = sm.new()
        assert session.session_dir.is_dir()

    def test_session_init_does_not_create_directory(self, tmp_path):
        # Session.__init__ must not call mkdir — only SessionManager.new() should.
        non_existent = tmp_path / "ghost_session"
        assert not non_existent.exists()
        Session("ghost", non_existent, _make_llm())
        assert not non_existent.exists()

    def test_new_raises_session_error_on_oserror(self, tmp_path):
        ws = _make_workspace(tmp_path / "proj")
        sessions_dir = tmp_path / "sessions"
        sm = SessionManager(ws, _make_llm(), sessions_dir)
        # Make sessions_dir a file so mkdir raises OSError (not FileExistsError)
        sessions_dir.rmdir()
        sessions_dir.write_text("not a directory")
        with pytest.raises(SessionError, match="Could not create session directory"):
            sm.new()

    def test_new_writes_metadata(self, tmp_path):
        ws = _make_workspace(tmp_path / "proj")
        sm = SessionManager(ws, _make_llm(), tmp_path / "sessions")
        session = sm.new()
        meta = session._read_meta()
        assert meta["session_id"] == session.session_id
        assert meta["message_count"] == 0
        assert meta["name"] is None

    def test_new_records_workspace_path(self, tmp_path):
        proj = tmp_path / "myproject"
        ws = _make_workspace(proj)
        sm = SessionManager(ws, _make_llm(), tmp_path / "sessions")
        session = sm.new()
        meta = session._read_meta()
        assert Path(meta["workspace_path"]) == proj

    def test_new_sessions_have_unique_ids(self, tmp_path):
        ws = _make_workspace(tmp_path / "proj")
        sm = SessionManager(ws, _make_llm(), tmp_path / "sessions")
        ids = {sm.new().session_id for _ in range(10)}
        assert len(ids) == 10


class TestSessionManagerLoad:
    def test_load_returns_session(self, tmp_path):
        ws = _make_workspace(tmp_path / "proj")
        sm = SessionManager(ws, _make_llm(), tmp_path / "sessions")
        created = sm.new()
        loaded = sm.load(created.session_id)
        assert loaded.session_id == created.session_id

    def test_load_raises_for_unknown_id(self, tmp_path):
        ws = _make_workspace(tmp_path / "proj")
        sm = SessionManager(ws, _make_llm(), tmp_path / "sessions")
        with pytest.raises(SessionError, match="not found"):
            sm.load("20240101T000000-ffffffff")

    def test_load_raises_when_metadata_missing(self, tmp_path):
        ws = _make_workspace(tmp_path / "proj")
        sessions_dir = tmp_path / "sessions"
        sm = SessionManager(ws, _make_llm(), sessions_dir)
        # Create directory without metadata
        (sessions_dir / "20240101T000000-00000001").mkdir(parents=True)
        with pytest.raises(SessionError, match="metadata missing"):
            sm.load("20240101T000000-00000001")

    def test_load_raises_on_corrupt_metadata(self, tmp_path):
        ws = _make_workspace(tmp_path / "proj")
        sessions_dir = tmp_path / "sessions"
        sm = SessionManager(ws, _make_llm(), sessions_dir)
        bad_dir = sessions_dir / "20240101T000000-badc0de1"
        bad_dir.mkdir(parents=True)
        (bad_dir / "metadata.yaml").write_text(": ][invalid yaml\n")
        with pytest.raises(SessionError):
            sm.load("20240101T000000-badc0de1")

    def test_load_raises_on_non_mapping_metadata(self, tmp_path):
        ws = _make_workspace(tmp_path / "proj")
        sessions_dir = tmp_path / "sessions"
        sm = SessionManager(ws, _make_llm(), sessions_dir)
        bad_dir = sessions_dir / "20240101T000000-badc0de2"
        bad_dir.mkdir(parents=True)
        (bad_dir / "metadata.yaml").write_text("- item1\n- item2\n")
        with pytest.raises(SessionError, match="not a YAML mapping"):
            sm.load("20240101T000000-badc0de2")

    def test_load_rejects_invalid_session_id_formats(self, tmp_path):
        ws = _make_workspace(tmp_path / "proj")
        sm = SessionManager(ws, _make_llm(), tmp_path / "sessions")
        bad_ids = [
            "",  # empty
            "../evil",  # path traversal
            "a/b",  # contains slash
            "a\\b",  # contains backslash
            "..",  # dots
            "nonexistent-id",  # wrong format
            "C:foo",  # Windows drive prefix
            "20240101T000000-FFFFFFFF",  # uppercase hex not allowed
            "20240101T000000-fffe",  # too short
            "20240101X000000-ffffffff",  # wrong separator
        ]
        for bad_id in bad_ids:
            with pytest.raises(SessionError, match="[Ii]nvalid"):
                sm.load(bad_id)


class TestSessionManagerList:
    def test_list_returns_empty_for_no_sessions(self, tmp_path):
        ws = _make_workspace(tmp_path / "proj")
        sm = SessionManager(ws, _make_llm(), tmp_path / "sessions")
        assert sm.list(tmp_path / "proj") == []

    def test_list_filters_by_workspace(self, tmp_path):
        proj_a = tmp_path / "proj_a"
        proj_b = tmp_path / "proj_b"
        sessions_dir = tmp_path / "sessions"

        sm_a = SessionManager(_make_workspace(proj_a), _make_llm(), sessions_dir)
        sm_b = SessionManager(_make_workspace(proj_b), _make_llm(), sessions_dir)

        sm_a.new()
        sm_b.new()

        results_a = sm_a.list(proj_a)
        results_b = sm_b.list(proj_b)
        assert len(results_a) == 1
        assert len(results_b) == 1
        assert results_a[0].workspace_path == proj_a
        assert results_b[0].workspace_path == proj_b

    def test_list_sorted_newest_first(self, tmp_path):
        proj = tmp_path / "proj"
        sm = SessionManager(_make_workspace(proj), _make_llm(), tmp_path / "sessions")
        s1 = sm.new()
        s2 = sm.new()
        # Pin distinct timestamps so sort order is deterministic regardless of
        # clock resolution.
        m1 = s1._read_meta()
        m2 = s2._read_meta()
        m1["started_at"] = "2024-01-01T00:00:00+00:00"
        m2["started_at"] = "2024-01-02T00:00:00+00:00"
        s1._write_meta(m1)
        s2._write_meta(m2)
        results = sm.list(proj)
        assert results[0].session_id == s2.session_id
        assert results[1].session_id == s1.session_id

    def test_list_sorted_uses_session_id_as_tiebreaker(self, tmp_path):
        proj = tmp_path / "proj"
        sm = SessionManager(_make_workspace(proj), _make_llm(), tmp_path / "sessions")
        s1 = sm.new()
        s2 = sm.new()
        # Give both sessions the same timestamp so the tie-breaker kicks in.
        same_ts = "2024-06-15T12:00:00+00:00"
        m1 = s1._read_meta()
        m2 = s2._read_meta()
        m1["started_at"] = same_ts
        m2["started_at"] = same_ts
        s1._write_meta(m1)
        s2._write_meta(m2)
        results = sm.list(proj)
        # Sorted descending by (started_at, session_id) — higher session_id first.
        ids = [r.session_id for r in results]
        assert ids == sorted(ids, reverse=True)

    def test_list_returns_session_meta_objects(self, tmp_path):
        proj = tmp_path / "proj"
        sm = SessionManager(_make_workspace(proj), _make_llm(), tmp_path / "sessions")
        sm.new()
        results = sm.list(proj)
        assert isinstance(results[0], SessionMeta)

    def test_list_raises_session_error_when_sessions_dir_unreadable(self, tmp_path):
        proj = tmp_path / "proj"
        sessions_dir = tmp_path / "sessions"
        sm = SessionManager(_make_workspace(proj), _make_llm(), sessions_dir)
        # Replace sessions_dir with a file so iterdir() raises OSError
        sessions_dir.rmdir()
        sessions_dir.write_text("not a directory")
        with pytest.raises(SessionError, match="Could not list"):
            sm.list(proj)

    def test_list_skips_dirs_without_metadata(self, tmp_path, caplog):
        proj = tmp_path / "proj"
        sessions_dir = tmp_path / "sessions"
        sm = SessionManager(_make_workspace(proj), _make_llm(), sessions_dir)
        sm.new()
        # Add an orphan directory with no metadata.yaml
        (sessions_dir / "orphan").mkdir()
        with caplog.at_level(logging.WARNING):
            results = sm.list(proj)
        assert len(results) == 1  # orphan is skipped

    def test_list_skips_corrupt_yaml_with_warning(self, tmp_path, caplog):
        proj = tmp_path / "proj"
        sessions_dir = tmp_path / "sessions"
        sm = SessionManager(_make_workspace(proj), _make_llm(), sessions_dir)
        good = sm.new()
        # Create a session directory with corrupt YAML metadata.
        bad_dir = sessions_dir / "20240101T000000-badbadba"
        bad_dir.mkdir(parents=True)
        (bad_dir / "metadata.yaml").write_text(": ][invalid yaml\n")
        with caplog.at_level(logging.WARNING):
            results = sm.list(proj)
        assert len(results) == 1
        assert results[0].session_id == good.session_id
        assert any("Could not read" in r.message for r in caplog.records)

    def test_list_skips_non_mapping_yaml_with_warning(self, tmp_path, caplog):
        proj = tmp_path / "proj"
        sessions_dir = tmp_path / "sessions"
        sm = SessionManager(_make_workspace(proj), _make_llm(), sessions_dir)
        good = sm.new()
        # Create a session directory whose metadata.yaml is a YAML list, not a mapping.
        bad_dir = sessions_dir / "20240101T000000-1a2b3c4d"
        bad_dir.mkdir(parents=True)
        (bad_dir / "metadata.yaml").write_text("- item1\n- item2\n")
        with caplog.at_level(logging.WARNING):
            results = sm.list(proj)
        assert len(results) == 1
        assert results[0].session_id == good.session_id
        assert any("not a YAML mapping" in r.message for r in caplog.records)


class TestSessionManagerMostRecent:
    def test_most_recent_returns_none_when_empty(self, tmp_path):
        ws = _make_workspace(tmp_path / "proj")
        sm = SessionManager(ws, _make_llm(), tmp_path / "sessions")
        assert sm.most_recent(tmp_path / "proj") is None

    def test_most_recent_returns_latest_session(self, tmp_path):
        proj = tmp_path / "proj"
        sm = SessionManager(_make_workspace(proj), _make_llm(), tmp_path / "sessions")
        s1 = sm.new()
        s2 = sm.new()
        # Pin distinct timestamps so most_recent() is deterministic.
        m1 = s1._read_meta()
        m2 = s2._read_meta()
        m1["started_at"] = "2024-01-01T00:00:00+00:00"
        m2["started_at"] = "2024-01-02T00:00:00+00:00"
        s1._write_meta(m1)
        s2._write_meta(m2)
        recent = sm.most_recent(proj)
        assert recent is not None
        assert recent.session_id == s2.session_id
