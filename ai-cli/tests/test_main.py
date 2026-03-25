"""Tests for ai_cli.__main__ entry point."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from ai_cli.__main__ import _RESUME_PICK, _pick_session, _show_resume_context
from ai_cli.__main__ import _cmd_repl as _real_cmd_repl
from ai_cli.core.session_manager import SessionError
from ai_cli.core.workspace import _DOT_AI_CLI, _INIT_TEMPLATES


def run_main(argv: list[str]) -> None:
    """Import and run main() with the given argv."""
    with patch.object(sys, "argv", ["ai-cli"] + argv):
        from ai_cli.__main__ import main

        main()


@pytest.fixture(autouse=True)
def isolate_global_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patch get_global_dir in both the workspace module and __main__'s local binding.
    fake_global = tmp_path_factory.mktemp("fake_global")
    monkeypatch.setattr("ai_cli.core.workspace.get_global_dir", lambda: fake_global)
    monkeypatch.setattr("ai_cli.__main__.get_global_dir", lambda: fake_global)
    # Prevent the REPL from actually starting in tests that only care about
    # startup/init logic.  Tests that specifically exercise _cmd_repl override this.
    monkeypatch.setattr("ai_cli.__main__._cmd_repl", lambda *_, **__: sys.exit(1))


# ---------------------------------------------------------------------------
# --init
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_creates_scaffold(self, tmp_path):
        run_main(["--init", "--workspace", str(tmp_path)])
        assert (tmp_path / _DOT_AI_CLI).is_dir()
        for filename in _INIT_TEMPLATES:
            assert (tmp_path / _DOT_AI_CLI / filename).is_file()

    def test_init_existing_dot_ai_cli_user_confirms(self, tmp_path):
        (tmp_path / _DOT_AI_CLI).mkdir()
        with patch("builtins.input", return_value="y"):
            run_main(["--init", "--workspace", str(tmp_path)])
        assert (tmp_path / _DOT_AI_CLI / "config.yaml").is_file()

    def test_init_existing_dot_ai_cli_user_aborts(self, tmp_path, capsys):
        (tmp_path / _DOT_AI_CLI).mkdir()
        with patch("builtins.input", return_value="n"):
            run_main(["--init", "--workspace", str(tmp_path)])
        assert not (tmp_path / _DOT_AI_CLI / "config.yaml").exists()
        assert "Aborted" in capsys.readouterr().out

    def test_init_eof_defaults_to_proceed(self, tmp_path):
        (tmp_path / _DOT_AI_CLI).mkdir()
        with patch("builtins.input", side_effect=EOFError):
            run_main(["--init", "--workspace", str(tmp_path)])
        assert (tmp_path / _DOT_AI_CLI / "config.yaml").is_file()

    def test_init_uses_cwd_by_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_main(["--init"])
        assert (tmp_path / _DOT_AI_CLI).is_dir()


# ---------------------------------------------------------------------------
# No subcommand
# ---------------------------------------------------------------------------


class TestNoSubcommand:
    def test_exits_nonzero_without_init(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            run_main([])
        assert exc_info.value.code != 0

    def test_no_workspace_exits_nonzero(self, tmp_path, capsys):
        """_cmd_repl exits nonzero when no .ai-cli/ project is found."""
        # Call the real _cmd_repl (imported at module level, before autouse patching).
        # tmp_path has no .ai-cli/ so find_root returns None.
        with pytest.raises(SystemExit) as exc_info:
            _real_cmd_repl(tmp_path, tmp_path)
        assert exc_info.value.code != 0
        assert ".ai-cli" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _ensure_global_dir
# ---------------------------------------------------------------------------


class TestEnsureGlobalDir:
    def _run_with_missing_global(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path_factory: pytest.TempPathFactory,
        input_val: str,
    ) -> tuple[int | None, str]:
        """Run main() with get_global_dir() returning a non-existent path."""
        missing = tmp_path_factory.mktemp("base") / "missing_global"
        monkeypatch.setattr("ai_cli.__main__.get_global_dir", lambda: missing)
        with (
            patch("builtins.input", return_value=input_val),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_main([])
        return exc_info.value.code, str(missing)

    def test_user_confirms_creates_global_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path_factory: pytest.TempPathFactory,
        capsys,
    ) -> None:
        missing = tmp_path_factory.mktemp("base") / "new_global"
        monkeypatch.setattr("ai_cli.__main__.get_global_dir", lambda: missing)
        with patch("builtins.input", return_value="y"), pytest.raises(SystemExit):
            run_main([])
        assert missing.is_dir()
        assert (missing / "config.yaml").is_file()

    def test_user_declines_exits_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path_factory: pytest.TempPathFactory,
        capsys,
    ) -> None:
        code, _ = self._run_with_missing_global(monkeypatch, tmp_path_factory, "n")
        assert code == 0

    def test_user_declines_prints_abort_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path_factory: pytest.TempPathFactory,
        capsys,
    ) -> None:
        self._run_with_missing_global(monkeypatch, tmp_path_factory, "n")
        out = capsys.readouterr().out
        assert "Aborted" in out

    def test_prompt_mentions_env_var(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path_factory: pytest.TempPathFactory,
        capsys,
    ) -> None:
        self._run_with_missing_global(monkeypatch, tmp_path_factory, "n")
        out = capsys.readouterr().out
        assert "AI_CLI_GLOBAL_DIR" in out

    def test_eof_defaults_to_create(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        missing = tmp_path_factory.mktemp("base") / "new_global_eof"
        monkeypatch.setattr("ai_cli.__main__.get_global_dir", lambda: missing)
        with patch("builtins.input", side_effect=EOFError), pytest.raises(SystemExit):
            run_main([])
        assert missing.is_dir()

    def test_existing_global_dir_skips_prompt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        existing = tmp_path_factory.mktemp("existing_global")
        monkeypatch.setattr("ai_cli.__main__.get_global_dir", lambda: existing)
        with patch("builtins.input") as mock_input, pytest.raises(SystemExit):
            run_main([])
        mock_input.assert_not_called()

    def test_path_is_file_exits_with_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path_factory: pytest.TempPathFactory,
        capsys,
    ) -> None:
        base = tmp_path_factory.mktemp("base")
        file_path = base / "not_a_dir"
        file_path.write_text("oops")
        monkeypatch.setattr("ai_cli.__main__.get_global_dir", lambda: file_path)
        with pytest.raises(SystemExit) as exc_info:
            run_main([])
        assert exc_info.value.code != 0
        assert "not a directory" in capsys.readouterr().err

    @pytest.mark.skipif(
        os.name == "nt",
        reason="symlink creation requires elevated privileges on Windows",
    )
    def test_broken_symlink_exits_with_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path_factory: pytest.TempPathFactory,
        capsys,
    ) -> None:
        base = tmp_path_factory.mktemp("base")
        symlink = base / "broken_link"
        symlink.symlink_to(base / "nonexistent_target")
        monkeypatch.setattr("ai_cli.__main__.get_global_dir", lambda: symlink)
        with pytest.raises(SystemExit) as exc_info:
            run_main([])
        assert exc_info.value.code != 0
        assert "not a directory" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Invalid AI_CLI_GLOBAL_DIR
# ---------------------------------------------------------------------------


class TestInvalidGlobalDirEnv:
    def test_empty_env_var_exits_with_error(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "ai_cli.__main__.get_global_dir",
            lambda: (_ for _ in ()).throw(
                ValueError("AI_CLI_GLOBAL_DIR is set but empty.")
            ),
        )
        with pytest.raises(SystemExit) as exc_info:
            run_main([])
        assert exc_info.value.code != 0
        err = capsys.readouterr().err
        assert "AI_CLI_GLOBAL_DIR" in err


# ---------------------------------------------------------------------------
# parse_args — new session flags
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_no_session_flags(self):
        with patch("sys.argv", ["ai-cli"]):
            from ai_cli.__main__ import parse_args

            args = parse_args()
        assert args.resume is None
        assert args.continue_ is False

    def test_resume_no_argument_stores_sentinel(self):
        with patch("sys.argv", ["ai-cli", "--resume"]):
            from ai_cli.__main__ import parse_args

            args = parse_args()
        assert args.resume is _RESUME_PICK

    def test_resume_with_session_id(self):
        sid = "lms-cli__2026-03-19T12h00m00.000s"
        with patch("sys.argv", ["ai-cli", "--resume", sid]):
            from ai_cli.__main__ import parse_args

            args = parse_args()
        assert args.resume == sid

    def test_continue_flag(self):
        with patch("sys.argv", ["ai-cli", "--continue"]):
            from ai_cli.__main__ import parse_args

            args = parse_args()
        assert args.continue_ is True

    def test_display_defaults_to_none(self):
        with patch("sys.argv", ["ai-cli"]):
            from ai_cli.__main__ import parse_args

            args = parse_args()
        assert args.display is None

    def test_display_plain(self):
        with patch("sys.argv", ["ai-cli", "--display", "plain"]):
            from ai_cli.__main__ import parse_args

            args = parse_args()
        assert args.display == "plain"

    def test_display_rich(self):
        with patch("sys.argv", ["ai-cli", "--display", "rich"]):
            from ai_cli.__main__ import parse_args

            args = parse_args()
        assert args.display == "rich"

    def test_display_invalid_exits(self):
        with patch("sys.argv", ["ai-cli", "--display", "curses"]):
            from ai_cli.__main__ import parse_args

            with pytest.raises(SystemExit):
                parse_args()

    def test_max_tool_rounds_valid(self):
        with patch("sys.argv", ["ai-cli", "--max-tool-rounds", "5"]):
            from ai_cli.__main__ import parse_args

            args = parse_args()
        assert args.max_tool_rounds == 5

    def test_max_tool_rounds_zero_exits(self):
        with patch("sys.argv", ["ai-cli", "--max-tool-rounds", "0"]):
            from ai_cli.__main__ import parse_args

            with pytest.raises(SystemExit):
                parse_args()

    def test_max_tool_rounds_negative_exits(self):
        with patch("sys.argv", ["ai-cli", "--max-tool-rounds", "-3"]):
            from ai_cli.__main__ import parse_args

            with pytest.raises(SystemExit):
                parse_args()

    def test_max_tool_rounds_non_integer_exits(self):
        with patch("sys.argv", ["ai-cli", "--max-tool-rounds", "abc"]):
            from ai_cli.__main__ import parse_args

            with pytest.raises(SystemExit):
                parse_args()


# ---------------------------------------------------------------------------
# _pick_session
# ---------------------------------------------------------------------------


def _make_session(session_id: str = "abc") -> MagicMock:
    s = MagicMock()
    s.session_id = session_id
    return s


def _make_meta(session_id: str = "abc") -> MagicMock:
    m = MagicMock()
    m.session_id = session_id
    return m


class TestPickSession:
    def _sm(
        self,
        *,
        new_session=None,
        loaded_session=None,
        recent_session=None,
        sessions=None,
    ):
        sm = MagicMock()
        sm.new.return_value = new_session or _make_session("new-id")
        sm.load.return_value = loaded_session or _make_session("loaded-id")
        sm.most_recent.return_value = recent_session
        sm.list.return_value = sessions if sessions is not None else []
        return sm

    def _display(self, choice=None):
        d = MagicMock()
        d.show_session_list.return_value = choice
        return d

    def test_no_flags_creates_new_session(self):
        new_sess = _make_session("new-id")
        sm = self._sm(new_session=new_sess)
        session, resumed = _pick_session(
            sm,
            self._display(),
            MagicMock(),
            resume_id=None,
            resume_list=False,
            continue_=False,
        )
        sm.new.assert_called_once()
        assert session is new_sess
        assert resumed is False

    def test_resume_id_loads_specific_session(self):
        loaded = _make_session("target-id")
        sm = self._sm(loaded_session=loaded)
        session, resumed = _pick_session(
            sm,
            self._display(),
            MagicMock(),
            resume_id="target-id",
            resume_list=False,
            continue_=False,
        )
        sm.load.assert_called_once_with("target-id")
        assert session is loaded
        assert resumed is True

    def test_resume_id_propagates_session_error(self):
        sm = self._sm()
        sm.load.side_effect = SessionError("not found")
        with pytest.raises(SessionError, match="not found"):
            _pick_session(
                sm,
                self._display(),
                MagicMock(),
                resume_id="bad-id",
                resume_list=False,
                continue_=False,
            )

    def test_resume_list_user_picks_session(self):
        meta = _make_meta("picked-id")
        loaded = _make_session("picked-id")
        sm = self._sm(loaded_session=loaded, sessions=[meta])
        display = self._display(choice=meta)
        workspace_root = MagicMock()

        session, resumed = _pick_session(
            sm,
            display,
            workspace_root,
            resume_id=None,
            resume_list=True,
            continue_=False,
        )
        sm.list.assert_called_once_with(workspace_root)
        display.show_session_list.assert_called_once_with([meta])
        sm.load.assert_called_once_with("picked-id")
        assert session is loaded
        assert resumed is True

    def test_resume_list_user_declines_creates_new(self):
        new_sess = _make_session("new-id")
        sm = self._sm(new_session=new_sess, sessions=[_make_meta()])
        display = self._display(choice=None)

        session, resumed = _pick_session(
            sm,
            display,
            MagicMock(),
            resume_id=None,
            resume_list=True,
            continue_=False,
        )
        sm.new.assert_called_once()
        assert session is new_sess
        assert resumed is False

    def test_resume_list_empty_sessions_creates_new(self):
        new_sess = _make_session("new-id")
        sm = self._sm(new_session=new_sess, sessions=[])
        display = self._display(choice=None)

        session, resumed = _pick_session(
            sm,
            display,
            MagicMock(),
            resume_id=None,
            resume_list=True,
            continue_=False,
        )
        sm.new.assert_called_once()
        assert session is new_sess
        assert resumed is False

    def test_continue_with_existing_session(self):
        recent = _make_session("recent-id")
        sm = self._sm(recent_session=recent)
        workspace_root = MagicMock()

        session, resumed = _pick_session(
            sm,
            self._display(),
            workspace_root,
            resume_id=None,
            resume_list=False,
            continue_=True,
        )
        sm.most_recent.assert_called_once_with(workspace_root)
        assert session is recent
        assert resumed is True

    def test_continue_no_sessions_creates_new(self):
        new_sess = _make_session("new-id")
        sm = self._sm(new_session=new_sess, recent_session=None)
        workspace_root = MagicMock()

        session, resumed = _pick_session(
            sm,
            self._display(),
            workspace_root,
            resume_id=None,
            resume_list=False,
            continue_=True,
        )
        sm.most_recent.assert_called_once_with(workspace_root)
        sm.new.assert_called_once()
        assert session is new_sess
        assert resumed is False


# ---------------------------------------------------------------------------
# _show_resume_context
# ---------------------------------------------------------------------------


class TestShowResumeContext:
    def _session(self, messages=None, error=False):
        s = MagicMock()
        s.session_id = "proj__2024-01-01T00h00m00.001s"
        if error:
            from ai_cli.core.session_manager import SessionError as _SE

            s.get_messages.side_effect = _SE("boom")
        else:
            s.get_messages.return_value = messages or []
        return s

    def _display(self):
        return MagicMock()

    def test_shows_session_id(self):
        ui = self._display()
        _show_resume_context(self._session(), ui)
        ui.show_status.assert_called()
        assert any(
            "proj__2024-01-01T00h00m00.001s" in str(c)
            for c in ui.show_status.call_args_list
        )

    def test_empty_history_no_turn_display(self):
        ui = self._display()
        _show_resume_context(self._session(messages=[]), ui)
        ui.begin_assistant_turn.assert_not_called()

    def test_get_messages_error_is_silenced(self):
        ui = self._display()
        _show_resume_context(self._session(error=True), ui)
        ui.begin_assistant_turn.assert_not_called()

    def test_last_assistant_message_replayed(self):
        msgs = [{"role": "assistant", "content": "Here is the answer."}]
        ui = self._display()
        _show_resume_context(self._session(messages=msgs), ui)
        ui.begin_assistant_turn.assert_called_once()
        ui.stream_text.assert_called_once_with("Here is the answer.")
        ui.end_assistant_turn.assert_called_once()

    def test_last_user_message_shows_status(self):
        msgs = [{"role": "user", "content": "What is 2+2?"}]
        ui = self._display()
        _show_resume_context(self._session(messages=msgs), ui)
        ui.begin_assistant_turn.assert_not_called()
        # A status message mentioning the unanswered state should be shown.
        combined = " ".join(str(c) for c in ui.show_status.call_args_list)
        assert "not yet answered" in combined or "What is 2+2?" in combined

    def test_non_string_assistant_content_not_replayed(self):
        msgs = [{"role": "assistant", "content": None}]
        ui = self._display()
        _show_resume_context(self._session(messages=msgs), ui)
        ui.begin_assistant_turn.assert_not_called()

    def test_tool_message_shows_only_session_id(self):
        msgs = [{"role": "tool", "content": "result", "tool_call_id": "x"}]
        ui = self._display()
        _show_resume_context(self._session(messages=msgs), ui)
        ui.begin_assistant_turn.assert_not_called()


# ---------------------------------------------------------------------------
# main() — flag routing and mutual-exclusion
# ---------------------------------------------------------------------------


class TestMainRouting:
    """Verify that main() routes flags to the correct _cmd_repl kwargs."""

    def _run(self, argv, monkeypatch):
        captured = {}

        def fake_repl(*args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr("ai_cli.__main__._cmd_repl", fake_repl)
        run_main(argv)
        return captured

    def test_no_flags_calls_default(self, monkeypatch):
        kwargs = self._run([], monkeypatch)
        assert not kwargs.get("resume_id")
        assert not kwargs.get("resume_list")
        assert not kwargs.get("continue_")

    def test_resume_no_arg_sets_resume_list(self, monkeypatch):
        kwargs = self._run(["--resume"], monkeypatch)
        assert kwargs.get("resume_list") is True

    def test_resume_with_id_sets_resume_id(self, monkeypatch):
        kwargs = self._run(["--resume", "20260319T120000-abcd1234"], monkeypatch)
        assert kwargs.get("resume_id") == "20260319T120000-abcd1234"

    def test_continue_sets_continue(self, monkeypatch):
        kwargs = self._run(["--continue"], monkeypatch)
        assert kwargs.get("continue_") is True

    def test_resume_and_continue_together_exits_nonzero(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc_info:
            run_main(["--resume", "--continue"])
        assert exc_info.value.code == 1
        assert "--resume" in capsys.readouterr().err

    def test_display_plain_passed_to_cmd_repl(self, monkeypatch):
        kwargs = self._run(["--display", "plain"], monkeypatch)
        assert kwargs.get("display") == "plain"

    def test_display_rich_passed_to_cmd_repl(self, monkeypatch):
        kwargs = self._run(["--display", "rich"], monkeypatch)
        assert kwargs.get("display") == "rich"

    def test_no_display_flag_passes_none(self, monkeypatch):
        kwargs = self._run([], monkeypatch)
        assert kwargs.get("display") is None

    def test_max_tool_rounds_passed_to_cmd_repl(self, monkeypatch):
        kwargs = self._run(["--max-tool-rounds", "20"], monkeypatch)
        assert kwargs.get("max_tool_rounds") == 20

    def test_no_max_tool_rounds_flag_passes_none(self, monkeypatch):
        kwargs = self._run([], monkeypatch)
        assert kwargs.get("max_tool_rounds") is None
