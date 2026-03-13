"""Tests for ai_cli.__main__ entry point."""

import os
import sys
from unittest.mock import patch

import pytest

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
