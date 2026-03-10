"""Tests for ai_cli.__main__ entry point."""

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
    fake_global = tmp_path_factory.mktemp("fake_global")
    monkeypatch.setattr("ai_cli.core.workspace._GLOBAL_DIR", fake_global)


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
