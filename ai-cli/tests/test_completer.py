"""Tests for ai_cli.cli.completer.REPLCompleter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from prompt_toolkit.document import Document

from ai_cli.cli.completer import _MAX_PATH_COMPLETIONS, REPLCompleter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CMDS = [
    "clear",
    "compact",
    "exit",
    "help",
    "history",
    "markdown",
    "rounds",
    "session",
    "tools",
    "verbose",
]


def _completer(
    cmds: list[str] | None = None,
    tool_names: list[str] | None = None,
    workspace=None,
) -> REPLCompleter:
    registry = None
    if tool_names is not None:
        registry = MagicMock()
        registry.all_tools_info.return_value = [{"name": n} for n in tool_names]
    return REPLCompleter(
        slash_commands=cmds if cmds is not None else CMDS,
        tool_registry=registry,
        workspace=workspace,
    )


def _completions(completer: REPLCompleter, text: str) -> list[str]:
    """Return the completion *text* strings for *text* as typed so far."""
    doc = Document(text, cursor_position=len(text))
    event = MagicMock()
    return [c.text for c in completer.get_completions(doc, event)]


def _make_workspace(root: Path, is_ignored: bool = False) -> MagicMock:
    ws = MagicMock()
    ws.root = root
    ws.is_ignored.return_value = is_ignored
    return ws


# ---------------------------------------------------------------------------
# Slash command — top-level names
# ---------------------------------------------------------------------------


class TestSlashTopLevel:
    def test_bare_slash_returns_all_commands(self):
        result = _completions(_completer(), "/")
        assert set(result) == {f"/{c}" for c in CMDS}

    def test_prefix_filters_commands(self):
        result = _completions(_completer(), "/h")
        assert set(result) == {"/help", "/history"}

    def test_exact_match_returned(self):
        result = _completions(_completer(), "/help")
        assert "/help" in result

    def test_no_match_returns_empty(self):
        result = _completions(_completer(), "/zzz")
        assert result == []

    def test_normal_text_returns_nothing(self):
        result = _completions(_completer(), "hello world")
        assert result == []

    def test_empty_input_returns_nothing(self):
        result = _completions(_completer(), "")
        assert result == []

    def test_trailing_space_after_unknown_cmd_returns_nothing(self):
        result = _completions(_completer(), "/zzz ")
        assert result == []

    def test_case_insensitive_prefix(self):
        result = _completions(_completer(), "/H")
        assert set(result) == {"/help", "/history"}

    def test_case_insensitive_mixed(self):
        result = _completions(_completer(), "/Ex")
        assert "/exit" in result


# ---------------------------------------------------------------------------
# /tools subcommands
# ---------------------------------------------------------------------------


class TestToolsSubcommands:
    def test_tools_space_offers_subcommands(self):
        result = _completions(_completer(), "/tools ")
        assert set(result) == {"list", "info", "enable", "disable", "allow", "disallow"}

    def test_tools_partial_subcommand(self):
        result = _completions(_completer(), "/tools e")
        assert set(result) == {"enable"}

    def test_tools_partial_subcommand_uppercase(self):
        result = _completions(_completer(), "/tools E")
        assert set(result) == {"enable"}

    def test_tools_partial_d(self):
        result = _completions(_completer(), "/tools d")
        assert set(result) == {"disable", "disallow"}

    def test_tools_list_no_further_completions(self):
        result = _completions(_completer(tool_names=["read_file"]), "/tools list ")
        assert result == []

    def test_tools_info_offers_tool_names(self):
        c = _completer(tool_names=["read_file", "write_file"])
        result = _completions(c, "/tools info ")
        assert set(result) == {"read_file", "write_file"}

    def test_tools_info_prefix_filters_names(self):
        c = _completer(tool_names=["read_file", "write_file"])
        result = _completions(c, "/tools info r")
        assert result == ["read_file"]

    def test_tools_enable_offers_session_flag_and_names(self):
        c = _completer(tool_names=["bash"])
        result = _completions(c, "/tools enable ")
        assert "--session" in result
        assert "bash" in result

    def test_tools_disable_offers_session_flag_and_names(self):
        c = _completer(tool_names=["bash"])
        result = _completions(c, "/tools disable ")
        assert "--session" in result
        assert "bash" in result

    def test_tools_allow_offers_session_flag_and_names(self):
        c = _completer(tool_names=["bash"])
        result = _completions(c, "/tools allow ")
        assert "--session" in result
        assert "bash" in result

    def test_tools_disallow_offers_session_flag_and_names(self):
        c = _completer(tool_names=["bash"])
        result = _completions(c, "/tools disallow ")
        assert "--session" in result
        assert "bash" in result

    def test_tools_enable_after_session_flag_offers_names_only(self):
        c = _completer(tool_names=["bash", "read_file"])
        result = _completions(c, "/tools enable --session ")
        assert set(result) == {"bash", "read_file"}
        assert "--session" not in result

    def test_tools_enable_no_registry_no_crash(self):
        c = _completer(tool_names=None)
        result = _completions(c, "/tools enable ")
        assert "--session" in result


# ---------------------------------------------------------------------------
# /session subcommands
# ---------------------------------------------------------------------------


class TestSessionSubcommands:
    def test_session_space_offers_name(self):
        result = _completions(_completer(), "/session ")
        assert result == ["name"]

    def test_session_n_prefix(self):
        result = _completions(_completer(), "/session n")
        assert result == ["name"]

    def test_session_uppercase_prefix(self):
        result = _completions(_completer(), "/session N")
        assert result == ["name"]

    def test_session_x_no_match(self):
        result = _completions(_completer(), "/session x")
        assert result == []

    def test_session_name_no_further_completions(self):
        result = _completions(_completer(), "/session name ")
        assert result == []


# ---------------------------------------------------------------------------
# /rounds
# ---------------------------------------------------------------------------


class TestRoundsCompletion:
    def test_rounds_space_offers_session_flag(self):
        result = _completions(_completer(), "/rounds ")
        assert "--session" in result

    def test_rounds_partial_dashes(self):
        result = _completions(_completer(), "/rounds --")
        assert "--session" in result

    def test_rounds_no_numeric_completion(self):
        result = _completions(_completer(), "/rounds 1")
        assert result == []

    def test_rounds_after_session_flag_no_completions(self):
        result = _completions(_completer(), "/rounds --session ")
        assert result == []


# ---------------------------------------------------------------------------
# @ file-path completion
# ---------------------------------------------------------------------------


class TestAtPathCompletion:
    def test_bare_at_lists_workspace_root(self, tmp_path):
        (tmp_path / "foo.py").touch()
        (tmp_path / "bar.py").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@")
        assert "foo.py" in result
        assert "bar.py" in result

    def test_at_prefix_filters_files(self, tmp_path):
        (tmp_path / "alpha.txt").touch()
        (tmp_path / "beta.txt").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@a")
        assert result == ["alpha.txt"]

    def test_at_directory_gets_trailing_slash(self, tmp_path):
        (tmp_path / "src").mkdir()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@")
        assert "src/" in result

    def test_at_symlinked_directory_gets_trailing_slash(self, tmp_path):
        real = tmp_path / "real_dir"
        real.mkdir()
        link = tmp_path / "link_dir"
        link.symlink_to(real)
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@")
        assert "link_dir/" in result

    def test_at_path_in_subdir(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@src/")
        assert "src/main.py" in result

    def test_ignored_files_excluded(self, tmp_path):
        (tmp_path / "secret.env").touch()
        (tmp_path / "ok.py").touch()
        ws = _make_workspace(tmp_path)
        ws.is_ignored.side_effect = lambda p, **_: p.name == "secret.env"
        c = _completer(workspace=ws)
        result = _completions(c, "@")
        assert "secret.env" not in result
        assert "ok.py" in result

    def test_no_workspace_returns_no_at_completions(self):
        c = _completer(workspace=None)
        result = _completions(c, "@foo")
        assert result == []

    def test_at_bang_prefix_also_completes(self, tmp_path):
        (tmp_path / "diagram.png").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@!")
        assert "diagram.png" in result

    def test_bypass_flag_skips_ignore_filter(self, tmp_path):
        """@! completions include files that would otherwise be ignored."""
        (tmp_path / "ignored.py").touch()
        ws = _make_workspace(tmp_path)
        ws.is_ignored.side_effect = lambda p, **_: p.name == "ignored.py"
        c = _completer(workspace=ws)
        assert "ignored.py" not in _completions(c, "@i")
        assert "ignored.py" in _completions(c, "@!i")

    def test_start_position_replaces_partial(self, tmp_path):
        (tmp_path / "alpha.py").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        doc = Document("@al", cursor_position=3)
        event = MagicMock()
        completions = list(c.get_completions(doc, event))
        assert len(completions) == 1
        assert completions[0].text == "alpha.py"
        assert completions[0].start_position == -len("al")

    def test_completions_capped_at_max(self, tmp_path):
        for i in range(_MAX_PATH_COMPLETIONS + 10):
            (tmp_path / f"file{i:04d}.txt").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@")
        assert len(result) == _MAX_PATH_COMPLETIONS

    def test_nonexistent_subdir_returns_nothing(self, tmp_path):
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@nonexistent/")
        assert result == []

    def test_at_in_middle_of_text_completes(self, tmp_path):
        (tmp_path / "main.py").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "look at @ma")
        assert "main.py" in result

    def test_dotdot_path_completes_parent_dir(self, tmp_path):
        """@../ completes files in the parent directory."""
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        (tmp_path / "sibling.txt").touch()
        c = _completer(workspace=_make_workspace(workspace_root))
        result = _completions(c, "@../")
        assert "../sibling.txt" in result

    def test_dotdot_outside_workspace_skips_ignore_check(self, tmp_path):
        """Files outside the workspace root are never filtered by ignore rules."""
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        (tmp_path / "outside.py").touch()
        ws = _make_workspace(workspace_root)
        ws.is_ignored.return_value = True  # would filter if checked
        c = _completer(workspace=ws)
        result = _completions(c, "@../")
        assert "../outside.py" in result
        ws.is_ignored.assert_not_called()

    def test_absolute_path_completes(self, tmp_path):
        """@/absolute/path completes from the filesystem root."""
        (tmp_path / "file.txt").touch()
        # Use a workspace rooted elsewhere so tmp_path is 'outside'.
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        c = _completer(workspace=_make_workspace(workspace_root))
        result = _completions(c, f"@{tmp_path}/")
        assert f"{tmp_path}/file.txt" in result

    def test_resolve_os_error_returns_nothing(self, tmp_path):
        """A dir_part that raises OSError on resolve() returns no completions."""
        from unittest.mock import patch

        c = _completer(workspace=_make_workspace(tmp_path))
        with patch("pathlib.Path.resolve", side_effect=OSError("symlink loop")):
            result = _completions(c, "@bad/")
        assert result == []

    def test_is_dir_os_error_skips_entry(self, tmp_path):
        """An entry whose is_dir() raises OSError is skipped; others still complete."""
        from unittest.mock import patch

        (tmp_path / "ok.py").touch()
        (tmp_path / "broken").touch()
        c = _completer(workspace=_make_workspace(tmp_path))

        import os as _os

        entry_type = type(next(iter(_os.scandir(tmp_path))))
        original_is_dir = entry_type.is_dir

        def _patched_is_dir(self, follow_symlinks=True):
            if self.name == "broken":
                raise OSError("stat failed")
            return original_is_dir(self, follow_symlinks=follow_symlinks)

        with patch.object(entry_type, "is_dir", _patched_is_dir):
            result = _completions(c, "@")
        assert "ok.py" in result
        assert "broken" not in result
