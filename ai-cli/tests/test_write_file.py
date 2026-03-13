"""Tests for ai_cli/tools/write_file.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_cli.tools.write_file import WriteFileTool

# ---------------------------------------------------------------------------
# Isolate _GLOBAL_DIR so real ~/.ai-cli/.ignore never influences results.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_global_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_global = tmp_path_factory.mktemp("fake_global_ai_cli")
    monkeypatch.setattr("ai_cli.core.workspace.get_global_dir", lambda: fake_global)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_real_tool(
    tmp_path: Path, *, permission_required: bool = True
) -> WriteFileTool:
    """Tool backed by a real Workspace for file-I/O tests."""
    from ai_cli.core.workspace import Workspace

    ws = Workspace(tmp_path, config_manager=MagicMock())
    pm = MagicMock()
    pm.request.return_value = (True, "")
    return WriteFileTool(
        workspace=ws,
        permission_manager=pm,
        permission_required=permission_required,
        name="write_file",
        description="Write or partially replace a file in the workspace.",
    )


def _init(tmp_path: Path) -> None:
    (tmp_path / ".ai-cli").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Class attributes
# ---------------------------------------------------------------------------


class TestClassAttributes:
    def test_name(self):
        assert WriteFileTool.NAME == "write_file"

    def test_permission_required_true_by_default(self):
        assert WriteFileTool.PERMISSION_REQUIRED is True

    def test_not_disabled_by_default(self):
        assert not getattr(WriteFileTool, "DISABLED_BY_DEFAULT", False)


# ---------------------------------------------------------------------------
# definition()
# ---------------------------------------------------------------------------


class TestDefinition:
    def test_schema_shape(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path)
        d = tool.definition()
        assert d["type"] == "function"
        fn = d["function"]
        assert fn["name"] == "write_file"
        params = fn["parameters"]
        props = params["properties"]
        assert "path" in props
        assert "content" in props
        assert "start_line" in props
        assert "end_line" in props
        assert params["required"] == ["path", "content"]


# ---------------------------------------------------------------------------
# execute() — full writes
# ---------------------------------------------------------------------------


class TestExecuteFullWrite:
    def test_creates_new_file(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="new.txt", content="hello\nworld\n")
        assert result["status"] == "success"
        assert (tmp_path / "new.txt").read_text() == "hello\nworld\n"

    def test_returns_path_summary_lines_written(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", content="a\nb\nc\n")
        d = result["data"]
        assert d["path"] == "f.txt"
        assert d["lines_written"] == 3
        assert "f.txt" in d["summary"]

    def test_overwrites_existing_file(self, tmp_path):
        _init(tmp_path)
        (tmp_path / "f.txt").write_text("old content\n")
        tool = make_real_tool(tmp_path)
        tool.execute(path="f.txt", content="new content\n")
        assert (tmp_path / "f.txt").read_text() == "new content\n"

    def test_creates_parent_directories(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="./a/b/c.txt", content="deep\n")
        assert result["status"] == "success"
        assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep\n"

    def test_empty_content(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="empty.txt", content="")
        assert result["status"] == "success"
        assert result["data"]["lines_written"] == 0

    def test_path_escape_returns_error(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="../outside.txt", content="x")
        assert result["status"] == "error"
        assert result["error"] == "write_error"

    def test_ignored_path_returns_error(self, tmp_path):
        _init(tmp_path)
        (tmp_path / ".ai-cli" / ".ignore").write_text("secret.txt\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="secret.txt", content="x")
        assert result["status"] == "error"
        assert result["error"] == "write_error"


# ---------------------------------------------------------------------------
# execute() — partial writes
# ---------------------------------------------------------------------------


class TestExecutePartialWrite:
    def test_replaces_line_range(self, tmp_path):
        _init(tmp_path)
        (tmp_path / "f.txt").write_text("a\nb\nc\nd\ne\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", content="X\nY\n", start_line=2, end_line=3)
        assert result["status"] == "success"
        assert (tmp_path / "f.txt").read_text() == "a\nX\nY\nd\ne\n"

    def test_partial_write_summary_mentions_lines(self, tmp_path):
        _init(tmp_path)
        (tmp_path / "f.txt").write_text("a\nb\nc\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", content="Z\n", start_line=2, end_line=2)
        assert "2" in result["data"]["summary"]

    def test_partial_write_requires_existing_file(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path)
        result = tool.execute(
            path="missing.txt", content="x\n", start_line=1, end_line=1
        )
        assert result["status"] == "error"
        assert result["error"] == "write_error"

    def test_only_start_line_returns_error(self, tmp_path):
        _init(tmp_path)
        (tmp_path / "f.txt").write_text("a\nb\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", content="x\n", start_line=1)
        assert result["status"] == "error"
        assert result["error"] == "invalid_range"

    def test_only_end_line_returns_error(self, tmp_path):
        _init(tmp_path)
        (tmp_path / "f.txt").write_text("a\nb\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", content="x\n", end_line=1)
        assert result["status"] == "error"
        assert result["error"] == "invalid_range"

    def test_start_line_less_than_one_returns_invalid_range(self, tmp_path):
        _init(tmp_path)
        (tmp_path / "f.txt").write_text("a\nb\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", content="x\n", start_line=0, end_line=1)
        assert result["status"] == "error"
        assert result["error"] == "invalid_range"

    def test_end_line_less_than_one_returns_invalid_range(self, tmp_path):
        _init(tmp_path)
        (tmp_path / "f.txt").write_text("a\nb\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", content="x\n", start_line=1, end_line=0)
        assert result["status"] == "error"
        assert result["error"] == "invalid_range"

    def test_start_line_after_end_line_returns_invalid_range(self, tmp_path):
        _init(tmp_path)
        (tmp_path / "f.txt").write_text("a\nb\nc\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", content="x\n", start_line=3, end_line=1)
        assert result["status"] == "error"
        assert result["error"] == "invalid_range"

    def test_out_of_range_lines_return_write_error(self, tmp_path):
        _init(tmp_path)
        (tmp_path / "f.txt").write_text("a\nb\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", content="x\n", start_line=1, end_line=99)
        assert result["status"] == "error"
        assert result["error"] == "write_error"


# ---------------------------------------------------------------------------
# extra_permission_options()
# ---------------------------------------------------------------------------


class TestExtraPermissionOptions:
    def test_generates_file_and_dir_options(self, tmp_path):
        _init(tmp_path)
        (tmp_path / "src").mkdir()
        tool = make_real_tool(tmp_path)
        opts = tool.extra_permission_options(path="./src/foo.py")
        assert opts[0] == "file:./src/foo.py"
        assert "dir:./src/" in opts
        assert "dir:./" in opts

    def test_nested_path_includes_all_ancestors(self, tmp_path):
        _init(tmp_path)
        (tmp_path / "a" / "b" / "c").mkdir(parents=True)
        tool = make_real_tool(tmp_path)
        opts = tool.extra_permission_options(path="./a/b/c/f.txt")
        assert opts[0] == "file:./a/b/c/f.txt"
        assert "dir:./a/b/c/" in opts
        assert "dir:./a/b/" in opts
        assert "dir:./a/" in opts
        assert "dir:./" in opts
        assert len(opts) == len(set(opts))

    def test_file_at_root_level(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path)
        opts = tool.extra_permission_options(path="./readme.txt")
        assert opts == ["file:./readme.txt", "dir:./"]

    def test_empty_path_returns_empty(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path)
        assert tool.extra_permission_options() == []

    def test_escaping_path_returns_empty(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path)
        assert tool.extra_permission_options(path="../escape") == []

    def test_path_resolving_to_root_returns_empty(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path)
        assert tool.extra_permission_options(path=".") == []
        assert tool.extra_permission_options(path="./") == []


# ---------------------------------------------------------------------------
# Session allow-list
# ---------------------------------------------------------------------------


class TestSessionAllowList:
    def test_file_grant_allows_exact_file(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path, permission_required=True)
        tool.on_permission_granted("file:./f.txt", path="./f.txt")
        allowed, _ = tool.request_permission("write f.txt", path="./f.txt")
        assert allowed is True

    def test_file_grant_does_not_allow_different_file(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path, permission_required=True)
        tool._permission_manager.request.return_value = (False, "denied")
        tool.on_permission_granted("file:./a.txt", path="./a.txt")
        allowed, _ = tool.request_permission("write b.txt", path="./b.txt")
        assert allowed is False

    def test_dir_grant_allows_file_in_dir(self, tmp_path):
        _init(tmp_path)
        (tmp_path / "src").mkdir()
        tool = make_real_tool(tmp_path, permission_required=True)
        tool.on_permission_granted("dir:./src/", path="./src/main.py")
        allowed, _ = tool.request_permission("write main.py", path="./src/main.py")
        assert allowed is True

    def test_dir_grant_allows_nested_file(self, tmp_path):
        _init(tmp_path)
        (tmp_path / "src" / "sub").mkdir(parents=True)
        tool = make_real_tool(tmp_path, permission_required=True)
        tool.on_permission_granted("dir:./src/", path="./src/sub/f.py")
        allowed, _ = tool.request_permission("write f.py", path="./src/sub/f.py")
        assert allowed is True

    def test_dir_grant_does_not_allow_sibling_dir(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path, permission_required=True)
        tool._permission_manager.request.return_value = (False, "denied")
        tool.on_permission_granted("dir:./src/", path="./src/x.py")
        allowed, _ = tool.request_permission("write t.py", path="./tests/t.py")
        assert allowed is False

    def test_reset_clears_allow_lists(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path, permission_required=True)
        tool.on_permission_granted("file:./f.txt", path="./f.txt")
        tool.reset_session_state()
        tool._permission_manager.request.return_value = (False, "denied")
        allowed, _ = tool.request_permission("write f.txt", path="./f.txt")
        assert allowed is False

    def test_permission_not_required_always_allowed(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path, permission_required=False)
        allowed, _ = tool.request_permission("write anything", path="./f.txt")
        assert allowed is True

    def test_unknown_choice_kind_is_ignored(self, tmp_path):
        _init(tmp_path)
        tool = make_real_tool(tmp_path)
        tool.on_permission_granted("unknown:./f.txt")
        assert not tool._session_allowed_files
        assert not tool._session_allowed_dirs
