"""Tests for ai_cli/tools/read_file.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_cli.tools.read_file import ReadFileTool

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


def make_tool(
    tmp_path: Path,
    *,
    permission_required: bool = False,
) -> ReadFileTool:
    workspace = MagicMock()
    workspace.root = tmp_path
    workspace.resolve.side_effect = lambda p: (tmp_path / p.lstrip("./")).resolve()
    pm = MagicMock()
    pm.request.return_value = (True, "")
    return ReadFileTool(
        workspace=workspace,
        permission_manager=pm,
        permission_required=permission_required,
        name="read_file",
        description="Read a file from the workspace.",
    )


def make_real_tool(
    tmp_path: Path, *, permission_required: bool = False
) -> ReadFileTool:
    """Tool backed by a real Workspace for file-I/O tests."""
    from ai_cli.core.workspace import Workspace

    ws = Workspace(tmp_path, config_manager=MagicMock())
    pm = MagicMock()
    pm.request.return_value = (True, "")
    return ReadFileTool(
        workspace=ws,
        permission_manager=pm,
        permission_required=permission_required,
        name="read_file",
        description="Read a file from the workspace.",
    )


# ---------------------------------------------------------------------------
# Class attributes
# ---------------------------------------------------------------------------


class TestClassAttributes:
    def test_name(self):
        assert ReadFileTool.NAME == "read_file"

    def test_permission_required_false_by_default(self):
        assert ReadFileTool.PERMISSION_REQUIRED is False

    def test_disabled_by_default(self):
        assert ReadFileTool.DISABLED_BY_DEFAULT is True


# ---------------------------------------------------------------------------
# definition()
# ---------------------------------------------------------------------------


class TestDefinition:
    def test_schema_shape(self, tmp_path):
        tool = make_tool(tmp_path)
        d = tool.definition().schema()
        assert d["type"] == "function"
        fn = d["function"]
        assert fn["name"] == "read_file"
        params = fn["parameters"]
        assert "path" in params["properties"]
        assert params["required"] == ["path"]
        assert "start_line" in params["properties"]
        assert "end_line" in params["properties"]


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


class TestExecute:
    def test_reads_full_file(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="hello.txt")
        assert result["status"] == "success"
        d = result["data"]
        assert d["content"] == "line1\nline2\nline3\n"
        assert d["total_lines"] == 3
        assert d["lines_returned"] == 3
        assert d["start_line"] == 1
        assert d["end_line"] == 3
        assert d["path"] == "hello.txt"

    def test_reads_line_range(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        f = tmp_path / "hello.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="hello.txt", start_line=2, end_line=4)
        d = result["data"]
        assert d["content"] == "b\nc\nd\n"
        assert d["start_line"] == 2
        assert d["end_line"] == 4
        assert d["lines_returned"] == 3
        assert d["total_lines"] == 5

    def test_reads_from_start_line_only(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "f.txt").write_text("a\nb\nc\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", start_line=2)
        d = result["data"]
        assert d["content"] == "b\nc\n"
        assert d["start_line"] == 2
        assert d["end_line"] == 3

    def test_reads_to_end_line_only(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "f.txt").write_text("a\nb\nc\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", end_line=2)
        d = result["data"]
        assert d["content"] == "a\nb\n"
        assert d["start_line"] == 1
        assert d["end_line"] == 2

    def test_file_not_found(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="missing.txt")
        assert result["status"] == "error"
        assert result["error"] == "read_error"

    def test_start_line_too_large(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "f.txt").write_text("a\nb\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", start_line=99)
        assert result["status"] == "error"
        assert result["error"] == "invalid_range"

    def test_end_line_too_large(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "f.txt").write_text("a\nb\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", end_line=99)
        assert result["status"] == "error"
        assert result["error"] == "invalid_range"

    def test_start_line_less_than_one(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "f.txt").write_text("a\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", start_line=0)
        assert result["status"] == "error"
        assert result["error"] == "invalid_range"

    def test_start_line_after_end_line(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "f.txt").write_text("a\nb\nc\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="f.txt", start_line=3, end_line=1)
        assert result["status"] == "error"
        assert result["error"] == "invalid_range"

    def test_empty_file_returns_zero_sentinels(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "empty.txt").write_text("")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="empty.txt")
        assert result["status"] == "success"
        d = result["data"]
        assert d["content"] == ""
        assert d["total_lines"] == 0
        assert d["lines_returned"] == 0
        assert d["start_line"] == 0
        assert d["end_line"] == 0

    def test_path_escape_returns_error(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="../outside.txt")
        assert result["status"] == "error"
        assert result["error"] == "read_error"

    def test_ignored_file_returns_error(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / ".ai-cli" / ".ignore").write_text("secret.txt\n")
        (tmp_path / "secret.txt").write_text("shh\n")
        tool = make_real_tool(tmp_path)
        result = tool.execute(path="secret.txt")
        assert result["status"] == "error"
        assert result["error"] == "read_error"


# ---------------------------------------------------------------------------
# extra_permission_options()
# ---------------------------------------------------------------------------


class TestExtraPermissionOptions:
    def test_generates_file_and_dir_options(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "foo.py").write_text("")
        tool = make_real_tool(tmp_path)
        opts = tool.extra_permission_options(path="./src/foo.py")
        assert opts[0] == "file:./src/foo.py"
        assert "dir:./src/" in opts
        assert "dir:./" in opts

    def test_nested_path_includes_all_ancestors(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "a" / "b" / "c").mkdir(parents=True)
        (tmp_path / "a" / "b" / "c" / "f.txt").write_text("")
        tool = make_real_tool(tmp_path)
        opts = tool.extra_permission_options(path="./a/b/c/f.txt")
        assert opts[0] == "file:./a/b/c/f.txt"
        assert "dir:./a/b/c/" in opts
        assert "dir:./a/b/" in opts
        assert "dir:./a/" in opts
        assert "dir:./" in opts
        # No duplicates
        assert len(opts) == len(set(opts))

    def test_file_at_root_has_only_root_dir_option(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "readme.txt").write_text("")
        tool = make_real_tool(tmp_path)
        opts = tool.extra_permission_options(path="./readme.txt")
        assert opts == ["file:./readme.txt", "dir:./"]

    def test_empty_path_returns_empty(self, tmp_path):
        tool = make_tool(tmp_path)
        assert tool.extra_permission_options() == []

    def test_invalid_path_returns_empty(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        tool = make_real_tool(tmp_path)
        assert tool.extra_permission_options(path="../escape") == []

    def test_path_resolving_to_root_returns_empty(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        tool = make_real_tool(tmp_path)
        assert tool.extra_permission_options(path=".") == []
        assert tool.extra_permission_options(path="./") == []


# ---------------------------------------------------------------------------
# on_permission_granted() and session allow-list
# ---------------------------------------------------------------------------


class TestSessionAllowList:
    def test_file_grant_allows_exact_file(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "f.txt").write_text("hi\n")
        tool = make_real_tool(tmp_path, permission_required=True)
        tool.on_permission_granted("file:./f.txt", path="./f.txt")
        allowed, _ = tool.request_permission("read f.txt", path="./f.txt")
        assert allowed is True

    def test_file_grant_does_not_allow_different_file(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.txt").write_text("")
        tool = make_real_tool(tmp_path, permission_required=True)
        tool._permission_manager.request.return_value = (False, "denied")
        tool.on_permission_granted("file:./a.txt", path="./a.txt")
        allowed, _ = tool.request_permission("read b.txt", path="./b.txt")
        assert allowed is False

    def test_dir_grant_allows_file_in_dir(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("")
        tool = make_real_tool(tmp_path, permission_required=True)
        tool.on_permission_granted("dir:./src/", path="./src/main.py")
        allowed, _ = tool.request_permission("read main.py", path="./src/main.py")
        assert allowed is True

    def test_dir_grant_allows_nested_file(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "src" / "sub").mkdir(parents=True)
        (tmp_path / "src" / "sub" / "f.py").write_text("")
        tool = make_real_tool(tmp_path, permission_required=True)
        tool.on_permission_granted("dir:./src/", path="./src/sub/f.py")
        allowed, _ = tool.request_permission("read f.py", path="./src/sub/f.py")
        assert allowed is True

    def test_dir_grant_does_not_allow_sibling_dir(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "t.py").write_text("")
        tool = make_real_tool(tmp_path, permission_required=True)
        tool._permission_manager.request.return_value = (False, "denied")
        tool.on_permission_granted("dir:./src/", path="./src/x.py")
        allowed, _ = tool.request_permission("read t.py", path="./tests/t.py")
        assert allowed is False

    def test_reset_session_state_clears_allow_lists(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "f.txt").write_text("")
        tool = make_real_tool(tmp_path, permission_required=True)
        tool.on_permission_granted("file:./f.txt", path="./f.txt")
        tool.reset_session_state()
        tool._permission_manager.request.return_value = (False, "denied")
        allowed, _ = tool.request_permission("read f.txt", path="./f.txt")
        assert allowed is False

    def test_unknown_choice_kind_is_ignored(self, tmp_path):
        (tmp_path / ".ai-cli").mkdir()
        tool = make_real_tool(tmp_path, permission_required=True)
        tool.on_permission_granted("unknown:./f.txt")
        assert not tool._session_allowed_files
        assert not tool._session_allowed_dirs

    def test_permission_not_required_always_allowed(self, tmp_path):
        tool = make_tool(tmp_path, permission_required=False)
        allowed, _ = tool.request_permission("read anything", path="./f.txt")
        assert allowed is True


# ---------------------------------------------------------------------------
# reset_session_state() via registry
# ---------------------------------------------------------------------------


class TestResetViaRegistry:
    def test_registry_reset_clears_tool_session_state(self, tmp_path):
        from unittest.mock import MagicMock

        from ai_cli.core.tool_registry import ToolRegistry
        from ai_cli.core.workspace import Workspace

        (tmp_path / ".ai-cli").mkdir()
        (tmp_path / "f.txt").write_text("hi\n")

        ws = Workspace(tmp_path, config_manager=MagicMock())
        config = MagicMock()
        config.get.return_value = {}
        config.get_project.return_value = {}
        pm = MagicMock()
        pm.request.return_value = (True, "")

        reg = ToolRegistry(ws, config, pm)
        reg.register(ReadFileTool, tier="bundled")
        reg._apply_config()

        tool = reg.get("read_file")
        tool.on_permission_granted("file:./f.txt", path="./f.txt")
        assert tool._session_allowed_files

        reg.reset_session_overrides()
        assert not tool._session_allowed_files
