"""Tests for ai_cli/tools/write.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_cli.tools.read import ReadTool
from ai_cli.tools.write import WriteTool

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
    with_read_tool: bool = False,
) -> tuple[WriteTool, ReadTool | None]:
    workspace = MagicMock()
    workspace.root = tmp_path
    workspace.contains.return_value = True
    workspace.is_ignored.return_value = False
    pm = MagicMock()
    pm.request.return_value = (True, "")
    tool = WriteTool(
        workspace=workspace,
        permission_manager=pm,
        permission_required=permission_required,
        name="write",
        description=WriteTool.DESCRIPTION,
    )
    read_tool = None
    if with_read_tool:
        read_tool = ReadTool(
            workspace=workspace,
            permission_manager=pm,
            permission_required=False,
            name="read",
            description=ReadTool.DESCRIPTION,
        )
        tool.set_read_tool(read_tool)
    return tool, read_tool


# ---------------------------------------------------------------------------
# Class attributes
# ---------------------------------------------------------------------------


class TestClassAttributes:
    def test_name(self):
        assert WriteTool.NAME == "write"

    def test_permission_required_true_by_default(self):
        assert WriteTool.PERMISSION_REQUIRED is True

    def test_disabled_by_default(self):
        assert WriteTool.DISABLED_BY_DEFAULT is True


# ---------------------------------------------------------------------------
# definition()
# ---------------------------------------------------------------------------


class TestDefinition:
    def test_schema_shape(self, tmp_path):
        tool, _ = make_tool(tmp_path)
        d = tool.definition().schema()
        assert d["type"] == "function"
        fn = d["function"]
        assert fn["name"] == "write"
        params = fn["parameters"]
        assert "file_path" in params["properties"]
        assert "content" in params["properties"]
        assert set(params["required"]) == {"file_path", "content"}


# ---------------------------------------------------------------------------
# execute() — path validation
# ---------------------------------------------------------------------------


class TestExecutePathValidation:
    def test_relative_path_returns_error(self, tmp_path):
        tool, _ = make_tool(tmp_path)
        result = tool.execute(file_path="relative/path.txt", content="hello")
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_directory_path_returns_error(self, tmp_path):
        tool, _ = make_tool(tmp_path)
        result = tool.execute(file_path=str(tmp_path), content="hello")
        assert result["status"] == "error"
        assert result["error"] == "write_error"


# ---------------------------------------------------------------------------
# execute() — new file creation
# ---------------------------------------------------------------------------


class TestNewFileCreation:
    def test_creates_new_file(self, tmp_path):
        f = tmp_path / "new.txt"
        tool, _ = make_tool(tmp_path)
        result = tool.execute(file_path=str(f), content="hello\n")
        assert result["status"] == "success"
        assert f.read_text() == "hello\n"

    def test_new_file_does_not_require_prior_read(self, tmp_path):
        f = tmp_path / "new.txt"
        tool, _ = make_tool(tmp_path, with_read_tool=True)
        result = tool.execute(file_path=str(f), content="created\n")
        assert result["status"] == "success"

    def test_creates_missing_parent_directories(self, tmp_path):
        f = tmp_path / "a" / "b" / "c" / "new.txt"
        tool, _ = make_tool(tmp_path)
        result = tool.execute(file_path=str(f), content="deep\n")
        assert result["status"] == "success"
        assert f.read_text() == "deep\n"

    def test_file_path_returned_in_result(self, tmp_path):
        f = tmp_path / "new.txt"
        tool, _ = make_tool(tmp_path)
        result = tool.execute(file_path=str(f), content="x")
        assert result["data"]["file_path"] == str(f)


# ---------------------------------------------------------------------------
# execute() — read-before-overwrite enforcement
# ---------------------------------------------------------------------------


class TestReadBeforeOverwrite:
    def test_existing_unread_file_rejected_when_read_tool_attached(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("original\n")
        tool, _ = make_tool(tmp_path, with_read_tool=True)
        result = tool.execute(file_path=str(f), content="overwritten\n")
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert f.read_text() == "original\n"

    def test_existing_read_file_can_be_overwritten(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("original\n")
        tool, read_tool = make_tool(tmp_path, with_read_tool=True)
        read_tool.execute(file_path=str(f))
        result = tool.execute(file_path=str(f), content="overwritten\n")
        assert result["status"] == "success"
        assert f.read_text() == "overwritten\n"

    def test_existing_changed_file_rejected(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("original\n")
        tool, read_tool = make_tool(tmp_path, with_read_tool=True)
        read_tool.execute(file_path=str(f))
        f.write_text("tampered\n")
        result = tool.execute(file_path=str(f), content="overwritten\n")
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_no_read_tool_skips_check_for_existing_file(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("original\n")
        tool, _ = make_tool(tmp_path, with_read_tool=False)
        result = tool.execute(file_path=str(f), content="overwritten\n")
        assert result["status"] == "success"
        assert f.read_text() == "overwritten\n"


# ---------------------------------------------------------------------------
# execute() — hash update after write
# ---------------------------------------------------------------------------


class TestHashUpdateAfterWrite:
    def test_new_file_hash_recorded_for_subsequent_update(self, tmp_path):
        from ai_cli.tools.update import UpdateTool

        f = tmp_path / "f.txt"
        workspace = MagicMock()
        workspace.root = tmp_path
        workspace.contains.return_value = True
        workspace.is_ignored.return_value = False
        pm = MagicMock()
        pm.request.return_value = (True, "")

        read_tool = ReadTool(
            workspace=workspace,
            permission_manager=pm,
            permission_required=False,
            name="read",
            description=ReadTool.DESCRIPTION,
        )
        write_tool = WriteTool(
            workspace=workspace,
            permission_manager=pm,
            permission_required=False,
            name="write",
            description=WriteTool.DESCRIPTION,
        )
        update_tool = UpdateTool(
            workspace=workspace,
            permission_manager=pm,
            permission_required=False,
            name="update",
            description=UpdateTool.DESCRIPTION,
        )
        write_tool.set_read_tool(read_tool)
        update_tool.set_read_tool(read_tool)

        write_tool.execute(file_path=str(f), content="hello world\n")
        result = update_tool.execute(
            file_path=str(f), old_string="hello", new_string="goodbye"
        )
        assert result["status"] == "success"

    def test_overwrite_updates_hash_for_subsequent_update(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("version one\n")
        tool, read_tool = make_tool(tmp_path, with_read_tool=True)
        read_tool.execute(file_path=str(f))
        tool.execute(file_path=str(f), content="version two\n")
        assert read_tool.has_been_read(str(f)) is True

    def test_write_hash_matches_written_content(self, tmp_path):
        f = tmp_path / "f.txt"
        tool, read_tool = make_tool(tmp_path, with_read_tool=True)
        tool.execute(file_path=str(f), content="fresh content\n")
        assert read_tool.has_been_read(str(f)) is True


# ---------------------------------------------------------------------------
# Cross-tool invalidation
# ---------------------------------------------------------------------------


class TestCrossToolInvalidation:
    def _make_both(self, tmp_path):
        from ai_cli.tools.update import UpdateTool

        workspace = MagicMock()
        workspace.root = tmp_path
        workspace.contains.return_value = True
        workspace.is_ignored.return_value = False
        pm = MagicMock()
        pm.request.return_value = (True, "")
        read_tool = ReadTool(
            workspace=workspace,
            permission_manager=pm,
            permission_required=False,
            name="read",
            description=ReadTool.DESCRIPTION,
        )
        write_tool = WriteTool(
            workspace=workspace,
            permission_manager=pm,
            permission_required=False,
            name="write",
            description=WriteTool.DESCRIPTION,
        )
        update_tool = UpdateTool(
            workspace=workspace,
            permission_manager=pm,
            permission_required=False,
            name="update",
            description=UpdateTool.DESCRIPTION,
        )
        write_tool.set_read_tool(read_tool)
        update_tool.set_read_tool(read_tool)
        return read_tool, write_tool, update_tool

    def test_update_then_write_is_rejected(self, tmp_path):
        """update modifies a file → write must not silently clobber the change."""
        f = tmp_path / "f.txt"
        f.write_text("original\n")
        read_tool, write_tool, update_tool = self._make_both(tmp_path)
        read_tool.execute(file_path=str(f))

        update_tool.execute(
            file_path=str(f), old_string="original", new_string="updated"
        )
        result = write_tool.execute(file_path=str(f), content="overwrite\n")
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert f.read_text() == "updated\n"

    def test_write_then_update_is_rejected(self, tmp_path):
        """write overwrites a file → update must not silently clobber the change."""
        f = tmp_path / "f.txt"
        f.write_text("original\n")
        read_tool, write_tool, update_tool = self._make_both(tmp_path)
        read_tool.execute(file_path=str(f))

        write_tool.execute(file_path=str(f), content="rewritten\n")
        result = update_tool.execute(
            file_path=str(f), old_string="rewritten", new_string="tweaked"
        )
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert f.read_text() == "rewritten\n"

    def test_reread_after_update_allows_write(self, tmp_path):
        """Re-reading after update clears the writer tag so write is allowed again."""
        f = tmp_path / "f.txt"
        f.write_text("original\n")
        read_tool, write_tool, update_tool = self._make_both(tmp_path)
        read_tool.execute(file_path=str(f))
        update_tool.execute(
            file_path=str(f), old_string="original", new_string="updated"
        )
        read_tool.execute(file_path=str(f))  # re-read

        result = write_tool.execute(file_path=str(f), content="overwrite\n")
        assert result["status"] == "success"

    def test_reread_after_write_allows_update(self, tmp_path):
        """Re-reading after write clears the writer tag so update is allowed again."""
        f = tmp_path / "f.txt"
        f.write_text("original\n")
        read_tool, write_tool, update_tool = self._make_both(tmp_path)
        read_tool.execute(file_path=str(f))
        write_tool.execute(file_path=str(f), content="rewritten\n")
        read_tool.execute(file_path=str(f))  # re-read

        result = update_tool.execute(
            file_path=str(f), old_string="rewritten", new_string="tweaked"
        )
        assert result["status"] == "success"

    def test_update_chain_still_works(self, tmp_path):
        """Multiple update calls in sequence are still allowed without re-reading."""
        f = tmp_path / "f.txt"
        f.write_text("alpha beta gamma\n")
        read_tool, _, update_tool = self._make_both(tmp_path)
        read_tool.execute(file_path=str(f))
        update_tool.execute(file_path=str(f), old_string="alpha", new_string="one")
        result = update_tool.execute(
            file_path=str(f), old_string="beta", new_string="two"
        )
        assert result["status"] == "success"

    def test_write_chain_still_works(self, tmp_path):
        """Multiple write calls in sequence are still allowed without re-reading."""
        f = tmp_path / "f.txt"
        f.write_text("original\n")
        read_tool, write_tool, _ = self._make_both(tmp_path)
        read_tool.execute(file_path=str(f))
        write_tool.execute(file_path=str(f), content="version two\n")
        result = write_tool.execute(file_path=str(f), content="version three\n")
        assert result["status"] == "success"
        assert f.read_text() == "version three\n"


class TestWriteEncodeError:
    def test_unicode_encode_error_returns_write_error(self, tmp_path):
        f = tmp_path / "new.txt"
        tool, _ = make_tool(tmp_path)
        with patch(
            "pathlib.Path.write_text",
            side_effect=UnicodeEncodeError("utf-8", "", 0, 1, "lone surrogate"),
        ):
            result = tool.execute(file_path=str(f), content="hello\n")
        assert result["error"] == "write_error"
        assert "Cannot write" in result["message"]
