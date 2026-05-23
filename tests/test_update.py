"""Tests for ai_cli/tools/update.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_cli.tools.read import ReadTool
from ai_cli.tools.update import UpdateTool

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
) -> tuple[UpdateTool, ReadTool | None]:
    workspace = MagicMock()
    workspace.root = tmp_path
    workspace.contains.return_value = True
    workspace.is_ignored.return_value = False
    pm = MagicMock()
    pm.request.return_value = (True, "")
    tool = UpdateTool(
        workspace=workspace,
        permission_manager=pm,
        permission_required=permission_required,
        name="update",
        description=UpdateTool.DESCRIPTION,
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
        assert UpdateTool.NAME == "update"

    def test_permission_required_true_by_default(self):
        assert UpdateTool.PERMISSION_REQUIRED is True

    def test_disabled_by_default(self):
        assert UpdateTool.DISABLED_BY_DEFAULT is True


# ---------------------------------------------------------------------------
# definition()
# ---------------------------------------------------------------------------


class TestDefinition:
    def test_schema_shape(self, tmp_path):
        tool, _ = make_tool(tmp_path)
        d = tool.definition().schema()
        assert d["type"] == "function"
        fn = d["function"]
        assert fn["name"] == "update"
        params = fn["parameters"]
        assert "file_path" in params["properties"]
        assert "old_string" in params["properties"]
        assert "new_string" in params["properties"]
        assert "replace_all" in params["properties"]
        assert set(params["required"]) == {"file_path", "old_string", "new_string"}

    def test_replace_all_is_boolean(self, tmp_path):
        tool, _ = make_tool(tmp_path)
        d = tool.definition().schema()
        prop = d["function"]["parameters"]["properties"]["replace_all"]
        assert prop["type"] == "boolean"


# ---------------------------------------------------------------------------
# execute() — path validation
# ---------------------------------------------------------------------------


class TestExecutePathValidation:
    def test_relative_path_returns_error(self, tmp_path):
        tool, _ = make_tool(tmp_path)
        result = tool.execute(
            file_path="relative/path.txt",
            old_string="x",
            new_string="y",
        )
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_file_not_found_returns_error(self, tmp_path):
        tool, _ = make_tool(tmp_path)
        result = tool.execute(
            file_path=str(tmp_path / "missing.txt"),
            old_string="x",
            new_string="y",
        )
        assert result["status"] == "error"
        assert result["error"] == "read_error"

    def test_directory_path_returns_error(self, tmp_path):
        tool, _ = make_tool(tmp_path)
        result = tool.execute(
            file_path=str(tmp_path),
            old_string="x",
            new_string="y",
        )
        assert result["status"] == "error"
        assert result["error"] == "read_error"

    def test_identical_old_and_new_returns_error(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello\n")
        tool, _ = make_tool(tmp_path)
        result = tool.execute(
            file_path=str(f),
            old_string="hello",
            new_string="hello",
        )
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"


# ---------------------------------------------------------------------------
# execute() — read-before-edit enforcement
# ---------------------------------------------------------------------------


class TestReadBeforeEdit:
    def test_unread_file_rejected_when_read_tool_attached(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello world\n")
        tool, _ = make_tool(tmp_path, with_read_tool=True)
        result = tool.execute(
            file_path=str(f),
            old_string="hello",
            new_string="goodbye",
        )
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_read_file_accepted(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello world\n")
        tool, read_tool = make_tool(tmp_path, with_read_tool=True)
        read_tool.execute(file_path=str(f))
        result = tool.execute(
            file_path=str(f),
            old_string="hello",
            new_string="goodbye",
        )
        assert result["status"] == "success"

    def test_changed_file_rejected(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("original\n")
        tool, read_tool = make_tool(tmp_path, with_read_tool=True)
        read_tool.execute(file_path=str(f))
        f.write_text("tampered\n")
        result = tool.execute(
            file_path=str(f),
            old_string="tampered",
            new_string="fixed",
        )
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_no_read_tool_skips_check(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello world\n")
        tool, _ = make_tool(tmp_path, with_read_tool=False)
        result = tool.execute(
            file_path=str(f),
            old_string="hello",
            new_string="goodbye",
        )
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# execute() — replacement logic
# ---------------------------------------------------------------------------


class TestExecuteReplacement:
    def test_basic_replacement(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello world\n")
        tool, _ = make_tool(tmp_path)
        result = tool.execute(
            file_path=str(f),
            old_string="hello",
            new_string="goodbye",
        )
        assert result["status"] == "success"
        assert f.read_text() == "goodbye world\n"
        assert result["data"]["replacements"] == 1

    def test_old_string_not_found(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello world\n")
        tool, _ = make_tool(tmp_path)
        result = tool.execute(
            file_path=str(f),
            old_string="missing",
            new_string="replacement",
        )
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_non_unique_old_string_rejected_without_replace_all(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("foo bar foo\n")
        tool, _ = make_tool(tmp_path)
        result = tool.execute(
            file_path=str(f),
            old_string="foo",
            new_string="baz",
        )
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert f.read_text() == "foo bar foo\n"

    def test_replace_all_replaces_every_occurrence(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("foo bar foo baz foo\n")
        tool, _ = make_tool(tmp_path)
        result = tool.execute(
            file_path=str(f),
            old_string="foo",
            new_string="qux",
            replace_all=True,
        )
        assert result["status"] == "success"
        assert f.read_text() == "qux bar qux baz qux\n"
        assert result["data"]["replacements"] == 3

    def test_replace_all_with_single_occurrence(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("once only\n")
        tool, _ = make_tool(tmp_path)
        result = tool.execute(
            file_path=str(f),
            old_string="once",
            new_string="just",
            replace_all=True,
        )
        assert result["status"] == "success"
        assert f.read_text() == "just only\n"
        assert result["data"]["replacements"] == 1

    def test_multiline_replacement(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("def foo():\n    return 1\n")
        tool, _ = make_tool(tmp_path)
        result = tool.execute(
            file_path=str(f),
            old_string="def foo():\n    return 1",
            new_string="def foo():\n    return 42",
        )
        assert result["status"] == "success"
        assert "return 42" in f.read_text()

    def test_file_path_returned_in_result(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("x\n")
        tool, _ = make_tool(tmp_path)
        result = tool.execute(
            file_path=str(f),
            old_string="x",
            new_string="y",
        )
        assert result["data"]["file_path"] == str(f)


# ---------------------------------------------------------------------------
# execute() — hash update after successful write
# ---------------------------------------------------------------------------


class TestHashUpdateAfterWrite:
    def test_subsequent_edit_succeeds_without_re_read(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("alpha beta gamma\n")
        tool, read_tool = make_tool(tmp_path, with_read_tool=True)
        read_tool.execute(file_path=str(f))

        tool.execute(file_path=str(f), old_string="alpha", new_string="one")
        result = tool.execute(file_path=str(f), old_string="beta", new_string="two")
        assert result["status"] == "success"
        assert f.read_text() == "one two gamma\n"

    def test_hash_updated_so_has_been_read_still_true(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello world\n")
        tool, read_tool = make_tool(tmp_path, with_read_tool=True)
        read_tool.execute(file_path=str(f))

        tool.execute(file_path=str(f), old_string="hello", new_string="goodbye")
        assert read_tool.has_been_read(str(f)) is True


class TestWriteEncodeError:
    def test_unicode_encode_error_returns_write_error(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello world\n")
        tool, read_tool = make_tool(tmp_path, with_read_tool=True)
        read_tool.execute(file_path=str(f))
        with patch(
            "pathlib.Path.write_text",
            side_effect=UnicodeEncodeError("utf-8", "", 0, 1, "lone surrogate"),
        ):
            result = tool.execute(
                file_path=str(f), old_string="hello", new_string="goodbye"
            )
        assert result["error"] == "write_error"
        assert "Cannot write" in result["message"]
