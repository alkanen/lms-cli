"""Tests for ai_cli/tools/read.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_cli.tools.read import ReadTool

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


def make_tool(tmp_path: Path, *, permission_required: bool = False) -> ReadTool:
    workspace = MagicMock()
    workspace.root = tmp_path
    workspace.contains.return_value = True
    workspace.is_ignored.return_value = False
    pm = MagicMock()
    pm.request.return_value = (True, "")
    return ReadTool(
        workspace=workspace,
        permission_manager=pm,
        permission_required=permission_required,
        name="read",
        description=ReadTool.DESCRIPTION,
    )


# ---------------------------------------------------------------------------
# Class attributes
# ---------------------------------------------------------------------------


class TestClassAttributes:
    def test_name(self):
        assert ReadTool.NAME == "read"

    def test_permission_required_true_by_default(self):
        assert ReadTool.PERMISSION_REQUIRED is True

    def test_disabled_by_default(self):
        assert ReadTool.DISABLED_BY_DEFAULT is True


# ---------------------------------------------------------------------------
# definition()
# ---------------------------------------------------------------------------


class TestDefinition:
    def test_schema_shape(self, tmp_path):
        tool = make_tool(tmp_path)
        d = tool.definition().schema()
        assert d["type"] == "function"
        fn = d["function"]
        assert fn["name"] == "read"
        params = fn["parameters"]
        assert "file_path" in params["properties"]
        assert params["required"] == ["file_path"]
        assert "limit" in params["properties"]
        assert "offset" in params["properties"]

    def test_limit_has_minimum_of_one(self, tmp_path):
        tool = make_tool(tmp_path)
        d = tool.definition().schema()
        assert d["function"]["parameters"]["properties"]["limit"]["minimum"] == 1

    def test_offset_has_minimum_of_zero(self, tmp_path):
        tool = make_tool(tmp_path)
        d = tool.definition().schema()
        assert d["function"]["parameters"]["properties"]["offset"]["minimum"] == 0


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


class TestExecute:
    def test_reads_full_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3\n")
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f))
        assert result["status"] == "success"
        d = result["data"]
        assert d["total_lines"] == 3
        assert d["lines_returned"] == 3
        assert d["file_path"] == str(f)

    def test_content_is_cat_n_format(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("alpha\nbeta\ngamma\n")
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f))
        content = result["data"]["content"]
        assert content.startswith("     1\talpha\n")
        assert "     2\tbeta\n" in content
        assert "     3\tgamma\n" in content

    def test_line_numbers_are_right_justified_six_chars(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("x\n")
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f))
        assert result["data"]["content"] == "     1\tx\n"

    def test_applies_default_limit_of_2000(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("".join(f"line{i}\n" for i in range(3000)))
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f))
        assert result["data"]["lines_returned"] == 2000
        assert result["data"]["total_lines"] == 3000

    def test_applies_custom_limit(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f), limit=3)
        assert result["data"]["lines_returned"] == 3
        assert result["data"]["total_lines"] == 5
        assert "     1\ta\n" in result["data"]["content"]
        assert "c" in result["data"]["content"]
        assert "d" not in result["data"]["content"]

    def test_applies_offset(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f), offset=2)
        d = result["data"]
        assert d["lines_returned"] == 3
        assert d["total_lines"] == 5
        assert "     3\tc\n" in d["content"]
        assert "a" not in d["content"]

    def test_applies_offset_and_limit_together(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f), offset=1, limit=2)
        d = result["data"]
        assert d["lines_returned"] == 2
        assert "     2\tb\n" in d["content"]
        assert "     3\tc\n" in d["content"]
        assert "a" not in d["content"]
        assert "d" not in d["content"]

    def test_offset_zero_reads_from_beginning(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("a\nb\n")
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f), offset=0)
        assert result["status"] == "success"
        assert "     1\ta\n" in result["data"]["content"]

    def test_relative_path_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.execute(file_path="relative/path.txt")
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_file_not_found_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(tmp_path / "missing.txt"))
        assert result["status"] == "error"
        assert result["error"] == "read_error"

    def test_directory_path_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(tmp_path))
        assert result["status"] == "error"
        assert result["error"] == "read_error"

    def test_undecodable_file_returns_error(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\xff\xfe\x00\x01")
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f))
        assert result["status"] == "error"
        assert result["error"] == "read_error"

    def test_empty_file_returns_success(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f))
        assert result["status"] == "success"
        d = result["data"]
        assert d["content"] == ""
        assert d["total_lines"] == 0
        assert d["lines_returned"] == 0

    def test_offset_beyond_file_length_returns_error(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("a\nb\n")
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f), offset=5)
        assert result["status"] == "error"
        assert result["error"] == "invalid_range"

    def test_offset_at_last_line_reads_that_line(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("a\nb\nc\n")
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f), offset=2)
        d = result["data"]
        assert d["lines_returned"] == 1
        assert "     3\tc\n" in d["content"]

    def test_file_without_trailing_newline(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("no newline")
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f))
        assert result["status"] == "success"
        assert result["data"]["total_lines"] == 1
        assert result["data"]["content"] == "     1\tno newline"

    def test_limit_larger_than_file_returns_all_lines(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("a\nb\n")
        tool = make_tool(tmp_path)
        result = tool.execute(file_path=str(f), limit=1000)
        assert result["data"]["lines_returned"] == 2


# ---------------------------------------------------------------------------
# has_been_read() / record_hash() / reset_session_state()
# ---------------------------------------------------------------------------


class TestReadTracking:
    def test_has_been_read_false_before_any_read(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello\n")
        tool = make_tool(tmp_path)
        assert tool.has_been_read(str(f)) is False

    def test_has_been_read_true_after_successful_read(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello\n")
        tool = make_tool(tmp_path)
        tool.execute(file_path=str(f))
        assert tool.has_been_read(str(f)) is True

    def test_has_been_read_false_after_file_changes(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("original\n")
        tool = make_tool(tmp_path)
        tool.execute(file_path=str(f))
        f.write_text("modified\n")
        assert tool.has_been_read(str(f)) is False

    def test_has_been_read_false_after_reset_session_state(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello\n")
        tool = make_tool(tmp_path)
        tool.execute(file_path=str(f))
        tool.reset_session_state()
        assert tool.has_been_read(str(f)) is False

    def test_has_been_read_false_for_relative_path(self, tmp_path):
        tool = make_tool(tmp_path)
        assert tool.has_been_read("relative/path.txt") is False

    def test_has_been_read_false_for_nonexistent_file(self, tmp_path):
        tool = make_tool(tmp_path)
        assert tool.has_been_read(str(tmp_path / "ghost.txt")) is False

    def test_record_hash_allows_has_been_read(self, tmp_path):
        f = tmp_path / "f.txt"
        content = "line1\nline2\n"
        f.write_text(content)
        tool = make_tool(tmp_path)
        tool.record_hash(str(f), content)
        assert tool.has_been_read(str(f)) is True

    def test_re_read_updates_hash(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("v1\n")
        tool = make_tool(tmp_path)
        tool.execute(file_path=str(f))
        f.write_text("v2\n")
        assert tool.has_been_read(str(f)) is False
        tool.execute(file_path=str(f))
        assert tool.has_been_read(str(f)) is True


# ---------------------------------------------------------------------------
# Writer-tag cross-tool invalidation
# ---------------------------------------------------------------------------


class TestWriterTag:
    def test_fresh_read_allows_any_caller(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello\n")
        tool = make_tool(tmp_path)
        tool.execute(file_path=str(f))
        assert tool.has_been_read(str(f), caller="update") is True
        assert tool.has_been_read(str(f), caller="write") is True

    def test_record_hash_with_writer_blocks_different_caller(self, tmp_path):
        f = tmp_path / "f.txt"
        content = "hello\n"
        f.write_text(content)
        tool = make_tool(tmp_path)
        tool.execute(file_path=str(f))
        tool.record_hash(str(f), content, writer="update")
        assert tool.has_been_read(str(f), caller="update") is True
        assert tool.has_been_read(str(f), caller="write") is False

    def test_record_hash_with_writer_allows_same_caller(self, tmp_path):
        f = tmp_path / "f.txt"
        content = "hello\n"
        f.write_text(content)
        tool = make_tool(tmp_path)
        tool.execute(file_path=str(f))
        tool.record_hash(str(f), content, writer="write")
        assert tool.has_been_read(str(f), caller="write") is True

    def test_execute_clears_writer_tag(self, tmp_path):
        f = tmp_path / "f.txt"
        content = "hello\n"
        f.write_text(content)
        tool = make_tool(tmp_path)
        tool.execute(file_path=str(f))
        tool.record_hash(str(f), content, writer="update")
        assert tool.has_been_read(str(f), caller="write") is False
        tool.execute(file_path=str(f))  # re-read clears the tag
        assert tool.has_been_read(str(f), caller="write") is True

    def test_no_caller_skips_writer_check(self, tmp_path):
        f = tmp_path / "f.txt"
        content = "hello\n"
        f.write_text(content)
        tool = make_tool(tmp_path)
        tool.execute(file_path=str(f))
        tool.record_hash(str(f), content, writer="update")
        assert tool.has_been_read(str(f)) is True  # no caller → no tag check

    def test_reset_clears_writer_tags(self, tmp_path):
        f = tmp_path / "f.txt"
        content = "hello\n"
        f.write_text(content)
        tool = make_tool(tmp_path)
        tool.execute(file_path=str(f))
        tool.record_hash(str(f), content, writer="update")
        tool.reset_session_state()
        assert tool.has_been_read(str(f), caller="update") is False

    def test_record_hash_without_prior_read_allows_any_caller(self, tmp_path):
        """New files created by a write tool (no prior read) carry no tag."""
        f = tmp_path / "f.txt"
        content = "brand new\n"
        f.write_text(content)
        tool = make_tool(tmp_path)
        tool.record_hash(str(f), content, writer="write")
        assert tool.has_been_read(str(f), caller="write") is True
        assert tool.has_been_read(str(f), caller="update") is True
