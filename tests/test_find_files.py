"""Tests for ai_cli/tools/find_files.py."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_cli.tools.find_files import FindFilesTool

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


def make_tool(tmp_path: Path) -> FindFilesTool:
    """Tool backed by a real Workspace (no ignore rules unless added)."""
    from ai_cli.core.workspace import Workspace

    ws = Workspace(tmp_path, config_manager=MagicMock())
    pm = MagicMock()
    return FindFilesTool(
        workspace=ws,
        permission_manager=pm,
        permission_required=False,
        name="find_files",
        description="Find files.",
    )


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Class attributes
# ---------------------------------------------------------------------------


class TestClassAttributes:
    def test_name(self):
        assert FindFilesTool.NAME == "find_files"

    def test_permission_required_false(self):
        assert FindFilesTool.PERMISSION_REQUIRED is False

    def test_disabled_by_default(self):
        assert FindFilesTool.DISABLED_BY_DEFAULT is True


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestDefinition:
    def test_returns_function_type(self, tmp_path):
        tool = make_tool(tmp_path)
        defn = tool.definition().schema()
        assert defn["type"] == "function"
        assert defn["function"]["name"] == "find_files"

    def test_pattern_is_required(self, tmp_path):
        tool = make_tool(tmp_path)
        defn = tool.definition().schema()
        assert "pattern" in defn["function"]["parameters"]["required"]

    def test_only_pattern_parameter(self, tmp_path):
        tool = make_tool(tmp_path)
        props = tool.definition().schema()["function"]["parameters"]["properties"]
        assert list(props.keys()) == ["pattern"]


# ---------------------------------------------------------------------------
# Basic matching
# ---------------------------------------------------------------------------


class TestBasicMatching:
    def test_finds_matching_files(self, tmp_path):
        _write(tmp_path / "foo.py")
        _write(tmp_path / "bar.py")
        _write(tmp_path / "baz.txt")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="*.py")
        assert result["status"] == "success"
        assert set(result["data"]["matches"]) == {"foo.py", "bar.py"}
        assert result["data"]["count"] == 2

    def test_no_matches_returns_empty_list(self, tmp_path):
        _write(tmp_path / "readme.md")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="*.py")
        assert result["status"] == "success"
        assert result["data"]["matches"] == []
        assert result["data"]["count"] == 0

    def test_results_are_sorted(self, tmp_path):
        _write(tmp_path / "z.py")
        _write(tmp_path / "a.py")
        _write(tmp_path / "m.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="*.py")
        assert result["data"]["matches"] == ["a.py", "m.py", "z.py"]

    def test_directories_excluded(self, tmp_path):
        (tmp_path / "subdir.py").mkdir()  # directory, not a file
        _write(tmp_path / "real.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="*.py")
        assert result["data"]["matches"] == ["real.py"]

    def test_pattern_echoed_in_result(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="*.py")
        assert result["data"]["pattern"] == "*.py"
        assert "directory" not in result["data"]


# ---------------------------------------------------------------------------
# Recursive search
# ---------------------------------------------------------------------------


class TestRecursiveSearch:
    def test_recursive_glob_finds_nested_files(self, tmp_path):
        _write(tmp_path / "a.py")
        _write(tmp_path / "sub" / "b.py")
        _write(tmp_path / "sub" / "deep" / "c.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="**/*.py")
        assert set(result["data"]["matches"]) == {
            "a.py",
            "sub/b.py",
            "sub/deep/c.py",
        }

    def test_non_recursive_glob_does_not_descend(self, tmp_path):
        _write(tmp_path / "top.py")
        _write(tmp_path / "sub" / "nested.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="*.py")
        assert result["data"]["matches"] == ["top.py"]

    def test_fixed_depth_pattern_ignores_files_in_ignored_directory(self, tmp_path):
        # "build/*.py" targets the build/ directory explicitly; its contents
        # should be excluded when build/ is in the ignore rules.
        _write(tmp_path / "build" / "out.py")
        ignore = tmp_path / ".ai-cli" / ".ignore"
        ignore.parent.mkdir(parents=True, exist_ok=True)
        ignore.write_text("build/\n", encoding="utf-8")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="build/*.py")
        assert result["data"]["matches"] == []

    def test_ignored_literal_prefix_returns_empty_immediately(self, tmp_path):
        # When the literal directory prefix of a pattern is ignored, the tool
        # should short-circuit rather than falling back to walking workspace_root.
        _write(tmp_path / "build" / "out.py")
        _write(tmp_path / "src" / "main.py")
        ignore = tmp_path / ".ai-cli" / ".ignore"
        ignore.parent.mkdir(parents=True, exist_ok=True)
        ignore.write_text("build/\n", encoding="utf-8")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="build/**/*.py")
        assert result["status"] == "success"
        assert result["data"]["matches"] == []

    def test_subdirectory_search_via_pattern_prefix(self, tmp_path):
        _write(tmp_path / "root.py")
        _write(tmp_path / "src" / "a.py")
        _write(tmp_path / "src" / "b.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="src/*.py")
        assert set(result["data"]["matches"]) == {"src/a.py", "src/b.py"}

    def test_fixed_depth_pattern_with_slash(self, tmp_path):
        # "src/*.py" contains a '/' but no '**' — only depth-1 files should match.
        _write(tmp_path / "root.py")
        _write(tmp_path / "src" / "a.py")
        _write(tmp_path / "src" / "deep" / "b.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="src/*.py")
        assert result["data"]["matches"] == ["src/a.py"]

    def test_fixed_depth_pattern_does_not_walk_beyond_depth(self, tmp_path):
        # Ensure depth-2 pattern doesn't scan beyond two levels.
        _write(tmp_path / "a" / "b" / "c.py")
        _write(tmp_path / "a" / "b" / "d" / "e.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="a/b/*.py")
        assert result["data"]["matches"] == ["a/b/c.py"]

    def test_literal_prefix_with_glob_subdir_descends_correctly(self, tmp_path):
        # "src/*/*.py" has a literal prefix "src/" and a wildcard subdir.
        # Regression test: depth limiting must not prune before reaching the
        # wildcard level, or nested files will never be found.
        _write(tmp_path / "src" / "lib" / "a.py")
        _write(tmp_path / "src" / "lib" / "sub" / "b.py")  # too deep
        _write(tmp_path / "src" / "c.py")  # too shallow
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="src/*/*.py")
        assert result["data"]["matches"] == ["src/lib/a.py"]

    def test_literal_prefix_narrowing_excludes_sibling_dirs(self, tmp_path):
        # "src/*.py" should match only files in src/, not in other sibling dirs.
        _write(tmp_path / "src" / "a.py")
        _write(tmp_path / "tests" / "b.py")
        _write(tmp_path / "docs" / "c.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="src/*.py")
        assert result["data"]["matches"] == ["src/a.py"]

    def test_literal_prefix_missing_returns_empty(self, tmp_path):
        # Pattern refers to a directory that doesn't exist — return empty, no error.
        _write(tmp_path / "other" / "a.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="src/*.py")
        assert result["status"] == "success"
        assert result["data"]["matches"] == []

    def test_recursive_pattern_with_literal_prefix_narrows_walk(self, tmp_path):
        # 'src/**/*.py' has a literal prefix 'src/', so the walk should start
        # at src/ and never enter sibling directories like tests/ or docs/.
        _write(tmp_path / "src" / "a.py")
        _write(tmp_path / "src" / "sub" / "b.py")
        _write(tmp_path / "tests" / "c.py")
        _write(tmp_path / "docs" / "d.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="src/**/*.py")
        assert result["status"] == "success"
        assert set(result["data"]["matches"]) == {"src/a.py", "src/sub/b.py"}

    def test_double_star_named_dir_matches_at_root(self, tmp_path):
        # '**/docs/*' must find files in a 'docs/' directory at the workspace
        # root.  Regression: this pattern stalled when large unignored
        # directories existed alongside docs/ because is_ignored() used
        # path.resolve() (expensive on WSL/NTFS) instead of normpath.
        _write(tmp_path / "docs" / "guide.md")
        _write(tmp_path / "docs" / "api.md")
        _write(tmp_path / "src" / "main.py")
        # Simulate an extra directory not covered by ignore rules.
        for i in range(5):
            _write(tmp_path / "other" / f"file_{i}.txt")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="**/docs/*")
        assert result["status"] == "success"
        assert set(result["data"]["matches"]) == {"docs/guide.md", "docs/api.md"}

    def test_double_star_named_dir_matches_in_subdirectory(self, tmp_path):
        # '**/docs/*' must also match when docs/ is nested, not just at root.
        _write(tmp_path / "pkg" / "docs" / "readme.md")
        _write(tmp_path / "pkg" / "src" / "main.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="**/docs/*")
        assert result["status"] == "success"
        assert result["data"]["matches"] == ["pkg/docs/readme.md"]

    def test_leading_dot_slash_equivalent_for_subdir_glob(self, tmp_path):
        _write(tmp_path / "my_directory" / "a.txt")
        _write(tmp_path / "my_directory" / "b.txt")
        tool = make_tool(tmp_path)
        plain = tool.execute(pattern="my_directory/*.txt")
        dotted = tool.execute(pattern="./my_directory/*.txt")
        expected = ["my_directory/a.txt", "my_directory/b.txt"]
        assert plain["status"] == "success"
        assert dotted["status"] == "success"
        assert plain["data"]["matches"] == expected
        assert dotted["data"]["matches"] == expected
        assert plain["data"]["matches"] == dotted["data"]["matches"]
        assert plain["data"]["count"] == dotted["data"]["count"]
        assert plain["data"].get("truncated") == dotted["data"].get("truncated")
        assert plain["data"].get("partial") == dotted["data"].get("partial")

    def test_leading_dot_slash_equivalent_for_recursive_glob(self, tmp_path):
        _write(tmp_path / "a.txt")
        _write(tmp_path / "sub" / "b.txt")
        _write(tmp_path / "sub" / "deep" / "c.txt")
        tool = make_tool(tmp_path)
        plain = tool.execute(pattern="**/*.txt")
        dotted = tool.execute(pattern="./**/*.txt")
        expected = ["a.txt", "sub/b.txt", "sub/deep/c.txt"]
        assert plain["status"] == "success"
        assert dotted["status"] == "success"
        assert plain["data"]["matches"] == expected
        assert dotted["data"]["matches"] == expected
        assert plain["data"]["matches"] == dotted["data"]["matches"]
        assert plain["data"]["count"] == dotted["data"]["count"]
        assert plain["data"].get("truncated") == dotted["data"].get("truncated")
        assert plain["data"].get("partial") == dotted["data"].get("partial")


# ---------------------------------------------------------------------------
# Ignore rules
# ---------------------------------------------------------------------------


class TestIgnoreRules:
    def test_ignored_files_excluded(self, tmp_path):
        _write(tmp_path / "keep.py")
        _write(tmp_path / "skip.py")
        ignore = tmp_path / ".ai-cli" / ".ignore"
        ignore.parent.mkdir(parents=True, exist_ok=True)
        ignore.write_text("skip.py\n", encoding="utf-8")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="*.py")
        assert "keep.py" in result["data"]["matches"]
        assert "skip.py" not in result["data"]["matches"]

    def test_ignored_directory_contents_excluded(self, tmp_path):
        _write(tmp_path / "src" / "good.py")
        _write(tmp_path / "build" / "bad.py")
        ignore = tmp_path / ".ai-cli" / ".ignore"
        ignore.parent.mkdir(parents=True, exist_ok=True)
        ignore.write_text("build/\n", encoding="utf-8")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="**/*.py")
        assert "src/good.py" in result["data"]["matches"]
        assert "build/bad.py" not in result["data"]["matches"]

    def test_gitignore_patterns_respected(self, tmp_path):
        _write(tmp_path / "keep.py")
        _write(tmp_path / "skip.py")
        (tmp_path / ".gitignore").write_text("skip.py\n", encoding="utf-8")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="*.py")
        assert "keep.py" in result["data"]["matches"]
        assert "skip.py" not in result["data"]["matches"]

    def test_ignored_directory_is_pruned_during_walk(self, tmp_path):
        # find_files prunes ignored directories for performance (env/, .git/,
        # node_modules/ etc. can contain thousands of files).  Once a directory
        # is pruned, no file inside it is ever returned — even if a negation
        # rule would re-include it.  This matches standard Git walk behaviour;
        # the IgnoreFilter negation semantic only applies when is_ignored() is
        # called directly on a known path.
        _write(tmp_path / "build" / "generated.py")
        _write(tmp_path / "build" / "important.py")
        ignore = tmp_path / ".ai-cli" / ".ignore"
        ignore.parent.mkdir(parents=True, exist_ok=True)
        ignore.write_text("build/\n!build/important.py\n", encoding="utf-8")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="**/*.py")
        # The whole build/ directory is pruned; neither file is returned.
        assert result["data"]["matches"] == []


# ---------------------------------------------------------------------------
# Square bracket patterns
# ---------------------------------------------------------------------------


class TestSquareBrackets:
    def test_bracket_class_matches_listed_chars(self, tmp_path):
        _write(tmp_path / "a.py")
        _write(tmp_path / "b.py")
        _write(tmp_path / "c.py")
        _write(tmp_path / "d.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="[ab].py")
        assert set(result["data"]["matches"]) == {"a.py", "b.py"}

    def test_bracket_range_matches_range(self, tmp_path):
        for name in ["file0.txt", "file3.txt", "file9.txt", "fileA.txt"]:
            _write(tmp_path / name)
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="file[0-9].txt")
        assert set(result["data"]["matches"]) == {"file0.txt", "file3.txt", "file9.txt"}

    def test_negated_bracket_class_caret(self, tmp_path):
        _write(tmp_path / "a.py")
        _write(tmp_path / "b.py")
        _write(tmp_path / "c.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="[^ab].py")
        assert result["data"]["matches"] == ["c.py"]

    def test_negated_bracket_class_exclamation(self, tmp_path):
        # '[!...]' is standard glob negation syntax (gitignore/fnmatch style).
        _write(tmp_path / "a.py")
        _write(tmp_path / "b.py")
        _write(tmp_path / "c.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="[!ab].py")
        assert result["data"]["matches"] == ["c.py"]

    def test_unclosed_bracket_treated_as_literal(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="[abc")
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Curly bracket patterns
# ---------------------------------------------------------------------------


class TestCurlyBrackets:
    def test_brace_expansion_matches_multiple_extensions(self, tmp_path):
        _write(tmp_path / "photo.jpg")
        _write(tmp_path / "diagram.png")
        _write(tmp_path / "notes.txt")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="*.{jpg,png}")
        assert set(result["data"]["matches"]) == {"photo.jpg", "diagram.png"}

    def test_brace_expansion_recursive(self, tmp_path):
        _write(tmp_path / "docs" / "a.png")
        _write(tmp_path / "docs" / "b.jpg")
        _write(tmp_path / "docs" / "c.txt")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="**/*.{png,jpg,jpeg,gif,svg,webp}")
        assert set(result["data"]["matches"]) == {"docs/a.png", "docs/b.jpg"}

    def test_unclosed_brace_treated_as_literal(self, tmp_path):
        # Should not crash; unclosed '{' is treated as a literal character.
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="*.{py")
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrorCases:
    def test_empty_pattern_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="")
        assert result["status"] == "error"

    def test_invalid_regex_from_bracket_returns_error(self, tmp_path):
        # [z-a] is an invalid regex range — should return a clean error, not raise.
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="[z-a].py")
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert result["code"] == 400

    def test_dotdot_pattern_segment_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="../*.py")
        assert result["status"] == "error"

    def test_dotdot_in_middle_of_pattern_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="src/../../*.py")
        assert result["status"] == "error"

    def test_dot_slash_dotdot_pattern_still_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="./../*.py")
        assert result["status"] == "error"

    def test_absolute_pattern_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="/etc/*.conf")
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Result limit / truncation
# ---------------------------------------------------------------------------


class TestResultLimit:
    def test_truncated_flag_set_when_limit_exceeded(self, tmp_path):
        from ai_cli.tools.find_files import _MAX_RESULTS

        for i in range(_MAX_RESULTS + 5):
            _write(tmp_path / f"file_{i:04d}.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="*.py")
        assert result["status"] == "success"
        assert result["data"]["count"] == _MAX_RESULTS
        assert result["data"].get("truncated") is True

    def test_no_truncated_flag_below_limit(self, tmp_path):
        _write(tmp_path / "a.py")
        _write(tmp_path / "b.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="*.py")
        assert result["status"] == "success"
        assert "truncated" not in result["data"]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions not on Windows")
    def test_partial_flag_set_on_unreadable_directory(self, tmp_path):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("chmod(0) has no effect when running as root")

        _write(tmp_path / "good.py")
        bad = tmp_path / "secret"
        bad.mkdir()
        _write(bad / "hidden.py")
        # Remove read+execute permission so os.walk raises an OSError.
        bad.chmod(0)
        try:
            tool = make_tool(tmp_path)
            result = tool.execute(pattern="**/*.py")
            assert result["status"] == "success"
            assert result["data"].get("partial") is True
            # The readable file should still appear.
            assert "good.py" in result["data"]["matches"]
        finally:
            bad.chmod(stat.S_IRWXU)

    def test_no_partial_flag_on_clean_walk(self, tmp_path):
        _write(tmp_path / "a.py")
        tool = make_tool(tmp_path)
        result = tool.execute(pattern="**/*.py")
        assert result["status"] == "success"
        assert "partial" not in result["data"]
