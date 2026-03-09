"""Tests for ai_cli.utils.ignore_filter.IgnoreFilter."""

import pytest
from pathlib import Path
from ai_cli.utils.ignore_filter import IgnoreFilter


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path


def make_filter(root: Path, *patterns: str) -> IgnoreFilter:
    return IgnoreFilter(root, list(patterns))


# ---------------------------------------------------------------------------
# Basic matching
# ---------------------------------------------------------------------------

def test_blank_and_comment_lines_ignored(root):
    # Blank lines and '#'-at-column-1 comments must not appear in raw_patterns.
    f = make_filter(root, "", "# comment", "*.pyc")
    assert f.raw_patterns == ["*.pyc"]


def test_indented_hash_is_pattern_not_comment(root):
    # A line with leading spaces followed by '#' is a pattern, not a comment —
    # it must appear in raw_patterns.
    f = make_filter(root, "  # not-a-comment", "*.pyc")
    assert "  # not-a-comment" in f.raw_patterns


def test_trailing_whitespace_stripped(root):
    # Trailing spaces should not affect matching.
    f = make_filter(root, "*.log   ")
    (root / "app.log").touch()
    assert f.is_ignored(root / "app.log")


def test_simple_glob_matches_any_component(root):
    f = make_filter(root, "*.log")
    (root / "a").mkdir()
    (root / "a" / "b.log").touch()
    assert f.is_ignored(root / "a" / "b.log")


def test_simple_glob_no_match(root):
    f = make_filter(root, "*.log")
    (root / "foo.txt").touch()
    assert not f.is_ignored(root / "foo.txt")


# ---------------------------------------------------------------------------
# Anchored patterns (leading slash)
# ---------------------------------------------------------------------------

def test_anchored_matches_top_level_only(root):
    f = make_filter(root, "/build")
    (root / "build").mkdir()
    (root / "src" / "build").mkdir(parents=True)
    assert f.is_ignored(root / "build")
    assert not f.is_ignored(root / "src" / "build")


# ---------------------------------------------------------------------------
# Directory-only patterns (trailing slash)
# ---------------------------------------------------------------------------

def test_dir_only_pattern_ignores_dir_not_file(root):
    f = make_filter(root, "__pycache__/")
    cache_dir = root / "__pycache__"
    cache_dir.mkdir()
    regular_file = root / "x__pycache__"
    regular_file.touch()

    assert f.is_ignored(cache_dir)
    assert not f.is_ignored(regular_file)


def test_dir_only_pattern_ignores_contents(root):
    f = make_filter(root, "__pycache__/")
    cache_dir = root / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "module.cpython-310.pyc").touch()

    assert f.is_ignored(cache_dir / "module.cpython-310.pyc")


# ---------------------------------------------------------------------------
# Multi-segment patterns
# ---------------------------------------------------------------------------

def test_multi_segment_pattern(root):
    f = make_filter(root, "docs/build")
    (root / "docs" / "build").mkdir(parents=True)
    assert f.is_ignored(root / "docs" / "build")


def test_multi_segment_pattern_ignores_descendants(root):
    f = make_filter(root, "docs/build")
    (root / "docs" / "build").mkdir(parents=True)
    (root / "docs" / "build" / "index.html").touch()
    assert f.is_ignored(root / "docs" / "build" / "index.html")


def test_multi_segment_does_not_match_wrong_parent(root):
    f = make_filter(root, "docs/build")
    (root / "src" / "build").mkdir(parents=True)
    assert not f.is_ignored(root / "src" / "build")


# ---------------------------------------------------------------------------
# Double-star patterns
# ---------------------------------------------------------------------------

def test_double_star_matches_zero_components(root):
    f = make_filter(root, "**/foo.txt")
    (root / "foo.txt").touch()
    assert f.is_ignored(root / "foo.txt")


def test_double_star_matches_nested(root):
    f = make_filter(root, "**/foo.txt")
    (root / "a" / "b" / "c").mkdir(parents=True)
    (root / "a" / "b" / "c" / "foo.txt").touch()
    assert f.is_ignored(root / "a" / "b" / "c" / "foo.txt")


def test_double_star_in_middle(root):
    f = make_filter(root, "src/**/test_*.py")
    (root / "src" / "unit").mkdir(parents=True)
    (root / "src" / "unit" / "test_foo.py").touch()
    (root / "src" / "test_bar.py").touch()
    assert f.is_ignored(root / "src" / "unit" / "test_foo.py")
    assert f.is_ignored(root / "src" / "test_bar.py")


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

def test_negation_re_includes(root):
    f = make_filter(root, "*.log", "!important.log")
    (root / "debug.log").touch()
    (root / "important.log").touch()
    assert f.is_ignored(root / "debug.log")
    assert not f.is_ignored(root / "important.log")


def test_later_pattern_wins(root):
    f = make_filter(root, "!*.log", "*.log")
    (root / "foo.log").touch()
    assert f.is_ignored(root / "foo.log")


def test_later_dir_pattern_overrides_earlier_negation(root):
    # Regression: pattern order must be respected across ancestor depth levels.
    # '!a/b.txt' comes first, 'a/' comes later — 'a/' should win and the file
    # should be ignored.
    f = make_filter(root, "!a/b.txt", "a/")
    (root / "a").mkdir()
    (root / "a" / "b.txt").touch()
    assert f.is_ignored(root / "a" / "b.txt")


def test_earlier_dir_pattern_overridden_by_later_negation(root):
    # Reverse: 'a/' first, '!a/b.txt' later — negation wins, file re-included.
    f = make_filter(root, "a/", "!a/b.txt")
    (root / "a").mkdir()
    (root / "a" / "b.txt").touch()
    assert not f.is_ignored(root / "a" / "b.txt")


# ---------------------------------------------------------------------------
# Paths outside root
# ---------------------------------------------------------------------------

def test_path_outside_root_not_ignored(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    f = make_filter(root, "**")
    assert not f.is_ignored(other)


# ---------------------------------------------------------------------------
# from_file constructor
# ---------------------------------------------------------------------------

def test_from_file_missing_file_returns_empty(root):
    f = IgnoreFilter.from_file(root, root / ".ignore")
    (root / "anything.txt").touch()
    assert not f.is_ignored(root / "anything.txt")


def test_from_file_loads_patterns(root):
    ignore = root / ".ignore"
    ignore.write_text("*.pyc\n__pycache__/\n")
    f = IgnoreFilter.from_file(root, ignore)
    (root / "module.pyc").touch()
    assert f.is_ignored(root / "module.pyc")
