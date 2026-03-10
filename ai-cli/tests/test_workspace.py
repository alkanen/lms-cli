"""Tests for ai_cli.core.workspace.Workspace."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_cli.core.workspace import (
    _DOT_AI_CLI,
    _INIT_TEMPLATES,
    Workspace,
    WorkspaceError,
)


@pytest.fixture(autouse=True)
def isolate_global_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirect _GLOBAL_DIR to an empty, isolated tmp dir so real ~/.ai-cli/.ignore is never read."""
    fake_global = tmp_path_factory.mktemp("fake_global_ai_cli")
    monkeypatch.setattr("ai_cli.core.workspace._GLOBAL_DIR", fake_global)


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A temporary directory with a .ai-cli/ folder (minimal project root)."""
    (tmp_path / _DOT_AI_CLI).mkdir()
    return tmp_path


@pytest.fixture()
def workspace(project: Path) -> Workspace:
    config = MagicMock()
    return Workspace(project, config)


# ---------------------------------------------------------------------------
# find_root
# ---------------------------------------------------------------------------


class TestFindRoot:
    def test_finds_direct_parent(self, tmp_path):
        (tmp_path / _DOT_AI_CLI).mkdir()
        assert Workspace.find_root(tmp_path) == tmp_path

    def test_finds_ancestor(self, tmp_path):
        (tmp_path / _DOT_AI_CLI).mkdir()
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        assert Workspace.find_root(deep) == tmp_path

    def test_returns_none_when_not_found(self, tmp_path):
        # No .ai-cli anywhere under tmp_path
        assert Workspace.find_root(tmp_path) is None

    def test_skips_global_dir(self, tmp_path, monkeypatch):
        # Pretend home is tmp_path so ~/.ai-cli is tmp_path/.ai-cli
        monkeypatch.setattr("ai_cli.core.workspace._GLOBAL_DIR", tmp_path / _DOT_AI_CLI)
        (tmp_path / _DOT_AI_CLI).mkdir(exist_ok=True)
        # A sub-project with its own .ai-cli should still be found
        sub = tmp_path / "project"
        (sub / _DOT_AI_CLI).mkdir(parents=True)
        assert Workspace.find_root(sub) == sub


# ---------------------------------------------------------------------------
# initialise
# ---------------------------------------------------------------------------


class TestInitialise:
    def test_creates_scaffold(self, tmp_path):
        Workspace.initialise(tmp_path)
        dot = tmp_path / _DOT_AI_CLI
        assert dot.is_dir()
        assert (dot / "tools").is_dir()
        for filename in _INIT_TEMPLATES:
            assert (dot / filename).is_file(), f"{filename} missing"

    def test_does_not_overwrite_existing_files(self, tmp_path):
        Workspace.initialise(tmp_path)
        config = tmp_path / _DOT_AI_CLI / "config.yaml"
        config.write_text("custom: true\n")
        Workspace.initialise(tmp_path)  # second call
        assert config.read_text() == "custom: true\n"


# ---------------------------------------------------------------------------
# resolve / path-escape protection
# ---------------------------------------------------------------------------


class TestResolve:
    def test_resolve_simple(self, workspace, project):
        p = workspace.resolve("src/foo.py")
        assert p == project / "src" / "foo.py"

    def test_resolve_rejects_escape(self, workspace):
        with pytest.raises(WorkspaceError, match="outside"):
            workspace.resolve("../../etc/passwd")


# ---------------------------------------------------------------------------
# file_exists
# ---------------------------------------------------------------------------


class TestFileExists:
    def test_existing_file(self, workspace, project):
        (project / "hello.txt").write_text("hi")
        assert workspace.file_exists("hello.txt")

    def test_missing_file(self, workspace):
        assert not workspace.file_exists("nope.txt")

    def test_escape_returns_false(self, workspace):
        assert not workspace.file_exists("../../etc/passwd")

    def test_ignored_file_returns_false(self, project):
        (project / _DOT_AI_CLI / ".ignore").write_text("secret.key\n")
        config = MagicMock()
        ws = Workspace(project, config)
        (project / "secret.key").write_text("key material")
        assert not ws.file_exists("secret.key")


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_read_full(self, workspace, project):
        (project / "f.txt").write_text("line1\nline2\nline3\n")
        assert workspace.read_file("f.txt") == "line1\nline2\nline3\n"

    def test_read_partial(self, workspace, project):
        (project / "f.txt").write_text("a\nb\nc\nd\n")
        assert workspace.read_file("f.txt", start_line=2, end_line=3) == "b\nc\n"

    def test_read_from_start(self, workspace, project):
        (project / "f.txt").write_text("a\nb\nc\n")
        assert workspace.read_file("f.txt", end_line=2) == "a\nb\n"

    def test_read_to_end(self, workspace, project):
        (project / "f.txt").write_text("a\nb\nc\n")
        assert workspace.read_file("f.txt", start_line=2) == "b\nc\n"

    def test_missing_raises(self, workspace):
        with pytest.raises(WorkspaceError, match="not found"):
            workspace.read_file("missing.txt")

    def test_ignored_file_raises(self, project):
        (project / _DOT_AI_CLI / ".ignore").write_text("secret.txt\n")
        config = MagicMock()
        ws = Workspace(project, config)
        (project / "secret.txt").write_text("sensitive")
        with pytest.raises(WorkspaceError, match="excluded"):
            ws.read_file("secret.txt")

    def test_invalid_start_line_raises(self, workspace, project):
        (project / "f.txt").write_text("a\nb\n")
        with pytest.raises(WorkspaceError, match="start_line"):
            workspace.read_file("f.txt", start_line=0)

    def test_start_after_end_raises(self, workspace, project):
        (project / "f.txt").write_text("a\nb\nc\n")
        with pytest.raises(WorkspaceError, match="start_line"):
            workspace.read_file("f.txt", start_line=3, end_line=1)

    def test_start_line_past_eof_raises(self, workspace, project):
        (project / "f.txt").write_text("a\nb\n")
        with pytest.raises(WorkspaceError, match="exceeds file length"):
            workspace.read_file("f.txt", start_line=5)

    def test_end_line_past_eof_raises(self, workspace, project):
        (project / "f.txt").write_text("a\nb\n")
        with pytest.raises(WorkspaceError, match="exceeds file length"):
            workspace.read_file("f.txt", end_line=10)


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_write_new_file(self, workspace, project):
        workspace.write_file("new.txt", "hello\n")
        assert (project / "new.txt").read_text() == "hello\n"

    def test_write_creates_parents(self, workspace, project):
        workspace.write_file("sub/dir/f.txt", "content\n")
        assert (project / "sub" / "dir" / "f.txt").is_file()

    def test_overwrite_full(self, workspace, project):
        (project / "f.txt").write_text("old\n")
        workspace.write_file("f.txt", "new\n")
        assert (project / "f.txt").read_text() == "new\n"

    def test_partial_write_replaces_lines(self, workspace, project):
        (project / "f.txt").write_text("a\nb\nc\nd\n")
        workspace.write_file("f.txt", "X\nY\n", start_line=2, end_line=3)
        assert (project / "f.txt").read_text() == "a\nX\nY\nd\n"

    def test_partial_write_at_start(self, workspace, project):
        (project / "f.txt").write_text("a\nb\nc\n")
        workspace.write_file("f.txt", "Z\n", start_line=1, end_line=1)
        assert (project / "f.txt").read_text() == "Z\nb\nc\n"

    def test_append_to_file_without_trailing_newline(self, workspace, project):
        # Existing file has no trailing newline — appended content must start
        # on a fresh line, not be concatenated onto the last line.
        (project / "f.txt").write_text("a\nb")
        workspace.write_file("f.txt", "c\n", start_line=3)
        assert (project / "f.txt").read_text() == "a\nb\nc\n"

    def test_escape_raises(self, workspace):
        with pytest.raises(WorkspaceError, match="outside"):
            workspace.write_file("../../evil.txt", "bad")

    def test_ignored_path_raises(self, project):
        (project / _DOT_AI_CLI / ".ignore").write_text("*.log\n")
        config = MagicMock()
        ws = Workspace(project, config)
        with pytest.raises(WorkspaceError, match="excluded"):
            ws.write_file("debug.log", "data")

    def test_partial_write_on_missing_file_raises(self, workspace, project):
        with pytest.raises(WorkspaceError, match="does not exist"):
            workspace.write_file("new.txt", "x\n", start_line=1, end_line=1)

    def test_empty_content_reports_zero_lines(self, workspace, project):
        result = workspace.write_file("empty.txt", "")
        assert "0 line(s)" in result

    def test_explicit_append_with_start_and_end_line(self, workspace, project):
        # start_line == end_line == file_len + 1 is the explicit append form.
        (project / "f.txt").write_text("a\nb\n")
        result = workspace.write_file("f.txt", "c\n", start_line=3, end_line=3)
        assert (project / "f.txt").read_text() == "a\nb\nc\n"
        assert "Appended" in result

    def test_invalid_line_range_raises(self, workspace, project):
        (project / "f.txt").write_text("a\nb\nc\n")
        with pytest.raises(WorkspaceError, match="start_line"):
            workspace.write_file("f.txt", "X\n", start_line=5, end_line=2)

    def test_start_line_past_eof_raises(self, workspace, project):
        (project / "f.txt").write_text("a\nb\n")
        with pytest.raises(WorkspaceError, match="past end of file"):
            workspace.write_file("f.txt", "X\n", start_line=5, end_line=5)

    def test_end_line_past_eof_raises(self, workspace, project):
        (project / "f.txt").write_text("a\nb\n")
        with pytest.raises(WorkspaceError, match="past end of file"):
            workspace.write_file("f.txt", "X\n", start_line=1, end_line=10)


# ---------------------------------------------------------------------------
# is_ignored
# ---------------------------------------------------------------------------


class TestIsIgnored:
    def test_ignores_via_project_ignore(self, project):
        (project / _DOT_AI_CLI / ".ignore").write_text("*.log\n")
        config = MagicMock()
        ws = Workspace(project, config)
        (project / "debug.log").touch()
        assert ws.is_ignored(project / "debug.log")

    def test_not_ignored(self, project):
        (project / _DOT_AI_CLI / ".ignore").write_text("*.log\n")
        config = MagicMock()
        ws = Workspace(project, config)
        (project / "main.py").touch()
        assert not ws.is_ignored(project / "main.py")

    def test_project_negation_overrides_global(self, project, tmp_path, monkeypatch):
        # Simulate a global .ignore that excludes *.log
        fake_global = tmp_path / "global_ai_cli"
        fake_global.mkdir()
        (fake_global / ".ignore").write_text("*.log\n")
        monkeypatch.setattr("ai_cli.core.workspace._GLOBAL_DIR", fake_global)
        # Project re-includes important.log
        (project / _DOT_AI_CLI / ".ignore").write_text("!important.log\n")
        config = MagicMock()
        ws = Workspace(project, config)
        (project / "debug.log").touch()
        (project / "important.log").touch()
        assert ws.is_ignored(project / "debug.log")
        assert not ws.is_ignored(project / "important.log")
