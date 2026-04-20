"""Tests for ai_cli.tools.search_files."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_cli.core.vector_store import SearchResult
from ai_cli.tools.search_files import SearchFilesTool, _read_snippet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_workspace(embedding_index: object = None) -> MagicMock:
    ws = MagicMock()
    ws.embedding_index = embedding_index
    return ws


def _make_tool(embedding_index: object = None) -> SearchFilesTool:
    ws = _make_workspace(embedding_index)
    pm = MagicMock()
    pm.request.return_value = (True, "")
    return SearchFilesTool(
        workspace=ws,
        permission_manager=pm,
        permission_required=False,
        name="search_files",
        description="search",
    )


def _make_search_result(
    id: str = "id1",
    score: float = 0.9,
    file_path: str = "/test/file.py",
    start_line: int = 1,
    end_line: int = 5,
    symbol_name: str | None = "my_func",
    symbol_kind: str | None = "function",
    chunk_type: str = "chunk",
) -> SearchResult:
    return SearchResult(
        id=id,
        score=score,
        metadata={
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "symbol_name": symbol_name,
            "symbol_kind": symbol_kind,
            "chunk_type": chunk_type,
            "char_hash": "abcdef12",
        },
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_definition_schema():
    """Tool schema includes required and optional parameters."""
    tool = _make_tool()
    schema = tool.definition().schema()
    fn = schema["function"]
    assert fn["name"] == "search_files"
    params = fn["parameters"]
    assert "query" in params["required"]
    props = params["properties"]
    assert "query" in props
    assert "k" in props
    assert "level" in props
    assert "path_glob" in props
    # level should have enum restriction.
    assert "enum" in props["level"]
    assert "chunk" in props["level"]["enum"]


def test_class_attributes():
    """CLASS-level attributes match specification."""
    assert SearchFilesTool.NAME == "search_files"
    assert SearchFilesTool.PERMISSION_REQUIRED is False
    assert SearchFilesTool.DISABLED_BY_DEFAULT is True


# ---------------------------------------------------------------------------
# execute() — embedding_index is None
# ---------------------------------------------------------------------------


def test_execute_no_embedding_index():
    """Returns not_enabled error when embedding_index is None."""
    tool = _make_tool(embedding_index=None)
    result = tool.execute(query="find something")
    assert result["status"] == "error"
    assert result["error"] == "not_enabled"
    assert result["code"] == 400


# ---------------------------------------------------------------------------
# execute() — with mock EmbeddingIndex
# ---------------------------------------------------------------------------


def test_execute_basic(tmp_path: Path) -> None:
    """execute() forwards query to embedding_index.search() and returns results."""
    ei = MagicMock()
    ei.search.return_value = [
        _make_search_result(
            id="c1",
            score=0.95,
            file_path=str(tmp_path / "file.py"),
            start_line=1,
            end_line=3,
        )
    ]
    tool = _make_tool(embedding_index=ei)

    # Create a real file to read from.
    f = tmp_path / "file.py"
    f.write_text("def hello():\n    pass\n# end\n")

    result = tool.execute(query="hello function")
    assert result["status"] == "success"
    data = result["data"]
    assert "results" in data
    assert len(data["results"]) == 1
    r = data["results"][0]
    assert r["file"] == str(f)
    assert r["score"] == pytest.approx(0.95, abs=0.001)
    assert "snippet" in r


def test_execute_forwards_correct_args() -> None:
    """execute() passes k, level, and path_glob to embedding_index.search()."""
    ei = MagicMock()
    ei.search.return_value = []
    tool = _make_tool(embedding_index=ei)

    tool.execute(query="test query", k=7, level="document", path_glob="**/*.py")

    ei.search.assert_called_once_with(
        "test query", k=7, level="document", path_glob="**/*.py"
    )


def test_execute_clamps_k() -> None:
    """k is clamped to [1, 20]."""
    ei = MagicMock()
    ei.search.return_value = []
    tool = _make_tool(embedding_index=ei)

    tool.execute(query="q", k=100)
    call_k = ei.search.call_args.kwargs["k"]
    assert call_k == 20

    tool.execute(query="q", k=0)
    call_k = ei.search.call_args.kwargs["k"]
    assert call_k == 1


def test_execute_invalid_level() -> None:
    """Invalid level value returns an error."""
    ei = MagicMock()
    tool = _make_tool(embedding_index=ei)
    result = tool.execute(query="q", level="invalid")
    assert result["status"] == "error"
    assert result["error"] == "invalid_input"


def test_execute_includes_query_time_ms() -> None:
    """Response includes query_time_ms."""
    ei = MagicMock()
    ei.search.return_value = []
    tool = _make_tool(embedding_index=ei)
    result = tool.execute(query="speed test")
    assert result["status"] == "success"
    assert "query_time_ms" in result["data"]
    assert isinstance(result["data"]["query_time_ms"], int)


def test_execute_document_level_snippet(tmp_path: Path) -> None:
    """Document-level results (start_line=None) return first 50 lines as snippet."""
    ei = MagicMock()
    f = tmp_path / "readme.md"
    lines = [f"line {i}\n" for i in range(100)]
    f.write_text("".join(lines))

    ei.search.return_value = [
        SearchResult(
            id="doc1",
            score=0.8,
            metadata={
                "file_path": str(f),
                "start_line": None,
                "end_line": None,
                "symbol_name": None,
                "symbol_kind": None,
                "chunk_type": "document",
                "char_hash": "h",
            },
        )
    ]
    tool = _make_tool(embedding_index=ei)
    result = tool.execute(query="q", level="document")
    assert result["status"] == "success"
    snippet = result["data"]["results"][0]["snippet"]
    # Should be at most first 50 lines.
    assert len(snippet.splitlines()) <= 50


def test_execute_search_exception() -> None:
    """Search exceptions are caught and returned as errors."""
    ei = MagicMock()
    ei.search.side_effect = RuntimeError("backend failure")
    tool = _make_tool(embedding_index=ei)
    result = tool.execute(query="q")
    assert result["status"] == "error"
    assert result["error"] == "search_error"
    assert result["code"] == 500


# ---------------------------------------------------------------------------
# _read_snippet
# ---------------------------------------------------------------------------


def test_read_snippet_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "code.py"
    f.write_text("line1\nline2\nline3\nline4\n")
    snippet = _read_snippet(f, 2, 3)
    assert "line2" in snippet
    assert "line3" in snippet
    assert "line4" not in snippet


def test_read_snippet_missing_file() -> None:
    snippet = _read_snippet(Path("/nonexistent/path.py"), 1, 5)
    assert snippet == ""


def test_read_snippet_none_lines(tmp_path: Path) -> None:
    """When start/end are None, returns up to first 50 lines."""
    f = tmp_path / "big.txt"
    f.write_text("".join(f"line{i}\n" for i in range(100)))
    snippet = _read_snippet(f, None, None)
    assert len(snippet.splitlines()) <= 50
