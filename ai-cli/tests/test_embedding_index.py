"""Tests for ai_cli.core.embedding_index."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("numpy")

from ai_cli.core.embedding_index import EmbeddingIndex  # noqa: E402
from ai_cli.core.vector_store import SQLiteVectorStore  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteVectorStore:
    return SQLiteVectorStore(tmp_path / "index.db")


def _make_provider(dimension: int = 4) -> MagicMock:
    """Return a mock EmbeddingProvider that returns deterministic unit vectors."""
    provider = MagicMock()
    provider.dimension = dimension

    # embed() is async.
    async def _embed(
        texts: list[str],
        on_batch: object = None,
    ) -> list[list[float]]:
        vecs = [[1.0] + [0.0] * (dimension - 1)] * len(texts)
        if on_batch is not None:
            on_batch(len(texts), len(texts))  # type: ignore[operator]
        return vecs

    provider.embed = _embed
    # embed_sync() is sync.
    provider.embed_sync.return_value = [[1.0] + [0.0] * (dimension - 1)]
    return provider


def _make_workspace(root: Path) -> MagicMock:
    ws = MagicMock()
    ws.root = root
    ws.is_ignored.return_value = False
    return ws


def _make_index(
    store: SQLiteVectorStore,
    workspace_root: Path,
    *,
    provider: MagicMock | None = None,
    config: dict | None = None,
) -> EmbeddingIndex:
    if provider is None:
        provider = _make_provider()
    ws = _make_workspace(workspace_root)
    return EmbeddingIndex(
        db_path=workspace_root / ".ai-cli" / "embeddings" / "index.db",
        provider=provider,
        store=store,
        config=config or {},
        workspace=ws,
    )


# ---------------------------------------------------------------------------
# Root management
# ---------------------------------------------------------------------------


def test_add_root_persists(store: SQLiteVectorStore, tmp_path: Path) -> None:
    """add_root persists the entry to the database."""
    idx = _make_index(store, tmp_path)
    ext_root = tmp_path / "external"
    ext_root.mkdir()
    idx.add_root(ext_root, label="My Corpus")
    roots = idx.roots
    # add_root resolves the path, so compare against resolved form.
    assert any(r.path == ext_root.resolve() and r.label == "My Corpus" for r in roots)


def test_add_root_rejects_nonexistent(store: SQLiteVectorStore, tmp_path: Path) -> None:
    """add_root raises ValueError for a path that does not exist."""
    idx = _make_index(store, tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        idx.add_root(tmp_path / "no_such_dir")


def test_add_root_rejects_file(store: SQLiteVectorStore, tmp_path: Path) -> None:
    """add_root raises ValueError when given a file path instead of a directory."""
    idx = _make_index(store, tmp_path)
    f = tmp_path / "file.txt"
    f.touch()
    with pytest.raises(ValueError, match="not a directory"):
        idx.add_root(f)


def test_remove_root_deletes_chunks(store: SQLiteVectorStore, tmp_path: Path) -> None:
    """remove_root removes the entry and all associated chunks."""
    idx = _make_index(store, tmp_path)
    ext_root = tmp_path / "corpus"
    ext_root.mkdir()

    # Manually insert a chunk for a file under ext_root.
    store.upsert(
        ids=["c1"],
        vectors=[[1.0, 0.0, 0.0, 0.0]],
        metadata=[
            {
                "file_path": str(ext_root / "file.txt"),
                "chunk_type": "chunk",
                "char_hash": "abc",
            }
        ],
    )
    idx.add_root(ext_root)
    idx.remove_root(ext_root)

    # Root should be gone.
    assert not any(r.path == ext_root for r in idx.roots)
    # Chunks for that file should be deleted.
    hashes = store.all_file_hashes()
    assert str(ext_root / "file.txt") not in hashes


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------


def test_is_indexed_path_workspace_root_false(
    store: SQLiteVectorStore, tmp_path: Path
) -> None:
    """Paths under the workspace root return False (handled by Workspace.contains)."""
    idx = _make_index(store, tmp_path)
    file_in_ws = tmp_path / "src" / "main.py"
    assert idx.is_indexed_path(file_in_ws) is False


def test_is_indexed_path_external_root_true(
    store: SQLiteVectorStore, tmp_path: Path
) -> None:
    """A path under an external indexed root returns True."""
    ext_root = tmp_path / "external"
    ext_root.mkdir()
    idx = _make_index(store, tmp_path)
    idx.add_root(ext_root)
    sub_file = ext_root / "sub" / "doc.md"
    assert idx.is_indexed_path(sub_file) is True


def test_is_indexed_path_unrelated_false(
    store: SQLiteVectorStore, tmp_path: Path
) -> None:
    """An unrelated path returns False."""
    idx = _make_index(store, tmp_path)
    assert idx.is_indexed_path(Path("/totally/unrelated/path.txt")) is False


# ---------------------------------------------------------------------------
# Incremental indexing
# ---------------------------------------------------------------------------


def test_index_incremental_skips_unchanged(
    store: SQLiteVectorStore, tmp_path: Path
) -> None:
    """Unchanged files (same hash) are skipped during incremental index."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("hello world")

    provider = _make_provider()
    idx = _make_index(store, root, provider=provider)
    store.upsert_root(str(root), None, "2024-01-01T00:00:00Z")

    # First index.
    asyncio.run(idx.index(roots=[root], incremental=True))
    # Count how many embed calls were made by checking store.
    hashes_after_first = store.all_file_hashes()
    assert str(root / "a.txt") in hashes_after_first

    # Second index with same content — should skip.
    stats = asyncio.run(idx.index(roots=[root], incremental=True))
    assert stats.files_skipped == 1
    assert stats.files_indexed == 0


def test_index_incremental_reindexes_changed(
    store: SQLiteVectorStore, tmp_path: Path
) -> None:
    """Changed files are re-indexed even in incremental mode."""
    root = tmp_path / "project"
    root.mkdir()
    f = root / "b.txt"
    f.write_text("original content")

    provider = _make_provider()
    idx = _make_index(store, root, provider=provider)
    store.upsert_root(str(root), None, "2024-01-01T00:00:00Z")

    asyncio.run(idx.index(roots=[root], incremental=True))

    # Modify the file.
    f.write_text("modified content that is different")
    stats = asyncio.run(idx.index(roots=[root], incremental=True))
    assert stats.files_indexed == 1
    assert stats.files_skipped == 0


def test_index_full_reindexes_all(store: SQLiteVectorStore, tmp_path: Path) -> None:
    """incremental=False re-embeds all files regardless of hash."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "c.txt").write_text("same content")

    provider = _make_provider()
    idx = _make_index(store, root, provider=provider)
    store.upsert_root(str(root), None, "2024-01-01T00:00:00Z")

    asyncio.run(idx.index(roots=[root], incremental=True))

    stats = asyncio.run(idx.index(roots=[root], incremental=False))
    assert stats.files_indexed == 1
    assert stats.files_skipped == 0


def test_index_removes_deleted_files(store: SQLiteVectorStore, tmp_path: Path) -> None:
    """Files deleted between index runs are removed from the store."""
    root = tmp_path / "project"
    root.mkdir()
    f = root / "temp.txt"
    f.write_text("will be deleted")

    provider = _make_provider()
    idx = _make_index(store, root, provider=provider)
    store.upsert_root(str(root), None, "2024-01-01T00:00:00Z")

    asyncio.run(idx.index(roots=[root], incremental=True))
    assert str(f) in store.all_file_hashes()

    # Delete the file.
    f.unlink()
    stats = asyncio.run(idx.index(roots=[root], incremental=True))
    assert stats.files_deleted == 1
    assert str(f) not in store.all_file_hashes()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_returns_ranked_results(
    store: SQLiteVectorStore, tmp_path: Path
) -> None:
    """search() returns results ranked by cosine similarity."""
    # Insert two chunks with known vectors.
    store.upsert(
        ids=["best", "ok"],
        vectors=[[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
        metadata=[
            {"file_path": "/f1.py", "chunk_type": "chunk", "char_hash": "h1"},
            {"file_path": "/f2.py", "chunk_type": "chunk", "char_hash": "h2"},
        ],
    )

    provider = _make_provider(dimension=4)
    # embed_sync returns a vector aligned with [1,0,0,0] → "best" should win.
    provider.embed_sync.return_value = [[1.0, 0.0, 0.0, 0.0]]

    idx = _make_index(store, tmp_path, provider=provider)
    results = idx.search("query", k=2, level="chunk")
    assert len(results) == 2
    assert results[0].id == "best"


def test_search_level_chunk_only(store: SQLiteVectorStore, tmp_path: Path) -> None:
    """level='chunk' only returns chunk-type rows."""
    store.upsert(
        ids=["ch", "doc"],
        vectors=[[1.0, 0.0], [1.0, 0.0]],
        metadata=[
            {"file_path": "/f", "chunk_type": "chunk", "char_hash": "h"},
            {"file_path": "/f", "chunk_type": "document", "char_hash": "h"},
        ],
    )
    provider = _make_provider(dimension=2)
    provider.embed_sync.return_value = [[1.0, 0.0]]

    idx = _make_index(store, tmp_path, provider=provider)
    results = idx.search("q", k=10, level="chunk")
    assert all(r.metadata["chunk_type"] == "chunk" for r in results)


def test_search_level_document_only(store: SQLiteVectorStore, tmp_path: Path) -> None:
    """level='document' only returns document-type rows."""
    store.upsert(
        ids=["ch", "doc"],
        vectors=[[1.0, 0.0], [1.0, 0.0]],
        metadata=[
            {"file_path": "/f", "chunk_type": "chunk", "char_hash": "h"},
            {"file_path": "/f", "chunk_type": "document", "char_hash": "h"},
        ],
    )
    provider = _make_provider(dimension=2)
    provider.embed_sync.return_value = [[1.0, 0.0]]

    idx = _make_index(store, tmp_path, provider=provider)
    results = idx.search("q", k=10, level="document")
    assert all(r.metadata["chunk_type"] == "document" for r in results)


def test_search_path_glob_filter(store: SQLiteVectorStore, tmp_path: Path) -> None:
    """path_glob restricts results to matching file paths."""
    store.upsert(
        ids=["py", "md"],
        vectors=[[1.0, 0.0], [0.9, 0.1]],
        metadata=[
            {"file_path": "/src/main.py", "chunk_type": "chunk", "char_hash": "h"},
            {"file_path": "/docs/readme.md", "chunk_type": "chunk", "char_hash": "h"},
        ],
    )
    provider = _make_provider(dimension=2)
    provider.embed_sync.return_value = [[1.0, 0.0]]

    idx = _make_index(store, tmp_path, provider=provider)
    results = idx.search("q", k=10, path_glob="*.py")
    assert all(r.metadata["file_path"].endswith(".py") for r in results)


# ---------------------------------------------------------------------------
# index() is awaitable / async task compatible
# ---------------------------------------------------------------------------


def test_index_is_coroutine(store: SQLiteVectorStore, tmp_path: Path) -> None:
    """EmbeddingIndex.index() returns a coroutine that can be awaited."""
    import inspect

    root = tmp_path / "p"
    root.mkdir()
    idx = _make_index(store, root)
    store.upsert_root(str(root), None, "2024-01-01T00:00:00Z")
    coro = idx.index(roots=[root])
    assert inspect.iscoroutine(coro)
    # Clean up.
    asyncio.run(coro)
