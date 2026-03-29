"""Tests for ai_cli.core.vector_store."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")

from ai_cli.core.vector_store import SQLiteVectorStore  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteVectorStore:
    """Return a fresh SQLiteVectorStore backed by a temp database."""
    return SQLiteVectorStore(tmp_path / "index.db")


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------


def test_upsert_and_search(store: SQLiteVectorStore) -> None:
    """Upserted vectors can be retrieved via search."""
    vec = [1.0, 0.0, 0.0]
    store.upsert(
        ids=["chunk-1"],
        vectors=[vec],
        metadata=[
            {"file_path": "/a/b.py", "chunk_type": "chunk", "char_hash": "abc123"}
        ],
    )
    results = store.search([1.0, 0.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0].id == "chunk-1"
    assert results[0].metadata["file_path"] == "/a/b.py"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_ordering(store: SQLiteVectorStore) -> None:
    """Results are returned in descending cosine similarity order."""
    store.upsert(
        ids=["c1", "c2", "c3"],
        vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        metadata=[
            {"file_path": "/f1", "chunk_type": "chunk", "char_hash": "h1"},
            {"file_path": "/f2", "chunk_type": "chunk", "char_hash": "h2"},
            {"file_path": "/f3", "chunk_type": "chunk", "char_hash": "h3"},
        ],
    )
    results = store.search([1.0, 0.0], k=3)
    # c1 is most similar to [1, 0]; c3 is second; c2 is least.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # The most similar should be c1 (perfectly aligned).
    assert results[0].id == "c1"


def test_search_by_chunk_type(store: SQLiteVectorStore) -> None:
    """chunk_type filter restricts results correctly."""
    store.upsert(
        ids=["ch", "doc"],
        vectors=[[1.0, 0.0], [0.9, 0.1]],
        metadata=[
            {"file_path": "/f", "chunk_type": "chunk", "char_hash": "h"},
            {"file_path": "/f", "chunk_type": "document", "char_hash": "h"},
        ],
    )
    chunk_results = store.search([1.0, 0.0], k=5, chunk_type="chunk")
    assert all(r.metadata["chunk_type"] == "chunk" for r in chunk_results)
    doc_results = store.search([1.0, 0.0], k=5, chunk_type="document")
    assert all(r.metadata["chunk_type"] == "document" for r in doc_results)


def test_delete_by_file(store: SQLiteVectorStore) -> None:
    """delete_by_file removes all rows with that file_path."""
    store.upsert(
        ids=["a1", "a2", "b1"],
        vectors=[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        metadata=[
            {"file_path": "/a.py", "chunk_type": "chunk", "char_hash": "ha"},
            {"file_path": "/a.py", "chunk_type": "document", "char_hash": "ha"},
            {"file_path": "/b.py", "chunk_type": "chunk", "char_hash": "hb"},
        ],
    )
    store.delete_by_file("/a.py")
    results = store.search([1.0, 0.0], k=10)
    file_paths = {r.metadata["file_path"] for r in results}
    assert "/a.py" not in file_paths
    assert "/b.py" in file_paths


def test_all_file_hashes(store: SQLiteVectorStore) -> None:
    """all_file_hashes returns one hash per file."""
    store.upsert(
        ids=["x1", "x2", "y1"],
        vectors=[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        metadata=[
            {"file_path": "/x.py", "chunk_type": "chunk", "char_hash": "hashX"},
            {"file_path": "/x.py", "chunk_type": "document", "char_hash": "hashX"},
            {"file_path": "/y.py", "chunk_type": "chunk", "char_hash": "hashY"},
        ],
    )
    hashes = store.all_file_hashes()
    assert hashes == {"/x.py": "hashX", "/y.py": "hashY"}


def test_clear(store: SQLiteVectorStore) -> None:
    """clear() removes all rows from the chunks table."""
    store.upsert(
        ids=["z1"],
        vectors=[[1.0, 0.0]],
        metadata=[{"file_path": "/z.py", "chunk_type": "chunk", "char_hash": "hz"}],
    )
    store.clear()
    assert store.all_file_hashes() == {}
    assert store.search([1.0, 0.0], k=5) == []


def test_wal_mode_enabled(tmp_path: Path) -> None:
    """WAL journal mode is enabled after construction."""
    import sqlite3

    db_path = tmp_path / "wal_test.db"
    SQLiteVectorStore(db_path)
    conn = sqlite3.connect(str(db_path))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode == "wal"


def test_upsert_replaces_existing(store: SQLiteVectorStore) -> None:
    """Upserting with the same id replaces the old row."""
    store.upsert(
        ids=["dup"],
        vectors=[[1.0, 0.0]],
        metadata=[{"file_path": "/old.py", "chunk_type": "chunk", "char_hash": "old"}],
    )
    store.upsert(
        ids=["dup"],
        vectors=[[0.0, 1.0]],
        metadata=[{"file_path": "/new.py", "chunk_type": "chunk", "char_hash": "new"}],
    )
    hashes = store.all_file_hashes()
    assert "/old.py" not in hashes
    assert hashes.get("/new.py") == "new"


def test_empty_search(store: SQLiteVectorStore) -> None:
    """Searching an empty store returns an empty list."""
    results = store.search([1.0, 0.0, 0.0], k=5)
    assert results == []


# ---------------------------------------------------------------------------
# index_roots helpers
# ---------------------------------------------------------------------------


def test_upsert_and_get_roots(store: SQLiteVectorStore) -> None:
    store.upsert_root("/some/path", "My Label", "2024-01-01T00:00:00Z")
    roots = store.get_all_roots()
    assert len(roots) == 1
    assert roots[0]["path"] == "/some/path"
    assert roots[0]["label"] == "My Label"


def test_delete_root(store: SQLiteVectorStore) -> None:
    store.upsert_root("/path/a", None, "2024-01-01T00:00:00Z")
    store.upsert_root("/path/b", None, "2024-01-01T00:00:00Z")
    store.delete_root("/path/a")
    paths = [r["path"] for r in store.get_all_roots()]
    assert "/path/a" not in paths
    assert "/path/b" in paths


# ---------------------------------------------------------------------------
# meta helpers
# ---------------------------------------------------------------------------


def test_get_set_meta(store: SQLiteVectorStore) -> None:
    assert store.get_meta("schema_version") is not None  # set at construction
    store.set_meta("my_key", "my_value")
    assert store.get_meta("my_key") == "my_value"


def test_get_meta_missing(store: SQLiteVectorStore) -> None:
    assert store.get_meta("nonexistent") is None
