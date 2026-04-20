"""
vector_store.py — VectorStore ABC and SQLiteVectorStore implementation.

Stores embeddings as float32 blobs in a WAL-mode SQLite database.
Cosine similarity search is implemented via numpy dot product on L2-normalised
vectors (normalisation happens at upsert time, so search is a plain matrix
multiply).

Optional dependency: numpy >= 1.24
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single result from a vector similarity search."""

    id: str
    score: float  # cosine similarity in [-1, 1]
    metadata: dict  # all non-vector columns from the chunks table


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    """Abstract base for vector storage backends."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict],
    ) -> None:
        """Insert or replace rows.  *ids*, *vectors*, and *metadata* are parallel lists."""

    @abstractmethod
    def delete_by_file(self, file_path: str) -> None:
        """Remove all chunks whose ``file_path`` matches exactly."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        k: int = 10,
        chunk_type: str | None = None,
        path_glob: str | None = None,
    ) -> list[SearchResult]:
        """Return the *k* most similar results by cosine similarity.

        When *chunk_type* is given only rows with that value are considered.
        When *path_glob* is given only rows whose ``file_path`` matches the
        pattern are included *before* the top-k selection, so exactly *k*
        matching results are returned when enough exist.

        .. note::
            The ``SQLiteVectorStore`` implementation evaluates *path_glob*
            using SQLite ``GLOB`` semantics (case-sensitive, ``*`` and ``?``
            wildcards, POSIX-style path separators).  This differs from
            :func:`fnmatch.fnmatch`, which is case-insensitive on some
            platforms.  Use forward slashes in glob patterns for portability.
        """

    @abstractmethod
    def all_file_hashes(self) -> dict[str, str]:
        """Return ``{file_path: char_hash}`` for every file that has any chunk."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all rows from the chunks table."""

    def close(self) -> None:  # noqa: B027
        """Release any resources held by this store (e.g. DB file handles).

        The default implementation is a no-op.  Subclasses that hold
        long-lived connections should override this and call it on teardown.
        """


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = "1"

_DDL = """\
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    file_path    TEXT NOT NULL,
    chunk_type   TEXT NOT NULL,
    start_line   INTEGER,
    end_line     INTEGER,
    symbol_name  TEXT,
    symbol_kind  TEXT,
    char_hash    TEXT NOT NULL,
    indexed_at   TEXT NOT NULL,
    vector       BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);

CREATE TABLE IF NOT EXISTS index_roots (
    path         TEXT PRIMARY KEY,
    label        TEXT,
    added_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SQLiteVectorStore(VectorStore):
    """SQLite-backed vector store using float32 blobs.

    All vectors are L2-normalised row-wise on upsert so that cosine similarity
    reduces to a dot product at search time.  No separate norms cache is needed.

    The database is opened in WAL mode for atomic incremental updates.

    Thread safety: a single ``threading.Lock`` serialises all connection
    operations so that the background indexing thread and the main thread
    (search / reads) can safely share one store instance.
    """

    def __init__(self, db_path: Path) -> None:
        if not _HAS_NUMPY:
            raise ImportError(
                "numpy is required for SQLiteVectorStore. "
                "Install it with: pip install numpy"
            )

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_DDL)
        self._conn.commit()

        # Ensure schema version is stored.
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", _SCHEMA_VERSION)

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict],
    ) -> None:
        """Insert or replace rows.

        Vectors are L2-normalised before storage.  All rows in one call are
        written in a single transaction.
        """
        if not ids:
            return

        rows = []
        for chunk_id, vec, meta in zip(ids, vectors, metadata, strict=True):
            file_path = meta.get("file_path")
            char_hash = meta.get("char_hash")
            if not file_path or not char_hash:
                raise ValueError(
                    "upsert requires non-empty 'file_path' and 'char_hash' in metadata "
                    f"for chunk {chunk_id!r}; "
                    f"got file_path={file_path!r}, char_hash={char_hash!r}"
                )
            arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            blob = arr.tobytes()
            rows.append(
                (
                    chunk_id,
                    file_path,
                    meta.get("chunk_type", "chunk"),
                    meta.get("start_line"),
                    meta.get("end_line"),
                    meta.get("symbol_name"),
                    meta.get("symbol_kind"),
                    char_hash,
                    meta.get("indexed_at", _now_iso()),
                    blob,
                )
            )

        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                  (id, file_path, chunk_type, start_line, end_line,
                   symbol_name, symbol_kind, char_hash, indexed_at, vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def delete_by_file(self, file_path: str) -> None:
        """Remove all chunks for the given file path."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))

    def search(
        self,
        query_vector: list[float],
        k: int = 10,
        chunk_type: str | None = None,
        path_glob: str | None = None,
    ) -> list[SearchResult]:
        """Return the top-*k* results by cosine similarity.

        ``chunk_type`` and ``path_glob`` are applied in SQL so that vector
        BLOBs for non-matching rows are never loaded into Python.
        """
        q = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        if k <= 0:
            return []

        # Build WHERE clause from optional filters so the DB does the
        # pre-filtering and we never deserialise vectors we won't use.
        conditions: list[str] = []
        params: list[object] = []
        if chunk_type is not None:
            conditions.append("chunk_type = ?")
            params.append(chunk_type)
        if path_glob is not None:
            # SQLite GLOB uses shell-style wildcards (* and ?) case-sensitively,
            # which matches fnmatch semantics for absolute paths on Unix.
            conditions.append("file_path GLOB ?")
            params.append(path_glob)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = (
            "SELECT id, file_path, chunk_type, start_line, end_line,"
            " symbol_name, symbol_kind, char_hash, indexed_at, vector"
            f" FROM chunks{where}"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        if not rows:
            return []

        # Build matrix of stored vectors.
        dim = len(q)
        ids = []
        meta_list = []
        vec_list = []

        for row in rows:
            (
                rid,
                file_path,
                ct,
                start_line,
                end_line,
                symbol_name,
                symbol_kind,
                char_hash,
                indexed_at,
                vec_blob,
            ) = row
            arr = np.frombuffer(vec_blob, dtype=np.float32)
            if len(arr) != dim:
                logger.warning(
                    "Dimension mismatch for chunk %s: expected %d, got %d — skipping",
                    rid,
                    dim,
                    len(arr),
                )
                continue
            ids.append(rid)
            vec_list.append(arr)
            meta_list.append(
                {
                    "file_path": file_path,
                    "chunk_type": ct,
                    "start_line": start_line,
                    "end_line": end_line,
                    "symbol_name": symbol_name,
                    "symbol_kind": symbol_kind,
                    "char_hash": char_hash,
                    "indexed_at": indexed_at,
                }
            )

        if not vec_list:
            return []

        matrix = np.stack(vec_list)  # shape (N, dim)
        scores = matrix @ q  # cosine similarities (vectors already normalised)

        k = min(k, len(ids))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            SearchResult(id=ids[i], score=float(scores[i]), metadata=meta_list[i])
            for i in top_indices
        ]

    def all_file_hashes(self) -> dict[str, str]:
        """Return ``{file_path: char_hash}`` for every file with stored chunks.

        Picks the ``char_hash`` from the row with the most-recent ``indexed_at``
        so the result is deterministic even if (due to a bug) a file has rows
        with differing hashes.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT c.file_path, MIN(c.char_hash) AS char_hash
                FROM chunks AS c
                JOIN (
                    SELECT file_path, MAX(indexed_at) AS max_indexed_at
                    FROM chunks
                    GROUP BY file_path
                ) AS latest
                ON  c.file_path  = latest.file_path
                AND c.indexed_at = latest.max_indexed_at
                GROUP BY c.file_path
                """
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def clear(self) -> None:
        """Delete all rows from the chunks table."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM chunks")

    # ------------------------------------------------------------------
    # index_roots table helpers
    # ------------------------------------------------------------------

    def upsert_root(self, path: str, label: str | None, added_at: str) -> None:
        """Insert or replace a root entry."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO index_roots (path, label, added_at) VALUES (?, ?, ?)",
                (path, label, added_at),
            )

    def delete_root(self, path: str) -> None:
        """Remove a root entry by path."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM index_roots WHERE path = ?", (path,))

    def get_all_roots(self) -> list[dict]:
        """Return all root entries as a list of dicts."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, label, added_at FROM index_roots"
            ).fetchall()
        return [{"path": r[0], "label": r[1], "added_at": r[2]} for r in rows]

    # ------------------------------------------------------------------
    # meta table helpers
    # ------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        """Return the value for *key* from the meta table, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Insert or replace a meta entry."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, value),
            )

    def close(self) -> None:
        """Close the underlying SQLite connection and release the file handle."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SQLiteVectorStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
