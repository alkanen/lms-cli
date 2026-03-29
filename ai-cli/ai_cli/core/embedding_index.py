"""
embedding_index.py — EmbeddingIndex orchestration layer.

Ties together chunking, embedding, and storage.  Handles incremental
re-indexing, root management, access control, and search.

Optional dependencies: xxhash (for hashing), numpy (via vector_store).
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ai_cli.core.chunker import make_chunker
from ai_cli.core.vector_store import SearchResult, SQLiteVectorStore, VectorStore

if TYPE_CHECKING:
    from ai_cli.core.embedding_provider import EmbeddingProvider
    from ai_cli.core.workspace import Workspace

logger = logging.getLogger(__name__)

# Keys that have already triggered a "not yet implemented" warning this
# process lifetime.  Prevents spamming logs once per indexed file.
_SUMMARY_WARNED_KEYS: set[str] = set()

try:
    import xxhash as _xxhash

    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False

try:
    import numpy as _np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Binary / large-file detection
# ---------------------------------------------------------------------------

# Files larger than this (in MB) are skipped with an INFO log.
# Override via the 'max_file_size_mb' key in the embedding config.
_DEFAULT_MAX_FILE_SIZE_MB: int = 10

# Number of bytes sampled at the start of a file for binary detection.
_BINARY_SAMPLE_BYTES: int = 8192

# If the ratio of NUL bytes in the sample exceeds this the file is treated as
# binary and skipped.  A value of 0.01 (1%) allows a handful of odd bytes in
# otherwise-text files while reliably rejecting compiled/compressed content.
_BINARY_NUL_RATIO: float = 0.01


def _is_likely_binary(content: bytes) -> bool:
    """Return True if *content* looks like a binary (non-text) file."""
    sample = content[:_BINARY_SAMPLE_BYTES]
    if not sample:
        return False
    return sample.count(b"\x00") / len(sample) > _BINARY_NUL_RATIO


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class IndexRoot:
    """A directory registered for indexing."""

    path: Path
    label: str | None
    added_at: str  # ISO-8601 UTC


@dataclass
class IndexStats:
    """Summary returned by ``EmbeddingIndex.index()``."""

    files_indexed: int = 0
    files_skipped: int = 0
    files_deleted: int = 0
    chunks_added: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class _FileEntry:
    """Collected file info used by the two-phase indexing pipeline."""

    path: Path
    char_hash: str
    needs_index: bool
    root: Path


# ---------------------------------------------------------------------------
# Chunk ID helper
# ---------------------------------------------------------------------------


def _chunk_id(
    file_path: str,
    chunk_type: str,
    start_line: int | None,
    ordinal: int = 0,
) -> str:
    """Return a stable 16-char hex ID for a chunk.

    Uses ``xxhash64(file_path + '\\x00' + chunk_type + '\\x00' + line_key
    + '\\x00' + ordinal)`` where *line_key* is ``str(start_line)`` for
    chunk-level rows and ``'doc'`` for document-level rows.

    *ordinal* is the 0-based position of the chunk within the file and
    disambiguates multiple chunks that share the same *start_line* (e.g.
    when a very long line is hard-capped into several fixed-size pieces).
    """
    if not _HAS_XXHASH:
        # Fallback: use a simple hash when xxhash is not available.
        import hashlib

        line_key = str(start_line) if start_line is not None else "doc"
        key = f"{file_path}\x00{chunk_type}\x00{line_key}\x00{ordinal}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    line_key = str(start_line) if start_line is not None else "doc"
    key = f"{file_path}\x00{chunk_type}\x00{line_key}\x00{ordinal}"
    return _xxhash.xxh64(key.encode()).hexdigest()


def _file_hash(content: bytes) -> str:
    """Return the xxhash64 hex digest of *content*."""
    if not _HAS_XXHASH:
        import hashlib

        return hashlib.sha256(content).hexdigest()[:16]
    return _xxhash.xxh64(content).hexdigest()


# ---------------------------------------------------------------------------
# EmbeddingIndex
# ---------------------------------------------------------------------------


class EmbeddingIndex:
    """Orchestrates chunking, embedding, storage, and semantic search.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    provider:
        Embedding model provider.
    store:
        Vector storage backend.
    config:
        Embedding configuration dict (from ``ConfigManager.get_embedding_config()``).
    workspace:
        The active workspace; used for ignore-rule filtering during indexing.
    llm_client:
        LLM client used for document-level summary embeddings (optional).
        Only required when ``document_embedding.strategy`` is ``"summary"``.
    """

    def __init__(
        self,
        db_path: Path,
        provider: EmbeddingProvider,
        store: VectorStore,
        config: dict,
        workspace: Workspace,
        llm_client: object = None,
    ) -> None:
        self._db_path = db_path
        self._provider = provider
        self._store = store
        self._config = config
        self._workspace = workspace
        self._llm_client = llm_client

    # ------------------------------------------------------------------
    # Root management
    # ------------------------------------------------------------------

    def add_root(self, path: Path, label: str | None = None) -> None:
        """Register an external directory for indexing.

        Resolves *path* to an absolute canonical form and validates that it
        exists and is a directory before persisting.  Raises ``ValueError``
        with a descriptive message when the path is invalid.

        Persists the entry to the ``index_roots`` table so it survives
        process restarts.
        """
        if not isinstance(self._store, SQLiteVectorStore):
            raise TypeError("Root management requires SQLiteVectorStore")
        resolved = path.resolve()
        if resolved == resolved.parent:
            raise ValueError(
                f"Cannot register the filesystem root '{resolved}' as an index root."
            )
        if not resolved.exists():
            raise ValueError(f"Index root does not exist: {path}")
        if not resolved.is_dir():
            raise ValueError(f"Index root is not a directory: {path}")
        added_at = datetime.now(timezone.utc).isoformat()
        self._store.upsert_root(resolved.as_posix(), label, added_at)
        logger.debug("Added index root: %s (label=%r)", resolved, label)

    def update_root_label(self, path: Path, label: str | None) -> None:
        """Update the label of an already-registered root, preserving ``added_at``.

        Raises ``ValueError`` if *path* is not currently registered.
        """
        if not isinstance(self._store, SQLiteVectorStore):
            raise TypeError("Root management requires SQLiteVectorStore")
        resolved = path.resolve()
        path_str = resolved.as_posix()
        existing = {r.path: r for r in self.roots}
        if resolved not in existing:
            raise ValueError(f"'{resolved}' is not a registered index root.")
        root = existing[resolved]
        self._store.upsert_root(path_str, label, root.added_at)
        logger.debug("Updated label for index root: %s (label=%r)", resolved, label)

    def remove_root(self, path: Path) -> None:
        """Unregister a root and delete all its chunks from the index.

        Raises ``ValueError`` if *path* is not a currently-registered root so
        that callers cannot accidentally wipe arbitrary filesystem prefixes.
        """
        if not isinstance(self._store, SQLiteVectorStore):
            raise TypeError("Root management requires SQLiteVectorStore")
        resolved = path.resolve()
        path_str = resolved.as_posix()

        # Validate that the resolved path is an actual registered root.
        # Stored paths are POSIX strings (written by add_root/update_root_label).
        registered = {r["path"] for r in self._store.get_all_roots()}
        if path_str not in registered:
            raise ValueError(
                f"'{resolved}' is not a registered index root. "
                "Use add_root() first, or check the path."
            )

        # Delete all chunks under this root.
        known = self._store.all_file_hashes()
        for file_path in list(known.keys()):
            try:
                Path(file_path).relative_to(resolved)
                self._store.delete_by_file(file_path)
            except ValueError:
                pass
        self._store.delete_root(path_str)  # path_str is already as_posix()
        logger.debug("Removed index root: %s", resolved)

    @property
    def roots(self) -> list[IndexRoot]:
        """All known index roots, loaded from the database."""
        if not isinstance(self._store, SQLiteVectorStore):
            return []
        rows = self._store.get_all_roots()
        return [
            IndexRoot(
                path=Path(r["path"]),
                label=r["label"],
                added_at=r["added_at"],
            )
            for r in rows
        ]

    @property
    def db_path(self) -> Path:
        """Path to the underlying SQLite database file."""
        return self._db_path

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def is_indexed_path(self, path: Path) -> bool:
        """Return True if *path* is under any external (non-workspace) index root.

        The workspace root is excluded — that is handled by
        ``Workspace.contains()``.  Only paths under explicitly added external
        roots return True here.
        """
        abs_path = path.resolve()
        for root in self.roots:
            if root.path == self._workspace.root:
                continue
            try:
                abs_path.relative_to(root.path)
                return True
            except ValueError:
                continue
        return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        level: str = "chunk",
        path_glob: str | None = None,
    ) -> list[SearchResult]:
        """Embed *query* synchronously and return ranked results.

        Parameters
        ----------
        query:
            Natural language or code snippet to search for.
        k:
            Maximum number of results to return.
        level:
            ``"chunk"`` (default) | ``"document"`` | ``"both"``
        path_glob:
            Restrict results to files matching this glob pattern.
        """
        query_vectors = self._provider.embed_sync([query])
        if not query_vectors:
            return []
        query_vec = query_vectors[0]

        chunk_type: str | None
        if level == "chunk":
            chunk_type = "chunk"
        elif level == "document":
            chunk_type = "document"
        else:
            chunk_type = None  # both

        return self._store.search(
            query_vec, k=k, chunk_type=chunk_type, path_glob=path_glob or None
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index(
        self,
        roots: list[Path] | None = None,
        *,
        incremental: bool = True,
        cancelled: threading.Event | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
        on_chunk_progress: Callable[[int, int], None] | None = None,
    ) -> IndexStats:
        """Walk each root, chunk and embed changed files.

        Parameters
        ----------
        roots:
            Specific roots to index.  ``None`` means all known roots.
        incremental:
            When ``True`` (default), skip files whose content hash matches
            the stored hash.  When ``False``, re-embed every file.
        cancelled:
            Optional ``threading.Event``; when set the indexing loop stops
            cleanly after the current file.  Already-indexed files are
            preserved so the next incremental run resumes where it left off.
        on_progress:
            Optional callback ``(current, total, file_path)`` called before
            each file is processed.  ``current`` is 1-based (1 = first file).
        on_chunk_progress:
            Optional callback ``(chunks_done, chunks_total)`` forwarded to
            the embedding provider so callers can track per-file chunk
            progress.  Called with ``(0, total_chunks)`` at the start of
            each file, then after every embedding batch.

        Returns
        -------
        IndexStats
            Summary of the run: files indexed/skipped/deleted and chunks added.
        """
        import time

        t0 = time.monotonic()

        target_roots: list[Path]
        if roots is None:
            known = self.roots
            if not known:
                # No external roots registered — default to the workspace root.
                logger.info(
                    "No index roots configured; defaulting to workspace root: %s",
                    self._workspace.root,
                )
                target_roots = [self._workspace.root]
            else:
                target_roots = [r.path for r in known]
        else:
            target_roots = list(roots)

        # Always load previous file hashes so Phase 3 can find files that were
        # deleted since the last index run.  The `incremental` flag only
        # controls whether hashes are used to *skip* re-embedding; it does not
        # prevent stale-chunk cleanup.
        known_hashes = self._store.all_file_hashes()

        # ------------------------------------------------------------------
        # Phase 1: collect all files (fast — no embedding yet)
        # ------------------------------------------------------------------
        logger.info("Phase 1: scanning %d root(s) for files", len(target_roots))
        all_entries: list[_FileEntry] = []
        for root in target_roots:
            if not root.is_dir():
                logger.warning(
                    "Index root does not exist or is not a directory: %s", root
                )
                continue
            all_entries.extend(
                self._collect_root_files(root, known_hashes, incremental, cancelled)
            )

        n_to_index = sum(1 for e in all_entries if e.needs_index)
        logger.info(
            "Phase 1 complete: %d files found — %d to embed, %d unchanged",
            len(all_entries),
            n_to_index,
            len(all_entries) - n_to_index,
        )

        # ------------------------------------------------------------------
        # Phase 2: embed files that need it, one at a time
        # ------------------------------------------------------------------
        stats = IndexStats()
        total = len(all_entries)

        for i, entry in enumerate(all_entries):
            if cancelled is not None and cancelled.is_set():
                logger.info("Indexing cancelled at file %d/%d.", i, total)
                break

            if on_progress is not None:
                on_progress(i + 1, total, str(entry.path))

            if not entry.needs_index:
                stats.files_skipped += 1
                continue

            logger.info("[%d/%d] %s", i + 1, total, entry.path)
            await self._embed_and_upsert_file(
                entry.path, entry.char_hash, stats, on_chunk_progress, cancelled
            )

        # ------------------------------------------------------------------
        # Phase 3: remove stale chunks for deleted files
        # ------------------------------------------------------------------
        seen_per_root: dict[str, set[str]] = {}
        for entry in all_entries:
            key = entry.root.as_posix()
            seen_per_root.setdefault(key, set()).add(entry.path.as_posix())

        for root in target_roots:
            if not root.is_dir():
                logger.warning(
                    "Skipping stale-chunk cleanup for unavailable root: %s", root
                )
                continue
            root_str = root.as_posix()
            seen = seen_per_root.get(root_str, set())
            for file_path_str in list(known_hashes.keys()):
                try:
                    Path(file_path_str).relative_to(root)
                except ValueError:
                    continue
                if file_path_str not in seen:
                    self._store.delete_by_file(file_path_str)
                    stats.files_deleted += 1
                    logger.debug("Removed deleted file from index: %s", file_path_str)

        stats.elapsed_seconds = time.monotonic() - t0
        logger.info(
            "Index complete: %d indexed, %d skipped, %d deleted, %d chunks in %.1fs",
            stats.files_indexed,
            stats.files_skipped,
            stats.files_deleted,
            stats.chunks_added,
            stats.elapsed_seconds,
        )
        return stats

    def _collect_root_files(
        self,
        root: Path,
        known_hashes: dict[str, str],
        incremental: bool,
        cancelled: threading.Event | None = None,
    ) -> list[_FileEntry]:
        """Walk *root* and return all indexable files with their hashes.

        This is the fast first phase: it reads file content for hashing but
        does not call the embedding provider.  When *cancelled* is set the
        walk stops after the current file and returns whatever was collected
        so far.
        """
        entries: list[_FileEntry] = []
        is_workspace_root = root == self._workspace.root
        _raw_mb = self._config.get("max_file_size_mb", _DEFAULT_MAX_FILE_SIZE_MB)
        try:
            _max_mb = (
                _DEFAULT_MAX_FILE_SIZE_MB if _raw_mb in (None, "") else int(_raw_mb)
            )
        except (TypeError, ValueError):
            logger.warning(
                "Invalid max_file_size_mb value %r; using default %d MB",
                _raw_mb,
                _DEFAULT_MAX_FILE_SIZE_MB,
            )
            _max_mb = _DEFAULT_MAX_FILE_SIZE_MB
        max_bytes = _max_mb * 1024 * 1024

        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            if cancelled is not None and cancelled.is_set():
                break

            current_dir = Path(dirpath)

            # Sort for deterministic ordering.
            dirnames[:] = sorted(dirnames)

            # Prune ignored directories when walking the workspace root.
            if is_workspace_root:
                dirnames[:] = [
                    d
                    for d in dirnames
                    if not self._workspace.is_ignored(current_dir / d, is_dir=True)
                ]

            for filename in sorted(filenames):
                if cancelled is not None and cancelled.is_set():
                    return entries

                file_path = current_dir / filename

                # Skip ignored files when walking workspace root.
                if is_workspace_root and self._workspace.is_ignored(
                    file_path, is_dir=False
                ):
                    continue

                # Skip files that exceed the size cap before reading content.
                try:
                    file_size = file_path.stat().st_size
                except OSError:
                    file_size = 0
                if file_size > max_bytes:
                    logger.info(
                        "Skipping %s — %.1f MB exceeds max_file_size_mb=%d",
                        file_path,
                        file_size / (1024 * 1024),
                        max_bytes // (1024 * 1024),
                    )
                    continue

                try:
                    content = file_path.read_bytes()
                except OSError as exc:
                    logger.warning("Cannot read %s: %s", file_path, exc)
                    continue

                if _is_likely_binary(content):
                    logger.debug("Skipping %s — likely binary file", file_path)
                    continue

                char_hash = _file_hash(content)
                needs_index = not (
                    incremental and known_hashes.get(file_path.as_posix()) == char_hash
                )
                entries.append(
                    _FileEntry(
                        path=file_path,
                        char_hash=char_hash,
                        needs_index=needs_index,
                        root=root,
                    )
                )

        return entries

    async def index_file(
        self,
        file_path: Path,
        *,
        incremental: bool = True,
        on_chunk_progress: Callable[[int, int], None] | None = None,
        cancelled: threading.Event | None = None,
    ) -> IndexStats:
        """Index a single file directly, bypassing root scanning.

        The file does not need to be under a registered root.  Useful for
        debugging and for re-indexing one file without scanning the whole tree.

        Parameters
        ----------
        file_path:
            Absolute path to the file to index.
        incremental:
            When ``True`` (default), skip the file if its hash is unchanged.
        on_chunk_progress:
            Optional ``(chunks_done, chunks_total)`` callback for progress.
        cancelled:
            Optional :class:`threading.Event`.  When set the method returns
            early without writing to the DB.
        """
        import time

        t0 = time.monotonic()

        if not file_path.is_file():
            logger.warning("index_file: not a regular file: %s", file_path)
            return IndexStats()

        stats = IndexStats()
        _raw_mb = self._config.get("max_file_size_mb", _DEFAULT_MAX_FILE_SIZE_MB)
        try:
            _max_mb = (
                _DEFAULT_MAX_FILE_SIZE_MB if _raw_mb in (None, "") else int(_raw_mb)
            )
        except (TypeError, ValueError):
            logger.warning(
                "Invalid max_file_size_mb value %r; using default %d MB",
                _raw_mb,
                _DEFAULT_MAX_FILE_SIZE_MB,
            )
            _max_mb = _DEFAULT_MAX_FILE_SIZE_MB
        max_bytes = _max_mb * 1024 * 1024
        try:
            file_size = file_path.stat().st_size
        except OSError:
            file_size = 0
        if file_size > max_bytes:
            logger.info(
                "index_file: skipping %s — %.1f MB exceeds max_file_size_mb=%d",
                file_path.name,
                file_size / (1024 * 1024),
                max_bytes // (1024 * 1024),
            )
            stats.files_skipped += 1
            stats.elapsed_seconds = time.monotonic() - t0
            return stats

        try:
            content = file_path.read_bytes()
        except OSError as exc:
            logger.warning("Cannot read %s: %s", file_path, exc)
            return IndexStats()

        if _is_likely_binary(content):
            logger.info(
                "index_file: skipping %s — detected as likely binary",
                file_path.name,
            )
            stats.files_skipped += 1
            stats.elapsed_seconds = time.monotonic() - t0
            return stats

        char_hash = _file_hash(content)

        if incremental:
            known = self._store.all_file_hashes()
            if known.get(file_path.as_posix()) == char_hash:
                logger.info(
                    "index_file: %s is unchanged (hash %s) — skipped",
                    file_path.name,
                    char_hash,
                )
                stats.files_skipped += 1
                stats.elapsed_seconds = time.monotonic() - t0
                return stats

        # Decode once here and pass through so _embed_and_upsert_file doesn't
        # need to re-read the same file.
        text = content.decode("utf-8", errors="replace")
        file_size_kb = len(content) / 1024
        logger.info(
            "index_file: %s — %.1f KB, hash %s",
            file_path.name,
            file_size_kb,
            char_hash,
        )
        if cancelled is not None and cancelled.is_set():
            return stats

        await self._embed_and_upsert_file(
            file_path, char_hash, stats, on_chunk_progress, cancelled, content=text
        )
        stats.elapsed_seconds = time.monotonic() - t0
        return stats

    async def aclose(self) -> None:
        """Close async resources held by the provider (HTTP client etc.).

        Call after each indexing run before the event loop closes, so that
        the next run creates a fresh HTTP client in the new event loop.
        The vector store connection is intentionally kept open across runs so
        that searches remain available; call ``self._store.close()`` only at
        final application shutdown.
        """
        await self._provider.aclose()

    async def _embed_and_upsert_file(
        self,
        file_path: Path,
        char_hash: str,
        stats: IndexStats,
        on_chunk_progress: Callable[[int, int], None] | None = None,
        cancelled: threading.Event | None = None,
        content: str | None = None,
    ) -> None:
        """Chunk, embed, and upsert a single file.

        *content* may be provided by the caller (e.g. already decoded in
        ``index_file``) to avoid reading the file a second time.
        """
        if content is None:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Cannot read %s: %s", file_path, exc)
                return
        text = content

        chunking_cfg = self._config.get("chunking", {})
        chunker = make_chunker(file_path, chunking_cfg, text=text)
        chunks = chunker.chunk(text, file_path)

        if not chunks:
            logger.warning("No chunks produced for %s — skipped", file_path)
            return

        # Check between chunking and embedding so a Ctrl+C before the first
        # network request is honoured without touching the DB.
        if cancelled is not None and cancelled.is_set():
            return

        logger.info(
            "Embedding %s: %d chunks (strategy: %s)",
            file_path.name,
            len(chunks),
            chunker.__class__.__name__,
        )

        file_path_str = file_path.as_posix()
        indexed_at = datetime.now(timezone.utc).isoformat()

        chunk_texts = [c.text for c in chunks]
        # Signal chunk bar start (0, total) so the bar can reset.
        if on_chunk_progress is not None:
            on_chunk_progress(0, len(chunks))
        chunk_vectors = await self._provider.embed(
            chunk_texts, on_batch=on_chunk_progress
        )

        # Check after embedding completes (each batch is bounded by
        # request_timeout, so cancellation is honoured between batches).
        # Critically, the deletion below has not happened yet, so the
        # previously-stored chunks are still intact if we bail out here.
        if cancelled is not None and cancelled.is_set():
            return

        # Guard against a truncated response from the embedding backend.
        # The zip(..., strict=True) below would raise anyway, but we want to
        # preserve the previous index state rather than deleting it first.
        if len(chunk_vectors) != len(chunks):
            logger.error(
                "_embed_and_upsert_file: embedding backend returned %d vectors "
                "for %d chunks in %s — skipping upsert to preserve previous index",
                len(chunk_vectors),
                len(chunks),
                file_path.name,
            )
            return

        # Delete existing chunks only after embeddings are successfully
        # computed so that a failure or cancellation mid-embed leaves the
        # previous index data intact rather than leaving the file unindexed.
        self._store.delete_by_file(file_path_str)

        ids: list[str] = []
        vectors: list[list[float]] = []
        metadata: list[dict] = []

        for ordinal, (chunk, vec) in enumerate(zip(chunks, chunk_vectors, strict=True)):
            cid = _chunk_id(file_path_str, "chunk", chunk.start_line, ordinal)
            ids.append(cid)
            vectors.append(vec)
            metadata.append(
                {
                    "file_path": file_path_str,
                    "chunk_type": "chunk",
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "symbol_name": chunk.symbol_name,
                    "symbol_kind": chunk.symbol_kind,
                    "char_hash": char_hash,
                    "indexed_at": indexed_at,
                }
            )

        # Document-level embedding.
        doc_vec = _compute_doc_vector(chunk_vectors, file_path, self._config)
        if doc_vec is not None:
            doc_id = _chunk_id(file_path_str, "document", None)
            ids.append(doc_id)
            vectors.append(doc_vec)
            metadata.append(
                {
                    "file_path": file_path_str,
                    "chunk_type": "document",
                    "start_line": None,
                    "end_line": None,
                    "symbol_name": None,
                    "symbol_kind": None,
                    "char_hash": char_hash,
                    "indexed_at": indexed_at,
                }
            )

        self._store.upsert(ids, vectors, metadata)
        stats.files_indexed += 1
        stats.chunks_added += len(ids)
        logger.info(
            "Indexed %s: %d chunk(s)%s",
            file_path.name,
            len(chunks),
            " + doc vector" if doc_vec is not None else "",
        )


# ---------------------------------------------------------------------------
# Document-vector helpers
# ---------------------------------------------------------------------------


def _compute_doc_vector(
    chunk_vectors: list[list[float]],
    file_path: Path,
    config: dict,
) -> list[float] | None:
    """Compute and return the document-level embedding vector, or ``None``.

    Returns ``None`` when ``document_embedding.enabled`` is false/absent,
    when numpy is not installed, when there are no chunk vectors, or when an
    unsupported strategy is configured.

    Supported strategy: ``average`` — element-wise mean of chunk vectors,
    L2-normalised.  The ``summary`` strategy (LLM call) is not yet
    implemented and will return ``None``.
    """
    doc_cfg = cast(dict, config.get("document_embedding", {}) or {})
    if not doc_cfg.get("enabled", False):
        return None

    strategy = doc_cfg.get("strategy", "auto")
    # "auto" maps to "average" until "summary" is implemented.
    if strategy in ("auto", "average"):
        # Warn once per process if summary-specific keys are set but have no
        # effect yet.  _SUMMARY_WARNED_KEYS guards against per-file log spam.
        for key in ("summary_model", "summary_max_tokens", "prose_extensions"):
            if doc_cfg.get(key) is not None and key not in _SUMMARY_WARNED_KEYS:
                _SUMMARY_WARNED_KEYS.add(key)
                logger.warning(
                    "document_embedding.%s is set but the 'summary' strategy is not "
                    "yet implemented — this key has no effect.",
                    key,
                )
    elif strategy == "summary":
        warn_key = "strategy:summary"
        if warn_key not in _SUMMARY_WARNED_KEYS:
            _SUMMARY_WARNED_KEYS.add(warn_key)
            logger.error(
                "document_embedding.strategy is set to 'summary', but this "
                "strategy is not yet implemented. No document-level vectors will be "
                "generated. Use 'auto' or 'average' instead.",
            )
        return None
    else:
        logger.warning(
            "Skipping document vector for %s: unknown strategy %r",
            file_path.name,
            strategy,
        )
        return None

    if not _HAS_NUMPY:
        return None
    if not chunk_vectors:
        return None

    arr = _np.array(chunk_vectors, dtype=_np.float32)
    mean_vec = arr.mean(axis=0)
    norm = _np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return cast("list[float]", mean_vec.tolist())
