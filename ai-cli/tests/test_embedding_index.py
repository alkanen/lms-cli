"""Tests for ai_cli.core.embedding_index."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("numpy")

import ai_cli.core.embedding_index as _ei_mod  # noqa: E402
from ai_cli.core.embedding_index import (  # noqa: E402
    EmbeddingIndex,
    _compute_doc_vector,
    _summarize_document,
)
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
    llm_client: object | None = None,
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
        llm_client=llm_client,
    )


def _make_llm_client(summary: str = "This is a summary.") -> MagicMock:
    """Return a mock LLMClient whose send() yields a text chunk then done."""
    client = MagicMock()
    client.send.return_value = iter(
        [
            {"type": "text", "delta": summary},
            {"type": "done", "stop_reason": "stop", "usage": {}},
        ]
    )
    return client


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


# ---------------------------------------------------------------------------
# _summarize_document unit tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_warned_keys():
    """Reset the process-level warning dedup set between tests."""
    _ei_mod._SUMMARY_WARNED_KEYS.clear()
    yield
    _ei_mod._SUMMARY_WARNED_KEYS.clear()


def test_summarize_document_returns_text(tmp_path: Path) -> None:
    """Returns the summary text from a successful LLM call."""
    client = _make_llm_client("Covers routing and middleware.")
    result = _summarize_document("long doc text", tmp_path / "readme.md", {}, client)
    assert result == "Covers routing and middleware."


def test_summarize_document_none_client(tmp_path: Path) -> None:
    """Returns None immediately when llm_client is None."""
    result = _summarize_document("text", tmp_path / "readme.md", {}, None)
    assert result is None


def test_summarize_document_llm_raises(tmp_path: Path) -> None:
    """Returns None and logs a warning when the LLM call raises."""
    client = MagicMock()
    client.send.side_effect = RuntimeError("connection refused")
    result = _summarize_document("text", tmp_path / "readme.md", {}, client)
    assert result is None


def test_summarize_document_empty_response(tmp_path: Path) -> None:
    """Returns None when the LLM yields no text deltas."""
    client = MagicMock()
    client.send.return_value = iter(
        [
            {"type": "done", "stop_reason": "stop", "usage": {}},
        ]
    )
    result = _summarize_document("text", tmp_path / "readme.md", {}, client)
    assert result is None


def test_summarize_document_truncates_input(tmp_path: Path) -> None:
    """Input is truncated to summary_max_tokens * 4 chars."""
    client = MagicMock()
    client.send.return_value = iter(
        [
            {"type": "text", "delta": "ok"},
            {"type": "done", "stop_reason": "stop", "usage": {}},
        ]
    )
    long_text = "x" * 1000
    config = {"document_embedding": {"summary_max_tokens": 10}}
    _summarize_document(long_text, tmp_path / "f.md", config, client)

    sent_messages = client.send.call_args[0][0]
    user_content = sent_messages[-1]["content"]
    # Extract the text portion after the instruction (everything after the last "\n\n").
    text_portion = user_content.split("\n\n", 1)[1]
    assert len(text_portion) <= 40  # summary_max_tokens=10 → 10*4=40 chars


def test_summarize_document_clamps_negative_max_input_tokens(tmp_path: Path) -> None:
    """A zero or negative summary_max_tokens is clamped to 1 (not passed raw to slice)."""
    client = MagicMock()
    client.send.return_value = iter(
        [
            {"type": "text", "delta": "ok"},
            {"type": "done", "stop_reason": "stop", "usage": {}},
        ]
    )
    config = {"document_embedding": {"summary_max_tokens": -10}}
    _summarize_document("hello world", tmp_path / "f.md", config, client)

    sent_messages = client.send.call_args[0][0]
    user_content = sent_messages[-1]["content"]
    text_portion = user_content.split("\n\n", 1)[1]
    # Clamped to 1 → budget = 4 chars; "hell" is sent, not an empty/reversed slice.
    assert len(text_portion) <= 4


def test_summarize_document_no_api_token_cap(tmp_path: Path) -> None:
    """send() is called without a max_tokens override — reasoning models keep their full budget."""
    client = MagicMock()
    client.send.return_value = iter(
        [
            {"type": "text", "delta": "ok"},
            {"type": "done", "stop_reason": "stop", "usage": {}},
        ]
    )
    _summarize_document("text", tmp_path / "f.md", {}, client)

    _, kwargs = client.send.call_args
    assert "max_tokens" not in kwargs


def test_summarize_document_prompt_includes_word_limit(tmp_path: Path) -> None:
    """User prompt mentions the derived word limit."""
    client = MagicMock()
    client.send.return_value = iter(
        [
            {"type": "text", "delta": "ok"},
            {"type": "done", "stop_reason": "stop", "usage": {}},
        ]
    )
    # chunk_size=400 → summary_response_tokens=100 → max_words=75
    config = {"chunking": {"chunk_size": 400}}
    _summarize_document("text", tmp_path / "f.md", config, client)

    sent_messages = client.send.call_args[0][0]
    user_content = sent_messages[-1]["content"]
    assert "75 words" in user_content


def test_summarize_document_warns_once_for_summary_model(tmp_path: Path) -> None:
    """summary_model warning is recorded in _SUMMARY_WARNED_KEYS after the first call."""
    config = {"document_embedding": {"summary_model": "gpt-4o"}}

    def _fresh_send(*_a, **_kw):
        return iter(
            [
                {"type": "text", "delta": "s"},
                {"type": "done", "stop_reason": "stop", "usage": {}},
            ]
        )

    client = MagicMock()
    client.send.side_effect = _fresh_send

    _summarize_document("t", tmp_path / "f.md", config, client)
    assert "summary_model:override" in _ei_mod._SUMMARY_WARNED_KEYS

    # Second call must NOT add a second entry (set semantics guarantee this, but
    # also confirm send is still called — only the warning is suppressed).
    _summarize_document("t", tmp_path / "f.md", config, client)
    assert client.send.call_count == 2


# ---------------------------------------------------------------------------
# _compute_doc_vector — strategy routing
# ---------------------------------------------------------------------------

_DOC_CFG_BASE = {"enabled": True}


def _avg_config() -> dict:
    return {"document_embedding": {**_DOC_CFG_BASE, "strategy": "average"}}


def _summary_config() -> dict:
    return {"document_embedding": {**_DOC_CFG_BASE, "strategy": "summary"}}


def _auto_config(prose_extensions: list[str] | None = None) -> dict:
    cfg: dict = {**_DOC_CFG_BASE, "strategy": "auto"}
    if prose_extensions is not None:
        cfg["prose_extensions"] = prose_extensions
    return {"document_embedding": cfg}


def test_compute_doc_vector_average_strategy(tmp_path: Path) -> None:
    """average strategy returns the L2-normalised mean of chunk vectors."""
    vecs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    result = _compute_doc_vector(vecs, tmp_path / "f.py", _avg_config())
    assert result is not None
    import math

    assert math.isclose(sum(x * x for x in result), 1.0, abs_tol=1e-5)


def test_compute_doc_vector_summary_calls_llm(tmp_path: Path) -> None:
    """summary strategy calls embed_sync with the LLM-generated text."""
    summary_text = "Describes the HTTP router."
    client = _make_llm_client(summary_text)
    provider = MagicMock()
    provider.embed_sync.return_value = [[0.1, 0.2, 0.3, 0.4]]

    vecs = [[1.0, 0.0, 0.0, 0.0]]
    result = _compute_doc_vector(
        vecs,
        tmp_path / "readme.md",
        _summary_config(),
        text="full doc text",
        llm_client=client,
        provider=provider,
    )
    assert result == [0.1, 0.2, 0.3, 0.4]
    provider.embed_sync.assert_called_once()
    embedded_text = provider.embed_sync.call_args[0][0][0]
    assert embedded_text == summary_text


def test_compute_doc_vector_summary_falls_back_when_no_client(tmp_path: Path) -> None:
    """summary strategy falls back to average when llm_client is None."""
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    provider = MagicMock()
    result = _compute_doc_vector(
        vecs,
        tmp_path / "f.md",
        _summary_config(),
        text="text",
        llm_client=None,
        provider=provider,
    )
    assert result is not None  # average fallback succeeded
    provider.embed_sync.assert_not_called()


def test_compute_doc_vector_summary_falls_back_on_llm_failure(tmp_path: Path) -> None:
    """summary strategy falls back to average when the LLM call raises."""
    client = MagicMock()
    client.send.side_effect = RuntimeError("timeout")
    provider = MagicMock()
    vecs = [[1.0, 0.0], [1.0, 0.0]]

    result = _compute_doc_vector(
        vecs,
        tmp_path / "f.md",
        _summary_config(),
        text="text",
        llm_client=client,
        provider=provider,
    )
    assert result is not None
    provider.embed_sync.assert_not_called()


def test_compute_doc_vector_summary_falls_back_when_embed_sync_empty(
    tmp_path: Path,
) -> None:
    """summary strategy falls back to average when embed_sync returns empty list."""
    client = _make_llm_client("summary text")
    provider = MagicMock()
    provider.embed_sync.return_value = []
    vecs = [[1.0, 0.0], [1.0, 0.0]]

    result = _compute_doc_vector(
        vecs,
        tmp_path / "f.md",
        _summary_config(),
        text="text",
        llm_client=client,
        provider=provider,
    )
    assert result is not None  # average fallback


def test_compute_doc_vector_auto_prose_uses_summary(tmp_path: Path) -> None:
    """auto strategy routes .md files to the summary path."""
    client = _make_llm_client("summary")
    provider = MagicMock()
    provider.embed_sync.return_value = [[0.5, 0.5]]

    _compute_doc_vector(
        [[1.0, 0.0]],
        tmp_path / "notes.md",
        _auto_config(),
        text="text",
        llm_client=client,
        provider=provider,
    )
    provider.embed_sync.assert_called_once()


def test_compute_doc_vector_auto_code_uses_average(tmp_path: Path) -> None:
    """auto strategy routes .py files to the average path (no LLM call)."""
    client = _make_llm_client("summary")
    provider = MagicMock()

    _compute_doc_vector(
        [[1.0, 0.0], [0.0, 1.0]],
        tmp_path / "main.py",
        _auto_config(),
        text="code text",
        llm_client=client,
        provider=provider,
    )
    provider.embed_sync.assert_not_called()
    client.send.assert_not_called()


def test_compute_doc_vector_auto_custom_prose_extensions(tmp_path: Path) -> None:
    """Custom prose_extensions override the default list."""
    client = _make_llm_client("summary")
    provider = MagicMock()
    provider.embed_sync.return_value = [[0.5, 0.5]]

    # .log is in the custom list → summary path.
    _compute_doc_vector(
        [[1.0, 0.0]],
        tmp_path / "app.log",
        _auto_config(prose_extensions=[".log"]),
        text="log text",
        llm_client=client,
        provider=provider,
    )
    provider.embed_sync.assert_called_once()

    provider.reset_mock()
    client.reset_mock()

    # .md is NOT in the custom list → average path.
    _compute_doc_vector(
        [[1.0, 0.0]],
        tmp_path / "readme.md",
        _auto_config(prose_extensions=[".log"]),
        text="md text",
        llm_client=client,
        provider=provider,
    )
    provider.embed_sync.assert_not_called()


def test_compute_doc_vector_auto_prose_extensions_case_insensitive(
    tmp_path: Path,
) -> None:
    """prose_extensions entries are matched case-insensitively."""
    client = _make_llm_client("summary")
    provider = MagicMock()
    provider.embed_sync.return_value = [[0.5, 0.5]]

    # Config uses uppercase ".MD"; file suffix is ".md" — should still match.
    _compute_doc_vector(
        [[1.0, 0.0]],
        tmp_path / "readme.md",
        _auto_config(prose_extensions=[".MD"]),
        text="text",
        llm_client=client,
        provider=provider,
    )
    provider.embed_sync.assert_called_once()


def test_compute_doc_vector_auto_prose_extensions_bare_string(tmp_path: Path) -> None:
    """A bare string for prose_extensions (YAML scalar mistake) is treated as one entry."""
    client = _make_llm_client("summary")
    provider = MagicMock()
    provider.embed_sync.return_value = [[0.5, 0.5]]

    # Simulates `prose_extensions: .md` in YAML (string, not list).
    cfg = {
        "document_embedding": {
            **_DOC_CFG_BASE,
            "strategy": "auto",
            "prose_extensions": ".md",
        }
    }
    _compute_doc_vector(
        [[1.0, 0.0]],
        tmp_path / "readme.md",
        cfg,
        text="text",
        llm_client=client,
        provider=provider,
    )
    provider.embed_sync.assert_called_once()


def test_compute_doc_vector_summary_embed_sync_raises_falls_back(
    tmp_path: Path,
) -> None:
    """embed_sync failure after summarisation falls back to chunk-average."""
    client = _make_llm_client("summary text")
    provider = MagicMock()
    provider.embed_sync.side_effect = RuntimeError("network error")
    vecs = [[1.0, 0.0], [0.0, 1.0]]

    result = _compute_doc_vector(
        vecs,
        tmp_path / "doc.md",
        {"document_embedding": {**_DOC_CFG_BASE, "strategy": "summary"}},
        text="some text",
        llm_client=client,
        provider=provider,
    )
    # Falls back to chunk-average — result is a normalised vector, not None.
    assert result is not None


# ---------------------------------------------------------------------------
# End-to-end: _embed_and_upsert_file stores a doc vector for prose files
# ---------------------------------------------------------------------------


def test_index_file_stores_document_vector_for_prose(
    store: SQLiteVectorStore, tmp_path: Path
) -> None:
    """Indexing a .md file with summary strategy stores a chunk_type=document row."""
    md_file = tmp_path / "notes.md"
    md_file.write_text("# Hello\nThis is a note.\n")

    provider = _make_provider(dimension=4)
    # embed_sync returns a distinct vector so we can identify the doc vector.
    provider.embed_sync.return_value = [[0.1, 0.2, 0.3, 0.4]]

    client = _make_llm_client("A note about hello.")
    config = {
        "document_embedding": {"enabled": True, "strategy": "summary"},
    }
    idx = _make_index(
        store, tmp_path, provider=provider, config=config, llm_client=client
    )
    asyncio.run(idx.index_file(md_file))

    results = store.search(
        query_vector=[0.1, 0.2, 0.3, 0.4],
        k=10,
        chunk_type="document",
    )
    assert any(r.metadata.get("chunk_type") == "document" for r in results)
    provider.embed_sync.assert_called_once()
