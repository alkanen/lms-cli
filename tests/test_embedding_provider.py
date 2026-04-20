"""Tests for ai_cli.core.embedding_provider."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_cli.core.embedding_provider import (
    _BATCH_SIZE_WARN_THRESHOLD,
    OpenAIEmbeddingProvider,
    _normalize_text,
)


def _run(coro: object) -> object:
    """Run a coroutine synchronously."""
    return asyncio.run(coro)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------


def test_normalize_text_nfc():
    """NFC normalization is applied."""
    # é as decomposed form (e + combining acute accent).
    decomposed = "e\u0301"
    result = _normalize_text(decomposed)
    assert result == "\xe9"  # precomposed é


def test_normalize_text_whitespace():
    """Multiple whitespace chars are collapsed to single space."""
    assert _normalize_text("hello   world\t\n") == "hello world"


def test_normalize_text_control_chars():
    """Control characters act as word separators (replaced with space, then collapsed)."""
    assert _normalize_text("hello\x00world\x1ftest") == "hello world test"


def test_normalize_text_curly_quotes():
    """Fancy quotes are replaced with straight ASCII equivalents."""
    assert _normalize_text("\u201cHello\u201d") == '"Hello"'
    assert _normalize_text("\u2018it\u2019s") == "'it's"


def test_normalize_text_dashes():
    """Em/en dashes are replaced with hyphens."""
    assert _normalize_text("a\u2013b\u2014c") == "a-b-c"


def test_normalize_text_empty():
    assert _normalize_text("") == ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding_response(vectors: list[list[float]]) -> MagicMock:
    """Build a fake openai embeddings response."""
    response = MagicMock()
    response.data = [MagicMock(embedding=v) for v in vectors]
    return response


# ---------------------------------------------------------------------------
# embed() — async path
# ---------------------------------------------------------------------------


def test_embed_basic():
    """embed() returns a list of vectors in order."""
    provider = OpenAIEmbeddingProvider(model="test-model")

    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(
        return_value=_make_embedding_response([[1.0, 2.0], [3.0, 4.0]])
    )
    provider._async_client = mock_client

    result = _run(provider.embed(["hello", "world"]))
    assert result == [[1.0, 2.0], [3.0, 4.0]]


def test_embed_batches_at_batch_size():
    """embed() splits requests according to batch_size and concatenates results."""
    batch_size = 10
    provider = OpenAIEmbeddingProvider(model="test-model", batch_size=batch_size)

    texts = ["text"] * (batch_size + 5)  # just over the batch boundary
    batch1_vecs = [[float(i), 0.0] for i in range(batch_size)]
    batch2_vecs = [[float(i + batch_size), 0.0] for i in range(5)]

    call_count = 0
    responses = [
        _make_embedding_response(batch1_vecs),
        _make_embedding_response(batch2_vecs),
    ]

    async def _fake_create(model: str, input: list[str]) -> MagicMock:
        nonlocal call_count
        resp = responses[call_count]
        call_count += 1
        return resp

    mock_client = AsyncMock()
    mock_client.embeddings.create = _fake_create
    provider._async_client = mock_client

    result = _run(provider.embed(texts))
    assert call_count == 2
    assert len(result) == batch_size + 5
    assert result[0] == batch1_vecs[0]
    assert result[batch_size] == batch2_vecs[0]


def test_embed_caches_dimension():
    """dimension is set from the first response and cached."""
    provider = OpenAIEmbeddingProvider(model="test-model")

    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(
        return_value=_make_embedding_response([[0.1, 0.2, 0.3]])
    )
    provider._async_client = mock_client

    _run(provider.embed(["test"]))
    assert provider.dimension == 3


def test_embed_empty_input():
    """embed() with empty list returns empty list without making API calls."""
    provider = OpenAIEmbeddingProvider(model="test-model")
    result = _run(provider.embed([]))
    assert result == []


# ---------------------------------------------------------------------------
# embed_sync() — sync path
# ---------------------------------------------------------------------------


def test_embed_sync_basic():
    """embed_sync() returns a list of vectors in order."""
    provider = OpenAIEmbeddingProvider(model="test-model")

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_embedding_response(
        [[1.0, 0.0], [0.0, 1.0]]
    )
    provider._sync_client = mock_client

    result = provider.embed_sync(["foo", "bar"])
    assert result == [[1.0, 0.0], [0.0, 1.0]]


def test_embed_sync_batches_at_batch_size():
    """embed_sync() splits requests according to batch_size."""
    batch_size = 10
    provider = OpenAIEmbeddingProvider(model="test-model", batch_size=batch_size)

    texts = ["t"] * (batch_size + 3)
    batch1_vecs = [[float(i)] for i in range(batch_size)]
    batch2_vecs = [[float(i + batch_size)] for i in range(3)]

    call_num = [0]
    responses = [
        _make_embedding_response(batch1_vecs),
        _make_embedding_response(batch2_vecs),
    ]

    def _fake_create(**kwargs: Any) -> MagicMock:
        resp = responses[call_num[0]]
        call_num[0] += 1
        return resp

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = _fake_create
    provider._sync_client = mock_client

    result = provider.embed_sync(texts)
    assert call_num[0] == 2
    assert len(result) == batch_size + 3


def test_embed_sync_caches_dimension():
    """embed_sync() caches the dimension from the first response."""
    provider = OpenAIEmbeddingProvider(model="test-model")
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _make_embedding_response([[0.1, 0.2]])
    provider._sync_client = mock_client

    provider.embed_sync(["query"])
    assert provider.dimension == 2


def test_embed_sync_empty_input():
    provider = OpenAIEmbeddingProvider(model="test-model")
    assert provider.embed_sync([]) == []


# ---------------------------------------------------------------------------
# dimension property
# ---------------------------------------------------------------------------


def test_dimension_raises_before_embed():
    """Accessing dimension before any embed call raises RuntimeError."""
    provider = OpenAIEmbeddingProvider(model="test-model")
    with pytest.raises(RuntimeError, match="dimension is not yet known"):
        _ = provider.dimension


# ---------------------------------------------------------------------------
# model property
# ---------------------------------------------------------------------------


def test_model_property():
    provider = OpenAIEmbeddingProvider(model="nomic-embed-text")
    assert provider.model == "nomic-embed-text"


# ---------------------------------------------------------------------------
# batch_size warning
# ---------------------------------------------------------------------------


def test_large_batch_size_emits_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A batch_size above the threshold logs a WARNING at construction time."""
    import logging

    oversized = _BATCH_SIZE_WARN_THRESHOLD + 1
    with caplog.at_level(logging.WARNING, logger="ai_cli.core.embedding_provider"):
        OpenAIEmbeddingProvider(model="test-model", batch_size=oversized)
    assert any("batch_size" in r.message for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_normal_batch_size_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A batch_size at or below the threshold does not log a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="ai_cli.core.embedding_provider"):
        OpenAIEmbeddingProvider(
            model="test-model", batch_size=_BATCH_SIZE_WARN_THRESHOLD
        )
    assert not caplog.records
