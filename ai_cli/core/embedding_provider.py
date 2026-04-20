"""
embedding_provider.py — EmbeddingProvider ABC and OpenAIEmbeddingProvider.

The provider handles text normalization, batching, and async/sync embedding
via the OpenAI-compatible ``/v1/embeddings`` endpoint.

Text normalization is applied before every embed call to ensure consistent
representation across Unicode variants, whitespace, and curly-quote variants.
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Default batch size.  Local servers (LM Studio, Ollama) can struggle with
# large batches; cloud APIs (OpenAI) handle 96+ comfortably.  Keep the
# default conservative so out-of-the-box local usage doesn't hang silently.
_DEFAULT_BATCH_SIZE = 32

# Warn when a configured batch_size exceeds this value.  There is no hard
# enforcement — users who know their backend can handle large batches may
# legitimately set higher values — but values this large are almost always
# a misconfiguration and can cause request timeouts or OOM on local servers.
_BATCH_SIZE_WARN_THRESHOLD = 512

# Default per-request read timeout in seconds.  Without an explicit timeout
# the openai library uses 600 s, which is long enough to look like an
# indefinite hang when a local server stalls on a large batch.
_DEFAULT_REQUEST_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Normalize *text* for embedding: consistent Unicode, whitespace, and quotes.

    Applies:
    - NFC normalization (composite form, e.g. "é" not "e"+"́")
    - Control characters treated as separators (U+0000–U+001F, U+007F → space)
    - Whitespace collapsing (all runs of whitespace → single space)
    - Curly-quote and em/en-dash normalisation to straight ASCII equivalents
    """
    text = unicodedata.normalize("NFC", text)
    # Replace control chars with a space *before* whitespace collapsing so
    # that they act as token separators rather than silent concatenators.
    # (e.g. "hello\x00world" → "hello world", not "helloworld")
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = " ".join(text.split())
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Abstract base class for embedding model backends."""

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        on_batch: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Async bulk embed — used by EmbeddingIndex.index() for indexing.

        Parameters
        ----------
        texts:
            Texts to embed.
        on_batch:
            Optional callback ``(chunks_done, chunks_total)`` invoked after
            each batch completes.  Useful for driving a progress bar.
        """

    @abstractmethod
    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Sync single-query embed — used by EmbeddingIndex.search().

        Must NOT use asyncio.run() internally; uses the synchronous openai
        client under the hood.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the vectors returned by embed() / embed_sync()."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Name of the embedding model in use."""

    async def aclose(self) -> None:  # noqa: B027
        """Close any async resources held by this provider.

        Called after each indexing run so the next run starts with a fresh
        HTTP client bound to the new event loop.  The default implementation
        is a no-op; override in subclasses that hold async clients.
        """


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible implementation
# ---------------------------------------------------------------------------


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that calls any OpenAI-compatible ``/v1/embeddings`` endpoint.

    Works with OpenAI, Ollama, LM Studio, and any other service that exposes
    the same API.

    Parameters
    ----------
    model:
        Name of the embedding model to use (e.g. ``"nomic-embed-text"``).
    base_url:
        Base URL for the API endpoint.  When ``None``, the default OpenAI URL
        is used.
    api_key:
        API key string.  When ``None`` and ``base_url`` is also ``None``, the
        OpenAI SDK reads ``OPENAI_API_KEY`` from the environment as usual.
        When ``None`` and ``base_url`` is set (local server), a placeholder
        ``"local"`` key is injected because local servers require *some* value
        but don't validate it.  To use a custom env var name, set
        ``api_key_env`` in config — ``ConfigManager`` resolves it to an actual
        key before constructing this provider.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        request_timeout: float = _DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._batch_size = max(1, int(batch_size))
        self._request_timeout = float(request_timeout)
        self._dimension: int | None = None
        if self._batch_size > _BATCH_SIZE_WARN_THRESHOLD:
            logger.warning(
                "OpenAIEmbeddingProvider: batch_size=%d is very large and may "
                "cause timeouts or errors on local embedding servers. "
                "Consider lowering it to 32 or fewer for LM Studio / Ollama.",
                self._batch_size,
            )
        # Lazy-initialized clients.  The lock guards _sync_client and
        # _dimension against concurrent access from the main thread (search)
        # and the background indexing thread (asyncio.to_thread).
        self._lock = threading.Lock()
        self._async_client: object | None = None
        self._sync_client: object | None = None

    # ------------------------------------------------------------------
    # Lazy client construction
    # ------------------------------------------------------------------

    def _get_async_client(self) -> object:
        """Return (or create) the async OpenAI client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai package is required. Install it with: pip install openai"
                ) from exc
            kwargs: dict = {"timeout": self._request_timeout}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._api_key:
                kwargs["api_key"] = self._api_key
            elif self._base_url:
                # Local servers (LM Studio, Ollama, etc.) require *some* key
                # but don't validate it.  ConfigManager resolves api_key_env
                # before constructing this provider, so if any key was
                # configured it is already in self._api_key.  Users pointing
                # base_url at a proxy that requires a real key should set
                # api_key or api_key_env in config.
                kwargs["api_key"] = "local"
            self._async_client = AsyncOpenAI(**kwargs)
        return self._async_client

    def _get_sync_client(self) -> object:
        """Return (or create) the sync OpenAI client (thread-safe)."""
        with self._lock:
            if self._sync_client is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise ImportError(
                        "openai package is required. Install it with: pip install openai"
                    ) from exc
                kwargs: dict = {"timeout": self._request_timeout}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                elif self._base_url:
                    kwargs["api_key"] = "local"
                # (No else — no base_url means standard OpenAI endpoint; let
                # the SDK read OPENAI_API_KEY from the environment as normal.)
                self._sync_client = OpenAI(**kwargs)
            return self._sync_client

    # ------------------------------------------------------------------
    # EmbeddingProvider interface
    # ------------------------------------------------------------------

    async def embed(
        self,
        texts: list[str],
        on_batch: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Embed *texts* asynchronously, in batches of ``batch_size``.

        Calls ``on_batch(chunks_done, chunks_total)`` after each batch so
        callers can drive a progress bar at chunk granularity.
        """
        if not texts:
            return []

        normalized = [_normalize_text(t) for t in texts]
        total = len(normalized)
        client = self._get_async_client()
        results: list[list[float]] = []

        for batch_start in range(0, total, self._batch_size):
            batch = normalized[batch_start : batch_start + self._batch_size]
            batch_end = min(batch_start + self._batch_size, total)
            logger.debug(
                "embed: sending batch %d-%d/%d to %s",
                batch_start + 1,
                batch_end,
                total,
                self._model,
            )
            try:
                response = await client.embeddings.create(  # type: ignore[attr-defined]
                    model=self._model, input=batch
                )
            except Exception as exc:
                logger.error(
                    "embed: batch %d-%d/%d failed (%s); "
                    "consider reducing 'batch_size' or increasing 'request_timeout' "
                    "in your embedding config",
                    batch_start + 1,
                    batch_end,
                    total,
                    exc,
                )
                raise
            batch_vecs = [item.embedding for item in response.data]
            results.extend(batch_vecs)

            if self._dimension is None and batch_vecs:
                with self._lock:
                    if self._dimension is None:
                        self._dimension = len(batch_vecs[0])

            if on_batch is not None:
                on_batch(batch_end, total)

        return results

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* synchronously using the sync OpenAI client.

        Does NOT use asyncio.run() — safe to call from synchronous code even
        when an event loop is running.
        """
        if not texts:
            return []

        normalized = [_normalize_text(t) for t in texts]
        client = self._get_sync_client()
        results: list[list[float]] = []

        for batch_start in range(0, len(normalized), self._batch_size):
            batch = normalized[batch_start : batch_start + self._batch_size]
            response = client.embeddings.create(  # type: ignore[attr-defined]
                model=self._model, input=batch
            )
            batch_vecs = [item.embedding for item in response.data]
            results.extend(batch_vecs)

            if self._dimension is None and batch_vecs:
                with self._lock:
                    if self._dimension is None:
                        self._dimension = len(batch_vecs[0])

        return results

    @property
    def dimension(self) -> int:
        """Return the vector dimension.

        Raises ``RuntimeError`` if neither ``embed()`` nor ``embed_sync()``
        has been called yet (dimension is learned from the first API response).
        """
        if self._dimension is None:
            raise RuntimeError(
                "dimension is not yet known; call embed() or embed_sync() first."
            )
        return self._dimension

    @property
    def model(self) -> str:
        """Name of the configured embedding model."""
        return self._model

    async def aclose(self) -> None:
        """Close the async HTTP client and reset it so the next call creates a fresh one.

        Must be awaited after each indexing run when the event loop is about to
        close; otherwise the httpx transport becomes unusable in the next loop.
        """
        if self._async_client is not None:
            client = self._async_client
            # OpenAI SDK versions differ: newer ones expose `aclose()`, older
            # ones expose `close()`. Try both so the HTTP transport is always
            # released, regardless of the installed SDK version.
            close_fn = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close_fn is not None:
                with contextlib.suppress(RuntimeError, OSError):
                    result = close_fn()
                    if hasattr(result, "__await__"):
                        await result
            self._async_client = None
