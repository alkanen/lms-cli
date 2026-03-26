"""
llm_client.py — LLM backend abstraction.

All backends yield the same Chunk dict format so the REPL can treat them
uniformly:

  {"type": "text",      "delta": str}
  {"type": "reasoning", "delta": str}
  {"type": "tool_call", "name": str, "call_id": str, "arguments": dict}
  {"type": "done",      "stop_reason": str,
                        "usage": {"prompt_tokens": int,
                                  "completion_tokens": int,
                                  "total_tokens": int}}

Text deltas are yielded immediately as they arrive.  Reasoning deltas come
from two sources: the ``reasoning_content`` delta field (OpenAI o1/o3 and
compatible models) and ``<think>…</think>`` tags embedded in the text stream
(opt-in via ``extract_think_tags: true`` in config; handled by
``_ThinkTagParser``).  Tool calls are buffered internally and emitted as a
single complete chunk — the caller never sees a partial tool call.  The "done"
chunk is always the last item yielded.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING, Any

import tiktoken

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

if TYPE_CHECKING:
    from ai_cli.core.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class _ThinkTagParser:
    """
    Splits a streaming text into ``"text"`` and ``"reasoning"`` chunk dicts.

    Handles ``<think>…</think>`` tags that may span multiple chunks.  Each
    call to :meth:`feed` returns zero or more ``{"type": …, "delta": …}``
    dicts that can be yielded directly to the REPL.  Call :meth:`flush` at
    end-of-stream to emit any content that was held back pending a potential
    tag boundary.

    State machine::

        OUTSIDE  → text before <think>   → emitted as "text" chunks
        INSIDE   → text between tags     → emitted as "reasoning" chunks
        OUTSIDE  ← text after </think>   → emitted as "text" chunks

    Tag boundaries may fall mid-chunk; the parser buffers characters until the
    full tag is confirmed or ruled out.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._inside = False
        self._buf = ""

    def feed(self, delta: str) -> list[dict]:
        """Feed a text delta; return a list of ``{type, delta}`` chunk dicts."""
        self._buf += delta
        chunks: list[dict] = []

        while self._buf:
            tag = self._CLOSE if self._inside else self._OPEN
            idx = self._buf.find(tag)
            if idx >= 0:
                # Tag found; emit everything before it, then flip state.
                if idx > 0:
                    chunk_type = "reasoning" if self._inside else "text"
                    chunks.append({"type": chunk_type, "delta": self._buf[:idx]})
                self._buf = self._buf[idx + len(tag) :]
                self._inside = not self._inside
            else:
                # No complete tag; emit whatever definitely is not a partial tag.
                safe = self._safe_prefix()
                if safe:
                    chunk_type = "reasoning" if self._inside else "text"
                    chunks.append({"type": chunk_type, "delta": safe})
                break  # rest of _buf may be a partial tag — wait for more input

        return chunks

    def _safe_prefix(self) -> str:
        """Return and consume the part of ``_buf`` that cannot be a partial tag start."""
        tag = self._CLOSE if self._inside else self._OPEN
        # Find the longest suffix of _buf that is a prefix of tag.
        max_check = min(len(tag) - 1, len(self._buf))
        for prefix_len in range(max_check, 0, -1):
            if self._buf.endswith(tag[:prefix_len]):
                safe_end = len(self._buf) - prefix_len
                safe = self._buf[:safe_end]
                self._buf = self._buf[safe_end:]
                return safe
        # No partial-tag match — entire buffer is safe to emit.
        safe = self._buf
        self._buf = ""
        return safe

    def flush(self) -> list[dict]:
        """Emit any remaining buffered content (call at end of stream)."""
        if self._buf:
            chunk_type = "reasoning" if self._inside else "text"
            result = [{"type": chunk_type, "delta": self._buf}]
            self._buf = ""
            return result
        return []


class LLMError(Exception):
    """Raised for unrecoverable LLM backend errors."""


class LLMClient(ABC):
    """Abstract base for all LLM backends."""

    @abstractmethod
    def send(
        self,
        messages: list[dict],
        tools: list[dict],
        stream: bool = True,
    ) -> Generator[dict, None, None]:
        """
        Send a conversation turn and yield Chunk dicts.

        When *stream* is ``True`` (default), text deltas are yielded
        immediately as they arrive.  Tool calls are buffered internally and
        yielded as a single complete chunk once all argument deltas have been
        received.  The "done" chunk is always last.

        When *stream* is ``False``, the entire response is awaited before any
        chunks are yielded; the same Chunk types are produced in the same order
        so callers need not handle the two modes differently.

        Yields
        ------
        dict
            One of: TextChunk, ToolCallChunk, DoneChunk (see module docstring).
        """

    @abstractmethod
    def get_model_metadata(self) -> dict:
        """Return ``{'model': str, 'context_window': int, 'max_response_tokens': int}``."""

    @abstractmethod
    def count_tokens(self, messages: list[dict]) -> int:
        """Estimate the number of tokens consumed by *messages*."""


class OpenAIClient(LLMClient):
    """
    OpenAI-compatible REST backend.

    Works with OpenAI's own API, LM Studio's REST endpoint, Ollama, and any
    other server that implements the OpenAI chat-completions API.

    Required config keys (from ``ConfigManager.get_model_config()``):
      model               — model identifier string
      context_window      — integer; not exposed reliably by the API
      max_response_tokens — integer; caps the response length

    Optional config keys:
      api_key             — API key; omit or use any string for local servers
      base_url            — override endpoint (e.g. http://localhost:1234/v1)
    """

    _MAX_RETRIES = 3
    _RETRY_BASE_DELAY = 1.0  # seconds; doubled on each retry

    def __init__(self, config: dict) -> None:
        _REQUIRED = ("model", "context_window", "max_response_tokens")
        missing = [k for k in _REQUIRED if k not in config]
        if missing:
            raise LLMError(
                f"Missing required config key(s): {', '.join(missing)}. "
                "Add them to your global config.yaml (default: ~/.ai-cli/config.yaml, "
                "override with AI_CLI_GLOBAL_DIR) or your project's .ai-cli/config.yaml."
            )
        self._model: str = config["model"]
        self._context_window: int = int(config["context_window"])
        self._max_response_tokens: int = int(config["max_response_tokens"])
        self._extract_think_tags: bool = bool(config.get("extract_think_tags", False))

        client_kwargs: dict[str, Any] = {
            "api_key": config.get("api_key", "no-key"),
        }
        if config.get("base_url"):
            client_kwargs["base_url"] = config["base_url"]

        self._client = OpenAI(**client_kwargs)
        self._encoding = _load_encoding(self._model)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def send(
        self,
        messages: list[dict],
        tools: list[dict],
        stream: bool = True,
    ) -> Generator[dict, None, None]:
        """Stream one conversation turn, retrying on rate-limit errors."""
        for attempt in range(self._MAX_RETRIES):
            try:
                yield from self._stream(messages, tools, stream=stream)
                return
            except RateLimitError as exc:
                if attempt == self._MAX_RETRIES - 1:
                    raise LLMError(
                        f"Rate limit exceeded after {self._MAX_RETRIES} attempts: {exc}"
                    ) from exc
                delay = self._RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Rate limited — retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    self._MAX_RETRIES,
                )
                time.sleep(delay)
            except APIConnectionError as exc:
                raise LLMError(f"Connection error: {exc}") from exc
            except APIStatusError as exc:
                raise LLMError(f"API error {exc.status_code}: {str(exc)}") from exc

    def get_model_metadata(self) -> dict:
        return {
            "model": self._model,
            "context_window": self._context_window,
            "max_response_tokens": self._max_response_tokens,
        }

    def count_tokens(self, messages: list[dict]) -> int:
        """
        Estimate token count via tiktoken.

        Uses 4 tokens per message as overhead (matches the OpenAI cookbook
        estimate for chat models).  The actual count returned by the API in
        the "done" chunk's ``usage`` field should be used for session
        accounting whenever available.
        """
        total = 0
        for message in messages:
            total += 4  # per-message overhead
            for value in message.values():
                if isinstance(value, str):
                    total += len(self._encoding.encode(value))
                elif isinstance(value, list):
                    # content can be a list of content blocks
                    for item in value:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            total += len(self._encoding.encode(item["text"]))
        total += 2  # reply-priming tokens
        return total

    # ------------------------------------------------------------------
    # Internal streaming
    # ------------------------------------------------------------------

    def _stream(
        self,
        messages: list[dict],
        tools: list[dict],
        stream: bool = True,
    ) -> Generator[dict, None, None]:
        """Single attempt — not responsible for retries."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_response_tokens,
            "stream": stream,
        }
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        stop_reason = ""
        usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if not stream:
            # Non-streaming: convert single response to the same Chunk sequence.
            yield from self._chunks_from_response(response)
            return

        # Streaming: buffers for assembling incremental tool-call argument deltas,
        # keyed by the tool-call index in the stream.
        tool_call_bufs: dict[int, dict[str, str]] = {}
        # Optional think-tag parser (created only when extract_think_tags is on).
        parser: _ThinkTagParser | None = (
            _ThinkTagParser() if self._extract_think_tags else None
        )

        try:
            for chunk in response:
                # Usage arrives in the final chunk when include_usage=True.
                if chunk.usage is not None:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                if choice.finish_reason:
                    stop_reason = choice.finish_reason

                delta = choice.delta

                # Yield reasoning_content first so it always precedes text chunks
                # from the same delta (OpenAI o1/o3 and compatible models).
                reasoning = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning, str) and reasoning:
                    yield {"type": "reasoning", "delta": reasoning}

                # Yield text content (or split via think-tag parser).
                if delta.content:
                    if parser is not None:
                        yield from parser.feed(delta.content)
                    else:
                        yield {"type": "text", "delta": delta.content}

                # Accumulate tool-call argument deltas.
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.index not in tool_call_bufs:
                            tool_call_bufs[tc.index] = {
                                "id": "",
                                "name": "",
                                "arguments_buf": "",
                            }
                        buf = tool_call_bufs[tc.index]
                        if tc.id and not buf["id"]:
                            buf["id"] = tc.id
                        if tc.function:
                            if tc.function.name and not buf["name"]:
                                buf["name"] = tc.function.name
                            if tc.function.arguments:
                                buf["arguments_buf"] += tc.function.arguments
        finally:
            # Release the underlying HTTP connection when the caller stops
            # iterating early (e.g. user abort via generator.close()).
            with contextlib.suppress(Exception):
                response.close()

        # Flush any content held back by the think-tag parser.
        if parser is not None:
            yield from parser.flush()

        # Yield fully assembled tool calls in index order.
        for idx in sorted(tool_call_bufs):
            buf = tool_call_bufs[idx]
            args_str = buf["arguments_buf"]
            try:
                arguments = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Could not parse arguments for tool '%s' (call_id=%s): %s "
                    "— using empty dict.",
                    buf["name"],
                    buf["id"],
                    exc,
                )
                arguments = {}
            yield {
                "type": "tool_call",
                "name": buf["name"],
                "call_id": buf["id"],
                "arguments": arguments,
            }
            del tool_call_bufs[idx]

        yield {"type": "done", "stop_reason": stop_reason, "usage": usage}

    def _chunks_from_response(self, response: Any) -> Iterator[dict]:
        """Convert a non-streaming API response to the standard Chunk sequence."""
        usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        stop_reason = ""
        if response.choices:
            choice = response.choices[0]
            stop_reason = choice.finish_reason or ""
            message = choice.message

            # Yield reasoning_content first so it always precedes text.
            reasoning = getattr(message, "reasoning_content", None)
            if isinstance(reasoning, str) and reasoning:
                yield {"type": "reasoning", "delta": reasoning}

            if message.content:
                if self._extract_think_tags:
                    parser = _ThinkTagParser()
                    yield from parser.feed(message.content)
                    yield from parser.flush()
                else:
                    yield {"type": "text", "delta": message.content}

            if message.tool_calls:
                for tc in message.tool_calls:
                    args_str = tc.function.arguments or ""
                    try:
                        arguments = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Could not parse arguments for tool '%s' (call_id=%s): %s "
                            "— using empty dict.",
                            tc.function.name,
                            tc.id,
                            exc,
                        )
                        arguments = {}
                    yield {
                        "type": "tool_call",
                        "name": tc.function.name,
                        "call_id": tc.id,
                        "arguments": arguments,
                    }

        yield {"type": "done", "stop_reason": stop_reason, "usage": usage}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_encoding(model: str) -> tiktoken.Encoding:
    """Return the tiktoken encoding for *model*, falling back to cl100k_base."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug(
            "No tiktoken encoding found for model '%s' — falling back to cl100k_base.",
            model,
        )
        return tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm_client(config_manager: ConfigManager) -> LLMClient:
    """
    Read the backend from config and return the appropriate ``LLMClient``.

    Currently only the OpenAI-compatible REST backend is supported.
    LM Studio's WebSocket backend is planned; in the meantime LM Studio can
    be used via its OpenAI-compatible REST endpoint with ``backend: openai``
    and an appropriate ``base_url``.
    """
    backend = config_manager.get_backend()
    if backend == "lmstudio":
        raise LLMError(
            "LM Studio WebSocket backend is not yet implemented. "
            "LM Studio also supports the OpenAI-compatible REST API — "
            "set 'backend: openai' and point 'base_url' at your LM Studio server "
            "(e.g. http://localhost:1234/v1)."
        )
    if backend != "openai":
        raise LLMError(f"Unknown backend '{backend}'. Currently supported: 'openai'.")
    return OpenAIClient(config_manager.get_model_config())
