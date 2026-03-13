"""Tests for ai_cli/core/llm_client.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_cli.core.llm_client import LLMClient, LLMError, OpenAIClient, create_llm_client

# ---------------------------------------------------------------------------
# Stub exceptions — used to patch ai_cli.core.llm_client.* so tests don't
# depend on the openai package's constructor signatures changing across
# unpinned versions.
# ---------------------------------------------------------------------------


class _FakeRateLimitError(Exception):
    pass


class _FakeAPIConnectionError(Exception):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
) -> MagicMock:
    """Build a fake streaming chunk matching the openai SDK shape."""
    chunk = MagicMock()
    chunk.usage = usage

    if (
        content is None
        and tool_calls is None
        and finish_reason is None
        and usage is not None
    ):
        # Usage-only chunk (no choices)
        chunk.choices = []
        return chunk

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.delta.content = content
    choice.delta.tool_calls = tool_calls
    chunk.choices = [choice]
    return chunk


def _make_tc_delta(
    index: int, call_id: str = "", name: str = "", arguments: str = ""
) -> MagicMock:
    """Build a fake tool-call delta."""
    tc = MagicMock()
    tc.index = index
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_usage(prompt: int = 10, completion: int = 5, total: int = 15) -> MagicMock:
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = total
    return u


def _make_client(chunks: list, model: str = "gpt-4o") -> OpenAIClient:
    """Return an OpenAIClient whose underlying OpenAI SDK is fully mocked."""
    config = {
        "model": model,
        "api_key": "test-key",
        "context_window": 128000,
        "max_response_tokens": 4096,
    }
    with patch("ai_cli.core.llm_client.OpenAI"):
        client = OpenAIClient(config)
    client._client.chat.completions.create.return_value = iter(chunks)
    return client


def _make_response(
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    usage: Any = None,
) -> MagicMock:
    """Build a fake non-streaming response matching the openai SDK shape."""
    response = MagicMock()
    response.usage = usage
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    response.choices = [choice]
    return response


def _make_tc_full(call_id: str, name: str, arguments: str) -> MagicMock:
    """Build a fake non-streaming tool call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_client_nonstream(response: Any, model: str = "gpt-4o") -> OpenAIClient:
    """Return an OpenAIClient configured for non-streaming with a fixed response."""
    config = {
        "model": model,
        "api_key": "test-key",
        "context_window": 128000,
        "max_response_tokens": 4096,
    }
    with patch("ai_cli.core.llm_client.OpenAI"):
        client = OpenAIClient(config)
    client._client.chat.completions.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class TestLLMClientInterface:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            LLMClient()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# OpenAIClient — text streaming
# ---------------------------------------------------------------------------


class TestOpenAIClientText:
    def test_text_chunks_yielded_immediately(self):
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=", world"),
            _make_chunk(content="!", finish_reason="stop"),
            _make_chunk(usage=_make_usage()),
        ]
        client = _make_client(chunks)
        results = list(client.send([], []))
        text_chunks = [c for c in results if c["type"] == "text"]
        assert [c["delta"] for c in text_chunks] == ["Hello", ", world", "!"]

    def test_done_chunk_always_last(self):
        chunks = [
            _make_chunk(content="Hi", finish_reason="stop"),
            _make_chunk(usage=_make_usage()),
        ]
        client = _make_client(chunks)
        results = list(client.send([], []))
        assert results[-1]["type"] == "done"

    def test_done_chunk_carries_usage(self):
        chunks = [
            _make_chunk(finish_reason="stop"),
            _make_chunk(usage=_make_usage(prompt=20, completion=10, total=30)),
        ]
        client = _make_client(chunks)
        done = list(client.send([], []))[-1]
        assert done["usage"] == {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        }

    def test_done_chunk_carries_stop_reason(self):
        chunks = [
            _make_chunk(finish_reason="stop"),
            _make_chunk(usage=_make_usage()),
        ]
        client = _make_client(chunks)
        done = list(client.send([], []))[-1]
        assert done["stop_reason"] == "stop"

    def test_empty_response(self):
        chunks = [
            _make_chunk(finish_reason="stop"),
            _make_chunk(usage=_make_usage()),
        ]
        client = _make_client(chunks)
        results = list(client.send([], []))
        text_chunks = [c for c in results if c["type"] == "text"]
        assert text_chunks == []
        assert results[-1]["type"] == "done"


# ---------------------------------------------------------------------------
# OpenAIClient — tool call assembly
# ---------------------------------------------------------------------------


class TestOpenAIClientToolCalls:
    def test_single_tool_call_assembled(self):
        chunks = [
            _make_chunk(
                tool_calls=[_make_tc_delta(0, call_id="id1", name="read_file")]
            ),
            _make_chunk(tool_calls=[_make_tc_delta(0, arguments='{"path":')]),
            _make_chunk(tool_calls=[_make_tc_delta(0, arguments=' "./foo.py"}')]),
            _make_chunk(finish_reason="tool_calls"),
            _make_chunk(usage=_make_usage()),
        ]
        client = _make_client(chunks)
        results = list(
            client.send([], [{"type": "function", "function": {"name": "read_file"}}])
        )
        tool_chunks = [c for c in results if c["type"] == "tool_call"]
        assert len(tool_chunks) == 1
        tc = tool_chunks[0]
        assert tc["name"] == "read_file"
        assert tc["call_id"] == "id1"
        assert tc["arguments"] == {"path": "./foo.py"}

    def test_multiple_tool_calls_assembled_in_order(self):
        chunks = [
            _make_chunk(
                tool_calls=[_make_tc_delta(0, call_id="id0", name="read_file")]
            ),
            _make_chunk(
                tool_calls=[_make_tc_delta(1, call_id="id1", name="write_file")]
            ),
            _make_chunk(tool_calls=[_make_tc_delta(0, arguments='{"path": "a.py"}')]),
            _make_chunk(
                tool_calls=[
                    _make_tc_delta(1, arguments='{"path": "b.py", "content": "x"}')
                ]
            ),
            _make_chunk(finish_reason="tool_calls"),
            _make_chunk(usage=_make_usage()),
        ]
        client = _make_client(chunks)
        results = list(client.send([], []))
        tool_chunks = [c for c in results if c["type"] == "tool_call"]
        assert len(tool_chunks) == 2
        assert tool_chunks[0]["name"] == "read_file"
        assert tool_chunks[0]["arguments"] == {"path": "a.py"}
        assert tool_chunks[1]["name"] == "write_file"
        assert tool_chunks[1]["arguments"] == {"path": "b.py", "content": "x"}

    def test_tool_calls_yielded_before_done(self):
        chunks = [
            _make_chunk(
                tool_calls=[
                    _make_tc_delta(
                        0, call_id="id0", name="read_file", arguments='{"path": "f.py"}'
                    )
                ]
            ),
            _make_chunk(finish_reason="tool_calls"),
            _make_chunk(usage=_make_usage()),
        ]
        client = _make_client(chunks)
        results = list(client.send([], []))
        types = [c["type"] for c in results]
        assert types.index("tool_call") < types.index("done")

    def test_malformed_tool_arguments_yields_empty_dict(self):
        chunks = [
            _make_chunk(
                tool_calls=[
                    _make_tc_delta(
                        0, call_id="id0", name="read_file", arguments="not valid json"
                    )
                ]
            ),
            _make_chunk(finish_reason="tool_calls"),
            _make_chunk(usage=_make_usage()),
        ]
        client = _make_client(chunks)
        results = list(client.send([], []))
        tc = next(c for c in results if c["type"] == "tool_call")
        assert tc["arguments"] == {}

    def test_no_tool_calls_when_none_in_stream(self):
        chunks = [
            _make_chunk(content="plain text", finish_reason="stop"),
            _make_chunk(usage=_make_usage()),
        ]
        client = _make_client(chunks)
        results = list(client.send([], []))
        assert not any(c["type"] == "tool_call" for c in results)

    def test_tools_not_sent_when_empty_list(self):
        chunks = [
            _make_chunk(finish_reason="stop"),
            _make_chunk(usage=_make_usage()),
        ]
        client = _make_client(chunks)
        list(client.send([], []))
        call_kwargs = client._client.chat.completions.create.call_args[1]
        assert "tools" not in call_kwargs

    def test_tools_sent_when_provided(self):
        chunks = [
            _make_chunk(finish_reason="stop"),
            _make_chunk(usage=_make_usage()),
        ]
        tools = [{"type": "function", "function": {"name": "read_file"}}]
        client = _make_client(chunks)
        list(client.send([], tools))
        call_kwargs = client._client.chat.completions.create.call_args[1]
        assert call_kwargs["tools"] == tools


# ---------------------------------------------------------------------------
# OpenAIClient — non-streaming path
# ---------------------------------------------------------------------------


class TestOpenAIClientNonStreaming:
    def test_text_chunk_yielded(self):
        response = _make_response(content="Hello", usage=_make_usage())
        client = _make_client_nonstream(response)
        results = list(client.send([], [], stream=False))
        text_chunks = [c for c in results if c["type"] == "text"]
        assert len(text_chunks) == 1
        assert text_chunks[0]["delta"] == "Hello"

    def test_done_chunk_always_last(self):
        response = _make_response(content="Hi", usage=_make_usage())
        client = _make_client_nonstream(response)
        results = list(client.send([], [], stream=False))
        assert results[-1]["type"] == "done"

    def test_done_chunk_carries_stop_reason(self):
        response = _make_response(finish_reason="stop", usage=_make_usage())
        client = _make_client_nonstream(response)
        done = list(client.send([], [], stream=False))[-1]
        assert done["stop_reason"] == "stop"

    def test_done_chunk_carries_usage(self):
        response = _make_response(usage=_make_usage(prompt=20, completion=10, total=30))
        client = _make_client_nonstream(response)
        done = list(client.send([], [], stream=False))[-1]
        assert done["usage"] == {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        }

    def test_tool_call_assembled(self):
        tc = _make_tc_full("id1", "read_file", '{"path": "foo.py"}')
        response = _make_response(tool_calls=[tc], usage=_make_usage())
        client = _make_client_nonstream(response)
        results = list(client.send([], [], stream=False))
        tool_chunks = [c for c in results if c["type"] == "tool_call"]
        assert len(tool_chunks) == 1
        assert tool_chunks[0]["name"] == "read_file"
        assert tool_chunks[0]["call_id"] == "id1"
        assert tool_chunks[0]["arguments"] == {"path": "foo.py"}

    def test_tool_call_yielded_before_done(self):
        tc = _make_tc_full("id1", "read_file", '{"path": "foo.py"}')
        response = _make_response(tool_calls=[tc], usage=_make_usage())
        client = _make_client_nonstream(response)
        results = list(client.send([], [], stream=False))
        types = [c["type"] for c in results]
        assert types.index("tool_call") < types.index("done")

    def test_malformed_tool_arguments_yields_empty_dict(self):
        tc = _make_tc_full("id1", "read_file", "not valid json")
        response = _make_response(tool_calls=[tc], usage=_make_usage())
        client = _make_client_nonstream(response)
        results = list(client.send([], [], stream=False))
        tool_chunk = next(c for c in results if c["type"] == "tool_call")
        assert tool_chunk["arguments"] == {}

    def test_no_tool_calls_when_none(self):
        response = _make_response(content="plain text", usage=_make_usage())
        client = _make_client_nonstream(response)
        results = list(client.send([], [], stream=False))
        assert not any(c["type"] == "tool_call" for c in results)

    def test_stream_false_passed_to_api_without_stream_options(self):
        response = _make_response(usage=_make_usage())
        client = _make_client_nonstream(response)
        list(client.send([], [], stream=False))
        call_kwargs = client._client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is False
        assert "stream_options" not in call_kwargs


# ---------------------------------------------------------------------------
# OpenAIClient — retry behaviour
# ---------------------------------------------------------------------------


_RETRY_CONFIG = {
    "model": "gpt-4o",
    "api_key": "key",
    "context_window": 128000,
    "max_response_tokens": 4096,
}


class TestOpenAIClientRetry:
    def test_retries_on_rate_limit_then_succeeds(self):
        chunks = [
            _make_chunk(finish_reason="stop"),
            _make_chunk(usage=_make_usage()),
        ]
        with patch("ai_cli.core.llm_client.OpenAI"):
            client = OpenAIClient(_RETRY_CONFIG)

        client._client.chat.completions.create.side_effect = [
            _FakeRateLimitError(),
            iter(chunks),
        ]

        with (
            patch("ai_cli.core.llm_client.RateLimitError", _FakeRateLimitError),
            patch("ai_cli.core.llm_client.time.sleep"),
        ):
            results = list(client.send([], []))
        assert results[-1]["type"] == "done"

    def test_raises_after_max_retries(self):
        with patch("ai_cli.core.llm_client.OpenAI"):
            client = OpenAIClient(_RETRY_CONFIG)

        client._client.chat.completions.create.side_effect = _FakeRateLimitError()

        with (
            patch("ai_cli.core.llm_client.RateLimitError", _FakeRateLimitError),
            patch("ai_cli.core.llm_client.time.sleep"),
            pytest.raises(LLMError, match="attempts"),
        ):
            list(client.send([], []))

    def test_raises_on_connection_error(self):
        with patch("ai_cli.core.llm_client.OpenAI"):
            client = OpenAIClient(_RETRY_CONFIG)

        client._client.chat.completions.create.side_effect = _FakeAPIConnectionError()

        with (
            patch("ai_cli.core.llm_client.APIConnectionError", _FakeAPIConnectionError),
            pytest.raises(LLMError, match="Connection error"),
        ):
            list(client.send([], []))


# ---------------------------------------------------------------------------
# OpenAIClient — metadata and token counting
# ---------------------------------------------------------------------------


class TestOpenAIClientInit:
    def test_missing_required_key_raises_llm_error(self):
        config = {
            "model": "gpt-4o",
            "api_key": "key",
        }  # missing context_window and max_response_tokens
        with (
            patch("ai_cli.core.llm_client.OpenAI"),
            pytest.raises(LLMError, match="context_window"),
        ):
            OpenAIClient(config)

    def test_all_required_keys_present_succeeds(self):
        config = {
            "model": "gpt-4o",
            "api_key": "key",
            "context_window": 128000,
            "max_response_tokens": 4096,
        }
        with patch("ai_cli.core.llm_client.OpenAI"):
            client = OpenAIClient(config)
        assert client._model == "gpt-4o"

    def test_usage_always_has_all_keys_when_server_omits_usage(self):
        """Servers that ignore include_usage should still get a normalised usage dict."""
        chunks = [
            _make_chunk(finish_reason="stop"),
            # No usage chunk at all
        ]
        client = _make_client(chunks)
        done = list(client.send([], []))[-1]
        assert done["usage"] == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


class TestOpenAIClientMetadata:
    def test_get_model_metadata(self):
        config = {
            "model": "gpt-4o",
            "api_key": "key",
            "context_window": 128000,
            "max_response_tokens": 4096,
        }
        with patch("ai_cli.core.llm_client.OpenAI"):
            client = OpenAIClient(config)
        meta = client.get_model_metadata()
        assert meta["model"] == "gpt-4o"
        assert meta["context_window"] == 128000
        assert meta["max_response_tokens"] == 4096

    def test_count_tokens_returns_positive_int(self):
        config = {
            "model": "gpt-4o",
            "api_key": "key",
            "context_window": 128000,
            "max_response_tokens": 4096,
        }
        with patch("ai_cli.core.llm_client.OpenAI"):
            client = OpenAIClient(config)
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am fine, thank you."},
        ]
        count = client.count_tokens(messages)
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_more_messages_means_more_tokens(self):
        config = {
            "model": "gpt-4o",
            "api_key": "key",
            "context_window": 128000,
            "max_response_tokens": 4096,
        }
        with patch("ai_cli.core.llm_client.OpenAI"):
            client = OpenAIClient(config)
        short = [{"role": "user", "content": "Hi"}]
        long = [{"role": "user", "content": "Hi " * 100}]
        assert client.count_tokens(long) > client.count_tokens(short)

    def test_count_tokens_unknown_model_falls_back(self):
        config = {
            "model": "some-unknown-model-xyz",
            "api_key": "key",
            "context_window": 8000,
            "max_response_tokens": 1024,
        }
        with patch("ai_cli.core.llm_client.OpenAI"):
            client = OpenAIClient(config)
        # Should not raise, just use fallback encoding
        count = client.count_tokens([{"role": "user", "content": "test"}])
        assert count > 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateLLMClient:
    def test_creates_openai_client(self):
        config_manager = MagicMock()
        config_manager.get_backend.return_value = "openai"
        config_manager.get_model_config.return_value = {
            "model": "gpt-4o",
            "api_key": "key",
            "context_window": 128000,
            "max_response_tokens": 4096,
        }
        with patch("ai_cli.core.llm_client.OpenAI"):
            client = create_llm_client(config_manager)
        assert isinstance(client, OpenAIClient)

    def test_lmstudio_raises_helpful_error(self):
        config_manager = MagicMock()
        config_manager.get_backend.return_value = "lmstudio"
        with pytest.raises(LLMError, match="not yet implemented"):
            create_llm_client(config_manager)

    def test_unknown_backend_raises_error(self):
        config_manager = MagicMock()
        config_manager.get_backend.return_value = "anthropic"
        with pytest.raises(LLMError, match="Unknown backend"):
            create_llm_client(config_manager)
