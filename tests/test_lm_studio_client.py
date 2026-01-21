"""
Unit tests for the LMStudioClient class.
"""

import pytest
from unittest.mock import patch
from lms_cli.core.lm_studio_client import LMStudioClient


@pytest.fixture
def lm_client():
    """Fixture to create an instance of LMStudioClient."""
    return LMStudioClient(model="test-model", base_url="http://localhost:1234")


def test_chat_completion_non_streaming(lm_client):
    """Test non-streaming chat completion."""
    messages = [{"role": "user", "content": "Hello"}]
    expected_response = {
        "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}]
    }

    with patch.object(lm_client, "_make_request", return_value=expected_response):
        response = lm_client.chat_completion(messages, stream=False)

        assert response["content"] == "Hi there!"
        assert response["tool_calls"] == []


def test_chat_completion_non_streaming_with_tools(lm_client):
    """Test non-streaming chat completion with tool calls."""
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool", "parameters": {}}}]

    expected_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Tool response",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "test_tool", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }

    with patch.object(lm_client, "_make_request", return_value=expected_response):
        response = lm_client.chat_completion(messages, tools=tools, stream=False)

        assert response["content"] == "Tool response"
        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0]["function"]["name"] == "test_tool"


def test_chat_completion_streaming(lm_client):
    """Test streaming chat completion."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock the streaming response
    mock_chunks = [
        'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
        'data: {"choices": [{"delta": {"content": " there!"}}]}\n\n',
    ]

    with patch.object(lm_client, "_make_streaming_request", return_value=mock_chunks):
        response = lm_client.chat_completion(messages, stream=True)

        assert response["content"] == "Hi there!"
        assert response["tool_calls"] == []


def test_chat_completion_streaming_with_tool_calls(lm_client):
    """Test streaming chat completion with tool calls."""
    messages = [{"role": "user", "content": "Hello"}]

    # Mock the streaming response with tool calls
    mock_chunks = [
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "type": "function", '
        '"id": "1", "function": {"name": "test_tool", "arguments": "{}"}}]}}]}\n\n'
    ]

    with patch.object(lm_client, "_make_streaming_request", return_value=mock_chunks):
        response = lm_client.chat_completion(messages, stream=True)

        assert response["content"] == ""
        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0]["function"]["name"] == "test_tool"


def test_chat_completion_with_callback(lm_client):
    """Test chat completion with a callback for chunks."""
    messages = [{"role": "user", "content": "Hello"}]
    callback_chunks = []

    def mock_callback(chunk):
        callback_chunks.append(chunk)

    # Mock the streaming response
    mock_chunks = [
        'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
        'data: {"choices": [{"delta": {"content": " there!"}}]}\n\n',
    ]

    with patch.object(lm_client, "_make_streaming_request", return_value=mock_chunks):
        response = lm_client.chat_completion(
            messages, stream=True, on_chunk_callback=mock_callback
        )

        assert response["content"] == "Hi there!"
        assert callback_chunks == ["Hi", " there!"]


def test_process_tool_chunks(lm_client):
    """Test the _process_tool_chunks method."""
    tool_chunks = [
        [
            {
                "index": 0,
                "type": "function",
                "id": "1",
                "function": {"name": "test_tool", "arguments": '{"arg1": '},
            },
            {"index": 0, "function": {"arguments": '"value"}'}},
        ]
    ]

    result = lm_client._process_tool_chunks(tool_chunks)

    assert len(result) == 1
    assert result[0]["id"] == "1"
    assert result[0]["function"]["name"] == "test_tool"
    assert result[0]["function"]["arguments"] == '{"arg1": "value"}'


def test_parse_tool_calls(lm_client):
    """Test the parse_tool_calls method."""
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "test_tool", "arguments": "{}"},
                        }
                    ]
                }
            }
        ]
    }

    result = lm_client.parse_tool_calls(response)

    assert len(result) == 1
    assert result[0]["function"]["name"] == "test_tool"
