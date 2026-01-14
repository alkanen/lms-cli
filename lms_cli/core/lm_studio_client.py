from typing import Dict, Iterator, Optional, Callable
from collections import defaultdict

import json
import requests
import yaml


enable_message_logs = False


class LMStudioClient:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Use provided values or fall back to config
        self.base_url = base_url if base_url is not None else self.config["lm_studio"]["base_url"]
        self.model = model if model is not None else self.config["lm_studio"]["model"]
        self.api_key = self.config["lm_studio"].get("api_key", "")
        # If no embedding model is specified, revert to main model
        self.embedding_model = self.config["embeddings"].get("model", self.model)

        self.system_message = self.config.get("agent", {}).get("system_message", None)
        if not self.system_message:
            self.system_message = (
                "You are a helpful coding assistant. Use the provided file contexts to "
                "answer questions about the code."
            )

    def _make_request(
        self, endpoint: str, method: str = "POST", data: Optional[Dict] = None
    ) -> Dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/{endpoint}"
        response = requests.request(method, url, json=data, headers=headers)
        response.raise_for_status()

        return response.json()

    def _make_streaming_request(
        self, endpoint: str, data: Optional[Dict] = None
    ) -> Iterator[Dict]:
        """Make a streaming request and yield chunks"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if enable_message_logs:
            with open("messages.log", "a") as f:
                f.write(f'User: {data["messages"][-1]}\n')

        url = f"{self.base_url}/{endpoint}"
        with requests.post(url, json=data, headers=headers, stream=True) as response:
            if response.status_code >= 400:
                import pdb
                pdb.set_trace()

            response.raise_for_status()
            if enable_message_logs:
                with open("messages.log", "a") as f:
                    f.write("Agent: ")
                    for line in response.iter_lines():
                        if line:
                            line = line.decode("utf-8")
                            f.write(f"{line}\n")
                            yield line
            else:
                for line in response.iter_lines():
                    if line:
                        yield line.decode("utf-8")

    def get_embedding(self, text: str) -> list:
        """Get embedding for a text snippet"""
        endpoint = "embeddings"
        data = {"model": self.embedding_model, "input": text}

        response = self._make_request(endpoint, data=data)
        try:
            return response["data"][0]["embedding"]
        except KeyError:
            print("No embeddings returned")
            return []

    def _process_tool_chunks(self, tool_chunks: list) -> list:
        """Process streaming tool call chunks into complete tool calls"""
        tool_calls = defaultdict(dict)
        tool_type = "<unknown>"
        tool_name = "<unknown>"
        # tool_id = "-1"

        for chunks in tool_chunks:
            for chunk in chunks:
                index = chunk["index"]
                # Initialize the function call if not already present
                if index not in tool_calls:
                    tool_calls[index] = {}

                try:
                    tool_calls[index]["id"] = chunk["id"]
                except KeyError:
                    pass
                tool_id = tool_calls[index]["id"]

                try:
                    tool_calls[index]["type"] = chunk["type"]
                except KeyError:
                    pass
                tool_type = tool_calls[index]["type"]

                # For now we only support function tools
                if tool_type != "function":
                    print(f"Error: tool request for unknown tool type '{tool_type}'")
                    del tool_calls[index]
                    break

                if not tool_type in tool_calls[index]:
                    tool_calls[index][tool_type] = {"name": "<unknown>", "arguments": ""}

                try:
                    tool_calls[index][tool_type]["name"] = chunk[tool_type]["name"]
                except KeyError:
                    pass
                tool_name = tool_calls[index][tool_type]["name"]

                # Append the arguments chunk
                tool_calls[index][tool_type]["arguments"] += chunk[tool_type].get(
                    "arguments", ""
                )

        # Convert to a list of completed function calls
        return list(tool_calls.values())

    def chat_completion(
        self,
        messages: list,
        tools: Optional[list] = None,
        stream: bool = False,
        on_chunk_callback: Optional[Callable[[str], None]] = None,
        usage_callback: Optional[Callable[[Dict], None]] = None,
    ) -> Dict:
        """
        Get chat completion with optional tool calls and streaming support.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            stream: Whether to stream the response
            on_chunk_callback: Optional callback for content chunks as they arrive
            usage_callback: Optional callback for token usage data

        Returns:
            Dict with 'content' (str) and 'tool_calls' (list)
        """
        endpoint = "chat/completions"
        data = {"model": self.model, "messages": messages}
        if tools:
            data["tools"] = tools
        if stream:
            data["stream"] = True

        data["stream_options"] = {"include_usage": True}

        if stream:
            # Handle streaming response
            full_content = []
            tool_chunks = []

            for chunk in self._make_streaming_request(endpoint, data=data):
                try:
                    chunk_data = chunk.strip()
                    if not chunk_data.startswith("data: "):
                        continue
                    json_str = chunk_data[6:]  # Remove "data: " prefix
                    if json_str == "[DONE]":
                        break

                    chunk_json = json.loads(json_str)
                    # Handle usage data
                    if len(chunk_json.get("choices", [])) == 0:
                        if "usage" in chunk_json and usage_callback:
                            usage_callback(chunk_json["usage"])
                        continue

                    delta = chunk_json["choices"][0]["delta"]

                    if "content" in delta:
                        content_chunk = delta["content"]
                        if type(content_chunk) is str:
                            full_content.append(content_chunk)
                            if on_chunk_callback:
                                on_chunk_callback(content_chunk)
                        elif type(content_chunk) is list:
                            tool_chunks.append(content_chunk)
                        else:
                            print(
                                "Unknown chunk of type",
                                type(content_chunk),
                                "-",
                                content_chunk,
                            )

                    if "tool_calls" in delta:
                        tool_chunks.append(delta["tool_calls"])

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error processing stream chunk: {e}")
                    continue

            # Process tool chunks into complete tool calls
            tool_calls = self._process_tool_chunks(tool_chunks) if tool_chunks else []

            return {"content": "".join(full_content), "tool_calls": tool_calls}

        else:
            # Non-streaming request
            response = self._make_request(endpoint, data=data)

            # Extract content and tool_calls
            if not response.get("choices"):
                return {"content": "", "tool_calls": []}

            choice = response["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")

            # Call callback with full content if provided
            if on_chunk_callback and content:
                on_chunk_callback(content)

            # Extract tool calls
            tool_calls = self.parse_tool_calls(response)

            return {"content": content, "tool_calls": tool_calls}

    def parse_tool_calls(self, response: Dict) -> list:
        """Parse tool calls from chat completion response"""
        if not response.get("choices"):
            return []

        choice = response["choices"][0]
        if "tool_calls" in choice["message"]:
            return [
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in choice["message"]["tool_calls"]
            ]

        return []
