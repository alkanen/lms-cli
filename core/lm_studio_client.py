from typing import Dict, Iterator, Optional

import json
import requests
import yaml


class LMStudioClient:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.base_url = self.config["lm_studio"]["base_url"]
        self.model = self.config["lm_studio"]["model"]
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

        url = f"{self.base_url}/{endpoint}"
        with requests.post(url, json=data, headers=headers, stream=True) as response:
            if response.status_code >= 400:
                import pdb
                pdb.set_trace()

            response.raise_for_status()
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

    def chat_completion(
        self, messages: list, tools: Optional[list] = None, stream: bool = False
    ) -> Dict:
        """Get chat completion with optional tool calls and streaming support"""
        endpoint = "chat/completions"
        data = {"model": self.model, "messages": messages}
        if tools:
            data["tools"] = tools
        if stream:
            data["stream"] = True

        if stream:
            # Handle streaming response
            for chunk in self._make_streaming_request(endpoint, data=data):
                try:
                    chunk_data = chunk.strip()
                    if not chunk_data.startswith("data: "):
                        continue
                    json_str = chunk_data[6:]  # Remove "data: " prefix
                    if json_str == "[DONE]":
                        break

                    chunk_json = json.loads(json_str)
                    delta = chunk_json["choices"][0]["delta"]

                    partial = {}

                    if "content" in delta:
                        chunk = delta["content"]
                        if type(chunk) is str:
                            partial["chunk"] = chunk
                        elif type(chunk) is list:
                            partial["tool_chunks"] = chunk
                        else:
                            print("Unknown chunk of type", type(chunk), "-", chunk)

                    if "tool_calls" in delta:
                        partial["tool_chunks"] = delta["tool_calls"]

                    if partial:
                        yield partial

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error processing stream chunk: {e}")
                    continue

            # After streaming is complete, stop iteration
            return

        else:
            # Non-streaming request
            return self._make_request(endpoint, data=data)

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
