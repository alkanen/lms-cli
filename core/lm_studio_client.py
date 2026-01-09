import requests
import yaml
from typing import Optional, Dict, Any


class LMStudioClient:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.base_url = self.config["lm_studio"]["base_url"]
        self.model = self.config["lm_studio"]["model"]
        self.api_key = self.config["lm_studio"].get("api_key", "")
        # If no embedding model is specified, revert to main model
        self.embedding_model = self.config["embeddings"].get("model", self.model)

        self.system_message = self.system_message = self.config.get("agent", {}).get(
            "system_message", None
        )
        if not self.system_message:
            self.system_message = (
                "You are a helpful coding assistant. Use the provided file contexts to answer "
                "questions about the code."
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

    def get_embedding(self, text: str) -> list:
        """Get embedding for a text snippet"""
        endpoint = f"embeddings"
        data = {"model": self.embedding_model, "input": text}

        response = self._make_request(endpoint, data=data)
        try:
            return response["data"][0]["embedding"]
        except KeyError:
            print("No embeddings returned")
            return []

    def chat_completion(self, messages: list, tools: Optional[list] = None) -> Dict:
        """Get chat completion with optional tool calls"""
        endpoint = f"chat/completions"
        data = {"model": self.model, "messages": messages}
        if tools:
            data["tools"] = tools

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
