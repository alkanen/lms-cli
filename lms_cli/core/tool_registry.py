from collections import defaultdict
import importlib
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List

import json
import yaml

from lms_cli.core.workspace import Workspace
from lms_cli.core.embedding_manager import EmbeddingManager


class ToolRegistry:
    def __init__(self, config_path: str = "config/config.yaml", workspace: str = "."):
        self.tools = {}
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.context = {
            "workspace": Workspace(workspace),
            "embedding_manager": EmbeddingManager(self.config_path),
        }

    def register_tool(
        self, name: str, func: Callable, description: str, parameters: List[Dict]
    ):
        """Register a tool with the registry"""
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters,
        }

    def get_tool_definition(self, name: str) -> Dict:
        """Get the definition of a registered tool"""
        if name not in self.tools:
            raise ValueError(
                f"ToolRegistry::get_tool_definition(): Tool {name} not found"
            )

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": self.tools[name]["description"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        param["name"]: {"type": param["type"]}
                        for param in self.tools[name]["parameters"]
                    },
                    "required": [
                        param["name"]
                        for param in self.tools[name]["parameters"]
                        if param.get("required", False)
                    ],
                },
            },
        }

    def execute_tool(self, name: str, arguments: str) -> Any:
        """Execute a registered tool with given arguments"""
        if name not in self.tools:
            raise ValueError(f"ToolRegistry::execute_tool(): Tool {name} not found")

        arguments_dict = json.loads(arguments)

        return self.tools[name]["function"](_context=self.context, **arguments_dict)

    def load_from_config(self):
        """Load tools from configuration file"""
        for tool in self.config.get("tools", []):
            # Dynamically import and register the tool module
            self._load_tool_module(tool)

    def _load_tool_module(self, tool: Dict[str, Any]) -> Callable:
        """Dynamically load a tool module from the tools folder"""

        tool_name = tool["name"]
        tool_description = tool["description"]
        tool_parameters = tool["parameters"]

        root_folder = Path(__file__).resolve().parent
        tools_folder = (root_folder / "../tools").resolve()
        module_path = (tools_folder / f"{tool_name}.py").resolve()

        if not module_path.exists():
            raise ValueError(
                f"Registered tool '{tool_name}' does not exist in '{tools_folder}'"
            )

        spec = importlib.util.spec_from_file_location(tool_name, module_path)
        if spec is None:
            raise ValueError(f"Could not load tool module {tool_name}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[tool_name] = module
        spec.loader.exec_module(module)

        self.tools[tool_name] = {
            "description": tool_description,
            "parameters": tool_parameters,
        }
        # Extract the function name from the tool name
        func_name = tool_name
        if hasattr(module, func_name):
            self.tools[tool_name]["function"] = getattr(module, func_name)

    @staticmethod
    def process_tool_chunks(tool_chunks):
        tool_calls = defaultdict(dict)
        tool_name = "<unknown>"
        tool_id = "-1"

        for chunks in tool_chunks:
            for chunk in chunks:
                index = chunk["index"]
                tool_type = chunk["type"]

                try:
                    tool_id = chunk["id"]
                except KeyError:
                    pass

                try:
                    tool_name = chunk["function"]["name"]
                except KeyError:
                    pass

                # Initialize the function call if not already present
                if index not in tool_calls:
                    tool_calls[index] = {
                        "id": tool_id,
                        "type": tool_type,
                        "function": {
                            "name": tool_name,
                            "arguments": ""
                        }
                    }

                # Append the arguments chunk
                tool_calls[index]["function"]["arguments"] += chunk["function"].get(
                    "arguments", ""
                )

        # Convert to a list of completed function calls
        completed_tools = list(tool_calls.values())
        return completed_tools
