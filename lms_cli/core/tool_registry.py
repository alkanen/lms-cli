from collections import defaultdict
import importlib
import inspect
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Tuple, Optional

import json
import yaml

from lms_cli.core.workspace import Workspace
from lms_cli.core.embedding_manager import EmbeddingManager


# Default permission request response values. If extra options are provided and
# selected, response value will be the index of the selected option.
TOOL_PERMISSION_YES = -1
TOOL_PERMISSION_ALWAYS = -2
TOOL_PERMISSION_NO = -3
TOOL_PERMISSION_USER_SUGGESTION = -4


class Tool:
    def __init__(
        self,
        _context: dict,
        name: str,
        description: Optional[str] = None,
        permission_required: bool = False,
    ):
        self.name = name
        self.description = description
        self.permission_required = permission_required

        registry = _context["tool_registry"]
        ws = _context["workspace"]
        em = _context["embedding_manager"]
        self.always_allow = False
        self.registry = registry
        self.workspace = ws
        self.embedding_manager = em

    def definition(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": []
        }

    def request_permission(self, *args, **kwargs) -> Tuple[bool, str]:
        # If no setting exists, assume not required
        if not self.permission_required:
            return True

        # If setting exists and is truthy, make a permission request
        return self.registry.request_permission([])

    def execute(self, *args, **kwargs) -> str:
        return "This tool performs no action and returns no useful data"


class ToolRegistry:
    def __init__(
        self,
        permission_request_cb: Callable[List[str], Tuple[int, str]],
        config_path: str = "config/config.yaml",
        workspace: str = ".",
    ):
        self.tools = {}
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.context = {
            "workspace": Workspace(workspace),
            "embedding_manager": EmbeddingManager(self.config_path),
            "tool_registry": self,
        }
        self.request_permission = permission_request_cb

    def register_tool(
        self, name: str, tool: Tool, description: str, parameters: List[Dict]
    ):
        """Register a tool with the registry"""
        self.tools[name] = {
            "object": func,
            "description": description,
            "parameters": parameters,
        }

    def get_tool_definition(self, name: str) -> Dict:
        """Get the definition of a registered tool"""
        if name not in self.tools:
            raise ValueError(
                f"ToolRegistry::get_tool_definition(): Tool {name} not found"
            )

        return self.tools[name]["class"].definition()

    def execute_tool(self, name: str, arguments: str) -> Any:
        """Execute a registered tool with given arguments"""
        if name not in self.tools:
            raise ValueError(f"ToolRegistry::execute_tool(): Tool {name} not found")

        arguments_dict = json.loads(arguments)
        tool = self.tools[name]["class"]

        # Make sure we have permission or report back a rejection reason to the agent
        allowed, reason = tool.request_permission(**arguments_dict)
        if not allowed:
            return reason

        # Permission granted, perform the tool function
        return self.tools[name]["class"].execute(**arguments_dict)

    def load_tools(self):
        """Load tools from tools folder in configuration file"""
        tools_conf = self.config.get("tools", {})
        if not tools_conf:
            print("No tools loaded")
            return

        # Copy any customization settings that might exist so we can send them to the initializer
        settings = tools_conf["tools_settings"]
        settings = {
            item["name"]: {
                key: value
                for key, value in item.items() if key != "name"
            } for item in settings
        }

        # Go through the tools folder looking for implementations of the Tool interface
        folder = Path(tools_conf["tools_folder"])
        if not folder.exists():
            print(f"Tools folder does not exist: '{folder}'")
            return

        if not folder.is_dir():
            print(f"'{folder}' is not a folder")
            return

        for script in folder.glob("*.py"):
            filename = str(script)

            # Skip special scripts
            if filename.startswith("_"):
                continue

            module_name = filename[:-3]  # Remove extension

            try:
                # Import the module dynamically
                spec = importlib.util.spec_from_file_location(module_name, script)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Iterate over all members in the module
                for name, obj in inspect.getmembers(module):
                    tool_settings = settings.get(name, {})
                    if tool_settings.get("disabled", False):
                        continue

                    # Remove disabled flag from setting to avoid problems initializing
                    tool_settings.pop("disabled", None)

                    if (
                        inspect.isclass(obj)
                        and hasattr(obj, "__bases__")
                        and any(base.__name__ == "Tool" for base in obj.__bases__)
                    ):
                        self.tools[name] = {
                            "class": obj(
                                _context=self.context,
                                **tool_settings
                            )
                        }
                        print(f"Loaded tool '{name}'")

            except Exception as e:
                print(f"Error importing module {module_name}: {e}")
                raise e

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
                        "function": {"name": tool_name, "arguments": ""},
                    }

                # Append the arguments chunk
                tool_calls[index]["function"]["arguments"] += chunk["function"].get(
                    "arguments", ""
                )

        # Convert to a list of completed function calls
        completed_tools = list(tool_calls.values())
        return completed_tools
