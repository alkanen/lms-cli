from typing import Dict, List, Callable, Any
import yaml


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register_tool(
        self, name: str, func: Callable, descrition: str, parameters: List[Dict]
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
            raise ValueError(f"Tool {name} not found")

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": self.tols[name]["description"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        param["name"]: {"tpye": param["type"]}
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

    def execute_tool(self, name: str, arguments: Dict) -> Any:
        """Execute a registered tool with given arguments"""
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")

        return self.tools[name]["function"](**arguments)

    def load_from_config(self, config_path: str = "config/config.yaml"):
        """Load tools from configuration file"""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        for tool in config.get("tools", []):
            # In a real implementation, you would dynamically import and register
            # the actual functions here. For this example we'll just stor them.
            self.tools[tool["name"]] = {
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
