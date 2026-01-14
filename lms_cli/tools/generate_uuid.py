from typing import Optional
import uuid

from lms_cli.core.embedding_manager import EmbeddingManager
from lms_cli.core.tool_registry import Tool, ToolRegistry
from lms_cli.core.tool_registry import (
    TOOL_PERMISSION_YES,
    TOOL_PERMISSION_ALWAYS,
    TOOL_PERMISSION_NO,
    TOOL_PERMISSION_USER_SUGGESTION,
)
from lms_cli.core.workspace import Workspace


class generate_uuid(Tool):
    def __init__(self, _context: dict, permission_required: bool):
        super().__init__(
            _context=_context,
            permission_required=permission_required,
            name="generate_uuid",
            description="Generate a UUID string based on the provided 'name'",
        )

        self.always_allowed_names = set()
        self.namespace = uuid.UUID("e00851a0-efc5-11f0-b00c-115acf816a78")

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": (
                                "The 'name' used to create the UUID, "
                                "e.g. a URL or document name"
                            ),
                        }
                    },
                },
            },
        }

    def request_permission(self, name: str):
        # If no setting exists, assume not required
        if not self.permission_required:
            return True, ""

        # User has determined the tool is always allowed
        if self.always_allow:
            return True, ""

        if name in self.always_allowed_names:
            return True, ""

        # If setting exists and is truthy, make a permission request
        option, reason = self.registry.request_permission(
            [f"Always allow UUID generation for '{name}'"]
        )

        if option == TOOL_PERMISSION_ALWAYS:
            self.always_allow = True
            return True, ""

        elif option == TOOL_PERMISSION_YES:
            return True, ""

        elif option == TOOL_PERMISSION_NO:
            return False, "User did not permit tool use"

        elif option == TOOL_PERMISSION_USER_SUGGESTION:
            return False, f"User aborted tool use and instead suggested: {reason}"

        elif option == 0:  # The first option sent by tool
            self.always_allowed_names.add(name)
            return True, ""

        else:
            return False, "User made an invalid choice which aborted the tool"

    def execute(self, name: str):
        return str(uuid.uuid5(self.namespace, name))
