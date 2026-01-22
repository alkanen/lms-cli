from typing import Optional, Set, Tuple

from lms_cli.core.context import CLIContext
from lms_cli.core.tool_registry import Tool


class file_search(Tool):
    def __init__(self, context: CLIContext, permission_required: bool = False):
        super().__init__(
            context=context,
            permission_required=permission_required,
            name="file_search",
            description="Search for files in the workspace",
        )
        self.allowed_folders: Set[str] = set()

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "extension": {
                            "type": "string",
                            "required": False,
                            "description": "An optional file extension to filter on",
                        }
                    },
                },
            },
        }

    def request_permission(self, extension: Optional[str] = None) -> Tuple[bool, str]:
        return True, ""

    def execute(self, extension: Optional[str] = None) -> str:
        included_set = self.context.embedding_manager.inclusion_paths
        excluded_set = self.context.embedding_manager.exclusion_paths

        files = self.context.workspace.list_files(
            extension=extension,
            included_folders=included_set,
            excluded_folders=excluded_set,
        )

        return [
            self.context.workspace.strip_workspace_folder_from_filename(f)
            for f in files
        ]
