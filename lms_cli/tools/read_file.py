from pathlib import Path
from typing import Optional, Set, Tuple

from lms_cli.core.tool_registry import Tool, ToolRegistry
from lms_cli.core.tool_registry import (
    TOOL_PERMISSION_YES,
    TOOL_PERMISSION_ALWAYS,
    TOOL_PERMISSION_NO,
    TOOL_PERMISSION_USER_SUGGESTION,
)


class read_file(Tool):
    def __init__(
        self,
        _context: dict,
        permission_required: bool = False,
        allow_outside_workspace: bool = False,
    ):
        super().__init__(
            _context=_context,
            permission_required=permission_required,
            name="read_file",
            description="Read contents from a file in the workspace",
        )
        self.allowed_files: Set[str] = set()
        self.allowed_folders: Set[str] = set()
        self.allow_outside_workspace = allow_outside_workspace

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "required": True,
                            "description": "A relative path to the file",
                        },
                        "start_line": {
                            "type": "integer",
                            "required": False,
                            "description": "Optional first line to read, starts from 1",
                        },
                        "end_line": {
                            "type": "integer",
                            "required": False,
                            "description": "Optional last line to read, starts from 1",
                        },
                    },
                },
            },
        }

    def request_permission(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Tuple[bool, str]:
        if self.always_allow:
            return True, ""

        if file_path in self.allowed_files:
            return True, ""

        if self._in_allowed_folders(file_path):
            return True, ""

        if self.allow_outside_workspace:
            if not Path(file_path).exists():
                return False, "File does not exist"
        else:
            if not self.workspace.file_exists(file_path):
                return False, "File does not exist in workspace"

        stripped_path = self.workspace.strip_root_path(file_path)
        stripped_parts = Path(stripped_path).parts

        progressive_paths = [
            Path(".").joinpath(*stripped_parts[:i])
            for i in range(1, len(stripped_parts) + 1)
        ]

        options = ["Always allow in workspace folder"]
        options.extend(
            [f"Always allow in '{folder}/'" for folder in progressive_paths[:-1]]
        )
        options.append(f"Allows allow on '{progressive_paths[-1]}'")

        lines = ""
        if start_line and end_line:
            lines = f" lines {start_line} through {end_line}"
        elif start_line:
            lines = f" from line {start_line}"
        elif end_line:
            lines = f" to line {end_line}"

        if len(file_path) < 60:
            question = f"Allow agent to read file '{file_path}'{lines}?"
        else:
            question = f"Allow agent to read file '{file_path[:26]}...{file_path[-26:]}'{lines}?"

        option, reason = self.registry.request_permission(question, options)

        if option == TOOL_PERMISSION_ALWAYS:
            self.always_allow = True
            return True, ""
        elif option == TOOL_PERMISSION_YES:
            return True, ""
        elif option == TOOL_PERMISSION_NO:
            return False, "User did not permit reading file '{file_path}'"
        elif option == TOOL_PERMISSION_USER_SUGGESTION:
            return False, f"User aborted read_file and instead suggested: {reason}"
        elif option == 0:  # Always allow entire workspace
            self.allowed_folders.add(str(self.workspace.root_path))
            return True, ""
        elif option == len(options) - 1:  # Always allow on specific file
            self.allowed_files.add(file_path)
            return True, ""
        elif option > 0 and option < len(options) - 1:  # Always allow on subfolder
            subfolder = str(progressive_paths[option - 1])
            self.allowed_folders.add(subfolder)
            return True, ""
        else:
            return False, "User made an invalid choice which aborted write_file"

    def execute(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        try:
            return self.workspace.read_file(file_path, start_line, end_line)
        except:
            return f"Unable to read from file '{file_path}'"

    def _in_allowed_folders(self, file_path: str) -> bool:
        """Returns true if the file_path is within the working directory"""
        file_parts = Path(file_path).parts

        # Go through all allowed folders until one of them matches fully
        for folder in self.allowed_folders:
            # See if we can loop through all the folder path parts without breaking
            for i, part in enumerate(Path(folder).parts):
                if part != file_parts[i]:
                    break
            else:  # Didn't break, means file is entirely in the folder
                return True

        return False
