from pathlib import Path
from typing import Set, Tuple

from lms_cli.core.context import CLIContext
from lms_cli.core.tool_registry import Tool
from lms_cli.core.tool_registry import (
    TOOL_PERMISSION_YES,
    TOOL_PERMISSION_ALWAYS,
    TOOL_PERMISSION_NO,
    TOOL_PERMISSION_USER_SUGGESTION,
)


class write_file(Tool):
    def __init__(
        self,
        context: CLIContext,
        permission_required: bool = False,
    ):
        super().__init__(
            context=context,
            permission_required=permission_required,
            name="write_file",
            description=(
                "Write contents to a file in the workspace, can append or overwrite"
            ),
        )
        self.allowed_files: Set[str] = set()
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
                        "file_path": {
                            "type": "string",
                            "required": True,
                            "description": "A relative path to the file",
                        },
                        "content": {
                            "type": "string",
                            "required": True,
                            "description": "The content to write to the file",
                        },
                        "append": {
                            "type": "boolean",
                            "required": False,
                            "description": "True to append, False to overwrite file",
                        },
                    },
                },
            },
        }

    def request_permission(
        self,
        file_path: str,
        content: str,
        append: bool = False,
    ) -> Tuple[bool, str]:
        if file_path in self.allowed_files:
            return True, ""

        if self._in_allowed_folders(file_path):
            return True, ""

        if (
            not Path(file_path)
            .resolve()
            .is_relative_to(self.context.workspace.root_path)
        ):
            return False, "Writing outside of the workspace is not allowed"

        if self.always_allow:
            return True, ""

        if not self.context.tool_registry.request_permission:
            print("Warning: No permission requester is registered, aborting tool")
            return False, f"Unable to grant access to '{self.name}'"

        stripped_path = self.context.workspace.strip_root_path(file_path)
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

        if Path(file_path).exists():
            if append:
                if len(file_path) < 60:
                    question = f"Allow agent to append to file '{file_path}'?"
                else:
                    question = (
                        "Allow agent to append to file "
                        f"'{file_path[:26]}...{file_path[-26:]}'?"
                    )
            else:
                if len(file_path) < 60:
                    question = f"Allow agent to overwrite file '{file_path}'?"
                else:
                    question = (
                        "Allow agent to overwrite file "
                        f"'{file_path[:26]}...{file_path[-26:]}'?"
                    )
        else:
            if len(file_path) < 60:
                question = f"Allow agent to write to file '{file_path}'?"
            else:
                question = (
                    "Allow agent to write to file "
                    f"'{file_path[:26]}...{file_path[-26:]}'?"
                )

        option, reason = self.context.tool_registry.request_permission(
            question, options
        )

        if option == TOOL_PERMISSION_ALWAYS:
            self.always_allow = True
            return True, ""
        elif option == TOOL_PERMISSION_YES:
            return True, ""
        elif option == TOOL_PERMISSION_NO:
            return False, "User did not permit writing to file"
        elif option == TOOL_PERMISSION_USER_SUGGESTION:
            return False, f"User aborted write_file and instead suggested: {reason}"
        elif option == 0:  # Always allow entire workspace
            self.allowed_folders.add(str(self.context.workspace.root_path))
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
        content: str,
        append: bool = False,
    ) -> str:
        try:
            return self.context.workspace.write_file(file_path, content, append)
        except Exception as e:
            return f"Unable to write to file '{file_path}': {e}"

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
