"""
update — perform exact string replacements in a file on the local filesystem.

When a compatible ``read`` tool is attached, the file must have a valid hash
recorded in it this session (from an explicit ``read`` call *or* from a previous
``write``/``update`` that recorded the hash), and the file must not have changed
since that hash was recorded.  The edit is rejected if no hash is present or if
the current content no longer matches — this prevents silently clobbering
modifications made outside the session.  Without an attached ``read`` tool the
hash gate is skipped and edits proceed unconditionally.

After a successful write the ``read`` tool's hash record is updated so that
subsequent edits to the same file do not require a re-read.

Permission required by default.  Disabled by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.workspace import Workspace
    from ai_cli.tools.read import ReadTool


class UpdateTool(Tool):
    NAME = "update"
    DESCRIPTION = (
        "Performs exact string replacements in files.\n\n"
        "Usage:\n"
        "- The file must have a valid hash recorded this session before editing — "
        "either by reading it with the `Read` tool, or by having written it with the "
        "`Write` tool earlier in the conversation. "
        "This check is enforced only when a compatible `read` tool is wired; "
        "without one, edits proceed without hash validation.\n"
        "- When editing text from Read tool output, ensure you preserve the exact indentation "
        "(tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format "
        "is: line number + tab. Everything after that is the actual file content to match. "
        "Never include any part of the line number prefix in the old_string or new_string.\n"
        "- ALWAYS prefer editing existing files in the codebase. "
        "NEVER write new files unless explicitly required.\n"
        "- Only use emojis if the user explicitly requests it. "
        "Avoid adding emojis to files unless asked.\n"
        "- The edit will FAIL if `old_string` is not unique in the file. "
        "Either provide a larger string with more surrounding context to make it unique "
        "or use `replace_all` to change every instance of `old_string`.\n"
        "- Use `replace_all` for replacing and renaming strings across the file. "
        "This parameter is useful if you want to rename a variable for instance."
    )
    PERMISSION_REQUIRED = True
    DISABLED_BY_DEFAULT = True

    def __init__(
        self,
        workspace: Workspace,
        permission_manager: PermissionManager,
        permission_required: bool,
        name: str,
        description: str,
    ) -> None:
        super().__init__(
            workspace, permission_manager, permission_required, name, description
        )
        self._read_tool: ReadTool | None = None

    def set_read_tool(self, read_tool: ReadTool | None) -> None:
        """Attach the ReadTool used for read-before-edit enforcement."""
        self._read_tool = read_tool

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def execute_log(self, **kwargs: object) -> str | None:
        file_path = kwargs.get("file_path", "")
        old_string = kwargs.get("old_string", "")
        new_string = kwargs.get("new_string", "")
        replace_all = kwargs.get("replace_all", False)
        return (
            f"file_path={file_path!r} replace_all={replace_all}"
            f" old_string=<str:{len(str(old_string))}ch>"
            f" new_string=<str:{len(str(new_string))}ch>"
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    name="file_path",
                    description="The absolute path to the file to modify",
                    argument_type="string",
                    required=True,
                ),
                ToolArgument(
                    name="old_string",
                    description="The text to replace",
                    argument_type="string",
                    required=True,
                ),
                ToolArgument(
                    name="new_string",
                    description="The text to replace it with (must be different from old_string)",
                    argument_type="string",
                    required=True,
                ),
                ToolArgument(
                    name="replace_all",
                    description="Replace all occurrences of old_string (default false)",
                    argument_type="boolean",
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(  # type: ignore[override]
        self,
        *,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> dict:
        logger.debug("update: '%s' (replace_all=%s)", file_path, replace_all)

        path = Path(file_path)
        if not path.is_absolute():
            return self._err_invalid_arguments(
                f"file_path must be an absolute path, got: '{file_path}'"
            )

        if not old_string:
            return self._err_invalid_arguments("old_string must not be empty.")

        if old_string == new_string:
            return self._err_invalid_arguments(
                "old_string and new_string must be different."
            )

        # Resolve symlinks so ignore checks and I/O both target the real file.
        resolved = path.resolve()

        if self._workspace.contains(resolved) and self._workspace.is_ignored(
            resolved, is_dir=False
        ):
            return self._err_read_error(
                f"Path is excluded by ignore rules: '{file_path}'"
            )

        if not resolved.exists():
            return self._err_read_error(f"File not found: '{file_path}'")

        if not resolved.is_file():
            return self._err_read_error(f"Path is not a file: '{file_path}'")

        # Read the file once so we can both validate the hash and perform the
        # replacement without a second read (which would introduce a TOCTOU gap).
        try:
            content = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return self._err_read_error(f"Cannot read '{file_path}': {exc}")

        if self._read_tool is not None and not self._read_tool.validate_content(
            str(resolved), content, caller="update"
        ):
            return self._err_invalid_arguments(
                f"'{file_path}' must be read with the read tool before editing. "
                "Read the file first, or re-read it if it has changed."
            )

        count = content.count(old_string)
        if count == 0:
            return self._err_invalid_arguments(
                f"old_string not found in '{file_path}'."
            )
        if not replace_all and count > 1:
            return self._err_invalid_arguments(
                f"old_string is not unique in '{file_path}' ({count} occurrences). "
                "Provide more surrounding context to make it unique, "
                "or pass replace_all=true to replace every occurrence."
            )

        new_content = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        replacements = count if replace_all else 1

        try:
            resolved.write_text(new_content, encoding="utf-8")
        except (OSError, UnicodeEncodeError) as exc:
            return self._err_write_error(f"Cannot write '{file_path}': {exc}")

        if self._read_tool is not None:
            self._read_tool.record_hash(str(resolved), new_content, writer="update")

        logger.debug(
            "update: replaced %d occurrence(s) in '%s'", replacements, file_path
        )
        return self._ok({"file_path": file_path, "replacements": replacements})
