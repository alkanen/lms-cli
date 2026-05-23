"""
write — write or overwrite a file on the local filesystem.

When a compatible ``read`` tool is attached, files that existed *before this
session* require a prior ``read`` call (no hash will be recorded for them yet),
and the file must not have changed since that read.  Files *created during this
session* by a ``write`` call already have a hash recorded, so a subsequent
overwrite by the same tool is allowed without an explicit re-read.  Without an
attached ``read`` tool the hash gate is skipped and overwrites proceed
unconditionally.  New files (path does not yet exist) may always be written
freely; any missing parent directories are created automatically.

After a successful write the ``read`` tool's hash record is updated so that
subsequent edits via ``update`` do not require a re-read.

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


class WriteTool(Tool):
    NAME = "write"
    DESCRIPTION = (
        "Writes a file to the local filesystem.\n\n"
        "Usage:\n"
        "- This tool will overwrite the existing file if there is one at the provided path.\n"
        "- If this is an existing file, you MUST use the Read tool first to read the file's "
        "contents. This guard is enforced only when a compatible `read` tool is wired; "
        "without one, overwrites proceed without hash validation.\n"
        "- Prefer the update tool for modifying existing files — it only sends the diff. "
        "Only use this tool to create new files or for complete rewrites.\n"
        "- NEVER create documentation files (*.md) or README files unless explicitly "
        "requested by the User.\n"
        "- Only use emojis if the user explicitly requests it. "
        "Avoid writing emojis to files unless asked."
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
        """Attach the ReadTool used for read-before-write enforcement on existing files."""
        self._read_tool = read_tool

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def execute_log(self, **kwargs: object) -> str | None:
        file_path = kwargs.get("file_path", "")
        content = kwargs.get("content", "")
        action = "overwrite" if Path(str(file_path)).exists() else "create"
        return f"file_path={file_path!r} action={action} content=<str:{len(str(content))}ch>"

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
                    description="The absolute path to the file to write (must be absolute, not relative)",
                    argument_type="string",
                    required=True,
                ),
                ToolArgument(
                    name="content",
                    description="The content to write to the file",
                    argument_type="string",
                    required=True,
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
        content: str,
    ) -> dict:
        logger.debug("write: '%s'", file_path)

        path = Path(file_path)
        if not path.is_absolute():
            return self._err_invalid_arguments(
                f"file_path must be an absolute path, got: '{file_path}'"
            )

        # Resolve symlinks so ignore checks and I/O both target the real location.
        resolved = path.resolve()

        if self._workspace.contains(resolved) and self._workspace.is_ignored(
            resolved, is_dir=False
        ):
            return self._err_write_error(
                f"Path is excluded by ignore rules: '{file_path}'"
            )

        if resolved.exists() and not resolved.is_file():
            return self._err_write_error(f"Path is not a file: '{file_path}'")

        if (
            resolved.is_file()
            and self._read_tool is not None
            and not self._read_tool.has_been_read(str(resolved), caller="write")
        ):
            return self._err_invalid_arguments(
                f"'{file_path}' already exists and must be read with the read tool "
                "before overwriting. Read the file first, or re-read it if it has changed."
            )

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except (OSError, UnicodeEncodeError) as exc:
            return self._err_write_error(f"Cannot write '{file_path}': {exc}")

        if self._read_tool is not None:
            self._read_tool.record_hash(str(resolved), content, writer="write")

        logger.debug("write: wrote %d chars to '%s'", len(content), file_path)
        return self._ok({"file_path": file_path})
