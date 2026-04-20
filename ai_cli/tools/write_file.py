"""
write_file — write or partially replace a file in the workspace.

Permission is required by default.  The tool manages its own session-scoped
allow-list so the user can grant permanent-for-session write access at the
file level or at any ancestor directory level up to the workspace root.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_cli.core.workspace import WorkspaceError
from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.workspace import Workspace


class WriteFileTool(Tool):
    NAME = "write_file"
    DESCRIPTION = "Write or partially replace a file in the workspace."
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
        # Session-scoped allow-lists; cleared by reset_session_state().
        self._session_allowed_files: set[Path] = set()
        self._session_allowed_dirs: set[Path] = set()

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------

    def reset_session_state(self) -> None:
        self._session_allowed_files.clear()
        self._session_allowed_dirs.clear()

    # ------------------------------------------------------------------
    # Permission helpers
    # ------------------------------------------------------------------

    def request_permission(self, action: str, **kwargs: Any) -> tuple[bool, str]:
        """Check the tool's own allow-list before delegating to PermissionManager."""
        if not self.permission_required:
            return True, ""
        path_str = kwargs.get("path", "")
        if path_str:
            try:
                resolved = self._workspace.resolve(path_str)
            except WorkspaceError:
                pass
            else:
                if resolved in self._session_allowed_files:
                    return True, ""
                if any(p in self._session_allowed_dirs for p in resolved.parents):
                    return True, ""
        return super().request_permission(action, **kwargs)

    def extra_permission_options(self, **kwargs: Any) -> list[str]:
        """
        Return one option per level of the path hierarchy, from the file
        itself up to (and including) the workspace root.

        Example for ``path="./src/foo/bar.py"``:

            file:./src/foo/bar.py
            dir:./src/foo/
            dir:./src/
            dir:./
        """
        path_str = kwargs.get("path", "")
        if not path_str:
            return []
        try:
            resolved = self._workspace.resolve(path_str)
        except WorkspaceError:
            return []

        root = self._workspace.root
        # A path that resolves to the workspace root is a directory, not a file.
        if resolved == root:
            return []

        # Normalise the file label to a workspace-relative ./… path with
        # forward slashes, matching the format used for dir: labels below.
        file_rel = resolved.relative_to(root)
        file_label = "./" + str(file_rel).replace("\\", "/")

        options: list[str] = [f"file:{file_label}"]
        current = resolved.parent
        while True:
            rel = current.relative_to(root)
            rel_str = str(rel).replace("\\", "/")
            dir_label = "./" if rel_str == "." else f"./{rel_str}/"
            options.append(f"dir:{dir_label}")
            if current == root:
                break
            current = current.parent
        return options

    def on_permission_granted(self, choice: str, **kwargs: Any) -> None:
        kind, _, path_str = choice.partition(":")
        if not path_str:
            return
        try:
            resolved = self._workspace.resolve(path_str.rstrip("/"))
        except WorkspaceError:
            return
        if kind == "file":
            self._session_allowed_files.add(resolved)
        elif kind == "dir":
            self._session_allowed_dirs.add(resolved)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def execute_log(self, **kwargs: Any) -> str | None:
        path = kwargs.get("path", "?")
        content = kwargs.get("content", "")
        start_line = kwargs.get("start_line")
        end_line = kwargs.get("end_line")
        size = f"{len(content)} chars"
        if start_line is not None and end_line is not None:
            range_info = f"lines {start_line}–{end_line}"
        else:
            range_info = "full write"
        return f"'{path}' — {range_info}, {size}"

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=(
                "Write content to a workspace file. "
                "If start_line and end_line are omitted, performs a full write — "
                "creating the file and any missing parent directories. "
                "If both are provided, replaces only those lines in an existing "
                "file (the file must already exist for partial writes)."
            ),
            arguments=[
                ToolArgument(
                    name="path",
                    description=(
                        "Path to the file, relative to the workspace root "
                        "(e.g. './src/main.py'). For full writes, parent "
                        "directories are created automatically. For partial "
                        "writes, the file must already exist."
                    ),
                    argument_type="string",
                    required=True,
                ),
                ToolArgument(
                    name="content",
                    description=(
                        "For a full write: the complete new file content. "
                        "For a partial write: the replacement text for lines "
                        "start_line through end_line."
                    ),
                    argument_type="string",
                    required=True,
                ),
                ToolArgument(
                    name="start_line",
                    description=(
                        "1-based first line to replace (inclusive). "
                        "Must be provided together with end_line for a "
                        "partial write. Omit for a full write."
                    ),
                    argument_type="integer",
                ),
                ToolArgument(
                    name="end_line",
                    description=(
                        "1-based last line to replace (inclusive). "
                        "Must be provided together with start_line for a "
                        "partial write. Omit for a full write."
                    ),
                    argument_type="integer",
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(  # type: ignore[override]
        self,
        *,
        path: str,
        content: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict:
        logger.debug(
            "write_file: '%s' (%s)",
            path,
            f"lines {start_line}–{end_line}"
            if start_line is not None
            else "full write",
        )
        # Reject supplying only one of the pair.
        if (start_line is None) != (end_line is None):
            return self._err(
                "invalid_range",
                "start_line and end_line must be provided together for a partial write.",
                400,
            )

        # Pre-validate range values so callers get 'invalid_range' (bad input)
        # rather than 'write_error' (I/O failure) for out-of-bounds line numbers.
        if start_line is not None and start_line < 1:
            return self._err(
                "invalid_range", f"start_line must be >= 1, got {start_line}.", 400
            )
        if end_line is not None and end_line < 1:
            return self._err(
                "invalid_range", f"end_line must be >= 1, got {end_line}.", 400
            )
        if start_line is not None and end_line is not None and start_line > end_line:
            return self._err(
                "invalid_range",
                f"start_line ({start_line}) must be <= end_line ({end_line}).",
                400,
            )

        try:
            summary = self._workspace.write_file(
                path, content, start_line=start_line, end_line=end_line
            )
        except WorkspaceError as exc:
            return self._err("write_error", str(exc), 400)

        lines_written = len(content.splitlines()) if content else 0
        logger.info(
            "write_file: wrote %d line(s) to '%s' (%s)", lines_written, path, summary
        )
        return self._ok(
            {
                "path": path,
                "summary": summary,
                "lines_written": lines_written,
            }
        )
