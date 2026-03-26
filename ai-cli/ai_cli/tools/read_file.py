"""
read_file — read a file (or line range) from the workspace.

Disabled by default, no permission required by default.  When
``permission_required`` is enabled via config, the tool maintains its own
session-scoped allow-list so the user can grant permanent-for-session access
at the file level or at any ancestor directory level up to the workspace root.
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


class ReadFileTool(Tool):
    NAME = "read_file"
    DESCRIPTION = (
        "Read a file (or line range) from the workspace. "
        "Returns start_line, end_line, lines_returned, and total_lines (1-based, inclusive). "
        "For an empty file, start_line and end_line are both 0."
    )
    PERMISSION_REQUIRED = False
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
    # Schema
    # ------------------------------------------------------------------

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    name="path",
                    description=(
                        "Path to the file, relative to the workspace root "
                        "(e.g. './src/main.py')."
                    ),
                    argument_type="string",
                    required=True,
                ),
                ToolArgument(
                    name="start_line",
                    description=(
                        "1-based first line to read (inclusive). "
                        "Omit to start from the beginning of the file."
                    ),
                    argument_type="integer",
                ),
                ToolArgument(
                    name="end_line",
                    description=(
                        "1-based last line to read (inclusive). "
                        "Omit to read to the end of the file."
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
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict:
        logger.debug(
            "read_file: '%s' (lines %s–%s)",
            path,
            start_line if start_line is not None else "start",
            end_line if end_line is not None else "end",
        )
        # Read the full file once; slicing and total_lines are derived here.
        try:
            full_text = self._workspace.read_file(path)
        except WorkspaceError as exc:
            logger.debug("read_file: error reading '%s': %s", path, exc)
            return self._err("read_error", str(exc), 400)

        all_lines = full_text.splitlines(keepends=True)
        total_lines = len(all_lines)

        # Validate requested range.
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
        if start_line is not None and start_line > total_lines:
            return self._err(
                "invalid_range",
                f"start_line ({start_line}) exceeds file length ({total_lines} line(s)) for '{path}'.",
                400,
            )
        if end_line is not None and end_line > total_lines:
            return self._err(
                "invalid_range",
                f"end_line ({end_line}) exceeds file length ({total_lines} line(s)) for '{path}'.",
                400,
            )

        # Empty file: return a consistent zero-based sentinel so that
        # start_line <= end_line always holds (both 0 means "no lines").
        if total_lines == 0:
            return self._ok(
                {
                    "content": "",
                    "path": path,
                    "start_line": 0,
                    "end_line": 0,
                    "lines_returned": 0,
                    "total_lines": 0,
                }
            )

        lo = (start_line - 1) if start_line is not None else 0
        hi = end_line if end_line is not None else total_lines
        content = "".join(all_lines[lo:hi])

        logger.debug(
            "read_file: returned %d/%d lines from '%s'",
            hi - lo,
            total_lines,
            path,
        )
        return self._ok(
            {
                "content": content,
                "path": path,
                "start_line": lo + 1,
                "end_line": hi,
                "lines_returned": hi - lo,
                "total_lines": total_lines,
            }
        )
