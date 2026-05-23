"""
read — read a file (or line range) from the local filesystem.

Accepts absolute paths only.  Returns content in cat -n format (1-based
line numbers followed by a tab and the line content).  Reads up to *limit*
lines (default 2000) starting at *offset* (0-based) lines from the beginning.

Permission required by default.  Disabled by default.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.workspace import Workspace

_DEFAULT_LIMIT = 2000


class ReadTool(Tool):
    NAME = "read"
    DESCRIPTION = (
        "Reads a file from the local filesystem. "
        "You can access any file directly by using this tool. "
        "Assume this tool is able to read all files on the machine. "
        "If the User provides a path to a file assume that path is valid. "
        "It is okay to read a file that does not exist; an error will be returned."
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
        # Maps resolved absolute path → SHA-256 hash of content at time of last read/write.
        # Cleared by reset_session_state().
        self._read_hashes: dict[str, str] = {}
        # Maps resolved absolute path → name of the tool that last wrote the file,
        # or None if the last operation was an actual read (no tool has written yet).
        # Used to prevent one tool from overwriting another tool's unseen changes.
        self._last_writer: dict[str, str | None] = {}

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------

    def reset_session_state(self) -> None:
        self._read_hashes.clear()
        self._last_writer.clear()

    # ------------------------------------------------------------------
    # Read-tracking API (used by UpdateTool)
    # ------------------------------------------------------------------

    def has_been_read(self, file_path: str, *, caller: str | None = None) -> bool:
        """Return True if *file_path* has a recorded hash (read or written this
        session), its content is still unchanged, and no *other* tool has written
        it since the last actual read.

        ``caller`` is the name of the requesting tool (e.g. ``"update"`` or
        ``"write"``).  If omitted the writer-tag check is skipped (legacy /
        test usage).

        Returns False if:
        - no hash has been recorded for the file (never read or written), or
        - its content has changed since the last read/write (external edit), or
        - a *different* tool wrote it since the last actual read (would silently
          clobber that tool's unseen changes).
        """
        path = Path(file_path)
        if not path.is_absolute():
            return False
        resolved = path.resolve()
        resolved_key = str(resolved)
        if resolved_key not in self._read_hashes:
            return False
        if not resolved.exists() or not resolved.is_file():
            return False
        try:
            current_text = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        if (
            hashlib.sha256(current_text.encode()).hexdigest()
            != self._read_hashes[resolved_key]
        ):
            return False
        if caller is not None:
            last = self._last_writer.get(resolved_key)  # None = freshly read
            if last is not None and last != caller:
                return False
        return True

    def validate_content(
        self, file_path: str, content: str, *, caller: str | None = None
    ) -> bool:
        """Return True if *content* matches the recorded hash for *file_path*
        and no other tool has written it since the last actual read.

        Like ``has_been_read`` but takes the already-read content directly,
        avoiding a second filesystem read and the TOCTOU window it would
        introduce between validation and use.
        """
        path = Path(file_path)
        if not path.is_absolute():
            return False
        resolved_key = str(path.resolve())
        if resolved_key not in self._read_hashes:
            return False
        if (
            hashlib.sha256(content.encode()).hexdigest()
            != self._read_hashes[resolved_key]
        ):
            return False
        if caller is not None:
            last = self._last_writer.get(resolved_key)
            if last is not None and last != caller:
                return False
        return True

    def record_hash(
        self, file_path: str, content: str, *, writer: str | None = None
    ) -> None:
        """Record the SHA-256 hash of *content* for *file_path* after a write.

        ``writer`` is the name of the tool that performed the write.  A
        subsequent call to ``has_been_read`` from a *different* tool will be
        rejected until the file is re-read with the ``read`` tool.

        The writer tag is only tracked for files that already had a read/write
        history (i.e. the key is present in ``_last_writer``).  Files that
        were created fresh by a write tool — and never read — carry no tag, so
        any tool may edit them freely.
        """
        path = Path(file_path)
        if path.is_absolute():
            resolved_key = str(path.resolve())
            self._read_hashes[resolved_key] = hashlib.sha256(
                content.encode()
            ).hexdigest()
            if resolved_key in self._last_writer:
                self._last_writer[resolved_key] = writer
            # else: brand-new file, no prior read → leave _last_writer absent

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
                    description="The absolute path to the file to read",
                    argument_type="string",
                    required=True,
                ),
                ToolArgument(
                    name="limit",
                    description=(
                        "The number of lines to read. "
                        "Only provide if the file is too large to read at once."
                    ),
                    argument_type="integer",
                    minimum=1,
                ),
                ToolArgument(
                    name="offset",
                    description=(
                        "The 0-based line offset to start reading from "
                        "(output line numbers remain 1-based). "
                        "Only provide if the file is too large to read at once."
                    ),
                    argument_type="integer",
                    minimum=0,
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
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict:
        logger.debug(
            "read: '%s' (offset=%s, limit=%s)",
            file_path,
            offset if offset is not None else 0,
            limit if limit is not None else _DEFAULT_LIMIT,
        )

        path = Path(file_path)
        if not path.is_absolute():
            return self._err_invalid_arguments(
                f"file_path must be an absolute path, got: '{file_path}'"
            )

        # Resolve symlinks so that all operations (ignore rules, existence,
        # I/O) refer to the same real target.  The tool intentionally accepts
        # arbitrary absolute paths; resolving here ensures the ignore check
        # and the actual I/O operate on the same file rather than two
        # different ends of a symlink swap.
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

        effective_offset = offset if offset is not None else 0
        effective_limit = limit if limit is not None else _DEFAULT_LIMIT

        try:
            hasher = hashlib.sha256()
            selected: list[str] = []
            total_lines = 0
            with resolved.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    hasher.update(line.encode())
                    total_lines += 1
                    if effective_offset <= i < effective_offset + effective_limit:
                        selected.append(line)
        except (OSError, UnicodeDecodeError) as exc:
            return self._err_read_error(f"Cannot read '{file_path}': {exc}")

        if effective_offset > 0 and effective_offset >= total_lines:
            return self._err_invalid_range(
                f"offset ({effective_offset}) exceeds file length "
                f"({total_lines} line(s)) for '{file_path}'."
            )

        resolved_key = str(resolved)
        self._read_hashes[resolved_key] = hasher.hexdigest()
        self._last_writer[resolved_key] = None  # fresh read clears any prior writer tag

        start_line_number = effective_offset + 1  # convert to 1-based for display

        content = "".join(
            f"{start_line_number + i:6d}\t{line}" for i, line in enumerate(selected)
        )

        lines_returned = len(selected)
        logger.debug(
            "read: returned %d/%d lines from '%s'",
            lines_returned,
            total_lines,
            file_path,
        )
        return self._ok(
            {
                "content": content,
                "file_path": file_path,
                "lines_returned": lines_returned,
                "total_lines": total_lines,
            }
        )
