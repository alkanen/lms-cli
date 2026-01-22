from pathlib import Path
from typing import Iterable, List, Optional


class Workspace:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        if not self.root_path.exists():
            raise ValueError(f"Workspace(): Workspace path {root_path} does not exist")

    def list_files(
        self,
        included_folders: Iterable[str],
        extension: str = None,
        excluded_folders: Optional[Iterable[str]] = None,
    ) -> List[Path]:
        """List all files in the workspace with optional extension filter"""
        files = []

        for ext in (
            ["*"]
            if not extension
            else [
                f"*.{extension}" if not extension.startswith(".") else f"*{extension}"
            ]
        ):
            for folder in included_folders:
                files.extend((self.root_path / folder).rglob(ext))

        extended_exclusion = [
            str(self.root_path / folder) for folder in excluded_folders
        ]

        accepted_files = []
        for file_path in sorted(files):
            accepted = True
            for excluded in extended_exclusion:
                if str(file_path).startswith(excluded):
                    accepted = False

            if accepted:
                accepted_files.append(file_path)

        return accepted_files

    def file_exists(self, file_path: Path | str) -> bool:
        """Returns True if file_path exists within workspace."""
        # Turn path into an absolute Path object
        file_path = Path(file_path).resolve()

        # Make sure the file exists
        if not file_path.exists():
            return False

        # Make sure it's actually a file
        if not file_path.is_file():
            return False

        root_parts = self.root_path.parts
        path_parts = file_path.parts

        # Make sure that the file is within the workspace
        try:
            for i, part in enumerate(root_parts):
                if part != path_parts[i]:
                    return False
        except IndexError:
            return False

        return True

    def strip_root_path(self, file_path: str | Path) -> str:
        """Removes the root directory of the workspace from the provided path and
        returns the relative path string.  Assumes that file_path is within the
        workspace or results are undefined and may cause exceptions."""
        file_path = str(Path(file_path).resolve())
        return "." + file_path[len(str(self.root_path)) :]

    def write_file(
        self,
        file_path: Path,
        content: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Write content to a file, optionally replacing specific lines."""
        full_path = (self.root_path / file_path).resolve()

        if not full_path.is_relative_to(self.root_path):
            return "Error: Trying to write to files outside of workspace not supported"

        # Ensure directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing content if the file exists
        if full_path.exists():
            existing_content = full_path.read_text()
            existing_lines = existing_content.splitlines(keepends=True)
        else:
            existing_lines = []

        # Convert start_line and end_line to 0-based index
        if start_line is not None:
            start_index = max(0, start_line - 1)
        else:
            start_index = 0  # Overwrite everything

        if end_line is not None:
            end_index = min(end_line, len(existing_lines))
        else:
            end_index = len(existing_lines)  # Go to the end of the file

        content_lines = content.splitlines(keepends=True)
        new_content_size = len(content)
        removed_lines = existing_lines[start_index:end_index]
        removed_content_size = len("".join(removed_lines))

        # Replace or insert content
        new_lines = existing_lines[:start_index] + content_lines
        if end_index < len(existing_lines):
            new_lines.extend(existing_lines[end_index:])

        # Write the updated content back to the file
        try:
            with full_path.open("w") as f:
                total_size = f.write("".join(new_lines))
        except Exception as e:
            return f"Error: Unable to write to file '{file_path}': {e}"

        return (
            f"Success: New size of file '{file_path}' is {total_size} bytes. Added {new_content_size} "
            f"bytes and removed {removed_content_size} bytes."
        )

    def read_file(
        self,
        file_path: Path,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Read content from a file"""
        full_path = (self.root_path / file_path).resolve()

        if not str(full_path).startswith(str(self.root_path)):
            return "Error: Trying to read from files outside of workspace not supported"

        if not full_path.exists():
            return f"Error: File '{file_path}' does not exist in workspace"

        content_lines = full_path.read_text().splitlines()

        if start_line is None:
            start_line = 0
        else:
            start_line = max(0, start_line - 1)

        if end_line is None:
            end_line = len(content_lines)
        else:
            end_line = min(end_line, len(content_lines))

        return "\n".join(content_lines[start_line:end_line])

    def get_file_context(self, file_path: Path, max_lines: int = 50) -> str:
        """Get context around a specific line in a file"""
        content = self.read_file(file_path)
        lines = content.split("\n")

        # Get last <max_lines> lines
        start_line = max(0, len(lines) - max_lines)
        return "\n".join(lines[start_line:])

    def strip_workspace_folder_from_filename(self, filepath: Path | str) -> str:
        file_string = str(filepath)
        root_string = str(self.root_path)

        if file_string.startswith(root_string):
            return "." + file_string[len(root_string) :]
