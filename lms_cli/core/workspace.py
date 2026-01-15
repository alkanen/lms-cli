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
        """Removes the root directory of the workspace from the provided path and returns the
        relative path string.  Assumes that file_path is within the workspace or results are
        undefined and may cause exceptions."""
        file_path = str(Path(file_path).resolve())
        return "." + file_path[len(str(self.root_path)) :]

    def read_file(self, file_path: Path) -> str:
        """Read content of a file"""
        full_path = self.root_path / file_path
        if not full_path.exists():
            raise FileNotFoundError(
                f"Workspace::read_file(): File {file_path} not found in workspace"
            )

        return full_path.read_text()

    def write_file(self, file_path: Path, content: str, append: bool=False) -> str:
        """Write content to a file"""
        full_path = (self.root_path / file_path).resolve()

        if not full_path.is_relative_to(self.root_path):
            return "Error: Trying to write to files outside of workspace not supported"

        # Ensure directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with full_path.open("a" if append else "w") as f:
            size = f.write(content)

        return f"Success: A total of {size} bytes written to '{file_path}'"

    def read_file(
        self,
        file_path: Path,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Read content from a file"""
        full_path = (self.root_path / file_path).resolve()

        if not str(full_path).startswith(str(self.root_path)):
            return (
                "Error: Trying to write from files outside of workspace not supported"
            )

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
