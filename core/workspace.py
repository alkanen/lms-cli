from pathlib import Path
import os
from typing import Dict, Iterable, List, Optional


class Workspace:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).absolute()
        if not self.root_path.exists():
            raise ValueError(f"Workspace path {root_path} does not exist")

    def list_files(
        self,
        included_folders: Iterable[str],
        extension: str = None,
        excluded_folders: Optional[Iterable[str]] = None
    ) -> List[Path]:
        """List all files in the workspace with optional extension filter"""
        files = []

        for ext in (
            ["*.py", "*.cpp", "*.h", "*cu"] if not extension else [f"*.{extension}"]
        ):
            for folder in included_folders:
                files.extend((self.root_path / folder).rglob(ext))

        extended_exclusion = [str(self.root_path / folder) for folder in excluded_folders]

        accepted_files = []
        for file_path in sorted(files):
            accepted = True
            for excluded in extended_exclusion:
                if str(file_path).startswith(excluded):
                    accepted = False

            if accepted:
                accepted_files.append(file_path)

        return accepted_files

    def read_file(self, file_path: Path) -> str:
        """Read content of a file"""
        full_path = self.root_path / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"File {file_path} not found in workspace")

        return full_path.read_text()

    def write_file(self, file_path: Path, content: str):
        """Write content to a file"""
        full_path = self.root_path / file_path
        # Ensure directory exists
        full.path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    def get_file_context(self, file_path: Path, max_lines: int = 50) -> str:
        """Get context around a specific line in a file"""
        content = self.read_file(file_path)
        lines = content.split("\n")

        # Get last <max_lines> lines
        start_line = max(0, len(lines) - max_lines)
        return "\n".join(lines[start_line:])
