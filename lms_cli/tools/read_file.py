from typing import Optional

from lms_cli.core.workspace import Workspace
from lms_cli.core.embedding_manager import EmbeddingManager


def read_file(
    _context: dict, file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None
) -> str:
    ws = _context["workspace"]

    return ws.read_file(file_path, start_line, end_line)
