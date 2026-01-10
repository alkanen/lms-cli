from typing import Optional

from core.workspace import Workspace
from core.embedding_manager import EmbeddingManager


def write_file(_context: dict, file_path: str, content: str) -> str:
    em = _context["embedding_manager"]
    ws = _context["workspace"]

    return ws.write_file(file_path, content)
