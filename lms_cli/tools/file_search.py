from typing import Optional

from lms_cli.core.workspace import Workspace
from lms_cli.core.embedding_manager import EmbeddingManager


def file_search(_context: dict, extension: Optional[str] = None) -> str:
    em = _context["embedding_manager"]
    ws = _context["workspace"]

    included_set = em.inclusion_paths
    excluded_set = em.exclusion_paths

    files = ws.list_files(
        extension=extension,
        included_folders=included_set,
        excluded_folders=excluded_set
    )

    return [ws.strip_workspace_folder_from_filename(f) for f in files]
