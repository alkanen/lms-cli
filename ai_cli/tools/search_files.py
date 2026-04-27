"""
search_files — semantic search over the indexed file corpus.

Disabled by default (``DISABLED_BY_DEFAULT = True``).  Only available when
``embeddings.enabled: true`` is set in config and the embedding index has
been initialised via ``/index``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

logger = logging.getLogger(__name__)


class SearchFilesTool(Tool):
    """Semantic search over indexed files using vector embeddings."""

    NAME = "search_files"
    DESCRIPTION = (
        "Semantic search over the indexed file corpus. "
        "Returns ranked file chunks or document-level results most relevant to the query. "
        "The index must be built first with /index. "
        "Use 'level' to control chunk vs. document granularity, "
        "and 'path_glob' to restrict results to a subset of the corpus."
    )
    PERMISSION_REQUIRED = False
    DISABLED_BY_DEFAULT = True

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def definition(self) -> ToolSchema:
        """Return the tool's function-calling schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    name="query",
                    description=(
                        "Natural language description or code snippet to search for."
                    ),
                    argument_type="string",
                    required=True,
                ),
                ToolArgument(
                    name="k",
                    description=("Number of results to return (default 5, max 20)."),
                    argument_type="integer",
                    minimum=1,
                    maximum=20,
                ),
                ToolArgument(
                    name="level",
                    description=(
                        "Granularity of results: "
                        "'chunk' (function/paragraph level, default), "
                        "'document' (whole-file level), "
                        "or 'both' (mix of chunk and document results)."
                    ),
                    argument_type="string",
                    enum=["chunk", "document", "both"],
                ),
                ToolArgument(
                    name="path_glob",
                    description=(
                        "Restrict results to files matching this glob pattern. "
                        "The pattern is matched against the absolute file paths "
                        "stored in the index (SQLite GLOB semantics: case-sensitive, "
                        "'*' and '?' wildcards, POSIX separators). "
                        "For workspace-relative matches prefix with '*/', e.g. "
                        "'*/src/**/*.py'. For absolute paths: '/home/user/docs/**/*.md'."
                    ),
                    argument_type="string",
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(  # type: ignore[override]
        self,
        *,
        query: str,
        k: int = 5,
        level: str = "chunk",
        path_glob: str | None = None,
    ) -> dict:
        """Run the semantic search and return ranked results."""
        import time

        ei = self._workspace.embedding_index
        if ei is None:
            return self._err(
                "not_enabled",
                "Embedding index is not enabled or not initialised. "
                "Enable embeddings in config and run /index.",
                503,
            )

        k = min(max(1, k), 20)

        if level not in ("chunk", "document", "both"):
            return self._err_invalid_arguments(
                f"'level' must be 'chunk', 'document', or 'both'; got {level!r}."
            )

        t0 = time.monotonic()

        try:
            results = ei.search(query, k=k, level=level, path_glob=path_glob)
        except Exception as exc:
            logger.exception("search_files: search failed")
            return self._err_internal_error(f"Search failed: {exc}")

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        output = []
        for r in results:
            meta = r.metadata
            file_path = meta.get("file_path", "")
            start_line = meta.get("start_line")
            end_line = meta.get("end_line")

            # Resolve symlinks once so the same path is used for the access
            # check and the read, eliminating a TOCTOU window.
            resolved_path = Path(file_path).resolve() if file_path else None

            # Only read snippet if the resolved path is within the workspace or
            # an indexed external root — guards against stale/tampered DB
            # entries pointing at arbitrary local files.
            if resolved_path and _is_accessible_path(
                resolved_path, self._workspace, ei
            ):
                snippet = _read_snippet(resolved_path, start_line, end_line)
            else:
                snippet = ""

            output.append(
                {
                    "file": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "symbol_name": meta.get("symbol_name"),
                    "symbol_kind": meta.get("symbol_kind"),
                    "score": round(r.score, 4),
                    "snippet": snippet,
                }
            )

        logger.debug(
            "search_files: query=%r, k=%d, level=%r, results=%d, elapsed=%dms",
            query,
            k,
            level,
            len(output),
            elapsed_ms,
        )

        return self._ok({"results": output, "query_time_ms": elapsed_ms})


def _read_snippet(
    path: Path,
    start_line: int | None,
    end_line: int | None,
) -> str:
    """Read the snippet for a search result directly from the file at query time.

    Reading live from the file ensures the snippet always reflects current
    content even if the file has changed since the last ``/index`` run.

    Returns an empty string if the file cannot be read.
    """
    try:
        if not path.is_file():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(
            keepends=True
        )
        if start_line is None or end_line is None:
            # Document-level result: return first 50 lines as a preview.
            return "".join(lines[:50])
        lo = max(0, start_line - 1)
        hi = min(len(lines), end_line)
        return "".join(lines[lo:hi])
    except OSError:
        return ""


def _is_accessible_path(
    path: Path,
    workspace: object,
    ei: object,
) -> bool:
    """Return True if *path* is within the workspace or an indexed root.

    Guards snippet reads against stale or tampered index DB entries that might
    point outside the intended corpus.
    """
    # Check workspace containment.
    try:
        contains = getattr(workspace, "contains", None)
        if callable(contains) and contains(path):
            return True
    except Exception:
        pass
    # Check indexed external roots via EmbeddingIndex.
    try:
        is_indexed = getattr(ei, "is_indexed_path", None)
        if callable(is_indexed) and is_indexed(path):
            return True
    except Exception:
        pass
    return False
