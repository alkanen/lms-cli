"""
find_files — search for files in the workspace matching a glob pattern.

No permission required by default.  Respects workspace ignore rules (global
~/.ai-cli/.ignore, root .gitignore, and project .ai-cli/.ignore), so
hidden/excluded files are never surfaced.

Patterns are always relative to the workspace root.

Glob pattern syntax:
  *.py                        — match all .py files in the workspace root
  **/*.py                     — match all .py files recursively
  src/**/*.json               — recursive under a specific sub-directory
  **/*.{png,jpg,jpeg,gif}     — brace expansion (multiple extensions at once)
  file[0-9].txt               — character range (matches file0.txt … file9.txt)
  [abc]*.py                   — character class (matches a*.py, b*.py, c*.py)
"""

from __future__ import annotations

import logging
import os
import re

from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

logger = logging.getLogger(__name__)

_MAX_RESULTS = 500


def _glob_to_regex(pattern: str) -> str:
    """Convert a glob pattern string to a regex string (not anchored).

    Supports:
      ``**/``       — zero or more directory levels  (``(.*/)?``)
      ``**``        — any sequence of chars including ``/``
      ``*``         — any sequence of non-separator chars (``[^/]*``)
      ``?``         — any single non-separator char (``[^/]``)
      ``[abcd]``    — character class
      ``[a-d]``     — character range
      ``[!abcd]``   — negated character class (``!`` or ``^`` both work)
      ``{a,b,c}``   — alternation  (``(a|b|c)``)
      All other characters are regex-escaped.
    """
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i : i + 3] == "**/":
            parts.append("(.*/)?")
            i += 3
        elif pattern[i : i + 2] == "**":
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            parts.append("[^/]")
            i += 1
        elif pattern[i] == "[":
            # Scan for the closing ']', respecting that ']' can appear as a
            # literal immediately after '[', '[^', or '[!'.
            j = i + 1
            if j < len(pattern) and pattern[j] in ("^", "!"):
                j += 1
            if j < len(pattern) and pattern[j] == "]":
                j += 1  # literal ']' at start of class
            end = pattern.find("]", j)
            if end == -1:
                # No closing bracket — treat '[' as a literal.
                parts.append(re.escape("["))
                i += 1
            else:
                # Translate glob negation '[!...]' → regex '[^...]'.
                content = pattern[i + 1 : end]
                if content.startswith("!"):
                    content = "^" + content[1:]
                parts.append("[" + content + "]")
                i = end + 1
        elif pattern[i] == "{":
            # Collect everything up to the matching closing brace.
            end = pattern.find("}", i + 1)
            if end == -1:
                # No closing brace — treat as a literal character.
                parts.append(re.escape("{"))
                i += 1
            else:
                alternatives = pattern[i + 1 : end].split(",")
                parts.append(
                    "(" + "|".join(_glob_to_regex(a) for a in alternatives) + ")"
                )
                i = end + 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return "".join(parts)


def _compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile a glob pattern to an anchored regex."""
    return re.compile("^" + _glob_to_regex(pattern) + "$")


class FindFilesTool(Tool):
    NAME = "find_files"
    DESCRIPTION = (
        "Find files in the workspace whose paths match a glob pattern. "
        "Patterns are relative to the workspace root: use '*.py' for root-level files, "
        "'**/*.py' to search recursively, or 'src/**/*.py' to restrict to a subtree. "
        "Use '{a,b}' for multiple extensions (e.g. '**/*.{png,jpg}'). "
        "Results are sorted and limited to files; directories are excluded. "
        "Workspace ignore rules (.ignore and .gitignore files) are always respected."
    )
    PERMISSION_REQUIRED = False
    DISABLED_BY_DEFAULT = True

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    name="pattern",
                    description=(
                        "Glob pattern relative to the workspace root. "
                        "Use '**/*.ext' for recursive search; prefix with a "
                        "directory to restrict the scope. "
                        "Examples: '*.py', '**/*.ts', 'src/**/*.json', '**/docs/*'."
                    ),
                    argument_type="string",
                    required=True,
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(  # type: ignore[override]
        self,
        *,
        pattern: str,
    ) -> dict:
        logger.debug("find_files: pattern=%r", pattern)
        if not pattern:
            return self._err("invalid_input", "'pattern' must not be empty.", 400)

        # Reject absolute patterns and any path traversal via ..
        # NOTE: Access control for external indexed roots (via
        # workspace.embedding_index.is_indexed_path()) is planned for a future
        # update when an optional 'path' parameter is added to this tool.
        # Currently all patterns are workspace-relative only.
        if pattern.startswith("/"):
            return self._err(
                "invalid_input", "Pattern must not be an absolute path.", 400
            )
        if re.search(r"(^|/)\.\.(/|$)", pattern):
            return self._err(
                "invalid_input",
                "Pattern must not contain '..' path traversal segments.",
                400,
            )

        try:
            compiled = _compile_glob(pattern)
        except re.error as exc:
            return self._err("invalid_input", f"Invalid glob pattern: {exc}", 400)

        workspace_root = self._workspace.root
        matches: list[str] = []
        truncated = False
        partial = False

        # Recursion is requested only when the pattern contains '**'.
        # Without '**', the maximum traversal depth equals the number of '/'
        # separators in the pattern (e.g. "src/*.py" needs depth 1).
        recursive = "**" in pattern
        max_depth = None if recursive else pattern.count("/")

        # Narrow the walk root by consuming any leading literal (non-glob)
        # directory segments from the pattern.  This applies to both fixed-depth
        # and recursive patterns: "src/lib/*.py" and "src/**/*.py" can both start
        # walking from workspace_root/src/lib (or workspace_root/src) instead of
        # workspace_root, skipping sibling trees like "tests/" or "docs/" entirely.
        walk_root = workspace_root
        if "/" in pattern:
            segments = pattern.split("/")
            # Collect leading segments that contain no glob characters.
            literal_dirs: list[str] = []
            for seg in segments[:-1]:
                if any(c in seg for c in ("*", "?", "[", "{")):
                    break
                literal_dirs.append(seg)
            if literal_dirs:
                candidate = workspace_root
                for d in literal_dirs:
                    candidate = candidate / d
                if not candidate.is_dir():
                    # Literal prefix path doesn't exist — no files can match.
                    return self._ok(
                        {
                            "matches": [],
                            "count": 0,
                            "pattern": pattern,
                        }
                    )
                # Short-circuit if the literal prefix itself is ignored — the
                # directory will be pruned during any os.walk anyway, so no
                # files under it can ever match.
                # candidate.is_dir() was already confirmed above, so is_dir=True.
                if self._workspace.is_ignored(candidate, is_dir=True):
                    return self._ok({"matches": [], "count": 0, "pattern": pattern})
                walk_root = candidate
                # max_depth stays relative to workspace_root (not walk_root),
                # matching how current_depth is computed in os.walk below.

        if max_depth == 0:
            # Fast path: only the immediate contents of walk_root matter.
            try:
                entries = sorted(walk_root.iterdir())
            except OSError as exc:
                return self._err("search_error", str(exc), 500)
            for entry in entries:
                if not entry.is_file():
                    continue
                if self._workspace.is_ignored(entry, is_dir=False):
                    continue
                try:
                    rel = entry.relative_to(workspace_root)
                except ValueError:
                    continue
                rel_str = str(rel).replace("\\", "/")
                if compiled.match(rel_str):
                    matches.append(rel_str)
                    if len(matches) >= _MAX_RESULTS:
                        truncated = True
                        break
        else:
            os_errors: list[str] = []

            def _onerror(err: OSError) -> None:
                os_errors.append(str(err))

            for dirpath, dirnames, filenames in os.walk(
                walk_root, topdown=True, onerror=_onerror
            ):
                rel_str = os.path.relpath(dirpath, workspace_root)
                current_depth = 0 if rel_str == "." else rel_str.count(os.sep) + 1
                current_dir = workspace_root / (rel_str if rel_str != "." else "")

                # Sort for deterministic ordering.
                dirnames[:] = sorted(dirnames)

                # For fixed-depth patterns, stop descending once we've reached
                # the level where matching files must live.
                if max_depth is not None and current_depth >= max_depth:
                    dirnames[:] = []
                else:
                    # Prune ignored directories so we never traverse env/,
                    # .git/, __pycache__/, node_modules/, etc.  This means
                    # files inside an ignored directory are never returned,
                    # even if a negation rule would re-include them — matching
                    # standard Git walk behaviour.
                    # Pass is_dir=True so is_ignored() can skip its stat() call.
                    dirnames[:] = [
                        d
                        for d in dirnames
                        if not self._workspace.is_ignored(current_dir / d, is_dir=True)
                    ]

                for filename in sorted(filenames):
                    filepath = current_dir / filename
                    # Pass is_dir=False — os.walk only puts regular files (and
                    # symlinks to files) in filenames, not directories.
                    if self._workspace.is_ignored(filepath, is_dir=False):
                        continue
                    try:
                        rel_ws = filepath.relative_to(workspace_root)
                    except ValueError:
                        continue
                    rel_str = str(rel_ws).replace("\\", "/")
                    if compiled.match(rel_str):
                        matches.append(rel_str)
                        if len(matches) >= _MAX_RESULTS:
                            truncated = True
                            break
                if truncated:
                    break

            if os_errors:
                partial = True

        matches.sort()
        logger.debug(
            "find_files: %d match(es) for %r (truncated=%s, partial=%s)",
            len(matches),
            pattern,
            truncated,
            partial,
        )
        result: dict = {
            "matches": matches,
            "count": len(matches),
            "pattern": pattern,
        }
        if truncated:
            result["truncated"] = True
        if partial:
            result["partial"] = True
        return self._ok(result)
