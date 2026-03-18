"""
find_files — search for files in the workspace matching a glob pattern.

No permission required by default.  Respects workspace ignore rules (global
~/.ai-cli/.ignore, root .gitignore, and project .ai-cli/.ignore), so
hidden/excluded files are never surfaced.

Glob pattern syntax:
  *.py                        — match all .py files in the search directory
  **/*.py                     — match all .py files recursively
  src/**/*.json               — recursive under a specific sub-directory
  **/*.{png,jpg,jpeg,gif}     — brace expansion (multiple extensions at once)
  file[0-9].txt               — character range (matches file0.txt … file9.txt)
  [abc]*.py                   — character class (matches a*.py, b*.py, c*.py)
"""

from __future__ import annotations

import os
import re

from ai_cli.core.workspace import WorkspaceError
from ai_cli.tools.base import Tool

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
        "Use '**/' to search recursively (e.g. '**/*.py' finds all Python files). "
        "Use '{a,b}' for multiple extensions (e.g. '**/*.{png,jpg}'). "
        "Results are sorted and limited to files; directories are excluded. "
        "Workspace ignore rules (.ignore and .gitignore files) are always respected."
    )
    PERMISSION_REQUIRED = False

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": (
                                "Glob pattern to match against file paths relative to "
                                "'directory'. Use '**/*.ext' for recursive search. "
                                "Examples: '*.py', '**/*.ts', 'src/**/*.json'."
                            ),
                        },
                        "directory": {
                            "type": "string",
                            "description": (
                                "Directory to search in, relative to the workspace root. "
                                "Defaults to '.' (the workspace root). "
                                "Example: 'src/components'."
                            ),
                        },
                    },
                    "required": ["pattern"],
                },
            },
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(  # type: ignore[override]
        self,
        *,
        pattern: str,
        directory: str = ".",
    ) -> dict:
        if not pattern:
            return self._err("invalid_input", "'pattern' must not be empty.", 400)

        # Reject absolute patterns and any path traversal via ..
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

        try:
            search_root = self._workspace.resolve(directory)
        except WorkspaceError as exc:
            return self._err("invalid_path", str(exc), 400)

        if not search_root.exists():
            return self._err(
                "not_found", f"Directory '{directory}' does not exist.", 404
            )
        if not search_root.is_dir():
            return self._err(
                "not_a_directory", f"'{directory}' is not a directory.", 400
            )

        workspace_root = self._workspace.root
        matches: list[str] = []
        truncated = False
        partial = False

        # Recursion is requested only when the pattern contains '**'.
        # Without '**', the maximum traversal depth equals the number of '/'
        # separators in the pattern (e.g. "src/*.py" needs depth 1).
        recursive = "**" in pattern
        max_depth = None if recursive else pattern.count("/")

        # For fixed-depth patterns, narrow the walk root by consuming any
        # leading literal (non-glob) directory segments.  For example, the
        # pattern "src/lib/*.py" can start walking directly from search_root/src/lib
        # rather than from search_root, avoiding traversal of unrelated sibling
        # directories like "tests/" or "docs/".
        walk_root = search_root
        if not recursive and "/" in pattern:
            segments = pattern.split("/")
            # Collect leading segments that contain no glob characters.
            literal_dirs: list[str] = []
            for seg in segments[:-1]:
                if any(c in seg for c in ("*", "?", "[", "{")):
                    break
                literal_dirs.append(seg)
            if literal_dirs:
                candidate = search_root
                for d in literal_dirs:
                    candidate = candidate / d
                if not candidate.is_dir():
                    # Literal prefix path doesn't exist — no files can match.
                    return self._ok(
                        {
                            "matches": [],
                            "count": 0,
                            "pattern": pattern,
                            "directory": directory,
                        }
                    )
                # Narrow the walk root only when the prefix is not ignored.
                # An ignored prefix would be pruned at the top of the os.walk
                # loop anyway, so there is nothing to gain by narrowing into it.
                if not self._workspace.is_ignored(candidate):
                    walk_root = candidate
                    # max_depth stays relative to search_root (not walk_root),
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
                if self._workspace.is_ignored(entry):
                    continue
                try:
                    rel_ws = entry.relative_to(workspace_root)
                    rel_search = entry.relative_to(search_root)
                except ValueError:
                    continue
                if compiled.match(str(rel_search).replace("\\", "/")):
                    matches.append(str(rel_ws).replace("\\", "/"))
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
                rel = os.path.relpath(dirpath, search_root)
                current_depth = 0 if rel == "." else rel.count(os.sep) + 1
                current_dir = workspace_root / os.path.relpath(dirpath, workspace_root)

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
                    dirnames[:] = [
                        d
                        for d in dirnames
                        if not self._workspace.is_ignored(current_dir / d)
                    ]

                for filename in sorted(filenames):
                    filepath = current_dir / filename
                    if self._workspace.is_ignored(filepath):
                        continue
                    try:
                        rel_ws = filepath.relative_to(workspace_root)
                        rel_search = filepath.relative_to(search_root)
                    except ValueError:
                        continue
                    rel_search_str = str(rel_search).replace("\\", "/")
                    if compiled.match(rel_search_str):
                        matches.append(str(rel_ws).replace("\\", "/"))
                        if len(matches) >= _MAX_RESULTS:
                            truncated = True
                            break
                if truncated:
                    break

            if os_errors:
                partial = True

        matches.sort()
        result: dict = {
            "matches": matches,
            "count": len(matches),
            "pattern": pattern,
            "directory": directory,
        }
        if truncated:
            result["truncated"] = True
        if partial:
            result["partial"] = True
        return self._ok(result)
