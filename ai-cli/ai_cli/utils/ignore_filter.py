"""
.gitignore-style ignore filter.

Patterns follow .gitignore semantics:
  - Blank lines and lines whose first character is '#' are ignored (comments
    must start at column 1; a line with leading spaces followed by '#' is
    treated as a pattern, matching .gitignore whitespace rules).
  - Trailing whitespace is stripped; leading whitespace is preserved as part
    of the pattern.
  - A leading '/' anchors the pattern to the root passed at construction time.
  - A trailing '/' matches directories only.
  - '**' matches any number of path components.
  - '!' negates a pattern (re-includes a previously excluded path).
    NOTE: unlike Git, negation CAN re-include a path whose ancestor directory
    was matched by an earlier pattern.  This is a deliberate simplification —
    it makes project-level overrides more predictable without requiring users
    to un-ignore the parent directory first.
  - Otherwise standard fnmatch glob rules apply.

When combining patterns from multiple sources (e.g. global + project), pass
all raw pattern strings to a single IgnoreFilter so that the "last match wins"
rule applies across both sources and project-level negations can override global
exclusions.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path


def _seg_match_exact(segs: list[str], parts: tuple[str, ...]) -> bool:
    """
    Return True if *segs* matches *parts* exactly (all parts consumed).

    ``**`` in segs consumes zero or more components of parts.

    Uses memoised DP over (segment-index, part-index) to avoid the
    exponential blow-up that plain recursion produces for patterns with
    multiple ``**`` segments.
    """
    memo: dict[tuple[int, int], bool] = {}

    def dp(i: int, j: int) -> bool:
        key = (i, j)
        if key in memo:
            return memo[key]
        if i == len(segs):
            result = j == len(parts)
            memo[key] = result
            return result
        seg = segs[i]
        if seg == "**":
            for k in range(j, len(parts) + 1):
                if dp(i + 1, k):
                    memo[key] = True
                    return True
            memo[key] = False
            return False
        if j == len(parts):
            memo[key] = False
            return False
        result = fnmatch.fnmatch(parts[j], seg) and dp(i + 1, j + 1)
        memo[key] = result
        return result

    return dp(0, 0)


class _Pattern:
    __slots__ = ("raw", "negated", "anchored", "dir_only", "segments")

    def __init__(self, raw: str) -> None:
        self.raw = raw
        pattern = raw

        self.negated = pattern.startswith("!")
        if self.negated:
            pattern = pattern[1:]

        # A leading slash anchors the pattern to the root.
        self.anchored = pattern.startswith("/")
        if self.anchored:
            pattern = pattern[1:]

        # A trailing slash means directories only.
        self.dir_only = pattern.endswith("/")
        if self.dir_only:
            pattern = pattern[:-1]

        # Split into path segments for multi-component matching.
        self.segments: list[str] = pattern.split("/") if "/" in pattern else [pattern]

    def matches(self, rel_parts: tuple[str, ...], is_dir: bool) -> bool:
        if self.dir_only and not is_dir:
            return False

        if len(self.segments) == 1:
            pat = self.segments[0]
            if self.anchored:
                # Must match the first component only.
                return len(rel_parts) >= 1 and fnmatch.fnmatch(rel_parts[0], pat)
            else:
                # May match any single component along the path.
                return any(fnmatch.fnmatch(part, pat) for part in rel_parts)
        else:
            # Multi-segment pattern — match against a contiguous slice of rel_parts.
            return self._match_segments(rel_parts)

    def _match_segments(self, rel_parts: tuple[str, ...]) -> bool:
        segs = self.segments
        if self.anchored:
            return _seg_match_exact(segs, rel_parts)
        # Unanchored: try matching segs starting at every position in rel_parts.
        for start in range(len(rel_parts) + 1):
            if _seg_match_exact(segs, rel_parts[start:]):
                return True
        return False


class IgnoreFilter:
    """
    Parses one .ignore file and answers is_ignored() queries.

    Usage::

        f = IgnoreFilter.from_file(root, ignore_path)
        if f.is_ignored(some_path):
            ...
    """

    def __init__(self, root: Path, patterns: list[str]) -> None:
        self._root = root.resolve()
        self._raw_patterns: list[str] = []
        self._patterns: list[_Pattern] = []
        for line in patterns:
            # Strip only trailing whitespace (gitignore ignores unescaped
            # trailing spaces; leading spaces are part of the pattern).
            line = line.rstrip(" \t\r\n")
            # Blank lines are skipped.
            if not line:
                continue
            # Comments must start with '#' at column 1 (not after leading spaces).
            if line.startswith("#"):
                continue
            self._raw_patterns.append(line)
            self._patterns.append(_Pattern(line))

    @property
    def raw_patterns(self) -> list[str]:
        """The parsed (non-blank, non-comment) pattern strings in load order."""
        return list(self._raw_patterns)

    @staticmethod
    def read_patterns(ignore_file: Path) -> list[str]:
        """
        Read and parse pattern strings from *ignore_file*.

        Returns the non-blank, non-comment pattern strings in file order.
        Returns an empty list if the file does not exist or cannot be read
        (e.g. permission error or non-UTF-8 encoding).  This is a lightweight
        alternative to constructing a full ``IgnoreFilter`` just to retrieve
        patterns for merging.
        """
        if not ignore_file.is_file():
            return []
        try:
            text = ignore_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return []
        raw: list[str] = []
        for line in text.splitlines():
            line = line.rstrip(" \t\r\n")
            if not line or line.startswith("#"):
                continue
            raw.append(line)
        return raw

    @classmethod
    def from_file(cls, root: Path, ignore_file: Path) -> IgnoreFilter:
        """Load patterns from *ignore_file*; return an empty filter if missing."""
        return cls(root, cls.read_patterns(ignore_file))

    @classmethod
    def empty(cls, root: Path) -> IgnoreFilter:
        return cls(root, [])

    def is_ignored(self, path: Path) -> bool:
        """
        Return True if *path* is excluded by this filter's patterns.

        Matching rules (simplified — see module docstring for divergence from Git):

        * Patterns are evaluated in order; the **last matching pattern wins**.
        * Each path component (and each ancestor directory) is tested
          independently, so ignoring a directory also ignores its contents.
        * A ``!`` negation pattern CAN re-include a path even when one of its
          ancestor directories was matched by an earlier pattern.  This differs
          from Git, where files inside an ignored directory cannot be
          re-included without first un-ignoring the directory itself.

        *path* may be absolute or relative; if absolute it must be under root.
        """
        path = path.resolve() if path.is_absolute() else (self._root / path).resolve()
        try:
            rel = path.relative_to(self._root)
        except ValueError:
            # Outside root — not governed by this filter.
            return False

        rel_parts = rel.parts
        is_dir = path.is_dir()

        # Iterate patterns in declaration order (outer loop) so that the last
        # matching pattern always wins, regardless of whether it matched the
        # path itself or an ancestor directory.  For each pattern, check every
        # ancestor directory (from root down) and then the path itself; stop at
        # the first level that matches (the negation status is fixed per pattern).
        result = False
        for pat in self._patterns:
            for depth in range(1, len(rel_parts) + 1):
                sub_parts = rel_parts[:depth]
                sub_is_dir = (depth < len(rel_parts)) or is_dir
                if pat.matches(sub_parts, sub_is_dir):
                    result = not pat.negated
                    break  # this pattern matched at some level; move to next pattern
        return result
