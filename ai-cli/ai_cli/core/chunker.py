"""
chunker.py — Text chunking strategies for the embedding index.

Provides a ``Chunk`` dataclass and a ``ChunkStrategy`` ABC with multiple
implementations:

- ``FixedSizeChunker``: character-window with line-boundary alignment
- ``TreeSitterChunker``: language-aware AST-based chunking (optional dep)
- ``MultiDocYamlChunker``: Kubernetes / multi-document YAML
- ``AnsibleChunker``: Ansible playbook task-level chunking
- ``ComposeChunker``: Docker Compose service-level chunking
- ``TomlChunker``: TOML section-level chunking

Use ``make_chunker(path, config)`` to obtain the best strategy for a given
file automatically.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single chunk of text extracted from a file.

    Line numbers are 1-based and inclusive.
    """

    start_line: int
    end_line: int
    text: str
    symbol_name: str | None = None
    symbol_kind: str | None = None


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class ChunkStrategy(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, path: Path) -> list[Chunk]:
        """Split *text* into chunks.  *path* is used only for language detection."""


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------


class FixedSizeChunker(ChunkStrategy):
    """Slides a character-window with line-boundary alignment.

    Configuration keys (all optional):
    - ``chunk_size`` (int, default 1200): target window size in characters.
    - ``chunk_overlap`` (int, default 200): overlap between adjacent chunks.
    - ``min_chunk_chars`` (int, default 80): merge final chunk if smaller.
    - ``max_file_chunks`` (int, default 300): skip files producing more chunks.
    """

    def __init__(self, config: dict) -> None:
        self._chunk_size: int = int(config.get("chunk_size", 1200))
        self._chunk_overlap: int = int(config.get("chunk_overlap", 200))
        self._min_chunk_chars: int = int(config.get("min_chunk_chars", 80))
        self._max_file_chunks: int = int(config.get("max_file_chunks", 300))

    def chunk(self, text: str, path: Path) -> list[Chunk]:
        """Return chunks aligned to line boundaries where possible.

        When an individual line exceeds ``chunk_size`` the chunker falls back
        to a hard mid-line split so that no chunk exceeds ``2 × chunk_size``
        characters regardless of line length.  This prevents embedding server
        timeouts caused by unexpectedly large inputs (e.g. minified HTML/JS).
        """
        if not text:
            return []

        lines = text.splitlines(keepends=True)
        n = len(lines)

        # Build cumulative character counts (offset of each line's first char).
        cum: list[int] = [0] * (n + 1)
        for i, line in enumerate(lines):
            cum[i + 1] = cum[i] + len(line)
        total_chars = cum[n]

        if total_chars == 0:
            return []

        # Absolute upper bound on a single chunk's character count.
        # Lines that exceed chunk_size are split mid-line rather than letting
        # one chunk grow arbitrarily large (which can hang local embedding
        # servers such as LM Studio when given oversized inputs).
        _hard_max: int = max(self._chunk_size, 1) * 2

        chunks: list[Chunk] = []
        pos = 0  # current start position in character space
        # Track whether pos was placed at a real line boundary.  When False
        # (mid-line cut from a previous hard-cap), we must not snap the chunk
        # start backwards or we'd replay the same content endlessly.
        at_line_start = True

        while pos < total_chars:
            end_pos = min(pos + self._chunk_size, total_chars)

            # Snap start to the beginning of its containing line (normal case).
            # Skip the backward snap when pos was placed mid-line by a hard cap
            # so that we don't re-cover already-emitted content.
            start_line_idx = _char_to_line(cum, pos)
            chunk_start_pos = cum[start_line_idx] if at_line_start else pos

            # Snap end_pos forward to the end of a line boundary.
            end_line_idx = _char_to_line(cum, end_pos)
            # end_line_idx is the line that *contains* end_pos;
            # we want to include the whole line.
            chunk_end_pos = cum[end_line_idx + 1]

            # Hard cap: if line snap would make the chunk exceed _hard_max
            # chars, cut mid-line instead and flag it so the next iteration
            # does not snap its start back inside the same long line.
            hard_cap_applied = chunk_end_pos - chunk_start_pos > _hard_max
            if hard_cap_applied:
                chunk_end_pos = min(chunk_start_pos + _hard_max, total_chars)
                end_line_idx = _char_to_line(cum, max(0, chunk_end_pos - 1))

            chunk_text = text[chunk_start_pos:chunk_end_pos]
            chunks.append(
                Chunk(
                    start_line=start_line_idx + 1,
                    end_line=end_line_idx + 1,
                    text=chunk_text,
                )
            )

            if chunk_end_pos >= total_chars:
                break

            # Advance with overlap: next chunk starts chunk_overlap chars before
            # the end of the current chunk, snapped to a line boundary.
            next_pos = max(pos + 1, chunk_end_pos - self._chunk_overlap)

            if hard_cap_applied:
                # Cut was mid-line: advance directly so the next iteration
                # starts inside the long line without snapping back to its start.
                pos = next_pos
                at_line_start = False
            else:
                next_line_idx = _char_to_line(cum, next_pos)
                new_pos = cum[next_line_idx]

                if new_pos >= chunk_end_pos or new_pos <= pos:
                    # The line-snap went past the chunk end or backward into
                    # already-covered content (can happen when a line is almost
                    # as long as chunk_size).  Skip the overlap and advance to
                    # the next line boundary.
                    pos = chunk_end_pos
                else:
                    pos = new_pos
                at_line_start = True

        # Merge final tiny chunk into the previous one.
        if len(chunks) >= 2 and len(chunks[-1].text) < self._min_chunk_chars:
            prev = chunks[-2]
            last = chunks.pop()
            chunks[-1] = Chunk(
                start_line=prev.start_line,
                end_line=last.end_line,
                text=prev.text + last.text,
            )

        # Skip files that would produce too many chunks.
        if len(chunks) > self._max_file_chunks:
            logger.warning(
                "FixedSizeChunker: %s skipped — %d chunks exceeds max_file_chunks=%d "
                "(set a higher max_file_chunks in your config to index this file)",
                path.name,
                len(chunks),
                self._max_file_chunks,
            )
            return []

        return chunks


def _char_to_line(cum: list[int], pos: int) -> int:
    """Return the 0-based line index that contains character offset *pos*.

    Uses binary search over the cumulative character array.
    """
    lo, hi = 0, len(cum) - 2  # last valid line index is len(cum)-2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cum[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# TreeSitterChunker
# ---------------------------------------------------------------------------

# Map file extension → (language name, list of AST node types to chunk at).
SUPPORTED_LANGUAGES: dict[str, tuple[str, list[str]]] = {
    ".py": (
        "python",
        ["function_definition", "class_definition", "decorated_definition"],
    ),
    ".cpp": (
        "cpp",
        ["function_definition", "class_specifier", "namespace_definition"],
    ),
    ".cc": (
        "cpp",
        ["function_definition", "class_specifier", "namespace_definition"],
    ),
    ".cxx": (
        "cpp",
        ["function_definition", "class_specifier", "namespace_definition"],
    ),
    ".h": (
        "cpp",
        ["function_definition", "class_specifier", "namespace_definition"],
    ),
    ".hpp": (
        "cpp",
        ["function_definition", "class_specifier", "namespace_definition"],
    ),
    ".rs": (
        "rust",
        ["function_item", "impl_item", "struct_item", "trait_item", "mod_item"],
    ),
    ".lua": (
        "lua",
        ["function_declaration", "local_function", "function_definition"],
    ),
    ".go": (
        "go",
        ["function_declaration", "method_declaration", "type_declaration"],
    ),
    ".js": (
        "javascript",
        ["function_declaration", "class_declaration", "method_definition"],
    ),
    ".ts": (
        "typescript",
        [
            "function_declaration",
            "class_declaration",
            "method_definition",
            "interface_declaration",
            "type_alias_declaration",
        ],
    ),
    ".sh": ("bash", ["function_definition"]),
    ".bash": ("bash", ["function_definition"]),
}

# Private alias used by make_chunker() to check supported extensions.
_TREESITTER_LANGUAGES = SUPPORTED_LANGUAGES


class TreeSitterChunker(ChunkStrategy):
    """Language-aware AST-based chunker using tree-sitter.

    Raises ``ImportError`` at construction time if ``tree_sitter`` or the
    requested language grammar package is not installed.  ``make_chunker()``
    catches this and falls back to ``FixedSizeChunker``.
    """

    def __init__(self, ext: str, config: dict) -> None:
        try:
            import tree_sitter  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "tree-sitter is not installed. "
                "Install it with: pip install ai-cli[embeddings,semantic]"
            ) from exc

        if ext not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported extension for tree-sitter: {ext!r}")

        lang_name, self._node_types = SUPPORTED_LANGUAGES[ext]

        # Import the grammar package.
        pkg_name = f"tree_sitter_{lang_name}"
        try:
            lang_pkg = __import__(pkg_name)
            lang_fn = getattr(lang_pkg, "language", None)
            if lang_fn is None:
                raise ImportError(f"{pkg_name} has no 'language' function")
        except ImportError as exc:
            raise ImportError(
                f"tree-sitter grammar for {lang_name!r} is not installed. "
                f"Install it with: pip install {pkg_name.replace('_', '-')}"
            ) from exc

        from tree_sitter import Language, Parser

        self._language = Language(lang_fn())
        # tree-sitter Python API varies by version: newer bindings use
        # Parser() + set_language(), older ones accept the language in __init__.
        try:
            self._parser = Parser()
            self._parser.set_language(self._language)  # type: ignore[attr-defined]
        except (TypeError, AttributeError):
            self._parser = Parser(self._language)
        self._min_chunk_chars: int = int(config.get("min_chunk_chars", 80))
        self._max_chunk_chars: int = int(config.get("max_chunk_chars", 3000))

    def chunk(self, text: str, path: Path) -> list[Chunk]:
        """Parse with tree-sitter and return symbol-level chunks."""
        encoded = text.encode("utf-8", errors="replace")
        tree = self._parser.parse(encoded)
        root = tree.root_node

        chunks: list[Chunk] = []

        # Collect top-level matching nodes.
        pending_short: list = []

        for child in root.children:
            if child.type in self._node_types:
                node_text = encoded[child.start_byte : child.end_byte].decode(
                    "utf-8", errors="replace"
                )
                start_line = child.start_point[0] + 1
                end_line = child.end_point[0] + 1
                name = _extract_symbol_name(child, text)
                kind = _node_kind(child.type)

                if len(node_text) < self._min_chunk_chars:
                    pending_short.append((start_line, end_line, node_text, name, kind))
                else:
                    # Flush any pending short nodes first.
                    if pending_short:
                        chunks.append(_merge_short(pending_short))
                        pending_short = []

                    if len(node_text) > self._max_chunk_chars:
                        # Split at blank-line boundaries within the node.
                        sub = _split_at_blank_lines(
                            node_text, start_line, name, kind, self._max_chunk_chars
                        )
                        chunks.extend(sub)
                    else:
                        chunks.append(
                            Chunk(
                                start_line=start_line,
                                end_line=end_line,
                                text=node_text,
                                symbol_name=name,
                                symbol_kind=kind,
                            )
                        )

        # Flush any remaining short nodes.
        if pending_short:
            chunks.append(_merge_short(pending_short))

        return chunks


def _extract_symbol_name(node: object, text: str) -> str | None:
    """Return the symbol name for *node* by inspecting child identifier nodes.

    Tree-sitter ``start_byte``/``end_byte`` are byte offsets into the UTF-8
    encoded source, so we slice the encoded bytes and decode rather than
    indexing the Python string directly (which would be wrong for non-ASCII).
    """
    try:
        encoded = text.encode("utf-8", errors="replace")
        for child in node.children:  # type: ignore[attr-defined]
            if child.type in ("identifier", "name"):
                return encoded[child.start_byte : child.end_byte].decode(
                    "utf-8", errors="replace"
                )
    except AttributeError:
        pass
    return None


def _node_kind(node_type: str) -> str:
    """Map tree-sitter node type string to a human-readable kind."""
    _map = {
        "function_definition": "function",
        "function_declaration": "function",
        "function_item": "function",
        "local_function": "function",
        "method_declaration": "method",
        "method_definition": "method",
        "class_definition": "class",
        "class_declaration": "class",
        "class_specifier": "class",
        "decorated_definition": "decorated",
        "impl_item": "impl",
        "struct_item": "struct",
        "trait_item": "trait",
        "mod_item": "module",
        "namespace_definition": "namespace",
        "type_declaration": "type",
        "type_alias_declaration": "type",
        "interface_declaration": "interface",
    }
    return _map.get(node_type, node_type)


def _merge_short(
    nodes: list[tuple[int, int, str, str | None, str]],
) -> Chunk:
    """Merge a list of short (start_line, end_line, text, name, kind) tuples."""
    start = nodes[0][0]
    end = nodes[-1][1]
    text = "".join(n[2] for n in nodes)
    name = nodes[0][3]
    kind = nodes[0][4]
    return Chunk(
        start_line=start, end_line=end, text=text, symbol_name=name, symbol_kind=kind
    )


def _split_at_blank_lines(
    text: str,
    base_line: int,
    symbol_name: str | None,
    symbol_kind: str | None,
    max_chars: int,
) -> list[Chunk]:
    """Split *text* at blank lines, keeping sub-chunks under *max_chars*."""
    lines = text.splitlines(keepends=True)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_start = base_line
    total = 0

    for i, line in enumerate(lines):
        buf.append(line)
        total += len(line)
        is_blank = not line.strip()

        if (is_blank and total >= max_chars) or total >= max_chars * 1.5:
            chunk_text = "".join(buf)
            chunk_end = base_line + i
            chunks.append(
                Chunk(
                    start_line=buf_start,
                    end_line=chunk_end,
                    text=chunk_text,
                    symbol_name=symbol_name,
                    symbol_kind=symbol_kind,
                )
            )
            buf = []
            total = 0
            buf_start = base_line + i + 1

    if buf:
        chunk_text = "".join(buf)
        chunk_end = base_line + len(lines) - 1
        chunks.append(
            Chunk(
                start_line=buf_start,
                end_line=chunk_end,
                text=chunk_text,
                symbol_name=symbol_name,
                symbol_kind=symbol_kind,
            )
        )

    return (
        chunks
        if chunks
        else [
            Chunk(
                start_line=base_line,
                end_line=base_line + len(lines) - 1,
                text=text,
                symbol_name=symbol_name,
                symbol_kind=symbol_kind,
            )
        ]
    )


# ---------------------------------------------------------------------------
# MultiDocYamlChunker
# ---------------------------------------------------------------------------


class MultiDocYamlChunker(ChunkStrategy):
    """Chunks a multi-document YAML file at ``---`` document boundaries.

    Each document becomes one chunk.  ``symbol_name`` is
    ``{kind}/{metadata.name}`` when those fields are present, else the
    document index.
    """

    def __init__(self, config: dict) -> None:
        pass  # no config needed

    def chunk(self, text: str, path: Path) -> list[Chunk]:
        """Return one chunk per YAML document."""
        import yaml

        chunks: list[Chunk] = []

        # Split on --- boundaries to preserve line numbers.
        doc_texts = _split_yaml_docs(text)

        for idx, (start_line, doc_text) in enumerate(doc_texts):
            line_count = len(doc_text.splitlines()) or 1
            end_line = start_line + line_count - 1
            symbol_name: str | None = None
            try:
                data = yaml.safe_load(doc_text)
                if isinstance(data, dict):
                    kind = data.get("kind")
                    meta = data.get("metadata", {})
                    name = meta.get("name") if isinstance(meta, dict) else None
                    if kind and name:
                        symbol_name = f"{kind}/{name}"
                    elif kind:
                        symbol_name = str(kind)
            except yaml.YAMLError:
                pass

            if symbol_name is None:
                symbol_name = str(idx)

            chunks.append(
                Chunk(
                    start_line=start_line,
                    end_line=end_line,
                    text=doc_text,
                    symbol_name=symbol_name,
                    symbol_kind="document",
                )
            )

        return chunks


def _split_yaml_docs(text: str) -> list[tuple[int, str]]:
    """Split a multi-document YAML string on ``---`` separators.

    Returns a list of ``(start_line, doc_text)`` tuples (1-based).
    """
    result: list[tuple[int, str]] = []
    lines = text.splitlines(keepends=True)
    buf: list[str] = []
    start = 1

    for i, line in enumerate(lines, start=1):
        if line.rstrip() == "---" and buf:
            result.append((start, "".join(buf)))
            buf = []
            start = i + 1
        else:
            buf.append(line)

    if buf:
        result.append((start, "".join(buf)))

    # If no separators found, return as a single document.
    if not result:
        result = [(1, text)]

    return result


# ---------------------------------------------------------------------------
# AnsibleChunker
# ---------------------------------------------------------------------------


class AnsibleChunker(ChunkStrategy):
    """Chunks an Ansible playbook at the task level.

    Each task in ``tasks``, ``pre_tasks``, and ``post_tasks`` becomes one
    chunk.  The play itself is also emitted as a chunk.
    """

    def __init__(self, config: dict) -> None:
        pass

    def chunk(self, text: str, path: Path) -> list[Chunk]:
        """Return task-level chunks from an Ansible playbook."""
        import yaml

        # Use yaml.compose() to obtain line-number information from the parse
        # tree, then pair each node with its yaml.safe_load()-parsed data so
        # field access (e.g. play.get("name")) stays simple.
        try:
            tree = yaml.compose(text)
            plays_data = yaml.safe_load(text)
        except yaml.YAMLError:
            return []

        if not isinstance(tree, yaml.SequenceNode) or not isinstance(plays_data, list):
            return []

        chunks: list[Chunk] = []

        for play_node, play in zip(tree.value, plays_data, strict=False):
            if not isinstance(play_node, yaml.MappingNode) or not isinstance(
                play, dict
            ):
                continue

            # Real line ranges from the parse tree.
            # start_mark.line is 0-based; + 1 converts to 1-based.
            # end_mark.line is 0-based; if column==0 the newline was consumed so
            # end_mark.line already equals the 1-based inclusive last line;
            # if column>0 (no trailing newline) we add 1 to convert to 1-based.
            play_start = play_node.start_mark.line + 1
            em = play_node.end_mark
            play_end = max(play_start, em.line if em.column == 0 else em.line + 1)

            try:
                play_text = yaml.dump(
                    [play], default_flow_style=False, allow_unicode=True
                )
            except yaml.YAMLError:
                play_text = str(play)

            chunks.append(
                Chunk(
                    start_line=play_start,
                    end_line=play_end,
                    text=play_text,
                    symbol_name=str(play.get("name", "play")),
                    symbol_kind="play",
                )
            )

            # Find task-section nodes in the compose tree to get line numbers.
            for key_node, val_node in play_node.value:
                if (
                    not isinstance(key_node, yaml.ScalarNode)
                    or key_node.value not in ("pre_tasks", "tasks", "post_tasks")
                    or not isinstance(val_node, yaml.SequenceNode)
                ):
                    continue

                task_list = play.get(key_node.value, [])
                if not isinstance(task_list, list):
                    continue

                for task_node, task in zip(val_node.value, task_list, strict=False):
                    if not isinstance(task_node, yaml.MappingNode) or not isinstance(
                        task, dict
                    ):
                        continue

                    task_start = task_node.start_mark.line + 1
                    em = task_node.end_mark
                    task_end = max(
                        task_start, em.line if em.column == 0 else em.line + 1
                    )

                    try:
                        task_text = yaml.dump(
                            task, default_flow_style=False, allow_unicode=True
                        )
                    except yaml.YAMLError:
                        task_text = str(task)

                    chunks.append(
                        Chunk(
                            start_line=task_start,
                            end_line=task_end,
                            text=task_text,
                            symbol_name=str(task.get("name", "task")),
                            symbol_kind="task",
                        )
                    )

        return chunks


# ---------------------------------------------------------------------------
# ComposeChunker
# ---------------------------------------------------------------------------


class ComposeChunker(ChunkStrategy):
    """Chunks a Docker Compose file at the service level.

    Each service under ``services:`` becomes one chunk.
    """

    def __init__(self, config: dict) -> None:
        pass

    def chunk(self, text: str, path: Path) -> list[Chunk]:
        """Return one chunk per service in a Docker Compose file."""
        import yaml

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            return []

        if not isinstance(data, dict):
            return []

        services = data.get("services", {})
        if not isinstance(services, dict):
            return []

        # Derive accurate line ranges from the original text rather than from
        # re-serialized YAML, so start_line/end_line match the actual file.
        lines = text.splitlines()

        # Locate the 'services:' key and its indentation level.
        services_start_line: int | None = None
        services_indent: int | None = None
        services_pattern = re.compile(r"^(\s*)services\s*:\s*(?:#.*)?$")
        for idx, line in enumerate(lines, start=1):
            m = services_pattern.match(line)
            if m:
                services_start_line = idx
                services_indent = len(m.group(1))
                break

        if services_start_line is None or services_indent is None:
            return []

        # Find where the services block ends (next non-blank, non-comment line
        # that is not more-indented than 'services:').
        services_end_line = len(lines)
        for idx in range(services_start_line + 1, len(lines) + 1):
            line = lines[idx - 1]
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(stripped)
            if indent <= services_indent:
                services_end_line = idx - 1
                break

        # Within the services block, find top-level service keys by matching
        # lines whose indentation is exactly one level below 'services:'.
        # Accepts unquoted, single-quoted, and double-quoted YAML keys.
        service_key_pattern = re.compile(
            r'^(\s*)(?:"([^"]+)"|\'([^\']+)\'|([A-Za-z0-9_.-]+))\s*:\s*(?:#.*)?$'
        )
        service_indent: int | None = None
        service_blocks: list[tuple[str, int, int]] = []
        current_name: str | None = None
        current_start: int | None = None

        for idx in range(services_start_line + 1, services_end_line + 1):
            line = lines[idx - 1]
            m = service_key_pattern.match(line)
            if not m:
                continue
            indent = len(m.group(1))
            # One of the three capturing groups holds the key name.
            name = m.group(2) or m.group(3) or m.group(4)

            # First key below 'services:' establishes the service indentation.
            if service_indent is None and indent > services_indent:
                service_indent = indent

            if service_indent is None or indent != service_indent:
                continue

            # Close the previous service block.
            if current_name is not None and current_start is not None:
                service_blocks.append((current_name, current_start, idx - 1))

            current_name = name
            current_start = idx

        # Close the final service block.
        if (
            current_name is not None
            and current_start is not None
            and services_end_line >= current_start
        ):
            service_blocks.append((current_name, current_start, services_end_line))

        chunks: list[Chunk] = []
        for svc_name, start, end in service_blocks:
            if svc_name not in services:
                continue
            svc_text = "\n".join(lines[start - 1 : end])
            chunks.append(
                Chunk(
                    start_line=start,
                    end_line=end,
                    text=svc_text,
                    symbol_name=str(svc_name),
                    symbol_kind="service",
                )
            )

        # Fallback: if no service keys could be located in the source text (e.g.
        # all keys are quoted in an unusual way), emit the entire services block
        # as a single chunk so the file is not silently skipped.
        if not chunks and services:
            block_text = "\n".join(lines[services_start_line - 1 : services_end_line])
            chunks.append(
                Chunk(
                    start_line=services_start_line,
                    end_line=services_end_line,
                    text=block_text,
                    symbol_name="services",
                    symbol_kind="service",
                )
            )

        return chunks


# ---------------------------------------------------------------------------
# TomlChunker
# ---------------------------------------------------------------------------


class TomlChunker(ChunkStrategy):
    """Chunks a TOML file at top-level table headers (``[section]``).

    This implementation uses a simple regex-based heuristic to detect
    top-level tables and does not perform full TOML parsing or validation.
    """

    def __init__(self, config: dict) -> None:
        pass

    def chunk(self, text: str, path: Path) -> list[Chunk]:
        """Return one chunk per top-level TOML section."""
        if not text:
            return []

        lines = text.splitlines(keepends=True)
        chunks: list[Chunk] = []

        # Find lines that start a new top-level section: lines matching [name]
        # but NOT [[name]] (array-of-tables).  The negative lookahead (?!\[)
        # explicitly rejects double-bracket headers.
        section_starts: list[tuple[int, str]] = []  # (0-based line idx, section name)
        _section_re = re.compile(r"^\[(?!\[)([^\[\]]+)\]")

        for i, line in enumerate(lines):
            m = _section_re.match(line)
            if m:
                section_starts.append((i, m.group(1).strip()))

        if not section_starts:
            # No sections — return the whole file as one chunk.
            return [Chunk(start_line=1, end_line=len(lines), text=text)]

        # Optionally add a "preamble" chunk before the first section.
        first_sec_line = section_starts[0][0]
        if first_sec_line > 0:
            preamble = "".join(lines[:first_sec_line])
            chunks.append(
                Chunk(
                    start_line=1,
                    end_line=first_sec_line,
                    text=preamble,
                    symbol_name="(preamble)",
                    symbol_kind="section",
                )
            )

        for idx, (sec_line, sec_name) in enumerate(section_starts):
            start = sec_line  # 0-based
            if idx + 1 < len(section_starts):
                end = section_starts[idx + 1][0]
            else:
                end = len(lines)
            sec_text = "".join(lines[start:end])
            chunks.append(
                Chunk(
                    start_line=start + 1,
                    end_line=end,
                    text=sec_text,
                    symbol_name=sec_name,
                    symbol_kind="section",
                )
            )

        return chunks


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _is_helm_template(path: Path, text: str | None = None) -> bool:
    """Return True if *path* looks like a Helm chart template.

    If *text* is provided it is used directly; otherwise the file is read.
    """
    parts = path.parts
    if not (
        len(parts) >= 2
        and parts[-2] == "templates"
        and path.suffix.lower() in (".yaml", ".yml")
    ):
        return False
    try:
        content = (
            text
            if text is not None
            else path.read_text(encoding="utf-8", errors="replace")
        )
        return "{{" in content and "}}" in content
    except OSError:
        return False


def _is_multi_doc_yaml(path: Path, text: str | None = None) -> bool:
    """Return True if *path* is a YAML file containing ``---`` separators.

    If *text* is provided it is used directly; otherwise the file is read.
    """
    if path.suffix.lower() not in (".yaml", ".yml"):
        return False
    try:
        content = (
            text
            if text is not None
            else path.read_text(encoding="utf-8", errors="replace")
        )
        # Look for a line that is exactly "---" (with optional trailing whitespace)
        for line in content.splitlines():
            if line.rstrip() == "---":
                return True
    except OSError:
        pass
    return False


def _is_ansible_playbook(path: Path, text: str | None = None) -> bool:
    """Return True if *path* looks like an Ansible playbook.

    If *text* is provided it is used directly; otherwise the file is read.
    """
    if path.suffix.lower() not in (".yaml", ".yml"):
        return False
    try:
        import yaml

        content = (
            text
            if text is not None
            else path.read_text(encoding="utf-8", errors="replace")
        )
        data = yaml.safe_load(content)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return "hosts" in data[0]
    except (OSError, yaml.YAMLError):
        pass
    return False


def _is_compose_file(path: Path) -> bool:
    """Return True if *path* looks like a Docker Compose file by name."""
    name = path.name.lower()
    compose_patterns = [
        "docker-compose*.yml",
        "docker-compose*.yaml",
        "compose.yml",
        "compose.yaml",
    ]
    return any(fnmatch.fnmatch(name, pat) for pat in compose_patterns)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_chunker(path: Path, config: dict, text: str | None = None) -> ChunkStrategy:
    """Return the best ``ChunkStrategy`` for *path* given *config*.

    *text* may be provided when the caller has already loaded the file content,
    to avoid redundant disk reads in the content-sniffing detection helpers.

    Strategy selection order:
    1. ``strategy: fixed`` in config → always use FixedSizeChunker.
    2. Helm template → FixedSizeChunker (go-template syntax invalid YAML).
    3. Multi-document YAML → MultiDocYamlChunker.
    4. Ansible playbook → AnsibleChunker.
    5. Docker Compose → ComposeChunker.
    6. TOML → TomlChunker.
    7. ``strategy: auto`` or ``semantic`` + known extension → try TreeSitterChunker,
       fall back to FixedSizeChunker if tree-sitter is not installed.
    8. Default → FixedSizeChunker.
    """
    chunking_cfg = config.get("chunking", config)
    strategy = chunking_cfg.get("strategy", "auto")

    if strategy == "fixed":
        return FixedSizeChunker(chunking_cfg)

    # Domain chunkers — checked before tree-sitter.
    if _is_helm_template(path, text):
        return FixedSizeChunker(chunking_cfg)
    if _is_multi_doc_yaml(path, text):
        return MultiDocYamlChunker(chunking_cfg)
    if _is_ansible_playbook(path, text):
        return AnsibleChunker(chunking_cfg)
    if _is_compose_file(path):
        return ComposeChunker(chunking_cfg)
    if path.suffix.lower() == ".toml":
        return TomlChunker(chunking_cfg)

    # Tree-sitter for known source languages.
    if strategy in ("auto", "semantic"):
        ext = path.suffix.lower()
        if ext in _TREESITTER_LANGUAGES:
            try:
                return TreeSitterChunker(ext, chunking_cfg)
            except ImportError:
                pass  # tree-sitter not installed — fall through

    return FixedSizeChunker(chunking_cfg)
