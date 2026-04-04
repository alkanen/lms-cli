"""
Workspace — project root resolution and file operations.

The workspace root is the nearest ancestor directory that contains a
`.ai-cli/` subdirectory.  Any directory whose `.ai-cli/` subdirectory
resolves to the global config directory (as returned by
``get_global_dir()``) is excluded as a project root.  For the default
setup this means the user's home directory is never returned as a project
root, because ``~/.ai-cli/`` is reserved for global user settings.

All file operations are expressed in terms of paths relative to the
workspace root.  Tools that need to operate outside the workspace must
obtain explicit permission via PermissionManager before calling the
relevant OS primitives directly.
"""

from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from ai_cli.utils.ignore_filter import IgnoreFilter

if TYPE_CHECKING:
    from ai_cli.core.embedding_index import EmbeddingIndex

logger = logging.getLogger(__name__)

# Sentinel directory name used for both global and project config.
_DOT_AI_CLI = ".ai-cli"


def get_global_dir() -> Path:
    """
    Return the global config directory as an absolute ``Path``.

    Defaults to ``~/.ai-cli/``.  Override at runtime by setting the
    ``AI_CLI_GLOBAL_DIR`` environment variable **before** (or via) ``.env``.

    Raises ``ValueError`` if ``AI_CLI_GLOBAL_DIR`` is set to an empty string.
    Relative paths and ``~`` are expanded and made absolute, but symlinks are
    *not* resolved so that callers can detect broken symlinks at the configured
    location (e.g. to report a useful error rather than silently creating a new
    directory at the resolved target).
    """
    env_val = os.getenv("AI_CLI_GLOBAL_DIR")
    if env_val is not None:
        if not env_val.strip():
            raise ValueError(
                "AI_CLI_GLOBAL_DIR is set but empty. "
                "Provide a non-empty path or unset the variable to use the default (~/.ai-cli)."
            )
        return Path(os.path.abspath(Path(env_val).expanduser()))
    return Path(os.path.abspath(Path.home() / _DOT_AI_CLI))


# Template contents written by --init.
_INIT_TEMPLATES: dict[str, str] = {
    "config.yaml": textwrap.dedent("""\
        # ai-cli project configuration
        # See docs/project_plan.md for available keys.
        #
        # backend: openai          # or: lmstudio
        # model: gpt-4o
        # base_url: https://api.openai.com/v1
        # api_key_env: OPENAI_API_KEY   # name of the env-var holding the key
        # context_window: 128000
        # max_response_tokens: 4096
        #
        # repl_behavior:
        #   complete_while_typing: false  # true = popup on every keystroke; false = Tab only (default)
        #
        # logging:
        #   level: WARNING   # ai_cli.* default; DEBUG/INFO for verbose output
        #   modules:         # per-module overrides (module name → level)
        #     ai_cli.tools: DEBUG
        #     ai_cli.core.llm_client: INFO
        #
        # ---------------------------------------------------------------------------
        # Embedding index and semantic search  (pip install ai-cli[embeddings])
        # ---------------------------------------------------------------------------
        # embeddings:
        #   enabled: false              # set to true to activate semantic search
        #   model: nomic-embed-text     # embedding model served at base_url
        #   # base_url: ~               # null = inherit from llm base_url above
        #   # api_key_env: ~            # null = inherit from llm api_key_env above
        #   # batch_size: 32            # texts per embedding API request
        #                               # lower values (8-16) help with local servers
        #                               # that stall on large batches (LM Studio etc.)
        #   # request_timeout: 120.0    # seconds before an embedding request times out
        #
        #   chunking:
        #     strategy: auto            # "auto" | "fixed" | "semantic"
        #     chunk_size: 1200          # characters per chunk (fixed/auto)
        #     chunk_overlap: 200        # character overlap between adjacent chunks
        #     max_file_chunks: 300      # skip files that would exceed this limit
        #     min_chunk_chars: 80       # merge tree-sitter nodes smaller than this
        #     max_chunk_chars: 3000     # split nodes larger than this
        #
        #   document_embedding:
        #     enabled: true
        #     strategy: auto            # "auto" | "average" | "summary"
        #                               # auto: prose files use LLM summary,
        #                               #       code files use chunk average
        #     # Options for the "summary" path (also used by "auto" for prose files):
        #     prose_extensions:         # extensions routed to LLM summary under "auto"
        #       - .md                   # (default: .md .txt .rst .adoc .tex .org etc.)
        #       - .txt
        #       - .rst
        #       - .adoc
        #     summary_model: ~          # null = use the configured main LLM model
        #                               # (non-null warns; per-call override unsupported)
        #     summary_max_tokens: 400   # caps input text chars sent to LLM (~4 chars/token)
        #     summary_response_tokens: ~ # word-count hint injected into the summary prompt
        #                               # null = chunk_size // 4; no per-call API cap is
        #                               # added — LLM client max_response_tokens still applies
    """),
    "system_prompt.md": textwrap.dedent("""\
        <!-- Project-specific system prompt (optional).
             Overrides the global default system prompt when present. -->
    """),
    ".ignore": textwrap.dedent("""\
        # Files and directories the LLM and tools should not read or modify.
        # Uses .gitignore syntax (globs, negation with !, comments with #).
        .git/
        __pycache__/
        *.pyc
        .env
        .env.*
    """),
}

_GLOBAL_INIT_TEMPLATES: dict[str, str] = {
    "config.yaml": textwrap.dedent("""\
        # ai-cli user configuration overrides
        # Values here apply to all projects unless overridden by a project-level config.
        # See docs/project_plan.md for available keys.
        #
        # backend: openai          # or: lmstudio
        # model: gpt-4o
        # base_url: https://api.openai.com/v1
        # api_key_env: OPENAI_API_KEY   # name of the env-var holding the key
        # context_window: 128000
        # max_response_tokens: 4096
        #
        # repl_behavior:
        #   complete_while_typing: false  # true = popup on every keystroke; false = Tab only (default)
        #
        # logging:
        #   level: WARNING   # ai_cli.* default; DEBUG/INFO for verbose output
        #   modules:         # per-module overrides (module name → level)
        #     ai_cli.tools: DEBUG
        #     ai_cli.core.llm_client: INFO
        #
        # ---------------------------------------------------------------------------
        # Embedding index and semantic search  (pip install ai-cli[embeddings])
        # ---------------------------------------------------------------------------
        # embeddings:
        #   enabled: false              # set to true to activate semantic search
        #   model: nomic-embed-text     # embedding model served at base_url
        #   # base_url: ~               # null = inherit from llm base_url above
        #   # api_key_env: ~            # null = inherit from llm api_key_env above
        #   # batch_size: 32            # texts per embedding API request
        #                               # lower values (8-16) help with local servers
        #                               # that stall on large batches (LM Studio etc.)
        #   # request_timeout: 120.0    # seconds before an embedding request times out
        #
        #   chunking:
        #     strategy: auto            # "auto" | "fixed" | "semantic"
        #     chunk_size: 1200          # characters per chunk (fixed/auto)
        #     chunk_overlap: 200        # character overlap between adjacent chunks
        #     max_file_chunks: 300      # skip files that would exceed this limit
        #     min_chunk_chars: 80       # merge tree-sitter nodes smaller than this
        #     max_chunk_chars: 3000     # split nodes larger than this
        #
        #   document_embedding:
        #     enabled: true
        #     strategy: auto            # "auto" | "average" | "summary"
        #                               # auto: prose files use LLM summary,
        #                               #       code files use chunk average
        #     # Options for the "summary" path (also used by "auto" for prose files):
        #     prose_extensions:         # extensions routed to LLM summary under "auto"
        #       - .md                   # (default: .md .txt .rst .adoc .tex .org etc.)
        #       - .txt
        #       - .rst
        #       - .adoc
        #     summary_model: ~          # null = use the configured main LLM model
        #                               # (non-null warns; per-call override unsupported)
        #     summary_max_tokens: 400   # caps input text chars sent to LLM (~4 chars/token)
        #     summary_response_tokens: ~ # word-count hint injected into the summary prompt
        #                               # null = chunk_size // 4; no per-call API cap is
        #                               # added — LLM client max_response_tokens still applies
    """),
    "system_prompt.md": textwrap.dedent("""\
        <!-- Default system prompt — applied to all projects unless a
             project-level system_prompt.md is present. -->
    """),
    ".ignore": _INIT_TEMPLATES[".ignore"],
}


class WorkspaceError(Exception):
    """Raised for workspace-level problems (path escapes root, file not found, …)."""


def _validate_line_range(start_line: int | None, end_line: int | None) -> None:
    """Raise ``WorkspaceError`` if the supplied line numbers are invalid."""
    if start_line is not None and start_line < 1:
        raise WorkspaceError(f"start_line must be >= 1, got {start_line}.")
    if end_line is not None and end_line < 1:
        raise WorkspaceError(f"end_line must be >= 1, got {end_line}.")
    if start_line is not None and end_line is not None and start_line > end_line:
        raise WorkspaceError(
            f"start_line ({start_line}) must be <= end_line ({end_line})."
        )


class Workspace:
    """
    Manages path resolution and file I/O relative to the project root.

    Parameters
    ----------
    root:
        Absolute path to the project's workspace root (the directory that
        *contains* the `.ai-cli/` folder).
    config_manager:
        Fully initialised ConfigManager for this project.  Typed as
        ``object`` until ``ai_cli.core.config_manager`` is implemented.
    """

    def __init__(self, root: Path, config_manager: object) -> None:
        self._root = root.resolve()
        self._config = config_manager

        # Build a single IgnoreFilter from three sources, evaluated in order
        # so that later sources override earlier ones ("last match wins"):
        #   1. global  ~/.ai-cli/.ignore
        #   2. project root  .gitignore
        #   3. project       .ai-cli/.ignore  (highest precedence)
        # Using one shared root ensures all patterns are evaluated against
        # workspace-relative paths, not `~/.ai-cli/`.
        combined = (
            IgnoreFilter.read_patterns(get_global_dir() / ".ignore")
            + IgnoreFilter.read_patterns(self._root / ".gitignore")
            + IgnoreFilter.read_patterns(self._root / _DOT_AI_CLI / ".ignore")
        )
        self._ignore_filter = IgnoreFilter(self._root, combined)

        # Optional embedding index — set by the startup sequence after
        # EmbeddingIndex is constructed.  None when embeddings are disabled.
        self.embedding_index: EmbeddingIndex | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """Absolute path to the workspace root directory."""
        return self._root

    @property
    def ai_cli_dir(self) -> Path:
        """Absolute path to the project-level `.ai-cli/` directory."""
        return self._root / _DOT_AI_CLI

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def find_root(start: Path) -> Path | None:
        """
        Walk up from *start*, returning the first directory that contains
        a `.ai-cli/` subdirectory.

        A candidate is skipped if its `.ai-cli/` subdirectory resolves to
        the global config directory returned by ``get_global_dir()``.  This
        prevents the directory *containing* the global config directory (e.g.
        the user's home directory) from being treated as a project root.

        Returns ``None`` if no project root is found before reaching the
        filesystem root.

        Raises
        ------
        ValueError
            If ``get_global_dir()`` raises — e.g. when ``AI_CLI_GLOBAL_DIR``
            is set to an empty or whitespace-only string.
        """
        start = start.resolve()
        global_dir = get_global_dir().resolve()
        candidate = start
        while True:
            dot = candidate / _DOT_AI_CLI
            # Never treat the global config directory itself as a project root.
            if dot.is_dir() and dot.resolve() != global_dir:
                logger.debug("Workspace root resolved to: %s", candidate)
                return candidate
            parent = candidate.parent
            if parent == candidate:
                # Reached filesystem root.
                return None
            candidate = parent

    @staticmethod
    def initialise(path: Path) -> None:
        """
        Create a `.ai-cli/` scaffold under *path*.

        If `.ai-cli/` already exists the caller is responsible for asking
        the user whether to proceed; this method does not overwrite existing
        files and does not delete unrecognised content.
        """
        dot = path.resolve() / _DOT_AI_CLI
        Workspace._write_scaffold(dot)

    @staticmethod
    def initialise_global(global_dir: Path) -> None:
        """
        Create the global config directory scaffold at *global_dir* itself.

        Unlike ``initialise()``, which creates a ``.ai-cli/`` subdirectory
        under the given path, this method writes template files directly into
        *global_dir* — because the global directory (e.g. ``~/.ai-cli/``) is
        already the config directory, not its parent.

        Does not overwrite existing files.
        """
        Workspace._write_scaffold(global_dir.resolve(), _GLOBAL_INIT_TEMPLATES)

    @staticmethod
    def _write_scaffold(
        dot: Path,
        templates: dict[str, str] | None = None,
    ) -> None:
        """Create template files and subdirectories inside *dot*."""
        if templates is None:
            templates = _INIT_TEMPLATES
        dot.mkdir(parents=True, exist_ok=True)
        (dot / "tools").mkdir(exist_ok=True)

        for filename, content in templates.items():
            target = dot / filename
            if not target.exists():
                target.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Ignore rules
    # ------------------------------------------------------------------

    def is_ignored(self, path: Path, *, is_dir: bool | None = None) -> bool:
        """
        Return ``True`` if *path* is excluded by the combined ignore rules.

        Sources evaluated in order (last match wins, so later sources
        override earlier ones):

        1. global ``~/.ai-cli/.ignore``
        2. project root ``.gitignore``
        3. project ``.ai-cli/.ignore``

        Parameters
        ----------
        is_dir:
            Whether *path* is a directory.  Pass ``True`` or ``False`` when the
            caller already knows to avoid an extra ``stat()`` call.
        """
        return self._ignore_filter.is_ignored(path, is_dir=is_dir)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def resolve(self, relative: str) -> Path:
        """
        Resolve *relative* against the workspace root and return an
        absolute ``Path``.

        Raises ``WorkspaceError`` if the resolved path escapes the root.
        """
        resolved = (self._root / relative).resolve()
        self._assert_within_root(resolved)
        return resolved

    def contains(self, path: Path) -> bool:
        """Return ``True`` if *path* is at or below the workspace root.

        Resolves symlinks and normalises relative paths before comparison so
        that symlink-based escapes are not treated as inside the workspace,
        matching the security properties of ``resolve()``/``_assert_within_root()``.
        """
        candidate = (
            (self._root / path).resolve() if not path.is_absolute() else path.resolve()
        )
        try:
            candidate.relative_to(self._root)
            return True
        except ValueError:
            return False

    def _assert_within_root(self, path: Path) -> None:
        try:
            path.relative_to(self._root)
        except ValueError:
            raise WorkspaceError(
                f"Path '{path}' is outside the workspace root '{self._root}'."
            ) from None

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def file_exists(self, relative: str) -> bool:
        """
        Return ``True`` if *relative* names an existing, non-ignored file
        within the root.  Returns ``False`` for ignored paths to avoid
        leaking information about excluded files.
        """
        try:
            path = self.resolve(relative)
        except WorkspaceError:
            return False
        if self.is_ignored(path):
            return False
        return path.is_file()

    def read_file(
        self,
        relative: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """
        Read a file relative to the workspace root.

        Parameters
        ----------
        relative:
            Path relative to the workspace root.
        start_line:
            1-based first line to include (inclusive). ``None`` means
            start from the beginning.
        end_line:
            1-based last line to include (inclusive). ``None`` means read
            to end of file.

        Returns the file content as a string.  Line endings are normalised
        to ``\\n`` by Python's universal-newlines mode (CRLF and CR are
        converted).  The last line may or may not end with a newline depending
        on the source file.

        Raises ``WorkspaceError`` for path-escape, ignored path, missing
        file, invalid line range, or I/O errors (including non-UTF-8 content).
        """
        path = self.resolve(relative)
        if self.is_ignored(path):
            raise WorkspaceError(f"Path '{relative}' is excluded by .ignore rules.")
        if not path.is_file():
            raise WorkspaceError(f"File not found: '{relative}'")

        _validate_line_range(start_line, end_line)

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            raise WorkspaceError(f"Cannot read '{relative}': {exc}") from exc

        if start_line is None and end_line is None:
            return text

        lines = text.splitlines(keepends=True)
        num_lines = len(lines)

        if start_line is not None and start_line > num_lines:
            raise WorkspaceError(
                f"start_line ({start_line}) exceeds file length "
                f"({num_lines} line(s)) for '{relative}'."
            )
        if end_line is not None and end_line > num_lines:
            raise WorkspaceError(
                f"end_line ({end_line}) exceeds file length "
                f"({num_lines} line(s)) for '{relative}'."
            )

        lo = (start_line - 1) if start_line is not None else 0
        hi = end_line if end_line is not None else num_lines
        return "".join(lines[lo:hi])

    def write_file(
        self,
        relative: str,
        content: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """
        Write *content* to a file relative to the workspace root.

        When *start_line* and/or *end_line* are provided, only that line
        range is replaced; the rest of the file is preserved.  This
        mirrors the partial-write behaviour of the existing ``lms_cli``
        write_file tool.

        Parameters
        ----------
        relative:
            Path relative to the workspace root.  Parent directories are
            created automatically if they do not exist.
        content:
            Text to write (or splice) into the file.
        start_line:
            1-based first line to replace (inclusive).
        end_line:
            1-based last line to replace (inclusive).

        Returns a human-readable summary string (for tool result ``data``).

        Raises ``WorkspaceError`` for path-escape, ignored path, invalid
        line range, or I/O errors (permission denied, disk full, etc.).
        """
        path = self.resolve(relative)
        if self.is_ignored(path):
            raise WorkspaceError(f"Path '{relative}' is excluded by .ignore rules.")

        _validate_line_range(start_line, end_line)

        if start_line is None and end_line is None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            except OSError as exc:
                raise WorkspaceError(f"Cannot write '{relative}': {exc}") from exc
            lines_written = len(content.splitlines()) if content else 0
            return f"Wrote {lines_written} line(s) to '{relative}'."

        # Partial write — file must exist; use write_file without line args to create.
        if not path.is_file():
            raise WorkspaceError(
                f"Cannot do a partial write: '{relative}' does not exist. "
                "Use write_file without start_line/end_line to create a new file."
            )
        try:
            existing_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except (UnicodeDecodeError, OSError) as exc:
            raise WorkspaceError(f"Cannot read '{relative}': {exc}") from exc

        file_len = len(existing_lines)

        # Validate line numbers against the actual file length.
        # start_line == file_len + 1 is allowed (append after last line).
        # end_line == file_len + 1 is only allowed when it equals start_line
        # (i.e. explicit append: start_line=end_line=file_len+1).
        if start_line is not None and start_line > file_len + 1:
            raise WorkspaceError(
                f"start_line ({start_line}) is past end of file ({file_len} line(s))."
            )
        if end_line is not None:
            append_pos = file_len + 1
            is_explicit_append = start_line == append_pos and end_line == append_pos
            if not is_explicit_append and end_line > file_len:
                raise WorkspaceError(
                    f"end_line ({end_line}) is past end of file ({file_len} line(s))."
                )

        new_lines = content.splitlines(keepends=True)
        # Ensure last new line ends with a newline.
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        lo = (start_line - 1) if start_line is not None else 0
        hi = end_line if end_line is not None else file_len

        # If appending and the existing last line has no trailing newline,
        # add one so the new content starts on a fresh line.
        if lo >= file_len and existing_lines and not existing_lines[-1].endswith("\n"):
            existing_lines[-1] += "\n"

        result_lines = existing_lines[:lo] + new_lines + existing_lines[hi:]
        try:
            path.write_text("".join(result_lines), encoding="utf-8")
        except OSError as exc:
            raise WorkspaceError(f"Cannot write '{relative}': {exc}") from exc

        # Use a meaningful summary: "appended" when inserting past EOF, otherwise
        # "replaced" with the actual line range.
        if lo >= file_len:
            return f"Appended {len(new_lines)} line(s) to '{relative}'."
        return (
            f"Replaced lines {lo + 1}–{hi} with {len(new_lines)} line(s) "
            f"in '{relative}'."
        )
