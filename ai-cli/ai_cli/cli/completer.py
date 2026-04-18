"""
completer.py — prompt_toolkit Completer for the ai-cli REPL.

Provides completions for:
- Slash command names and their subcommands / flags
- Tool names for ``/tools info|enable|disable|allow|disallow``
- File paths for ``@path`` references (workspace-relative, ``../``, and
  absolute ``/`` paths are all supported)
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from ai_cli.core.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ai_cli.core.mcp_manager import MCPManager
    from ai_cli.core.task_manager import TaskManager
    from ai_cli.core.tool_registry import ToolRegistry
    from ai_cli.core.workspace import Workspace


# Matches a partial @ reference at the very end of the text before the cursor.
# Group 1: optional '!' (bypass-ignore flag)
# Group 2: the partial path being typed (may be empty).
#   Allows backslash-escaped characters (e.g. ``\\ `` for a space in a name)
#   in addition to the usual non-whitespace, non-special characters.
_AT_PARTIAL_RE = re.compile(r"@(!?)((?:\\.|[^\s\\,;:!?()\[\]{}'\"<>])*)$")

_INDEX_FLAGS = ["--file", "--full", "--label", "--remove"]
_TOOLS_SUBCOMMANDS = ["allow", "disable", "disallow", "enable", "info", "list"]
_TASKS_SUBCOMMANDS = [
    "add",
    "close",
    "delete",
    "edit",
    "info",
    "list",
    "note",
    "open",
    "tree",
]
_MCP_SUBCOMMANDS = ["allow", "disable", "disallow", "enable", "info", "list"]
_SKILLS_SUBCOMMANDS = ["info", "list", "reload"]
_MCP_FLAG_SUBCMDS = frozenset({"allow", "disable", "disallow", "enable"})
_TASKS_REQUIRED_PATH_SUBCMDS = frozenset({"close", "edit", "info", "open"})
_TASKS_OPTIONAL_PATH_SUBCMDS = frozenset({"add", "delete", "list"})


# ---------------------------------------------------------------------------
# Shell-style tokenizer helpers
# ---------------------------------------------------------------------------


def _tokenize_command(text: str) -> tuple[list[str], str]:
    """Parse *text* into completed tokens and the raw partial being typed.

    Handles backslash-escaping (``\\ `` → space, ``\\\\`` → backslash, etc.)
    and single/double quoting.

    Returns ``(completed_tokens, partial_raw)`` where:

    - *completed_tokens*: whitespace-delimited tokens already finished,
      decoded (unescaped).
    - *partial_raw*: the raw (un-decoded) suffix of *text* that forms the
      token currently being typed; may be empty when the cursor is after
      trailing whitespace.  Its ``len()`` is used as the negated
      ``start_position`` for completions so the raw partial is fully
      replaced.
    """
    completed: list[str] = []
    buf: list[str] = []  # decoded chars of the token in progress
    raw_start: int = 0  # index in *text* where the current token's raw text begins
    in_single = False
    in_double = False
    i = 0

    while i < len(text):
        c = text[i]

        if in_single:
            if c == "'":
                in_single = False
            else:
                buf.append(c)
            i += 1
        elif in_double:
            if c == '"':
                in_double = False
                i += 1
            elif c == "\\" and i + 1 < len(text) and text[i + 1] in ('"', "\\", " "):
                buf.append(text[i + 1])
                i += 2
            else:
                buf.append(c)
                i += 1
        elif c == "\\" and i + 1 < len(text):
            buf.append(text[i + 1])
            i += 2
        elif c == "'":
            if not buf:
                # Opening quote starts a new token: raw_start must point past
                # the quote so partial_raw (used for completion) is the quoted
                # content only, not the quote character itself.
                raw_start = i + 1
            in_single = True
            i += 1
        elif c == '"':
            if not buf:
                raw_start = i + 1
            in_double = True
            i += 1
        elif c in " \t":
            if buf:
                completed.append("".join(buf))
                buf = []
            i += 1
            raw_start = i
        else:
            buf.append(c)
            i += 1

    return completed, text[raw_start:]


def _unescape(s: str) -> str:
    """Remove backslash escaping: ``\\x`` → ``x``."""
    return re.sub(r"\\(.)", r"\1", s)


def _escape_path(s: str) -> str:
    """Backslash-escape characters that need quoting in a command token.

    Currently escapes backslashes and spaces so that a filename like
    ``terry pratchett.txt`` becomes ``terry\\ pratchett.txt`` when inserted
    as a completion, making it parseable by :func:`_tokenize_command`.
    """
    return s.replace("\\", "\\\\").replace(" ", "\\ ")


_TOOLS_NAME_SUBCMDS = frozenset({"allow", "disable", "disallow", "enable", "info"})
_TOOLS_FLAG_SUBCMDS = frozenset({"allow", "disable", "disallow", "enable"})
_SESSION_SUBCOMMANDS = ["name"]

# Default cap on file-path completions to keep the UI responsive.
# Overridable via ``repl_behavior.completion_max_results`` in config.
DEFAULT_MAX_PATH_COMPLETIONS = 200


class REPLCompleter(Completer):
    """Tab completer for the ai-cli REPL.

    Parameters
    ----------
    slash_commands:
        Top-level slash command names *without* the leading ``/``
        (e.g. ``["help", "exit", "tools", ...]``).
    tool_registry:
        Live registry used to enumerate tool names for ``/tools`` subcommands.
        Queried lazily on each completion request.
    workspace:
        Workspace used to resolve and filter file-path completions for
        ``@path`` references.
    task_manager:
        Live TaskManager used to enumerate task paths for ``/tasks``
        subcommands. Queried lazily on each completion request.
    mcp_manager:
        Live MCPManager used to enumerate server and tool names for ``/mcp``
        subcommands.  Queried lazily on each completion request.
    skill_registry_getter:
        Callable returning the current SkillRegistry for ``/skills info``
        completions. Queried lazily so completions reflect runtime reloads.
    max_path_completions:
        Maximum number of ``@path`` completions returned per keystroke.
        Defaults to :data:`DEFAULT_MAX_PATH_COMPLETIONS`.
    """

    def __init__(
        self,
        slash_commands: list[str],
        tool_registry: ToolRegistry | None = None,
        workspace: Workspace | None = None,
        task_manager: TaskManager | None = None,
        mcp_manager: MCPManager | None = None,
        skill_registry_getter: Callable[[], SkillRegistry | None] | None = None,
        skill_aliases_getter: Callable[[], dict[str, str]] | None = None,
        max_path_completions: int = DEFAULT_MAX_PATH_COMPLETIONS,
    ) -> None:
        if max_path_completions < 1:
            raise ValueError(
                f"max_path_completions must be >= 1, got {max_path_completions}"
            )
        self._slash_commands = sorted(slash_commands)
        self._tool_registry = tool_registry
        self._workspace = workspace
        self._task_manager = task_manager
        self._mcp_manager = mcp_manager
        self._skill_registry_getter = skill_registry_getter
        self._skill_aliases_getter = skill_aliases_getter
        self._max_path_completions = max_path_completions
        self._skill_names_cache_registry: SkillRegistry | None = None
        self._skill_names_cache: tuple[str, ...] | None = None

    # ------------------------------------------------------------------
    # Completer API
    # ------------------------------------------------------------------

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        try:
            yield from self._get_completions(document)
        except Exception:
            logger.warning("Unexpected error during tab completion", exc_info=True)

    def _get_completions(self, document: Document) -> Iterable[Completion]:
        text = document.text_before_cursor

        # Check for a partial @ reference at the cursor position first, so
        # that e.g. "@/foo" doesn't accidentally trigger slash completion.
        at_m = _AT_PARTIAL_RE.search(text)
        if at_m:
            bypass_ignore = bool(at_m.group(1))
            partial_raw = at_m.group(2)
            yield from self._complete_path(
                _unescape(partial_raw),
                bypass_ignore=bypass_ignore,
                raw_len=len(partial_raw),
            )
            return

        stripped = text.lstrip()
        if stripped.startswith("/"):
            yield from self._complete_slash(stripped)

    # ------------------------------------------------------------------
    # Slash command completion
    # ------------------------------------------------------------------

    def _complete_slash(self, text: str) -> Iterable[Completion]:
        """Yield completions for *text* which starts with ``/``."""
        parts = text.split()

        # Still typing the top-level command word (no trailing space yet).
        if len(parts) == 1 and not text.endswith(" "):
            prefix = parts[0][1:].lower()  # strip "/" and normalise case
            dynamic_commands = sorted(
                set(self._slash_commands) | set(self._skill_aliases())
            )
            for cmd in dynamic_commands:
                if cmd.startswith(prefix):
                    yield Completion(
                        "/" + cmd,
                        start_position=-len(parts[0]),
                        display="/" + cmd,
                    )
            return

        if not parts:
            return

        cmd = parts[0][1:].lower()
        if cmd == "tools":
            yield from self._complete_tools(parts, text)
        elif cmd == "tasks":
            yield from self._complete_tasks(parts, text)
        elif cmd == "session":
            yield from self._complete_session(parts, text)
        elif cmd == "rounds":
            yield from self._complete_rounds(parts, text)
        elif cmd == "index":
            yield from self._complete_index(parts, text)
        elif cmd == "mcp":
            yield from self._complete_mcp(parts, text)
        elif cmd == "skills":
            yield from self._complete_skills(text)

    def _complete_skills(self, text: str) -> Iterable[Completion]:
        completed, partial_raw = _tokenize_command(text)
        trailing = text.endswith((" ", "\t"))
        parts = list(completed)
        if not trailing and partial_raw:
            parts.append(_unescape(partial_raw))
        if not parts:
            return

        if len(parts) == 1 or (len(parts) == 2 and not trailing):
            prefix = (parts[1] if len(parts) > 1 else "").lower()
            raw_len = len(partial_raw) if len(parts) > 1 and not trailing else 0
            for sub in _SKILLS_SUBCOMMANDS:
                if sub.startswith(prefix):
                    yield Completion(sub, start_position=-raw_len)
            return

        subcmd = parts[1].lower()
        if subcmd != "info":
            return

        if (len(parts) == 2 and trailing) or (len(parts) == 3 and not trailing):
            prefix = parts[2] if len(parts) == 3 else ""
            for name in self._skill_names():
                if name.startswith(prefix):
                    raw_len = (
                        len(partial_raw) if len(parts) == 3 and not trailing else 0
                    )
                    yield Completion(
                        _escape_path(name),
                        start_position=-raw_len,
                        display=name,
                    )

    def _complete_tools(self, parts: list[str], text: str) -> Iterable[Completion]:
        trailing = text.endswith(" ")

        # Completing the subcommand.
        if len(parts) == 1 or (len(parts) == 2 and not trailing):
            prefix = (parts[1] if len(parts) > 1 else "").lower()
            raw_len = len(parts[1]) if len(parts) > 1 else 0
            for sub in _TOOLS_SUBCOMMANDS:
                if sub.startswith(prefix):
                    yield Completion(sub, start_position=-raw_len)
            return

        subcmd = parts[1].lower()
        if subcmd not in _TOOLS_NAME_SUBCMDS:
            return

        after = parts[2:]  # tokens after "/tools <subcmd>"

        if subcmd in _TOOLS_FLAG_SUBCMDS:
            # /tools enable|disable|allow|disallow [--session] <name>
            if not after or (len(after) == 1 and not trailing):
                prefix = after[0] if after else ""
                yield from self._flag_then_tool_names(prefix)
                return
            if after[0] == "--session" and (
                len(after) == 1 or (len(after) == 2 and not trailing)
            ):
                # --session already present; next token is the tool name.
                prefix = after[1] if len(after) > 1 else ""
                for name in self._tool_names():
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))
        else:
            # /tools info <name>
            if not after or (len(after) == 1 and not trailing):
                prefix = after[0] if after else ""
                for name in self._tool_names():
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))

    def _flag_then_tool_names(self, prefix: str) -> Iterable[Completion]:
        """Yield ``--session`` then all matching tool names for *prefix*."""
        if "--session".startswith(prefix):
            yield Completion("--session", start_position=-len(prefix))
        for name in self._tool_names():
            if name.startswith(prefix):
                yield Completion(name, start_position=-len(prefix))

    def _complete_tasks(self, parts: list[str], text: str) -> Iterable[Completion]:
        trailing = text.endswith(" ")
        if len(parts) == 1 or (len(parts) == 2 and not trailing):
            prefix = (parts[1] if len(parts) > 1 else "").lower()
            raw_len = len(parts[1]) if len(parts) > 1 else 0
            for sub in _TASKS_SUBCOMMANDS:
                if sub.startswith(prefix):
                    yield Completion(sub, start_position=-raw_len)
            return

        subcmd = parts[1].lower()
        if subcmd == "note":
            after = parts[2:]

            # Complete verb: only "obsolete" is supported for now.
            if not after or (len(after) == 1 and not trailing):
                prefix = (after[0] if after else "").lower()
                raw_len = len(after[0]) if after else 0
                if "obsolete".startswith(prefix):
                    yield Completion("obsolete", start_position=-raw_len)
                return

            if after[0].lower() != "obsolete":
                return

            note_args = after[1:]

            # Complete the task path: /tasks note obsolete <path>
            path_partial = self._tasks_note_obsolete_path_partial(note_args, trailing)
            if path_partial is not None:
                yield from self._complete_task_path(path_partial)
                return

            # Offer --reason after index is provided.
            reason_prefix = self._tasks_note_obsolete_reason_prefix(note_args, trailing)
            if reason_prefix is not None and "--reason".startswith(reason_prefix):
                yield Completion("--reason", start_position=-len(reason_prefix))
            return

        path_partial = self._tasks_path_partial(subcmd, parts[2:], trailing)
        if path_partial is None:
            return

        yield from self._complete_task_path(path_partial)

    def _tasks_note_obsolete_path_partial(
        self, note_args: list[str], trailing: bool
    ) -> str | None:
        if not note_args:
            return "" if trailing else None
        if len(note_args) == 1 and not trailing:
            return note_args[0]
        return None

    def _tasks_note_obsolete_reason_prefix(
        self, note_args: list[str], trailing: bool
    ) -> str | None:
        # note_args shape after 'obsolete': [<path>, <index>, ...]
        if len(note_args) == 2 and trailing:
            return ""
        if len(note_args) == 3 and not trailing:
            return note_args[2]
        return None

    def _tasks_path_partial(
        self, subcmd: str, args: list[str], trailing: bool
    ) -> str | None:
        if subcmd in _TASKS_REQUIRED_PATH_SUBCMDS | _TASKS_OPTIONAL_PATH_SUBCMDS:
            if not args:
                return ""
            if len(args) == 1 and not trailing:
                return args[0]
            return None

        if subcmd != "tree":
            return None

        if args and args[-1] == "--depth":
            return None
        if not trailing and len(args) >= 2 and args[-2] == "--depth":
            return None

        positional_count = 0
        i = 0
        while i < len(args):
            token = args[i]
            if token == "--depth":
                i += 2
                continue
            positional_count += 1
            i += 1

        if trailing:
            return "" if positional_count == 0 else None

        partial = args[-1]
        if partial == "--depth" or partial.startswith("-"):
            return None
        return partial if positional_count <= 1 else None

    def _complete_task_path(self, partial: str) -> Iterable[Completion]:
        detail_map = self._task_detail_map()
        if not detail_map:
            return
        detail_index = self._build_task_detail_index(detail_map)

        parent_path = ""
        prefix = partial
        if partial.endswith("."):
            parent_path = partial[:-1]
            prefix = ""
        elif "." in partial:
            parent_path, prefix = partial.rsplit(".", 1)

        parent_id: str | None = None
        if parent_path:
            parent = self._resolve_task_path(parent_path, detail_map, detail_index)
            if parent is None:
                return
            parent_id = parent["id"]

        for task in sorted(
            detail_index.get(parent_id, {}).values(),
            key=lambda t: str(t.get("name", "")),
        ):
            name = str(task.get("name", ""))
            if not name.startswith(prefix):
                continue
            candidate = f"{parent_path}.{name}" if parent_path else name
            subtask_count = len(task.get("subtasks", []))
            label = (
                f"{candidate} ({subtask_count} subtask)"
                if subtask_count == 1
                else f"{candidate} ({subtask_count} subtasks)"
            )
            yield Completion(
                candidate,
                start_position=-len(partial),
                display=label,
            )

    def _complete_session(self, parts: list[str], text: str) -> Iterable[Completion]:
        trailing = text.endswith(" ")
        if len(parts) == 1 or (len(parts) == 2 and not trailing):
            prefix = (parts[1] if len(parts) > 1 else "").lower()
            raw_len = len(parts[1]) if len(parts) > 1 else 0
            for sub in _SESSION_SUBCOMMANDS:
                if sub.startswith(prefix):
                    yield Completion(sub, start_position=-raw_len)

    def _complete_rounds(self, parts: list[str], text: str) -> Iterable[Completion]:
        trailing = text.endswith(" ")
        # Only offer --session (the numeric argument is free-form).
        if len(parts) == 1 or (len(parts) == 2 and not trailing):
            prefix = parts[1] if len(parts) > 1 else ""
            if "--session".startswith(prefix):
                yield Completion("--session", start_position=-len(prefix))

    def _complete_mcp(self, parts: list[str], text: str) -> Iterable[Completion]:
        """Completions for ``/mcp [subcommand] [server] [tool]``."""
        trailing = text.endswith(" ")

        # Completing the subcommand word.
        if len(parts) == 1 or (len(parts) == 2 and not trailing):
            prefix = (parts[1] if len(parts) > 1 else "").lower()
            raw_len = len(parts[1]) if len(parts) > 1 else 0
            for sub in _MCP_SUBCOMMANDS:
                if sub.startswith(prefix):
                    yield Completion(sub, start_position=-raw_len)
            return

        subcmd = parts[1].lower()

        # ``/mcp info <server>``  or  ``/mcp <action> [--persist] <server> [<tool>]``
        if subcmd not in (*_MCP_FLAG_SUBCMDS, "info"):
            return

        after = parts[2:]  # tokens after "/mcp <subcmd>"

        # Recognize --persist only in its documented flag position,
        # immediately after the subcommand.
        persist_present = bool(after and after[0] == "--persist")
        after_clean = after[1:] if persist_present else after

        if subcmd in _MCP_FLAG_SUBCMDS:
            # Position 0 (after optional --persist): server name or --persist.
            if not after_clean or (len(after_clean) == 1 and not trailing):
                prefix = after_clean[0] if after_clean else ""
                if not persist_present and "--persist".startswith(prefix):
                    yield Completion("--persist", start_position=-len(prefix))
                for name in self._mcp_server_names():
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))
                return

            server_name = after_clean[0]

            # Position 1: tool name within that server.
            if len(after_clean) == 1 or (len(after_clean) == 2 and not trailing):
                prefix = after_clean[1] if len(after_clean) > 1 else ""
                for t in self._mcp_tool_names(server_name):
                    if t.startswith(prefix):
                        yield Completion(t, start_position=-len(prefix))
            return

        # subcmd == "info": complete server name only.
        if not after or (len(after) == 1 and not trailing):
            prefix = after[0] if after else ""
            for name in self._mcp_server_names():
                if name.startswith(prefix):
                    yield Completion(name, start_position=-len(prefix))

    def _complete_index(self, parts: list[str], text: str) -> Iterable[Completion]:
        """Completions for ``/index [path] [--label <name>] [--file <path>] ...``.

        Uses :func:`_tokenize_command` so paths containing backslash-escaped
        spaces (e.g. ``terry\\ pratchett/``) are handled correctly.
        Completions with spaces in their names are inserted backslash-escaped.
        """
        completed_toks, partial_raw = _tokenize_command(text)
        after = completed_toks[1:]  # decoded tokens after "/index"
        partial = _unescape(partial_raw)
        raw_len = len(partial_raw)

        # Suppress completions when typing the free-form --label value.
        if after and after[-1] == "--label":
            return

        # After --file, complete the next token as a file/directory path.
        if after and after[-1] == "--file":
            yield from self._complete_path(partial, raw_len=raw_len)
            return

        # When partial starts with "-" offer only flags (no path noise).
        # When partial is empty, offer flags first then fall through to paths.
        if partial_raw == "" or partial_raw.startswith("-"):
            already = set(after)
            for flag in _INDEX_FLAGS:
                if flag.startswith(partial_raw) and flag not in already:
                    yield Completion(flag, start_position=-raw_len)
            if partial_raw.startswith("-"):
                return  # don't mix paths when explicitly typing a flag

        # Complete as a filesystem path (positional root argument).
        yield from self._complete_path(partial, raw_len=raw_len)

    # ------------------------------------------------------------------
    # @ file-path completion
    # ------------------------------------------------------------------

    def _complete_path(
        self,
        partial: str,
        bypass_ignore: bool = False,
        raw_len: int | None = None,
    ) -> Iterable[Completion]:
        """Yield file-path completions for *partial* (an unescaped path prefix).

        ``..`` and absolute paths (``/``) are allowed — Python's pathlib
        treats a leading ``/`` as absolute when joined, so
        ``root / "/etc"`` resolves to ``/etc``.  Ignore rules are only
        applied for paths inside the workspace root; paths that escape it
        (or when *bypass_ignore* is ``True``) are listed as-is.

        Spaces in entry names are backslash-escaped in the inserted text so
        the result is parseable by :func:`_tokenize_command`.

        Parameters
        ----------
        partial:
            The decoded (unescaped) path prefix being completed.
        bypass_ignore:
            When ``True``, skip workspace ignore-filter checks.
        raw_len:
            Length of the raw (escaped) partial in the original input.
            Used as the negated ``start_position`` so the raw text is fully
            replaced.  Defaults to ``len(partial)`` when ``None``.
        """
        if self._workspace is None:
            return

        root = self._workspace.root
        start = -(raw_len if raw_len is not None else len(partial))

        # Split into a confirmed directory prefix and the stem being typed.
        if "/" in partial:
            dir_part, stem = partial.rsplit("/", 1)
        else:
            dir_part = ""
            stem = partial

        # pathlib: root / "/absolute" == Path("/absolute"); root / ".." goes up.
        if dir_part:
            try:
                base_dir = (root / dir_part).resolve()
            except OSError:
                return
        else:
            base_dir = root

        if not base_dir.is_dir():
            return

        within_workspace = base_dir.is_relative_to(root)

        count = 0
        try:
            with os.scandir(base_dir) as it:
                for entry in it:
                    if not entry.name.startswith(stem):
                        continue

                    try:
                        is_dir = entry.is_dir(follow_symlinks=True)
                    except OSError:
                        continue
                    abs_path = base_dir / entry.name

                    if not bypass_ignore and within_workspace:
                        try:
                            if self._workspace.is_ignored(abs_path, is_dir=is_dir):
                                continue
                        except OSError:
                            continue

                    suffix = "/" if is_dir else ""
                    # Escape spaces so the inserted text survives re-tokenization.
                    escaped_name = _escape_path(entry.name)
                    rel = (
                        f"{_escape_path(dir_part)}/{escaped_name}"
                        if dir_part
                        else escaped_name
                    )
                    yield Completion(
                        rel + suffix,
                        start_position=start,
                        display=entry.name + suffix,
                    )
                    count += 1
                    if count >= self._max_path_completions:
                        break
        except OSError:
            return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tool_names(self) -> list[str]:
        if self._tool_registry is None:
            return []
        return sorted(t["name"] for t in self._tool_registry.all_tools_info())

    def _task_detail_map(self) -> dict[str, dict]:
        if self._task_manager is None:
            return {}
        try:
            return self._task_manager.get_all_task_details_map()
        except Exception:
            logger.warning("Task completion lookup failed", exc_info=True)
            return {}

    @staticmethod
    def _build_task_detail_index(
        detail_map: dict[str, dict],
    ) -> dict[str | None, dict[str, dict]]:
        index: dict[str | None, dict[str, dict]] = {}
        for task in detail_map.values():
            parent_id = task.get("parent_id")
            name = task.get("name")
            if not isinstance(name, str):
                continue
            index.setdefault(parent_id, {})[name] = task
        return index

    @staticmethod
    def _resolve_task_path(
        path: str,
        detail_map: dict[str, dict],
        detail_index: dict[str | None, dict[str, dict]] | None = None,
    ) -> dict | None:
        if detail_index is None:
            detail_index = REPLCompleter._build_task_detail_index(detail_map)

        current_parent_id: str | None = None
        found: dict | None = None
        for segment in path.split("."):
            found = detail_index.get(current_parent_id, {}).get(segment)
            if found is None:
                return None
            current_parent_id = str(found["id"])
        return found

    def _mcp_server_names(self) -> list[str]:
        if self._mcp_manager is None:
            return []
        return sorted(self._mcp_manager.server_names())

    def _mcp_tool_names(self, server_name: str) -> list[str]:
        if self._mcp_manager is None:
            return []
        return sorted(self._mcp_manager.get_server_tools(server_name))

    def _skill_names(self) -> list[str]:
        if self._skill_registry_getter is None:
            return []
        try:
            registry = self._skill_registry_getter()
        except Exception:
            logger.warning("Skill completion lookup failed", exc_info=True)
            return []
        if registry is None:
            self._skill_names_cache_registry = None
            self._skill_names_cache = None
            return []

        if (
            registry is self._skill_names_cache_registry
            and self._skill_names_cache is not None
        ):
            return list(self._skill_names_cache)

        names = tuple(sorted(registry.names()))
        self._skill_names_cache_registry = registry
        self._skill_names_cache = names
        return list(names)

    def _skill_aliases(self) -> dict[str, str]:
        if self._skill_aliases_getter is None:
            return {}
        try:
            return dict(self._skill_aliases_getter())
        except Exception:
            logger.warning("Skill alias completion lookup failed", exc_info=True)
            return {}
