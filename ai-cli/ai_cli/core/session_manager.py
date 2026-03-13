"""
session_manager.py — Session lifecycle: create, persist, resume, compact.

Each session is stored as a directory under the global sessions directory
(``get_global_dir()/sessions/<session-id>/``) and contains three files:

  metadata.yaml         — workspace path, timestamps, counters, optional name
  history_full.jsonl    — every message ever added, with timestamps (append-only)
  history_current.jsonl — messages the LLM sees; may be replaced by compaction

``Session`` manages a single conversation.  ``SessionManager`` is the
factory and registry that creates and looks up sessions.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ai_cli.core.llm_client import LLMError

if TYPE_CHECKING:
    from ai_cli.core.llm_client import LLMClient
    from ai_cli.core.workspace import Workspace

logger = logging.getLogger(__name__)

# Strict session ID format produced by _generate_session_id(): YYYYMMDDTHHMMSS-<8 hex>
_SESSION_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}-[0-9a-f]{8}$")

# Valid message roles accepted by the LLM API.
_VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


_PREVIEW_LEN = 120  # max chars kept for truncated preview fields in metadata
_COMPACT_THRESHOLD = 0.9  # compact when token usage exceeds this fraction of context
_COMPACT_OVERHEAD_TOKENS = (
    500  # reserved for compaction system prompt + summarize instruction
)

# System prompt injected into the one-shot summarisation call.
_COMPACT_SYSTEM = (
    "You are a helpful assistant that summarizes conversations. "
    "Produce a concise but complete summary that preserves all facts, "
    "decisions, file paths, code snippets, and action items. "
    "The summary will replace the conversation history; no context outside "
    "it will be available to you in future turns."
)


class SessionError(Exception):
    """Raised for unrecoverable session-level errors."""


# ---------------------------------------------------------------------------
# SessionMeta — lightweight descriptor returned by SessionManager.list()
# ---------------------------------------------------------------------------


@dataclass
class SessionMeta:
    """Lightweight session descriptor, populated from metadata.yaml."""

    session_id: str
    workspace_path: Path
    started_at: datetime
    message_count: int
    name: str | None
    first_user_message: str
    last_message_role: str
    last_message_preview: str


# ---------------------------------------------------------------------------
# Session — the live conversation handle
# ---------------------------------------------------------------------------


class Session:
    """
    Manages a single conversation: history files, metadata, and compaction.

    Parameters
    ----------
    session_id:
        Unique identifier for this session (directory name under sessions_dir).
    session_dir:
        Absolute path to the session's storage directory.
    llm_client:
        Backend client used by :meth:`compact` and :meth:`token_usage`.
    """

    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        llm_client: LLMClient,
    ) -> None:
        self._id = session_id
        self._dir = session_dir
        self._llm = llm_client
        self._meta_path = self._dir / "metadata.yaml"
        self._full_path = self._dir / "history_full.jsonl"
        self._current_path = self._dir / "history_current.jsonl"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """The session's unique identifier."""
        return self._id

    @property
    def session_dir(self) -> Path:
        """Absolute path to this session's storage directory."""
        return self._dir

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """
        Append *role*/*content* to both history files and update metadata.

        ``history_full.jsonl`` records every message with a UTC timestamp.
        ``history_current.jsonl`` stores only the messages the LLM will see
        on the next turn (may be shorter after compaction).
        """
        if not isinstance(role, str) or not isinstance(content, str):
            raise SessionError(
                f"role and content must be strings; got role={type(role).__name__!r}, "
                f"content={type(content).__name__!r}"
            )
        if role not in _VALID_ROLES:
            raise SessionError(
                f"Invalid role {role!r}; must be one of {sorted(_VALID_ROLES)}"
            )

        now = datetime.now(timezone.utc).isoformat()
        current_entry = json.dumps({"role": role, "content": content})
        full_entry = json.dumps({"role": role, "content": content, "timestamp": now})

        current_size = (
            self._current_path.stat().st_size if self._current_path.exists() else 0
        )
        full_size = self._full_path.stat().st_size if self._full_path.exists() else 0

        try:
            with self._current_path.open("a", encoding="utf-8") as fh:
                fh.write(current_entry + "\n")
            with self._full_path.open("a", encoding="utf-8") as fh:
                fh.write(full_entry + "\n")
        except OSError as exc:
            self._rollback_history(current_size, full_size)
            raise SessionError(
                f"Could not append message to history files for session {self._id}: {exc}"
            ) from exc

        try:
            self._update_meta_after_message(role, content)
        except SessionError:
            # History writes succeeded but metadata failed — roll back history so
            # the two history files and metadata stay in sync.
            self._rollback_history(current_size, full_size)
            raise

    def get_messages(self) -> list[dict]:
        """Return messages from ``history_current.jsonl`` as a list of dicts.

        Lines that are not valid JSON or that lack string ``role``/``content``
        fields are skipped with a warning rather than raising.

        Raises
        ------
        SessionError
            If the history file cannot be opened or read (e.g. permission
            error or non-UTF-8 content).
        """
        if not self._current_path.exists():
            return []
        messages: list[dict] = []
        try:
            with self._current_path.open("r", encoding="utf-8") as fh:
                for lineno, raw in enumerate(fh, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Skipping malformed JSONL line %d in %s: %s",
                            lineno,
                            self._current_path,
                            exc,
                        )
                        continue
                    if not isinstance(entry, dict):
                        logger.warning(
                            "Skipping line %d in %s: expected a JSON object, got %s",
                            lineno,
                            self._current_path,
                            type(entry).__name__,
                        )
                        continue
                    if not isinstance(entry.get("role"), str) or not isinstance(
                        entry.get("content"), str
                    ):
                        logger.warning(
                            "Skipping line %d in %s: missing or non-string role/content",
                            lineno,
                            self._current_path,
                        )
                        continue
                    messages.append(
                        {"role": entry["role"], "content": entry["content"]}
                    )
        except (OSError, UnicodeDecodeError) as exc:
            raise SessionError(
                f"Could not read history file {self._current_path}: {exc}"
            ) from exc
        return messages

    def compact(self, instructions: str = "") -> None:
        """
        Replace ``history_current.jsonl`` with an LLM-generated summary.

        The full history (``history_full.jsonl``) is never modified — it
        retains every message for archival purposes.

        Parameters
        ----------
        instructions:
            Optional hint forwarded to the LLM, e.g. "focus on code changes".

        Raises
        ------
        SessionError
            If the LLM call fails or returns an empty summary, or if the
            compacted history cannot be written to disk.
        """
        messages = self.get_messages()
        if not messages:
            return

        summarize_instruction = (
            "Summarize the conversation above into a concise but complete summary "
            "that preserves all facts, decisions, file paths, code snippets, and "
            "action items."
        )
        if instructions:
            summarize_instruction += f" Focus on: {instructions}."

        summary_request = [
            {"role": "system", "content": _COMPACT_SYSTEM},
            *messages,
            {"role": "user", "content": summarize_instruction},
        ]

        text_parts: list[str] = []
        try:
            for chunk in self._llm.send(summary_request, tools=[], stream=False):
                if chunk["type"] == "text":
                    text_parts.append(chunk["delta"])
        except LLMError as exc:
            raise SessionError(f"LLM call failed during compaction: {exc}") from exc
        summary = "".join(text_parts).strip()

        if not summary:
            raise SessionError("LLM returned an empty summary during compaction.")

        summary_entry = json.dumps(
            {"role": "system", "content": f"Previous conversation summary:\n{summary}"}
        )

        # Save old history bytes so we can restore them if the metadata write fails.
        old_history = (
            self._current_path.read_bytes() if self._current_path.exists() else b""
        )

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._dir,
                suffix=".jsonl.tmp",
                delete=False,
            ) as fh:
                tmp_path = Path(fh.name)
                fh.write(summary_entry + "\n")
            tmp_path.replace(self._current_path)
        except OSError as exc:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise SessionError(
                f"Could not write compacted history to {self._current_path}: {exc}"
            ) from exc

        try:
            meta = self._read_meta()
            meta["message_count"] = 1
            meta["last_message_role"] = "system"
            meta["last_message_preview"] = _truncate(summary)
            self._write_meta(meta)
        except SessionError:
            # Restore the previous history so compacted content and metadata stay in sync.
            try:
                self._current_path.write_bytes(old_history)
            except OSError:
                logger.warning(
                    "Could not restore history after metadata failure in compact() for session %s",
                    self._id,
                )
            raise

        logger.debug(
            "Session %s compacted: %d messages → summary (%d chars).",
            self._id,
            len(messages),
            len(summary),
        )

    def set_name(self, name: str) -> None:
        """Persist *name* to ``metadata.yaml``."""
        meta = self._read_meta()
        meta["name"] = name
        self._write_meta(meta)

    def token_usage(self) -> tuple[int, int]:
        """
        Return ``(used_tokens, context_window)``.

        *used_tokens* is estimated via the LLM client's token counter.
        *context_window* comes from the LLM client's model metadata.
        """
        messages = self.get_messages()
        used = self._llm.count_tokens(messages)
        context_window: int = self._llm.get_model_metadata()["context_window"]
        return used, context_window

    def should_compact(self) -> bool:
        """Return ``True`` when token usage plus compaction overhead exceeds ``_COMPACT_THRESHOLD`` of the context window."""
        used, window = self.token_usage()
        return (used + _COMPACT_OVERHEAD_TOKENS) > window * _COMPACT_THRESHOLD

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rollback_history(self, current_size: int, full_size: int) -> None:
        """Best-effort truncate of both history files back to pre-write offsets."""
        for path, size in (
            (self._current_path, current_size),
            (self._full_path, full_size),
        ):
            try:
                if path.exists() and not path.is_symlink():
                    with path.open("r+b") as fh:
                        fh.truncate(size)
            except OSError:
                logger.warning("Could not roll back partial write to %s", path)

    def _read_meta(self) -> dict:
        if not self._meta_path.exists():
            return {}
        try:
            with self._meta_path.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise SessionError(
                f"Could not parse session metadata at {self._meta_path}: {exc}"
            ) from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise SessionError(
                f"Could not read session metadata at {self._meta_path}: {exc}"
            ) from exc
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise SessionError(
                f"Session metadata at {self._meta_path} is not a YAML mapping "
                f"(got {type(data).__name__})."
            )
        return data

    def _write_meta(self, meta: dict) -> None:
        """Write *meta* to ``metadata.yaml`` atomically via a temp file + replace."""
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._dir,
                suffix=".yaml.tmp",
                delete=False,
            ) as fh:
                tmp_path = Path(fh.name)
                yaml.safe_dump(
                    meta,
                    fh,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            tmp_path.replace(self._meta_path)
        except OSError as exc:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise SessionError(
                f"Could not write session metadata to {self._meta_path}: {exc}"
            ) from exc

    def _update_meta_after_message(self, role: str, content: str) -> None:
        meta = self._read_meta()
        try:
            current_count = int(meta.get("message_count", 0))
        except (TypeError, ValueError):
            logger.warning(
                "Session %s had non-numeric message_count in metadata; resetting to 0.",
                self._id,
            )
            current_count = 0
        meta["message_count"] = current_count + 1
        meta["last_message_role"] = role
        meta["last_message_preview"] = _truncate(content)
        if role == "user" and not meta.get("first_user_message"):
            meta["first_user_message"] = _truncate(content)
        self._write_meta(meta)


# ---------------------------------------------------------------------------
# SessionManager — factory and registry
# ---------------------------------------------------------------------------


class SessionManager:
    """
    Creates and looks up :class:`Session` objects.

    Parameters
    ----------
    workspace:
        The active project workspace.  Its ``root`` path is recorded in
        each new session's metadata so sessions can be filtered by project.
    llm_client:
        Passed through to every :class:`Session` it creates.
    sessions_dir:
        Root directory that contains one sub-directory per session.
        Typically ``get_global_dir() / "sessions"``.
    """

    def __init__(
        self,
        workspace: Workspace,
        llm_client: LLMClient,
        sessions_dir: Path,
    ) -> None:
        self._workspace = workspace
        self._llm = llm_client
        self._sessions_dir = sessions_dir
        try:
            self._sessions_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SessionError(
                f"Could not create sessions directory {self._sessions_dir}: {exc}"
            ) from exc

    def new(self) -> Session:
        """Create a fresh session, persist its metadata, and return it."""
        for _ in range(3):
            session_id = _generate_session_id()
            session_dir = self._sessions_dir / session_id
            try:
                session_dir.mkdir(parents=False, exist_ok=False)
                break
            except FileExistsError:
                continue  # extremely unlikely UUID collision — retry
            except OSError as exc:
                raise SessionError(
                    f"Could not create session directory {session_dir}: {exc}"
                ) from exc
        else:
            raise SessionError(
                "Could not create a unique session directory after 3 attempts."
            )

        session = Session(session_id, session_dir, self._llm)
        meta: dict = {
            "session_id": session_id,
            "workspace_path": str(self._workspace.root),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "message_count": 0,
            "name": None,
            "first_user_message": "",
            "last_message_role": "",
            "last_message_preview": "",
        }
        session._write_meta(meta)
        return session

    def list(self, workspace_path: Path) -> list[SessionMeta]:
        """
        Return :class:`SessionMeta` objects for *workspace_path*, newest first.

        Directories with no ``metadata.yaml`` are silently skipped (they are
        not session directories).  Directories whose ``metadata.yaml`` exists
        but cannot be read or parsed are skipped with a warning.
        """
        results: list[SessionMeta] = []
        try:
            entries = list(self._sessions_dir.iterdir())
        except OSError as exc:
            raise SessionError(
                f"Could not list sessions directory {self._sessions_dir}: {exc}"
            ) from exc
        for entry in entries:
            if not entry.is_dir():
                continue
            meta_file = entry / "metadata.yaml"
            if not meta_file.exists():
                continue
            try:
                with meta_file.open(encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                logger.warning(
                    "Could not read session metadata at %s: %s", meta_file, exc
                )
                continue
            if data is None:
                data = {}
            if not isinstance(data, dict):
                logger.warning(
                    "Skipping session at %s: metadata is not a YAML mapping (got %s)",
                    meta_file,
                    type(data).__name__,
                )
                continue
            try:
                meta = _parse_session_meta(data)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not parse session metadata at %s: %s", meta_file, exc
                )
                continue
            if meta.workspace_path != workspace_path:
                continue
            results.append(meta)
        results.sort(key=lambda m: (m.started_at, m.session_id), reverse=True)
        return results

    def load(self, session_id: str) -> Session:
        """
        Return a :class:`Session` for an existing *session_id*.

        Raises
        ------
        SessionError
            If *session_id* is invalid, or the session directory or its
            metadata file does not exist.
        """
        if not _SESSION_ID_RE.match(session_id or ""):
            raise SessionError(f"Invalid session ID: {session_id!r}")
        session_dir = self._sessions_dir / session_id
        if not session_dir.is_dir():
            raise SessionError(f"Session not found: {session_id!r}")
        if not (session_dir / "metadata.yaml").exists():
            raise SessionError(f"Session metadata missing for {session_id!r}")
        session = Session(session_id, session_dir, self._llm)
        session._read_meta()  # raises SessionError if corrupt or not a mapping
        return session

    def most_recent(self, workspace_path: Path) -> Session | None:
        """
        Return the most recently started session for *workspace_path*, or ``None``.
        """
        sessions = self.list(workspace_path)
        if not sessions:
            return None
        return self.load(sessions[0].session_id)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _truncate(text: str) -> str:
    """Truncate *text* to at most ``_PREVIEW_LEN`` characters."""
    if len(text) <= _PREVIEW_LEN:
        return text
    return text[: _PREVIEW_LEN - 1] + "…"


def _generate_session_id() -> str:
    """Return a unique, time-sortable session ID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _parse_session_meta(data: dict) -> SessionMeta:
    """Construct a :class:`SessionMeta` from a raw metadata dict."""
    raw_ts = data.get("started_at", "")
    try:
        started_at = datetime.fromisoformat(str(raw_ts))
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        started_at = datetime.min.replace(tzinfo=timezone.utc)

    try:
        message_count = int(data.get("message_count", 0))
    except (TypeError, ValueError):
        message_count = 0

    raw_name = data.get("name")
    return SessionMeta(
        session_id=str(data.get("session_id") or ""),
        workspace_path=Path(str(data.get("workspace_path") or "")),
        started_at=started_at,
        message_count=message_count,
        name=str(raw_name) if raw_name is not None else None,
        first_user_message=str(data.get("first_user_message") or ""),
        last_message_role=str(data.get("last_message_role") or ""),
        last_message_preview=str(data.get("last_message_preview") or ""),
    )
