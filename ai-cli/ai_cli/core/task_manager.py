"""
task_manager.py — Persistent task tree for the task system.

Tasks are stored in ``<session_dir>/tasks.json`` as a flat map of
``task_id → Task``.  The manager owns all file I/O, ID generation, validation,
and status-transition enforcement.

The file is created on the first write; reads against a missing file return
empty results rather than raising.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import re
import string
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

# Valid status values and the allowed transitions from each.
# "done" is only reachable via mark_done(), not update_task().
_VALID_STATUSES = frozenset(
    {"not_started", "in_progress", "blocked", "in_review", "done"}
)
_UPDATABLE_STATUSES = _VALID_STATUSES - frozenset({"done"})

_VALID_PRIORITIES = frozenset({"low", "medium", "high"})

# Public aliases — importable by task tools so their ToolSchema enums stay
# in sync with TaskManager's validation without duplicating the sets.
VALID_STATUSES: frozenset[str] = _VALID_STATUSES
UPDATABLE_STATUSES: frozenset[str] = _UPDATABLE_STATUSES
VALID_PRIORITIES: frozenset[str] = _VALID_PRIORITIES

# Allowed status transitions for update_task().  Self-transitions are included
# so that setting the same status is a no-op rather than an error.
# "done" has no outbound transitions — only mark_done() can reach it.
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "not_started": frozenset({"not_started", "in_progress", "blocked"}),
    "in_progress": frozenset({"in_progress", "in_review", "blocked"}),
    # in_review → in_progress allows reviewer rejection.
    "in_review": frozenset({"in_review", "in_progress", "blocked"}),
    # blocked → in_progress is the unblocking path.
    "blocked": frozenset({"blocked", "in_progress"}),
    # done tasks cannot change status via update_task().
    "done": frozenset(),
}

_MIN_DOD_CHARS = 5
_ID_CHARS = string.ascii_lowercase + string.digits
_ID_LENGTH = 6
_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _generate_id() -> str:
    return "task_" + "".join(random.choices(_ID_CHARS, k=_ID_LENGTH))


class TaskValidationError(ValueError):
    """Raised when a task operation violates a business rule."""


class TaskNotFoundError(LookupError):
    """Raised when a referenced task_id does not exist."""


class TaskStorageError(OSError):
    """Raised when tasks.json exists but cannot be read or parsed.

    A missing file is *not* an error — callers receive an empty store.
    A corrupt or unreadable file raises this so callers can distinguish
    "no tasks yet" from "storage is broken" and avoid overwriting data.
    """


class TaskManager:
    """Read/write the task tree in ``<session_dir>/tasks.json``.

    All public methods that mutate state write atomically via a temp-file
    rename so the file is never left in a partial state.
    """

    def __init__(self, session_dir: Path) -> None:
        self._path = session_dir / "tasks.json"
        self._last_ts: str = ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now_iso(self) -> str:
        """Return a monotonically increasing ISO-8601 timestamp with microseconds.

        Forces ``timespec="microseconds"`` so the string always contains
        ``.ffffff`` before the trailing ``Z``.  Without a fixed timespec,
        when microseconds are zero the format omits ``.ffffff``, producing
        ``...T12:00:00Z`` which sorts *after* ``...T12:00:00.000001Z``
        because ``'.' < 'Z'`` in ASCII — breaking lexicographic ordering.

        If the wall clock returns the same microsecond as the previous call
        (possible on coarse-resolution platforms or in tight test loops), the
        timestamp is bumped forward by 1 µs to preserve strict monotonicity
        within this ``TaskManager`` instance.
        """
        from datetime import timedelta

        ts = (
            datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
        if ts <= self._last_ts:
            # Parse last_ts, add 1µs, reformat.
            last_dt = datetime.fromisoformat(self._last_ts.replace("Z", "+00:00"))
            ts = (
                (last_dt + timedelta(microseconds=1))
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z")
            )
        self._last_ts = ts
        return ts

    def _load(self) -> dict[str, Any]:
        """Load the task store from disk, returning an empty store if absent.

        A missing file is normal (no tasks created yet) and returns
        ``{"tasks": {}}``.  A file that exists but cannot be read or parsed
        raises :exc:`TaskStorageError` to prevent silent data loss.
        """
        if not self._path.exists():
            return {"tasks": {}}
        try:
            text = self._path.read_text(encoding="utf-8")
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TaskStorageError(
                f"tasks.json contains invalid JSON and cannot be loaded: {exc}"
            ) from exc
        except OSError as exc:
            raise TaskStorageError(f"tasks.json could not be read: {exc}") from exc

        if (
            not isinstance(data, dict)
            or "tasks" not in data
            or not isinstance(data.get("tasks"), dict)
        ):
            # Use microsecond-precision ISO timestamp so repeated corruptions
            # within the same second don't collide.
            ts = (
                datetime.now(timezone.utc)
                .isoformat(timespec="microseconds")
                .replace(":", "")
            )
            corrupt_path = self._path.with_name(f"tasks.{ts}.corrupt")
            quarantined = False
            try:
                self._path.rename(corrupt_path)
                quarantined = True
                logger.error(
                    "tasks.json has unexpected structure; quarantined to %s",
                    corrupt_path,
                )
            except OSError as rename_exc:
                logger.error(
                    "tasks.json has unexpected structure and could not be quarantined: %s",
                    rename_exc,
                )
            detail = f" A backup was saved to {corrupt_path}." if quarantined else ""
            raise TaskStorageError(f"tasks.json has unexpected structure.{detail}")
        return cast(dict[str, Any], data)

    def _save(self, data: dict[str, Any]) -> None:
        """Write *data* to disk atomically.

        The temp file is fully written, fsync-ed, and *closed* before the
        atomic rename so that cleanup (unlink) never races with an open
        handle — important on Windows where unlinking an open file fails.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=".tasks_",
                suffix=".json.tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                json.dump(data, tmp, indent=2, ensure_ascii=False)
                tmp.flush()
                os.fsync(tmp.fileno())
            # File handle is now closed; safe to rename on all platforms.
            tmp_path.replace(self._path)
            # fsync the parent directory so the rename is durable.
            with contextlib.suppress(OSError):
                dir_fd = os.open(str(self._path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except Exception:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    tmp_path.unlink(missing_ok=True)
            raise

    def _get_or_raise(self, data: dict[str, Any], task_id: str) -> dict[str, Any]:
        task = data["tasks"].get(task_id)
        if task is None:
            raise TaskNotFoundError(f"Task not found: {task_id!r}")
        return self._validate_task_record(task)

    _REQUIRED_TASK_FIELDS: frozenset[str] = frozenset(
        {"id", "name", "status", "priority"}
    )

    def _validate_task_record(self, task: object) -> dict[str, Any]:
        """Raise ``TaskStorageError`` if *task* is not a well-formed task dict.

        Checks that the record is a ``dict`` and contains all required string
        fields.  Called before accessing ``task["id"]`` etc. so that partial
        corruption produces a clear error rather than a raw ``KeyError``.
        """
        if not isinstance(task, dict):
            raise TaskStorageError(
                f"Malformed task record: expected dict, got {type(task).__name__}."
            )
        missing = self._REQUIRED_TASK_FIELDS - task.keys()
        if missing:
            raise TaskStorageError(
                f"Malformed task record is missing required field(s): "
                f"{sorted(missing)!r}."
            )
        for field in self._REQUIRED_TASK_FIELDS:
            if not isinstance(task[field], str):
                raise TaskStorageError(
                    f"Malformed task record: field {field!r} must be a str, "
                    f"got {type(task[field]).__name__}."
                )
        if task["status"] not in _VALID_STATUSES:
            raise TaskStorageError(
                f"Malformed task record: 'status' {task['status']!r} is not a "
                f"valid status. Expected one of {sorted(_VALID_STATUSES)}."
            )
        if task["priority"] not in _VALID_PRIORITIES:
            raise TaskStorageError(
                f"Malformed task record: 'priority' {task['priority']!r} is not a "
                f"valid priority. Expected one of {sorted(_VALID_PRIORITIES)}."
            )
        return cast(dict[str, Any], task)

    # ------------------------------------------------------------------
    # Response shapes
    # ------------------------------------------------------------------

    def _task_summary(self, task: dict) -> dict:
        task = self._validate_task_record(task)
        return {
            "id": task["id"],
            "name": task["name"],
            "status": task["status"],
            "priority": task["priority"],
            "has_subtasks": bool(task.get("subtask_ids")),
        }

    def _task_detail(self, data: dict, task: dict) -> dict:
        task_id = task["id"]

        subtask_ids = task.get("subtask_ids", [])
        if not isinstance(subtask_ids, list) or not all(
            isinstance(s, str) for s in subtask_ids
        ):
            raise TaskStorageError(
                f"Task {task_id!r} has a corrupted 'subtask_ids' field."
            )
        missing = [sid for sid in subtask_ids if sid not in data["tasks"]]
        if missing:
            raise TaskValidationError(
                f"Task {task_id!r} references missing subtask(s): {missing!r}"
            )
        subtasks = [self._task_summary(data["tasks"][sid]) for sid in subtask_ids]

        blockers = task.get("blockers", [])
        if not isinstance(blockers, list) or not all(
            isinstance(b, str) for b in blockers
        ):
            raise TaskStorageError(
                f"Task {task_id!r} has a corrupted 'blockers' field."
            )

        notes = task.get("notes", [])
        if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
            raise TaskStorageError(f"Task {task_id!r} has a corrupted 'notes' field.")

        parent_id = task.get("parent_id")
        if parent_id is not None and not isinstance(parent_id, str):
            raise TaskStorageError(
                f"Task {task_id!r} has a corrupted 'parent_id' field."
            )
        if parent_id is not None and parent_id not in data["tasks"]:
            raise TaskStorageError(
                f"Task {task_id!r} references a missing parent task: {parent_id!r}"
            )

        return {
            "id": task_id,
            "parent_id": parent_id,
            "name": task["name"],
            "description": task.get("description", ""),
            "definition_of_done": task.get("definition_of_done", ""),
            "status": task["status"],
            "priority": task["priority"],
            "next_action": task.get("next_action", ""),
            "blockers": list(blockers),
            "notes": list(notes),
            "subtasks": subtasks,
        }

    # ------------------------------------------------------------------
    # Goal
    # ------------------------------------------------------------------

    def set_goal(self, goal: str) -> None:
        if not isinstance(goal, str) or not goal.strip():
            raise TaskValidationError("'goal' must be a non-empty string.")
        data = self._load()
        data["goal"] = goal
        self._save(data)

    def get_goal(self) -> str | None:
        goal = self._load().get("goal")
        if goal is None:
            return None
        if not isinstance(goal, str):
            raise TaskStorageError("Stored 'goal' must be a string or null.")
        return goal

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_task(
        self,
        name: str,
        definition_of_done: str,
        description: str = "",
        parent_id: str | None = None,
        priority: str = "medium",
    ) -> dict:
        """Create a new task and return its ``task_detail``."""
        if not isinstance(name, str) or not name.strip():
            raise TaskValidationError("'name' must be a non-empty string.")
        name = name.strip()
        if not _NAME_RE.match(name):
            raise TaskValidationError(
                "'name' must contain only letters, digits, and underscores."
            )
        if not isinstance(description, str):
            raise TaskValidationError("'description' must be a string.")
        if (
            not isinstance(definition_of_done, str)
            or len(definition_of_done.strip()) < _MIN_DOD_CHARS
        ):
            raise TaskValidationError(
                f"'definition_of_done' is required and must be at least {_MIN_DOD_CHARS} non-whitespace characters."
            )
        if not isinstance(priority, str) or priority not in _VALID_PRIORITIES:
            raise TaskValidationError(
                f"'priority' must be one of {sorted(_VALID_PRIORITIES)}."
            )

        data = self._load()

        if parent_id is not None:
            if not isinstance(parent_id, str):
                raise TaskValidationError("'parent_id' must be a string or None.")
            if parent_id not in data["tasks"]:
                raise TaskNotFoundError(f"Parent task not found: {parent_id!r}")

        # Enforce sibling-scoped name uniqueness.
        sibling_names: set[str] = set()
        for t in data["tasks"].values():
            if not isinstance(t, dict):
                raise TaskStorageError(
                    "Malformed task store: each task record must be a JSON object."
                )
            if t.get("parent_id") != parent_id:
                continue
            sibling_name = t.get("name")
            if not isinstance(sibling_name, str):
                raise TaskStorageError(
                    "Malformed task record in storage: sibling task is missing a valid 'name'."
                )
            sibling_names.add(sibling_name)
        if name in sibling_names:
            raise TaskValidationError(
                f"A task named {name!r} already exists under this parent."
            )

        # Generate a unique ID (collision is astronomically unlikely but guard anyway).
        for _ in range(10):
            task_id = _generate_id()
            if task_id not in data["tasks"]:
                break
        else:
            raise RuntimeError("Failed to generate a unique task ID after 10 attempts.")

        now = self._now_iso()
        task: dict = {
            "id": task_id,
            "parent_id": parent_id,
            "name": name,
            "description": description,
            "definition_of_done": definition_of_done,
            "status": "not_started",
            "priority": priority,
            "next_action": "",
            "blockers": [],
            "notes": [],
            "subtask_ids": [],
            "created_at": now,
            "updated_at": now,
        }
        data["tasks"][task_id] = task

        if parent_id is not None:
            parent = self._validate_task_record(data["tasks"][parent_id])
            parent_subtask_ids = parent.get("subtask_ids")
            if not isinstance(parent_subtask_ids, list):
                raise TaskStorageError(
                    f"Parent task {parent_id!r} has a corrupted 'subtask_ids' field."
                )
            parent_subtask_ids.append(task_id)
            parent["updated_at"] = now

        self._save(data)
        return self._task_detail(data, task)

    def get_task(self, task_id: str) -> dict:
        """Return the full ``task_detail`` for *task_id*."""
        data = self._load()
        task = self._get_or_raise(data, task_id)
        return self._task_detail(data, task)

    def list_tasks(self, parent_id: str | None = None) -> list[dict]:
        """Return lightweight summaries of direct children of *parent_id*.

        Pass ``None`` (default) for root-level tasks.
        """
        data = self._load()
        results = []
        for raw in data["tasks"].values():
            task = self._validate_task_record(
                raw
            )  # raises TaskStorageError if malformed
            if task.get("parent_id") == parent_id:
                results.append(self._task_summary(task))
        # Stable order: created_at ascending, then id for tie-breaking.
        results.sort(
            key=lambda t: (data["tasks"][t["id"]].get("created_at", ""), t["id"])
        )
        return results

    def update_task(self, task_id: str, **fields: object) -> dict:
        """Update allowed fields on an existing task.

        ``"done"`` is not an accepted status value here — use
        :meth:`mark_done` instead.
        """
        data = self._load()
        task = self._get_or_raise(data, task_id)

        allowed = {
            "name",
            "description",
            "definition_of_done",
            "status",
            "priority",
            "next_action",
            "blockers",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise TaskValidationError(
                f"Unknown field(s): {sorted(unknown)}. Allowed: {sorted(allowed)}."
            )

        if "name" in fields and (
            not isinstance(fields["name"], str) or not fields["name"].strip()
        ):
            raise TaskValidationError("'name' must be a non-empty string.")
        if "name" in fields and isinstance(fields["name"], str):
            new_name = fields["name"].strip()
            if not _NAME_RE.match(new_name):
                raise TaskValidationError(
                    "'name' must contain only letters, digits, and underscores."
                )
            parent_id = task.get("parent_id")
            sibling_names_upd: set[str] = set()
            for tid, t in data["tasks"].items():
                if tid == task_id:
                    continue
                if not isinstance(t, dict):
                    raise TaskStorageError(
                        f"Stored task {tid!r} is malformed: expected object."
                    )
                if t.get("parent_id") != parent_id:
                    continue
                sibling_name = t.get("name")
                if not isinstance(sibling_name, str) or not sibling_name.strip():
                    raise TaskStorageError(
                        f"Stored task {tid!r} is malformed: missing valid 'name'."
                    )
                sibling_names_upd.add(sibling_name)
            if new_name in sibling_names_upd:
                raise TaskValidationError(
                    f"A task named {new_name!r} already exists under this parent."
                )
            fields = dict(fields)
            fields["name"] = new_name

        for str_field in ("description", "next_action"):
            if str_field in fields and not isinstance(fields[str_field], str):
                raise TaskValidationError(f"'{str_field}' must be a string.")

        if "status" in fields:
            new_status = fields["status"]
            if not isinstance(new_status, str):
                raise TaskValidationError("'status' must be a string.")
            if new_status == "done":
                raise TaskValidationError(
                    "Use 'tasks_mark_done' to mark a task as done."
                )
            if new_status not in _UPDATABLE_STATUSES:
                raise TaskValidationError(
                    f"Invalid status {new_status!r}. Must be one of {sorted(_UPDATABLE_STATUSES)}."
                )
            # Enforce documented status transition rules.
            current_status = task.get("status", "not_started")
            valid_next: frozenset[str] = _ALLOWED_TRANSITIONS.get(
                str(current_status), frozenset()
            )
            if new_status not in valid_next:
                raise TaskValidationError(
                    f"Invalid status transition from {current_status!r} to {new_status!r}. "
                    f"Allowed: {sorted(valid_next)}."
                )

        if "priority" in fields and (
            not isinstance(fields["priority"], str)
            or fields["priority"] not in _VALID_PRIORITIES
        ):
            raise TaskValidationError(
                f"'priority' must be one of {sorted(_VALID_PRIORITIES)}."
            )

        if "definition_of_done" in fields:
            dod = fields["definition_of_done"]
            if not isinstance(dod, str) or len(dod.strip()) < _MIN_DOD_CHARS:
                raise TaskValidationError(
                    f"'definition_of_done' must be at least {_MIN_DOD_CHARS} non-whitespace characters."
                )

        if "blockers" in fields:
            blk = fields["blockers"]
            if not isinstance(blk, list) or not all(isinstance(b, str) for b in blk):
                raise TaskValidationError("'blockers' must be a list of strings.")

        for key, value in fields.items():
            task[key] = value
        task["updated_at"] = self._now_iso()

        self._save(data)
        return self._task_detail(data, task)

    def add_note(self, task_id: str, note: str) -> dict:
        """Append a timestamped note to *task_id*."""
        if not isinstance(note, str) or not note.strip():
            raise TaskValidationError("'note' must be a non-empty string.")

        data = self._load()
        task = self._get_or_raise(data, task_id)

        notes = task.setdefault("notes", [])
        if not isinstance(notes, list):
            raise TaskStorageError(
                f"Task {task_id!r} has a corrupted 'notes' field: "
                f"expected list, got {type(notes).__name__}."
            )

        timestamp = self._now_iso()
        notes.append(f"[{timestamp}] {note}")
        task["updated_at"] = timestamp

        self._save(data)
        return self._task_detail(data, task)

    def mark_done(self, task_id: str) -> dict:
        """Mark *task_id* as ``done``, enforcing structural constraints."""
        data = self._load()
        task = self._get_or_raise(data, task_id)

        dod = task.get("definition_of_done", "")
        if not isinstance(dod, str) or len(dod.strip()) < _MIN_DOD_CHARS:
            raise TaskValidationError(
                f"'definition_of_done' is required and must be at least {_MIN_DOD_CHARS} non-whitespace characters."
            )

        subtask_ids = task.get("subtask_ids", [])
        if not isinstance(subtask_ids, list) or not all(
            isinstance(s, str) for s in subtask_ids
        ):
            raise TaskStorageError(
                f"Task {task_id!r} has a corrupted 'subtask_ids' field."
            )
        for sub_id in subtask_ids:
            raw_sub = data["tasks"].get(sub_id)
            if raw_sub is None:
                raise TaskValidationError(
                    f"Subtask reference {sub_id!r} is missing for task {task_id!r}."
                )
            sub = self._validate_task_record(raw_sub)
            if sub["status"] != "done":
                raise TaskValidationError(
                    f"Subtask {sub_id!r} ({sub['name']!r}) is not done."
                )

        task["status"] = "done"
        task["updated_at"] = self._now_iso()
        self._save(data)
        return self._task_detail(data, task)

    def close_task(self, task_id: str) -> dict:
        """Force-close *task_id* and all its descendants.

        Sets ``status = "done"`` on the target and every descendant regardless
        of DoD validation.  Unlike :meth:`mark_done`, no structural checks are
        enforced — this is an override for human use.

        Returns the updated detail dict for the target task.
        """
        data = self._load()
        task = self._get_or_raise(data, task_id)

        now = self._now_iso()

        # DFS to collect the target + all descendants.
        ids_to_close: list[str] = []
        visited: set[str] = set()
        stack = [task_id]
        while stack:
            tid = stack.pop()
            if tid in visited:
                continue
            visited.add(tid)
            ids_to_close.append(tid)
            t = data["tasks"].get(tid)
            if t is None:
                raise TaskStorageError(
                    f"Task {tid!r} is referenced in the task tree but missing from storage"
                )
            if not isinstance(t, dict):
                raise TaskStorageError(
                    f"Task {tid!r} has invalid record: expected dict, "
                    f"got {type(t).__name__}"
                )
            raw_subs = t.get("subtask_ids", [])
            if not isinstance(raw_subs, list):
                raise TaskStorageError(
                    f"Task {tid!r} has invalid subtask_ids: expected list, "
                    f"got {type(raw_subs).__name__}"
                )
            for sub_id in raw_subs:
                if not isinstance(sub_id, str):
                    raise TaskStorageError(
                        f"Task {tid!r} has invalid subtask_ids entry: "
                        f"expected str, got {type(sub_id).__name__}"
                    )
                if sub_id not in visited:
                    stack.append(sub_id)

        for tid in ids_to_close:
            t = data["tasks"].get(tid)
            if isinstance(t, dict):
                t["status"] = "done"
                t["updated_at"] = now

        self._save(data)
        return self._task_detail(data, self._validate_task_record(task))

    def open_task(self, task_id: str) -> dict:
        """Re-open *task_id* and all of its ``done`` ancestors.

        Sets ``status = "not_started"`` on the target task, then walks up the
        ancestor chain and re-opens every ancestor whose status is ``"done"``,
        ensuring no ``done`` task has an unfinished descendant.

        Returns the updated detail dict for the target task.
        """
        data = self._load()
        task = self._get_or_raise(data, task_id)

        if task["status"] != "done":
            raise TaskValidationError(
                f"Task {task_id!r} is not done (status={task['status']!r}); "
                f"only done tasks can be re-opened."
            )

        now = self._now_iso()
        task["status"] = "not_started"
        task["updated_at"] = now

        # Walk up the parent chain, re-opening any "done" ancestor.
        current = task
        while True:
            parent_id = current.get("parent_id")
            if parent_id is None:
                break
            if not isinstance(parent_id, str):
                raise TaskStorageError(
                    f"Task {current['id']!r} has invalid parent_id: "
                    f"expected str or None, got {type(parent_id).__name__}"
                )
            if not parent_id:
                raise TaskStorageError(
                    f"Task {current['id']!r} has invalid parent_id: "
                    f"empty string is not allowed"
                )
            parent_raw = data["tasks"].get(parent_id)
            if parent_raw is None:
                raise TaskStorageError(
                    f"Task {current['id']!r} references missing parent {parent_id!r}"
                )
            if not isinstance(parent_raw, dict):
                raise TaskStorageError(
                    f"Task {current['id']!r} references invalid parent {parent_id!r}: "
                    f"expected dict, got {type(parent_raw).__name__}"
                )
            parent = self._validate_task_record(parent_raw)
            if parent["status"] != "done":
                break
            parent["status"] = "not_started"
            parent["updated_at"] = now
            current = parent

        self._save(data)
        return self._task_detail(data, self._validate_task_record(task))

    # ------------------------------------------------------------------
    # Queries (used by the orchestrator)
    # ------------------------------------------------------------------

    def find(self, status: str) -> list[dict]:
        """Return task summaries filtered by *status*, ordered by created_at then id."""
        data = self._load()
        results = []
        for raw in data["tasks"].values():
            summary = self._task_summary(raw)  # validates record shape
            if summary["status"] == status:
                results.append(summary)
        results.sort(
            key=lambda t: (data["tasks"][t["id"]].get("created_at", ""), t["id"])
        )
        return results

    def find_incomplete(self) -> list[dict]:
        """Return task summaries for all tasks that are not ``done``, ordered by created_at then id."""
        data = self._load()
        results = []
        for raw in data["tasks"].values():
            summary = self._task_summary(raw)  # validates record shape
            if summary["status"] != "done":
                results.append(summary)
        results.sort(
            key=lambda t: (data["tasks"][t["id"]].get("created_at", ""), t["id"])
        )
        return results

    def list_task_details(self, parent_id: str | None = None) -> list[dict]:
        """Return full ``task_detail`` dicts for direct children of *parent_id*.

        Like :meth:`list_tasks` but loads the store only once and returns
        full detail rather than lightweight summaries.  Sorted by created_at
        ascending, then id.
        """
        data = self._load()
        results = []
        for raw in data["tasks"].values():
            task = self._validate_task_record(raw)
            if task.get("parent_id") == parent_id:
                results.append(self._task_detail(data, task))
        results.sort(
            key=lambda t: (
                str(data["tasks"][t["id"]].get("created_at") or ""),
                t["id"],
            )
        )
        return results

    def get_all_task_details_map(self) -> dict[str, dict]:
        """Return a ``{task_id: task_detail}`` mapping for all tasks in one load.

        Useful for building display trees without N+1 disk reads.
        """
        data = self._load()
        return {
            task_id: self._task_detail(data, self._validate_task_record(raw))
            for task_id, raw in data["tasks"].items()
        }

    def all_tasks(self) -> list[dict]:
        """Return all tasks as full task records (not summaries).

        Used by the orchestrator's ``_pick_next_task()`` which needs
        ``subtask_ids``, ``created_at``, and ``status`` on every task.
        Returns shallow copies of the loaded records; mutations are not
        persisted unless saved through the manager.
        """
        data = self._load()
        tasks = []
        for raw in data["tasks"].values():
            tasks.append(dict(self._validate_task_record(raw)))
        return tasks

    # ------------------------------------------------------------------
    # Path addressing
    # ------------------------------------------------------------------

    def find_by_path(self, path: str) -> dict:
        """Resolve a dot-separated name path to a full ``task_detail``.

        Each segment is validated against ``_NAME_RE``.  Raises
        :exc:`TaskValidationError` for invalid segments and
        :exc:`TaskNotFoundError` if any segment is not found.
        """
        if not isinstance(path, str) or not path.strip():
            raise TaskValidationError("'path' must be a non-empty string.")
        segments = path.strip().split(".")
        for seg in segments:
            if not _NAME_RE.match(seg):
                raise TaskValidationError(
                    f"Invalid path segment {seg!r}: must match ^[A-Za-z0-9_]+$."
                )
        data = self._load()
        tasks = data["tasks"]
        # Validate every record up-front so malformed entries always raise TaskStorageError
        # rather than silently appearing as TaskNotFoundError.
        for t in tasks.values():
            self._validate_task_record(t)  # raises TaskStorageError if malformed
            pid = t.get("parent_id")
            if pid is not None:
                if not isinstance(pid, str):
                    raise TaskStorageError(
                        f"Task {t.get('id')!r} has a corrupted 'parent_id' field."
                    )
                if pid not in tasks:
                    raise TaskStorageError(
                        f"Task {t.get('id')!r} references missing parent {pid!r}."
                    )
        current_parent_id: str | None = None
        found: dict | None = None
        for seg in segments:
            found = None
            for t in tasks.values():
                if t.get("parent_id") == current_parent_id and t["name"] == seg:
                    found = t
                    break
            if found is None:
                raise TaskNotFoundError(f"Task not found at path segment {seg!r}")
            current_parent_id = found["id"]
        assert found is not None  # guaranteed by loop above (segments is non-empty)
        return self._task_detail(data, found)

    def delete_task(self, task_id: str) -> None:
        """Delete *task_id* and all its descendants.

        Removes the task from its parent's ``subtask_ids`` and saves atomically.
        """
        data = self._load()
        task = self._get_or_raise(data, task_id)

        # Collect all descendant IDs, guarding against cycles and corrupted subtask_ids.
        ids_to_delete: list[str] = []
        visited: set[str] = set()
        stack = [task_id]
        while stack:
            tid = stack.pop()
            if tid in visited:
                continue
            visited.add(tid)
            ids_to_delete.append(tid)
            t = data["tasks"].get(tid)
            if t is None:
                continue
            if not isinstance(t, dict):
                raise TaskStorageError(
                    f"Corrupted task store: record {tid!r} is not a JSON object."
                )
            raw_subtask_ids = t.get("subtask_ids", [])
            if not isinstance(raw_subtask_ids, list) or not all(
                isinstance(s, str) for s in raw_subtask_ids
            ):
                raise TaskStorageError(
                    f"Task {tid!r} has a corrupted 'subtask_ids' field."
                )
            stack.extend(s for s in raw_subtask_ids if s not in visited)

        # Cross-check parent_id links using a full BFS so orphaned descendants at
        # any depth are caught, regardless of dict iteration order or subtask_ids
        # corruption.  Build a parent->children index first, then traverse from
        # every already-queued ID.
        ids_to_delete_set = set(ids_to_delete)
        children_by_parent: dict[str, list[str]] = {}
        for tid, t in data["tasks"].items():
            if not isinstance(t, dict):
                continue
            pid = t.get("parent_id")
            if isinstance(pid, str):
                children_by_parent.setdefault(pid, []).append(tid)

        parent_stack = list(ids_to_delete)
        while parent_stack:
            pid = parent_stack.pop()
            for child_id in children_by_parent.get(pid, []):
                if child_id in ids_to_delete_set:
                    continue
                ids_to_delete.append(child_id)
                ids_to_delete_set.add(child_id)
                parent_stack.append(child_id)

        # Remove the root task from its parent's subtask_ids list.
        parent_id = task.get("parent_id")
        if parent_id is not None and not isinstance(parent_id, str):
            raise TaskStorageError(
                f"Task {task_id!r} has a corrupted 'parent_id' field."
            )
        if parent_id is not None and parent_id not in data["tasks"]:
            raise TaskStorageError(
                f"Corrupted task store: parent task {parent_id!r} referenced by"
                f" {task_id!r} is missing."
            )
        if parent_id is not None and parent_id in data["tasks"]:
            parent = data["tasks"][parent_id]
            if not isinstance(parent, dict):
                raise TaskStorageError(
                    f"Corrupted task store: parent record {parent_id!r} is not a JSON object."
                )
            subtask_ids = parent.get("subtask_ids", [])
            if not isinstance(subtask_ids, list) or not all(
                isinstance(s, str) for s in subtask_ids
            ):
                raise TaskStorageError(
                    f"Task {parent_id!r} has a corrupted 'subtask_ids' field."
                )
            new_subtask_ids = [s for s in subtask_ids if s != task_id]
            if new_subtask_ids != subtask_ids:
                parent["subtask_ids"] = new_subtask_ids
                parent["updated_at"] = self._now_iso()

        for tid in ids_to_delete:
            data["tasks"].pop(tid, None)

        self._save(data)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete tasks.json, removing all tasks and the goal."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.error("Failed to delete tasks.json: %s", exc)
            raise
