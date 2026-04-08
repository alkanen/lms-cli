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


def _generate_id() -> str:
    return "task_" + "".join(random.choices(_ID_CHARS, k=_ID_LENGTH))


class TaskValidationError(ValueError):
    """Raised when a task operation violates a business rule."""


class TaskNotFoundError(KeyError):
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

        return {
            "id": task_id,
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
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete tasks.json, removing all tasks and the goal."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.error("Failed to delete tasks.json: %s", exc)
            raise
