"""
Task tools — read and mutate the persistent task tree via TaskManager.

Seven tools are provided:

  tasks_list           List task summaries under a parent (or root).
  tasks_get            Retrieve full details of a single task.
  tasks_create         Create a new (sub)task.
  tasks_update         Update fields of an existing task.
  tasks_add_note       Append a timestamped note to a task.
  tasks_obsolete_note  Mark a note obsolete and remove it from active context.
  tasks_mark_done      Mark a task done (validates subtasks and DoD).

All seven share a :class:`~ai_cli.core.task_manager.TaskManager` instance
injected at construction time.  They use ``REGISTER_VIA_INSTANCE = True``
because their constructors are non-standard; they are wired into registries
in PR 3 via :meth:`~ai_cli.core.tool_registry.ToolRegistry.register_instance`.

Tasks are referenced by **name path** — a dot-separated string of task names
from the root to the target (e.g. ``"root_task.sub_task.leaf_task"``).
A single segment (e.g. ``"root_task"``) addresses a root-level task.
Name characters are restricted to ``[A-Za-z0-9_]``, so dots unambiguously
act as separators.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ai_cli.core.task_manager import (
    UPDATABLE_STATUSES,
    VALID_PRIORITIES,
    TaskManager,
    TaskNotFoundError,
    TaskStorageError,
    TaskValidationError,
    normalize_task_path,
)
from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

if TYPE_CHECKING:
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.workspace import Workspace

logger = logging.getLogger(__name__)

# Sorted lists for ToolSchema enums — derived from TaskManager's canonical sets
# so schema and validation never drift apart.
_UPDATABLE_STATUSES: list[str] = sorted(UPDATABLE_STATUSES)
_VALID_PRIORITIES: list[str] = sorted(VALID_PRIORITIES)


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _TaskTool(Tool):
    """Shared base for all task tools.

    Holds the ``TaskManager`` reference and provides the ``_handle`` helper
    that catches ``TaskManager`` exceptions and converts them to canonical
    error responses.
    """

    PERMISSION_REQUIRED = False
    REGISTER_VIA_INSTANCE = True

    def __init__(
        self,
        task_manager: TaskManager,
        workspace: Workspace,
        permission_manager: PermissionManager,
    ) -> None:
        super().__init__(
            workspace,
            permission_manager,
            self.PERMISSION_REQUIRED,
            self.NAME,  # type: ignore[attr-defined]
            self.DESCRIPTION,  # type: ignore[attr-defined]
        )
        self._tm = task_manager

    def _handle(self, fn: Any, *args: Any, **kwargs: Any) -> dict:
        """Call *fn* with *args*/*kwargs* and map TaskManager exceptions to errors."""
        try:
            result = fn(*args, **kwargs)
            logger.debug("Task tool '%s' succeeded", self.name)
            return self._ok(result)
        except TaskNotFoundError as exc:
            logger.info("Task tool '%s' not_found: %s", self.name, exc)
            return self._err(
                "not_found", exc.args[0] if exc.args else str(exc), code=404
            )
        except TaskValidationError as exc:
            logger.info("Task tool '%s' validation_error: %s", self.name, exc)
            return self._err("validation_error", str(exc), code=400)
        except TaskStorageError as exc:
            logger.info("Task tool '%s' storage_error: %s", self.name, exc)
            return self._err("storage_error", str(exc), code=500)

    def _parse_task_path(
        self, kwargs: dict[str, Any], key: str = "task_path"
    ) -> tuple[str, dict | None]:
        """Return ``(stripped_path, None)`` or ``("", error_dict)``."""
        raw = kwargs.get(key, "")
        try:
            return normalize_task_path(raw), None
        except TaskValidationError:
            return "", self._err(
                "validation_error", f"'{key}' must be a non-empty string.", code=400
            )

    def _parse_parent_path(
        self, kwargs: dict[str, Any]
    ) -> tuple[str | None, dict | None]:
        """Return ``(parent_path_or_None, None)`` or ``(None, error_dict)``."""
        raw = kwargs.get("parent_path")
        if raw is None or raw == "":
            return None, None
        if not isinstance(raw, str):
            return None, self._err(
                "validation_error", "'parent_path' must be a string.", code=400
            )
        if not raw.strip():
            return None, None
        try:
            return normalize_task_path(raw), None
        except TaskValidationError:
            return None, self._err(
                "validation_error",
                "'parent_path' must be a non-empty string.",
                code=400,
            )


# ---------------------------------------------------------------------------
# TasksListTool
# ---------------------------------------------------------------------------


class TasksListTool(_TaskTool):
    """List task summaries for direct children of *parent_path* (or root)."""

    NAME = "tasks_list"
    DESCRIPTION = (
        "List tasks as lightweight summaries. "
        "Pass parent_path to list subtasks; omit it for root-level tasks."
    )

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    "parent_path",
                    (
                        "Dot-separated name path of the parent task whose subtasks "
                        "to list (e.g. 'root_task.sub_task'). Omit for root tasks."
                    ),
                    "string",
                    required=False,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        parent_path, err = self._parse_parent_path(kwargs)
        if err:
            return err

        logger.debug("Task tool '%s' invoked: parent_path=%r", self.name, parent_path)

        def _go() -> dict:
            parent_id = (
                self._tm.resolve_path_to_id(parent_path) if parent_path else None
            )
            tasks = self._tm.list_tasks(parent_id=parent_id)
            logger.debug(
                "Task tool '%s': listed %d task(s) for parent_path=%r parent_id=%r",
                self.name,
                len(tasks),
                parent_path,
                parent_id,
            )
            return {"tasks": tasks}

        return self._handle(_go)


# ---------------------------------------------------------------------------
# TasksGetTool
# ---------------------------------------------------------------------------


class TasksGetTool(_TaskTool):
    """Retrieve full details of a single task."""

    NAME = "tasks_get"
    DESCRIPTION = "Get full details of a task, including subtask summaries."

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    "task_path",
                    (
                        "Dot-separated name path to the task "
                        "(e.g. 'root_task' or 'root_task.sub_task.leaf_task')."
                    ),
                    "string",
                    required=True,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        path, err = self._parse_task_path(kwargs)
        if err:
            return err
        logger.debug("Task tool '%s' invoked: task_path=%r", self.name, path)

        def _go() -> dict:
            task = self._tm.find_by_path(path)
            logger.debug(
                "Task tool '%s': retrieved task id=%s path=%r",
                self.name,
                task.get("id"),
                path,
            )
            return {"task": task}

        return self._handle(_go)


# ---------------------------------------------------------------------------
# TasksCreateTool
# ---------------------------------------------------------------------------


class TasksCreateTool(_TaskTool):
    """Create a new task (optionally as a subtask of an existing task)."""

    NAME = "tasks_create"
    DESCRIPTION = (
        "Create a new task. Provide parent_path to attach it as a subtask. "
        "definition_of_done is required and must be at least 5 non-whitespace characters."
    )

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    "name",
                    "Short task name (letters, digits, underscores only).",
                    "string",
                    required=True,
                ),
                ToolArgument(
                    "definition_of_done",
                    "Completion criteria (minimum 5 non-whitespace characters).",
                    "string",
                    required=True,
                ),
                ToolArgument(
                    "description",
                    "Detailed description of the task (default: empty).",
                    "string",
                    required=False,
                ),
                ToolArgument(
                    "parent_path",
                    (
                        "Dot-separated name path of the parent task to attach this "
                        "task as a subtask (e.g. 'root_task.sub_task'). "
                        "Omit to create a root-level task."
                    ),
                    "string",
                    required=False,
                ),
                ToolArgument(
                    "priority",
                    "Task priority.",
                    "string",
                    required=False,
                    enum=_VALID_PRIORITIES,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        name: str = kwargs.get("name", "")
        dod: str = kwargs.get("definition_of_done", "")
        description: str = kwargs.get("description", "")
        priority: str = kwargs.get("priority", "medium")

        parent_path, err = self._parse_parent_path(kwargs)
        if err:
            return err

        if not isinstance(name, str) or not name.strip():
            return self._err(
                "validation_error", "'name' must be a non-empty string.", code=400
            )
        if not isinstance(dod, str):
            return self._err(
                "validation_error", "'definition_of_done' must be a string.", code=400
            )
        if not isinstance(description, str):
            return self._err(
                "validation_error", "'description' must be a string.", code=400
            )
        if not isinstance(priority, str):
            return self._err(
                "validation_error", "'priority' must be a string.", code=400
            )

        logger.debug(
            "Task tool '%s' invoked: name=%r parent_path=%r priority=%r",
            self.name,
            name,
            parent_path,
            priority,
        )

        def _go() -> dict:
            parent_id = (
                self._tm.resolve_path_to_id(parent_path) if parent_path else None
            )
            task = self._tm.create_task(
                name=name,
                definition_of_done=dod,
                description=description,
                parent_id=parent_id,
                priority=priority,
            )
            logger.debug(
                "Task tool '%s': created task id=%s name=%r parent_id=%r",
                self.name,
                task.get("id"),
                task.get("name"),
                parent_id,
            )
            return {"task": task}

        return self._handle(_go)


# ---------------------------------------------------------------------------
# TasksUpdateTool
# ---------------------------------------------------------------------------


class TasksUpdateTool(_TaskTool):
    """Update one or more fields of an existing task."""

    NAME = "tasks_update"
    DESCRIPTION = (
        "Update fields of an existing task. "
        "Only include fields that need to change. "
        "Use tasks_mark_done to mark a task as done."
    )

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    "task_path",
                    (
                        "Dot-separated name path to the task to update "
                        "(e.g. 'root_task' or 'root_task.sub_task.leaf_task')."
                    ),
                    "string",
                    required=True,
                ),
                ToolArgument(
                    "name",
                    "New task name.",
                    "string",
                    required=False,
                ),
                ToolArgument(
                    "description",
                    "New description.",
                    "string",
                    required=False,
                ),
                ToolArgument(
                    "definition_of_done",
                    "New completion criteria (minimum 5 non-whitespace characters).",
                    "string",
                    required=False,
                ),
                ToolArgument(
                    "status",
                    "New status. Use tasks_mark_done to set status to 'done'.",
                    "string",
                    required=False,
                    enum=_UPDATABLE_STATUSES,
                ),
                ToolArgument(
                    "priority",
                    "New priority.",
                    "string",
                    required=False,
                    enum=_VALID_PRIORITIES,
                ),
                ToolArgument(
                    "next_action",
                    "Suggested next action for this task.",
                    "string",
                    required=False,
                ),
                ToolArgument(
                    "blockers",
                    "Replacement list of blocker descriptions.",
                    "array",
                    required=False,
                    items={"type": "string"},
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        path, err = self._parse_task_path(kwargs)
        if err:
            return err

        update_fields: dict[str, Any] = {
            k: v for k, v in kwargs.items() if k != "task_path"
        }
        if not update_fields:
            return self._err(
                "validation_error",
                "At least one field must be provided to update.",
                code=400,
            )

        logger.debug(
            "Task tool '%s' invoked: task_path=%r fields=%s",
            self.name,
            path,
            sorted(update_fields.keys()),
        )

        def _go() -> dict:
            task_id = self._tm.resolve_path_to_id(path)
            task = self._tm.update_task(task_id, **update_fields)
            logger.debug(
                "Task tool '%s': updated task id=%s path=%r",
                self.name,
                task_id,
                path,
            )
            return {"task": task}

        return self._handle(_go)


# ---------------------------------------------------------------------------
# TasksAddNoteTool
# ---------------------------------------------------------------------------


class TasksAddNoteTool(_TaskTool):
    """Append a timestamped note to a task."""

    NAME = "tasks_add_note"
    DESCRIPTION = "Append a timestamped note to a task. Notes accumulate — they are never replaced."

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    "task_path",
                    (
                        "Dot-separated name path to the task to annotate "
                        "(e.g. 'root_task' or 'root_task.sub_task.leaf_task')."
                    ),
                    "string",
                    required=True,
                ),
                ToolArgument(
                    "note",
                    "The note content to append.",
                    "string",
                    required=True,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        path, err = self._parse_task_path(kwargs)
        if err:
            return err
        note: str = kwargs.get("note", "")
        if not isinstance(note, str) or not note.strip():
            return self._err(
                "validation_error", "'note' must be a non-empty string.", code=400
            )

        logger.debug(
            "Task tool '%s' invoked: task_path=%r note_chars=%d",
            self.name,
            path,
            len(note),
        )

        def _go() -> dict:
            task_id = self._tm.resolve_path_to_id(path)
            task = self._tm.add_note(task_id, note)
            logger.debug(
                "Task tool '%s': added note to task id=%s path=%r",
                self.name,
                task_id,
                path,
            )
            return {"task": task}

        return self._handle(_go)


# ---------------------------------------------------------------------------
# TasksObsoleteNoteTool
# ---------------------------------------------------------------------------


class TasksObsoleteNoteTool(_TaskTool):
    """Mark an active task note obsolete by index."""

    NAME = "tasks_obsolete_note"
    DESCRIPTION = (
        "Mark an active note obsolete by index and remove it from active task "
        "context used by later planning/execution/review rounds."
    )

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    "task_path",
                    (
                        "Dot-separated name path to the task whose note should be "
                        "obsoleted (e.g. 'root_task.sub_task')."
                    ),
                    "string",
                    required=True,
                ),
                ToolArgument(
                    "note_index",
                    "0-based index in the task's active notes list.",
                    "integer",
                    required=True,
                ),
                ToolArgument(
                    "reason",
                    "Optional reason for obsoleting the note.",
                    "string",
                    required=False,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        path, err = self._parse_task_path(kwargs)
        if err:
            return err

        note_index = kwargs.get("note_index")
        if not isinstance(note_index, int):
            return self._err(
                "validation_error", "'note_index' must be an integer.", code=400
            )

        reason = kwargs.get("reason", "")
        if not isinstance(reason, str):
            return self._err("validation_error", "'reason' must be a string.", code=400)

        logger.debug(
            "Task tool '%s' invoked: task_path=%r note_index=%d",
            self.name,
            path,
            note_index,
        )

        def _go() -> dict:
            task_id = self._tm.resolve_path_to_id(path)
            task = self._tm.obsolete_note(task_id, note_index, reason=reason)
            logger.debug(
                "Task tool '%s': obsoleted note for task id=%s path=%r note_index=%d",
                self.name,
                task_id,
                path,
                note_index,
            )
            return {"task": task}

        return self._handle(_go)


# ---------------------------------------------------------------------------
# TasksMarkDoneTool
# ---------------------------------------------------------------------------


class TasksMarkDoneTool(_TaskTool):
    """Mark a task as done, enforcing DoD and subtask completion."""

    NAME = "tasks_mark_done"
    DESCRIPTION = (
        "Mark a task as done. "
        "Requires definition_of_done to contain at least 5 non-whitespace characters "
        "and all subtasks to be done first."
    )

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    "task_path",
                    (
                        "Dot-separated name path to the task to mark as done "
                        "(e.g. 'root_task' or 'root_task.sub_task.leaf_task')."
                    ),
                    "string",
                    required=True,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        path, err = self._parse_task_path(kwargs)
        if err:
            return err

        logger.debug("Task tool '%s' invoked: task_path=%r", self.name, path)

        def _go() -> dict:
            task_id = self._tm.resolve_path_to_id(path)
            task = self._tm.mark_done(task_id)
            logger.debug(
                "Task tool '%s': marked task done id=%s path=%r",
                self.name,
                task_id,
                path,
            )
            return {"task": task}

        return self._handle(_go)
