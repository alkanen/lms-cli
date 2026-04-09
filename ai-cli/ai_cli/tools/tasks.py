"""
Task tools — read and mutate the persistent task tree via TaskManager.

Six tools are provided:

  tasks_list        List task summaries under a parent (or root).
  tasks_get         Retrieve full details of a single task.
  tasks_create      Create a new (sub)task.
  tasks_update      Update fields of an existing task.
  tasks_add_note    Append a timestamped note to a task.
  tasks_mark_done   Mark a task done (validates subtasks and DoD).

All six share a :class:`~ai_cli.core.task_manager.TaskManager` instance
injected at construction time.  They use ``REGISTER_VIA_INSTANCE = True``
because their constructors are non-standard; they are wired into registries
in PR 3 via :meth:`~ai_cli.core.tool_registry.ToolRegistry.register_instance`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_cli.core.task_manager import (
    UPDATABLE_STATUSES,
    VALID_PRIORITIES,
    TaskManager,
    TaskNotFoundError,
    TaskStorageError,
    TaskValidationError,
)
from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

if TYPE_CHECKING:
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.workspace import Workspace

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
            return self._ok(fn(*args, **kwargs))
        except TaskNotFoundError as exc:
            return self._err(
                "not_found", exc.args[0] if exc.args else str(exc), code=404
            )
        except TaskValidationError as exc:
            return self._err("validation_error", str(exc), code=400)
        except TaskStorageError as exc:
            return self._err("storage_error", str(exc), code=500)

    def _parse_task_id(self, kwargs: dict[str, Any]) -> tuple[str, dict | None]:
        """Return ``(stripped_task_id, None)`` or ``("", error_dict)``."""
        raw = kwargs.get("task_id", "")
        if not isinstance(raw, str) or not raw.strip():
            return "", self._err(
                "validation_error", "'task_id' must be a non-empty string.", code=400
            )
        return raw.strip(), None

    def _parse_parent_id(
        self, kwargs: dict[str, Any]
    ) -> tuple[str | None, dict | None]:
        """Return ``(parent_id_or_None, None)`` or ``(None, error_dict)``."""
        raw = kwargs.get("parent_id")
        if raw is None or raw == "":
            return None, None
        if not isinstance(raw, str):
            return None, self._err(
                "validation_error", "'parent_id' must be a string.", code=400
            )
        return raw.strip() or None, None


# ---------------------------------------------------------------------------
# TasksListTool
# ---------------------------------------------------------------------------


class TasksListTool(_TaskTool):
    """List task summaries for direct children of *parent_id* (or root)."""

    NAME = "tasks_list"
    DESCRIPTION = (
        "List tasks as lightweight summaries. "
        "Pass parent_id to list subtasks; omit it for root-level tasks."
    )

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    "parent_id",
                    "Parent task ID whose subtasks to list, or omit for root tasks.",
                    "string",
                    required=False,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        parent_id, err = self._parse_parent_id(kwargs)
        if err:
            return err
        return self._handle(lambda: {"tasks": self._tm.list_tasks(parent_id=parent_id)})


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
                    "task_id",
                    "The ID of the task to retrieve.",
                    "string",
                    required=True,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        task_id, err = self._parse_task_id(kwargs)
        if err:
            return err
        return self._handle(lambda: {"task": self._tm.get_task(task_id)})


# ---------------------------------------------------------------------------
# TasksCreateTool
# ---------------------------------------------------------------------------


class TasksCreateTool(_TaskTool):
    """Create a new task (optionally as a subtask of an existing task)."""

    NAME = "tasks_create"
    DESCRIPTION = (
        "Create a new task. Provide parent_id to attach it as a subtask. "
        "definition_of_done is required and must be at least 5 non-whitespace characters."
    )

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    "name",
                    "Short task name.",
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
                    "parent_id",
                    "Parent task ID to attach this task as a subtask (omit for root).",
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

        parent_id, err = self._parse_parent_id(kwargs)
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

        return self._handle(
            lambda: {
                "task": self._tm.create_task(
                    name=name,
                    definition_of_done=dod,
                    description=description,
                    parent_id=parent_id,
                    priority=priority,
                )
            }
        )


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
                    "task_id",
                    "The ID of the task to update.",
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
        task_id, err = self._parse_task_id(kwargs)
        if err:
            return err

        update_fields: dict[str, Any] = {
            k: v for k, v in kwargs.items() if k != "task_id"
        }
        if not update_fields:
            return self._err(
                "validation_error",
                "At least one field must be provided to update.",
                code=400,
            )

        return self._handle(
            lambda: {"task": self._tm.update_task(task_id, **update_fields)}
        )


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
                    "task_id",
                    "The ID of the task to annotate.",
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
        task_id, err = self._parse_task_id(kwargs)
        if err:
            return err
        note: str = kwargs.get("note", "")
        if not isinstance(note, str) or not note.strip():
            return self._err(
                "validation_error", "'note' must be a non-empty string.", code=400
            )
        return self._handle(lambda: {"task": self._tm.add_note(task_id, note)})


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
                    "task_id",
                    "The ID of the task to mark as done.",
                    "string",
                    required=True,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        task_id, err = self._parse_task_id(kwargs)
        if err:
            return err
        return self._handle(lambda: {"task": self._tm.mark_done(task_id)})
