"""Tests for ai_cli/tools/tasks.py — task tool wrappers around TaskManager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_cli.core.task_manager import (
    UPDATABLE_STATUSES,
    VALID_PRIORITIES,
    TaskManager,
    TaskStorageError,
)
from ai_cli.tools.tasks import (
    TasksAddNoteTool,
    TasksCreateTool,
    TasksGetTool,
    TasksListTool,
    TasksMarkDoneTool,
    TasksObsoleteNoteTool,
    TasksUpdateTool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tm(tmp_path: Path) -> TaskManager:
    return TaskManager(tmp_path)


def _make_tool(cls, tmp_path: Path):
    """Instantiate *cls* with a real TaskManager and mock workspace/pm."""
    workspace = MagicMock()
    pm = MagicMock()
    tm = _make_tm(tmp_path)
    return cls(tm, workspace, pm), tm


def _make_tool_with_mock_tm(cls):
    """Instantiate *cls* with a MagicMock TaskManager for delegation tests."""
    workspace = MagicMock()
    pm = MagicMock()
    tm = MagicMock(spec=TaskManager)
    return cls(tm, workspace, pm), tm


def _ok(data: dict) -> dict:
    return {"status": "success", "data": data}


def _err(error: str, message: str, code: int = 400) -> dict:
    return {"status": "error", "error": error, "message": message, "code": code}


# ---------------------------------------------------------------------------
# REGISTER_VIA_INSTANCE / class-level attributes
# ---------------------------------------------------------------------------


class TestClassAttributes:
    @pytest.mark.parametrize(
        "cls",
        [
            TasksListTool,
            TasksGetTool,
            TasksCreateTool,
            TasksUpdateTool,
            TasksAddNoteTool,
            TasksObsoleteNoteTool,
            TasksMarkDoneTool,
        ],
    )
    def test_register_via_instance(self, cls):
        assert cls.REGISTER_VIA_INSTANCE is True

    @pytest.mark.parametrize(
        "cls",
        [
            TasksListTool,
            TasksGetTool,
            TasksCreateTool,
            TasksUpdateTool,
            TasksAddNoteTool,
            TasksObsoleteNoteTool,
            TasksMarkDoneTool,
        ],
    )
    def test_permission_not_required(self, cls):
        assert cls.PERMISSION_REQUIRED is False

    @pytest.mark.parametrize(
        "cls, name",
        [
            (TasksListTool, "tasks_list"),
            (TasksGetTool, "tasks_get"),
            (TasksCreateTool, "tasks_create"),
            (TasksUpdateTool, "tasks_update"),
            (TasksAddNoteTool, "tasks_add_note"),
            (TasksObsoleteNoteTool, "tasks_obsolete_note"),
            (TasksMarkDoneTool, "tasks_mark_done"),
        ],
    )
    def test_tool_names(self, cls, name):
        assert name == cls.NAME


# ---------------------------------------------------------------------------
# Definition schemas
# ---------------------------------------------------------------------------


class TestDefinitions:
    def test_tasks_list_schema(self, tmp_path):
        tool, _ = _make_tool(TasksListTool, tmp_path)
        schema = tool.definition().schema()
        assert schema["function"]["name"] == "tasks_list"
        params = schema["function"]["parameters"]
        assert "parent_path" in params["properties"]
        assert "parent_path" not in params.get("required", [])

    def test_tasks_get_schema(self, tmp_path):
        tool, _ = _make_tool(TasksGetTool, tmp_path)
        schema = tool.definition().schema()
        assert schema["function"]["name"] == "tasks_get"
        assert "task_path" in schema["function"]["parameters"]["required"]

    def test_tasks_create_schema(self, tmp_path):
        tool, _ = _make_tool(TasksCreateTool, tmp_path)
        schema = tool.definition().schema()
        required = schema["function"]["parameters"]["required"]
        assert "name" in required
        assert "definition_of_done" in required
        props = schema["function"]["parameters"]["properties"]
        assert "description" in props
        assert "parent_path" in props
        priority_prop = props["priority"]
        assert set(priority_prop["enum"]) == VALID_PRIORITIES

    def test_tasks_update_schema(self, tmp_path):
        tool, _ = _make_tool(TasksUpdateTool, tmp_path)
        schema = tool.definition().schema()
        required = schema["function"]["parameters"]["required"]
        assert "task_path" in required
        props = schema["function"]["parameters"]["properties"]
        status_enum = set(props["status"]["enum"])
        assert status_enum == UPDATABLE_STATUSES
        assert "done" not in status_enum
        assert "blockers" in props
        assert props["blockers"]["type"] == "array"

    def test_tasks_add_note_schema(self, tmp_path):
        tool, _ = _make_tool(TasksAddNoteTool, tmp_path)
        schema = tool.definition().schema()
        required = schema["function"]["parameters"]["required"]
        assert "task_path" in required
        assert "note" in required

    def test_tasks_mark_done_schema(self, tmp_path):
        tool, _ = _make_tool(TasksMarkDoneTool, tmp_path)
        schema = tool.definition().schema()
        assert "task_path" in schema["function"]["parameters"]["required"]

    def test_tasks_obsolete_note_schema(self, tmp_path):
        tool, _ = _make_tool(TasksObsoleteNoteTool, tmp_path)
        schema = tool.definition().schema()
        required = schema["function"]["parameters"]["required"]
        assert "task_path" in required
        assert "note_index" in required


# ---------------------------------------------------------------------------
# TasksListTool
# ---------------------------------------------------------------------------


class TestTasksListTool:
    def test_list_root_tasks(self, tmp_path):
        tool, tm = _make_tool(TasksListTool, tmp_path)
        tm.create_task("T1", "DoD here")
        result = tool.execute()
        assert result["status"] == "success"
        tasks = result["data"]["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["name"] == "T1"

    def test_list_subtasks(self, tmp_path):
        tool, tm = _make_tool(TasksListTool, tmp_path)
        tm.create_task("Parent", "Parent DoD")
        tm.create_task("Child", "Child DoD", parent_id=tm.find_by_path("Parent")["id"])
        result = tool.execute(parent_path="Parent")
        assert result["status"] == "success"
        assert len(result["data"]["tasks"]) == 1
        assert result["data"]["tasks"][0]["name"] == "Child"

    def test_list_empty_parent_path_treated_as_root(self, tmp_path):
        tool, tm = _make_tool(TasksListTool, tmp_path)
        tm.create_task("Root", "DoD here")
        result = tool.execute(parent_path="")
        assert result["status"] == "success"
        assert len(result["data"]["tasks"]) == 1

    def test_list_non_string_parent_path_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksListTool, tmp_path)
        result = tool.execute(parent_path=0)
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_list_unknown_parent_path_returns_not_found(self, tmp_path):
        tool, _ = _make_tool(TasksListTool, tmp_path)
        result = tool.execute(parent_path="nonexistent_task")
        assert result["status"] == "error"
        assert result["error"] == "not_found"

    def test_delegates_to_task_manager(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksListTool)
        tm.resolve_path_to_id.return_value = "p1"
        tm.list_tasks.return_value = []
        result = tool.execute(parent_path="some_task")
        tm.resolve_path_to_id.assert_called_once_with("some_task")
        tm.list_tasks.assert_called_once_with(parent_id="p1")
        assert result == _ok({"tasks": []})

    def test_delegates_root_list_without_resolve(self, tmp_path):
        """Omitting parent_path lists root tasks without calling resolve_path_to_id."""
        tool, tm = _make_tool_with_mock_tm(TasksListTool)
        tm.list_tasks.return_value = []
        result = tool.execute()
        tm.resolve_path_to_id.assert_not_called()
        tm.list_tasks.assert_called_once_with(parent_id=None)
        assert result == _ok({"tasks": []})

    def test_parent_path_strips_trailing_dot(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksListTool)
        tm.resolve_path_to_id.return_value = "p1"
        tm.list_tasks.return_value = []
        result = tool.execute(parent_path="some_task.")
        tm.resolve_path_to_id.assert_called_once_with("some_task")
        tm.list_tasks.assert_called_once_with(parent_id="p1")
        assert result == _ok({"tasks": []})

    def test_parent_path_dot_only_returns_validation_error(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksListTool)
        result = tool.execute(parent_path=".")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"
        tm.resolve_path_to_id.assert_not_called()
        tm.list_tasks.assert_not_called()

    def test_storage_error_returns_error_response(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksListTool)
        tm.list_tasks.side_effect = TaskStorageError("disk failure")
        result = tool.execute()
        assert result["status"] == "error"
        assert result["error"] == "storage_error"
        assert result["code"] == 500


# ---------------------------------------------------------------------------
# TasksGetTool
# ---------------------------------------------------------------------------


class TestTasksGetTool:
    def test_get_existing_task(self, tmp_path):
        tool, tm = _make_tool(TasksGetTool, tmp_path)
        created = tm.create_task("T1", "DoD here")
        result = tool.execute(task_path="T1")
        assert result["status"] == "success"
        assert result["data"]["task"]["id"] == created["id"]

    def test_get_nested_task(self, tmp_path):
        tool, tm = _make_tool(TasksGetTool, tmp_path)
        parent = tm.create_task("Parent", "Parent DoD")
        child = tm.create_task("Child", "Child DoD", parent_id=parent["id"])
        result = tool.execute(task_path="Parent.Child")
        assert result["status"] == "success"
        assert result["data"]["task"]["id"] == child["id"]

    def test_unknown_task_returns_not_found(self, tmp_path):
        tool, _ = _make_tool(TasksGetTool, tmp_path)
        result = tool.execute(task_path="nonexistent_task")
        assert result["status"] == "error"
        assert result["error"] == "not_found"
        assert result["code"] == 404

    def test_empty_task_path_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksGetTool, tmp_path)
        result = tool.execute(task_path="")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_missing_task_path_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksGetTool, tmp_path)
        result = tool.execute()
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_delegates_to_task_manager(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksGetTool)
        detail = {"id": "task_abc", "name": "T"}
        tm.find_by_path.return_value = detail
        result = tool.execute(task_path="T")
        tm.find_by_path.assert_called_once_with("T")
        assert result == _ok({"task": detail})

    def test_task_path_strips_trailing_dot(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksGetTool)
        detail = {"id": "task_abc", "name": "T"}
        tm.find_by_path.return_value = detail
        result = tool.execute(task_path="T.")
        tm.find_by_path.assert_called_once_with("T")
        assert result == _ok({"task": detail})


# ---------------------------------------------------------------------------
# TasksCreateTool
# ---------------------------------------------------------------------------


class TestTasksCreateTool:
    def test_create_minimal(self, tmp_path):
        tool, _ = _make_tool(TasksCreateTool, tmp_path)
        result = tool.execute(name="T1", definition_of_done="DoD here")
        assert result["status"] == "success"
        task = result["data"]["task"]
        assert task["name"] == "T1"
        assert task["status"] == "not_started"
        assert task["priority"] == "medium"

    def test_create_with_all_fields(self, tmp_path):
        tool, _ = _make_tool(TasksCreateTool, tmp_path)
        result = tool.execute(
            name="T",
            definition_of_done="Long enough DoD",
            description="Desc",
            priority="high",
        )
        assert result["status"] == "success"
        assert result["data"]["task"]["priority"] == "high"

    def test_create_subtask(self, tmp_path):
        tool, tm = _make_tool(TasksCreateTool, tmp_path)
        parent = tm.create_task("Parent", "Parent DoD")
        result = tool.execute(
            name="Child", definition_of_done="Child DoD", parent_path="Parent"
        )
        assert result["status"] == "success"
        child_id = result["data"]["task"]["id"]
        # The child should appear in the parent's subtask list.
        subtasks = tm.list_tasks(parent_id=parent["id"])
        assert any(s["id"] == child_id for s in subtasks)

    def test_create_deeply_nested_subtask(self, tmp_path):
        tool, tm = _make_tool(TasksCreateTool, tmp_path)
        root = tm.create_task("Root", "Root DoD")
        child = tm.create_task("Child", "Child DoD", parent_id=root["id"])
        result = tool.execute(
            name="Leaf", definition_of_done="Leaf DoD", parent_path="Root.Child"
        )
        assert result["status"] == "success"
        subtasks = tm.list_tasks(parent_id=child["id"])
        assert any(s["name"] == "Leaf" for s in subtasks)

    def test_short_dod_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksCreateTool, tmp_path)
        result = tool.execute(name="T", definition_of_done="hi")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_empty_name_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksCreateTool, tmp_path)
        result = tool.execute(name="", definition_of_done="DoD here")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_missing_name_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksCreateTool, tmp_path)
        result = tool.execute(definition_of_done="DoD here")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_non_string_parent_path_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksCreateTool, tmp_path)
        result = tool.execute(
            name="T", definition_of_done="DoD here", parent_path=False
        )
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_unknown_parent_returns_not_found(self, tmp_path):
        tool, _ = _make_tool(TasksCreateTool, tmp_path)
        result = tool.execute(
            name="Child",
            definition_of_done="DoD here",
            parent_path="ghost_parent",
        )
        assert result["status"] == "error"
        assert result["error"] == "not_found"

    def test_delegates_to_task_manager(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksCreateTool)
        detail = {"id": "task_x", "name": "T"}
        tm.create_task.return_value = detail
        result = tool.execute(name="T", definition_of_done="DoD here")
        tm.create_task.assert_called_once_with(
            name="T",
            definition_of_done="DoD here",
            description="",
            parent_id=None,
            priority="medium",
        )
        assert result == _ok({"task": detail})

    def test_delegates_with_parent_path_resolved(self, tmp_path):
        """parent_path is resolved to parent_id before calling create_task."""
        tool, tm = _make_tool_with_mock_tm(TasksCreateTool)
        tm.resolve_path_to_id.return_value = "task_parent"
        detail = {"id": "task_x", "name": "Child"}
        tm.create_task.return_value = detail
        result = tool.execute(
            name="Child", definition_of_done="DoD here", parent_path="Parent"
        )
        tm.resolve_path_to_id.assert_called_once_with("Parent")
        tm.create_task.assert_called_once_with(
            name="Child",
            definition_of_done="DoD here",
            description="",
            parent_id="task_parent",
            priority="medium",
        )
        assert result == _ok({"task": detail})


# ---------------------------------------------------------------------------
# TasksUpdateTool
# ---------------------------------------------------------------------------


class TestTasksUpdateTool:
    def test_update_name(self, tmp_path):
        tool, tm = _make_tool(TasksUpdateTool, tmp_path)
        tm.create_task("Old", "DoD here")
        result = tool.execute(task_path="Old", name="New")
        assert result["status"] == "success"
        assert result["data"]["task"]["name"] == "New"

    def test_update_status(self, tmp_path):
        tool, tm = _make_tool(TasksUpdateTool, tmp_path)
        tm.create_task("T", "DoD here")
        result = tool.execute(task_path="T", status="in_progress")
        assert result["status"] == "success"
        assert result["data"]["task"]["status"] == "in_progress"

    def test_update_nested_task(self, tmp_path):
        tool, tm = _make_tool(TasksUpdateTool, tmp_path)
        root = tm.create_task("Root", "Root DoD")
        tm.create_task("Child", "Child DoD", parent_id=root["id"])
        result = tool.execute(task_path="Root.Child", status="in_progress")
        assert result["status"] == "success"
        assert result["data"]["task"]["status"] == "in_progress"

    def test_update_blocked_with_blockers(self, tmp_path):
        tool, tm = _make_tool(TasksUpdateTool, tmp_path)
        tm.create_task("T", "DoD here")
        result = tool.execute(
            task_path="T", status="blocked", blockers=["waiting for review"]
        )
        assert result["status"] == "success"
        assert result["data"]["task"]["status"] == "blocked"

    def test_update_done_status_returns_validation_error(self, tmp_path):
        tool, tm = _make_tool(TasksUpdateTool, tmp_path)
        tm.create_task("T", "DoD here")
        result = tool.execute(task_path="T", status="done")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_unknown_task_returns_not_found(self, tmp_path):
        tool, _ = _make_tool(TasksUpdateTool, tmp_path)
        result = tool.execute(task_path="ghost_task", name="X")
        assert result["status"] == "error"
        assert result["error"] == "not_found"

    def test_empty_task_path_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksUpdateTool, tmp_path)
        result = tool.execute(task_path="")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_only_task_path_returns_validation_error(self, tmp_path):
        """Calling with only task_path (no fields to update) must return validation_error."""
        tool, tm = _make_tool(TasksUpdateTool, tmp_path)
        tm.create_task("T", "DoD here")
        result = tool.execute(task_path="T")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_delegates_only_update_fields(self, tmp_path):
        """task_path must not be passed as an update field to TaskManager."""
        tool, tm = _make_tool_with_mock_tm(TasksUpdateTool)
        tm.resolve_path_to_id.return_value = "task_x"
        detail = {"id": "task_x", "name": "New"}
        tm.update_task.return_value = detail
        result = tool.execute(task_path="T", name="New")
        tm.resolve_path_to_id.assert_called_once_with("T")
        call_kwargs = tm.update_task.call_args
        assert call_kwargs[0][0] == "task_x"
        assert "task_path" not in call_kwargs[1]
        assert call_kwargs[1]["name"] == "New"
        assert result == _ok({"task": detail})


# ---------------------------------------------------------------------------
# TasksAddNoteTool
# ---------------------------------------------------------------------------


class TestTasksAddNoteTool:
    def test_add_note(self, tmp_path):
        tool, tm = _make_tool(TasksAddNoteTool, tmp_path)
        tm.create_task("T", "DoD here")
        result = tool.execute(task_path="T", note="Progress made.")
        assert result["status"] == "success"
        notes = result["data"]["task"]["notes"]
        assert any("Progress made." in n for n in notes)

    def test_notes_accumulate(self, tmp_path):
        tool, tm = _make_tool(TasksAddNoteTool, tmp_path)
        tm.create_task("T", "DoD here")
        tool.execute(task_path="T", note="First note.")
        result = tool.execute(task_path="T", note="Second note.")
        notes = result["data"]["task"]["notes"]
        assert len(notes) == 2

    def test_add_note_nested_task(self, tmp_path):
        tool, tm = _make_tool(TasksAddNoteTool, tmp_path)
        root = tm.create_task("Root", "Root DoD")
        tm.create_task("Child", "Child DoD", parent_id=root["id"])
        result = tool.execute(task_path="Root.Child", note="Working on it.")
        assert result["status"] == "success"
        assert any("Working on it." in n for n in result["data"]["task"]["notes"])

    def test_empty_note_returns_validation_error(self, tmp_path):
        tool, tm = _make_tool(TasksAddNoteTool, tmp_path)
        tm.create_task("T", "DoD here")
        result = tool.execute(task_path="T", note="")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_empty_task_path_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksAddNoteTool, tmp_path)
        result = tool.execute(task_path="", note="Something.")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_unknown_task_returns_not_found(self, tmp_path):
        tool, _ = _make_tool(TasksAddNoteTool, tmp_path)
        result = tool.execute(task_path="ghost_task", note="A note.")
        assert result["status"] == "error"
        assert result["error"] == "not_found"

    def test_delegates_to_task_manager(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksAddNoteTool)
        tm.resolve_path_to_id.return_value = "task_x"
        detail = {"id": "task_x", "notes": ["[ts] note"]}
        tm.add_note.return_value = detail
        result = tool.execute(task_path="T", note="note")
        tm.resolve_path_to_id.assert_called_once_with("T")
        tm.add_note.assert_called_once_with("task_x", "note")
        assert result == _ok({"task": detail})


# ---------------------------------------------------------------------------
# TasksObsoleteNoteTool
# ---------------------------------------------------------------------------


class TestTasksObsoleteNoteTool:
    def test_obsolete_note(self, tmp_path):
        workspace = MagicMock()
        pm = MagicMock()
        tm = _make_tm(tmp_path)
        add_tool = TasksAddNoteTool(tm, workspace, pm)
        tool = TasksObsoleteNoteTool(tm, workspace, pm)
        tm.create_task("T", "DoD here")
        add_tool.execute(task_path="T", note="Stale")
        add_tool.execute(task_path="T", note="Current")

        result = tool.execute(task_path="T", note_index=0, reason="resolved")

        assert result["status"] == "success"
        notes = result["data"]["task"]["notes"]
        assert len(notes) == 1
        assert "Current" in notes[0]

    def test_obsolete_note_unknown_task_returns_not_found(self, tmp_path):
        tool, _ = _make_tool(TasksObsoleteNoteTool, tmp_path)
        result = tool.execute(task_path="ghost", note_index=0)
        assert result["status"] == "error"
        assert result["error"] == "not_found"

    def test_obsolete_note_requires_integer_index(self, tmp_path):
        tool, tm = _make_tool(TasksObsoleteNoteTool, tmp_path)
        tm.create_task("T", "DoD here")
        result = tool.execute(task_path="T", note_index="0")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_delegates_to_task_manager(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksObsoleteNoteTool)
        tm.resolve_path_to_id.return_value = "task_x"
        detail = {"id": "task_x", "notes": []}
        tm.obsolete_note.return_value = detail
        result = tool.execute(task_path="T", note_index=1, reason="stale")
        tm.resolve_path_to_id.assert_called_once_with("T")
        tm.obsolete_note.assert_called_once_with("task_x", 1, reason="stale")
        assert result == _ok({"task": detail})


# ---------------------------------------------------------------------------
# TasksMarkDoneTool
# ---------------------------------------------------------------------------


class TestTasksMarkDoneTool:
    def test_mark_done_success(self, tmp_path):
        tool, tm = _make_tool(TasksMarkDoneTool, tmp_path)
        tm.create_task("T", "DoD here")
        result = tool.execute(task_path="T")
        assert result["status"] == "success"
        assert result["data"]["task"]["status"] == "done"

    def test_mark_done_nested_task(self, tmp_path):
        tool, tm = _make_tool(TasksMarkDoneTool, tmp_path)
        root = tm.create_task("Root", "Root DoD")
        tm.create_task("Child", "Child DoD", parent_id=root["id"])
        # Mark child done first (required before root can be marked done)
        result = tool.execute(task_path="Root.Child")
        assert result["status"] == "success"
        assert result["data"]["task"]["status"] == "done"

    def test_subtask_not_done_returns_validation_error(self, tmp_path):
        tool, tm = _make_tool(TasksMarkDoneTool, tmp_path)
        parent = tm.create_task("Parent", "Parent DoD")
        tm.create_task("Child", "Child DoD", parent_id=parent["id"])
        result = tool.execute(task_path="Parent")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_missing_dod_returns_validation_error(self, tmp_path):
        tool, tm = _make_tool(TasksMarkDoneTool, tmp_path)
        task = tm.create_task("T", "DoD here")
        # Corrupt the DoD directly in the in-memory cache (TaskManager no longer
        # re-reads the file on every call; write to cache to simulate corruption).
        tm._cache["tasks"][task["id"]]["definition_of_done"] = ""
        result = tool.execute(task_path="T")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_unknown_task_returns_not_found(self, tmp_path):
        tool, _ = _make_tool(TasksMarkDoneTool, tmp_path)
        result = tool.execute(task_path="ghost_task")
        assert result["status"] == "error"
        assert result["error"] == "not_found"

    def test_empty_task_path_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksMarkDoneTool, tmp_path)
        result = tool.execute(task_path="")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_delegates_to_task_manager(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksMarkDoneTool)
        tm.resolve_path_to_id.return_value = "task_x"
        detail = {"id": "task_x", "status": "done"}
        tm.mark_done.return_value = detail
        result = tool.execute(task_path="T")
        tm.resolve_path_to_id.assert_called_once_with("T")
        tm.mark_done.assert_called_once_with("task_x")
        assert result == _ok({"task": detail})

    def test_storage_error_returns_error_response(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksMarkDoneTool)
        tm.resolve_path_to_id.return_value = "task_x"
        tm.mark_done.side_effect = TaskStorageError("I/O error")
        result = tool.execute(task_path="T")
        assert result["status"] == "error"
        assert result["error"] == "storage_error"
        assert result["code"] == 500
