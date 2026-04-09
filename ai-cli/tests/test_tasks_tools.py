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
        assert "parent_id" in params["properties"]
        assert "parent_id" not in params.get("required", [])

    def test_tasks_get_schema(self, tmp_path):
        tool, _ = _make_tool(TasksGetTool, tmp_path)
        schema = tool.definition().schema()
        assert schema["function"]["name"] == "tasks_get"
        assert "task_id" in schema["function"]["parameters"]["required"]

    def test_tasks_create_schema(self, tmp_path):
        tool, _ = _make_tool(TasksCreateTool, tmp_path)
        schema = tool.definition().schema()
        required = schema["function"]["parameters"]["required"]
        assert "name" in required
        assert "definition_of_done" in required
        props = schema["function"]["parameters"]["properties"]
        assert "description" in props
        assert "parent_id" in props
        priority_prop = props["priority"]
        assert set(priority_prop["enum"]) == VALID_PRIORITIES

    def test_tasks_update_schema(self, tmp_path):
        tool, _ = _make_tool(TasksUpdateTool, tmp_path)
        schema = tool.definition().schema()
        required = schema["function"]["parameters"]["required"]
        assert "task_id" in required
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
        assert "task_id" in required
        assert "note" in required

    def test_tasks_mark_done_schema(self, tmp_path):
        tool, _ = _make_tool(TasksMarkDoneTool, tmp_path)
        schema = tool.definition().schema()
        assert "task_id" in schema["function"]["parameters"]["required"]


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
        parent = tm.create_task("Parent", "Parent DoD")
        tm.create_task("Child", "Child DoD", parent_id=parent["id"])
        result = tool.execute(parent_id=parent["id"])
        assert result["status"] == "success"
        assert len(result["data"]["tasks"]) == 1
        assert result["data"]["tasks"][0]["name"] == "Child"

    def test_list_empty_parent_id_treated_as_root(self, tmp_path):
        tool, tm = _make_tool(TasksListTool, tmp_path)
        tm.create_task("Root", "DoD here")
        result = tool.execute(parent_id="")
        assert result["status"] == "success"
        assert len(result["data"]["tasks"]) == 1

    def test_list_non_string_parent_id_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksListTool, tmp_path)
        result = tool.execute(parent_id=0)
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_delegates_to_task_manager(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksListTool)
        tm.list_tasks.return_value = []
        result = tool.execute(parent_id="p1")
        tm.list_tasks.assert_called_once_with(parent_id="p1")
        assert result == _ok({"tasks": []})

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
        result = tool.execute(task_id=created["id"])
        assert result["status"] == "success"
        assert result["data"]["task"]["id"] == created["id"]

    def test_unknown_task_returns_not_found(self, tmp_path):
        tool, _ = _make_tool(TasksGetTool, tmp_path)
        result = tool.execute(task_id="task_nonexistent")
        assert result["status"] == "error"
        assert result["error"] == "not_found"
        assert result["code"] == 404

    def test_empty_task_id_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksGetTool, tmp_path)
        result = tool.execute(task_id="")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_missing_task_id_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksGetTool, tmp_path)
        result = tool.execute()
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_delegates_to_task_manager(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksGetTool)
        detail = {"id": "task_abc", "name": "T"}
        tm.get_task.return_value = detail
        result = tool.execute(task_id="task_abc")
        tm.get_task.assert_called_once_with("task_abc")
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
            name="Child", definition_of_done="Child DoD", parent_id=parent["id"]
        )
        assert result["status"] == "success"
        child_id = result["data"]["task"]["id"]
        # The child should appear in the parent's subtask list.
        subtasks = tm.list_tasks(parent_id=parent["id"])
        assert any(s["id"] == child_id for s in subtasks)

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

    def test_non_string_parent_id_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksCreateTool, tmp_path)
        result = tool.execute(name="T", definition_of_done="DoD here", parent_id=False)
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_unknown_parent_returns_not_found(self, tmp_path):
        tool, _ = _make_tool(TasksCreateTool, tmp_path)
        result = tool.execute(
            name="Child",
            definition_of_done="DoD here",
            parent_id="task_ghost",
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


# ---------------------------------------------------------------------------
# TasksUpdateTool
# ---------------------------------------------------------------------------


class TestTasksUpdateTool:
    def test_update_name(self, tmp_path):
        tool, tm = _make_tool(TasksUpdateTool, tmp_path)
        task = tm.create_task("Old", "DoD here")
        result = tool.execute(task_id=task["id"], name="New")
        assert result["status"] == "success"
        assert result["data"]["task"]["name"] == "New"

    def test_update_status(self, tmp_path):
        tool, tm = _make_tool(TasksUpdateTool, tmp_path)
        task = tm.create_task("T", "DoD here")
        result = tool.execute(task_id=task["id"], status="in_progress")
        assert result["status"] == "success"
        assert result["data"]["task"]["status"] == "in_progress"

    def test_update_blocked_with_blockers(self, tmp_path):
        tool, tm = _make_tool(TasksUpdateTool, tmp_path)
        task = tm.create_task("T", "DoD here")
        result = tool.execute(
            task_id=task["id"], status="blocked", blockers=["waiting for review"]
        )
        assert result["status"] == "success"
        assert result["data"]["task"]["status"] == "blocked"

    def test_update_done_status_returns_validation_error(self, tmp_path):
        tool, tm = _make_tool(TasksUpdateTool, tmp_path)
        task = tm.create_task("T", "DoD here")
        result = tool.execute(task_id=task["id"], status="done")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_unknown_task_returns_not_found(self, tmp_path):
        tool, _ = _make_tool(TasksUpdateTool, tmp_path)
        result = tool.execute(task_id="task_ghost", name="X")
        assert result["status"] == "error"
        assert result["error"] == "not_found"

    def test_empty_task_id_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksUpdateTool, tmp_path)
        result = tool.execute(task_id="")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_only_task_id_returns_validation_error(self, tmp_path):
        """Calling with only task_id (no fields to update) must return validation_error."""
        tool, tm = _make_tool(TasksUpdateTool, tmp_path)
        task = tm.create_task("T", "DoD here")
        result = tool.execute(task_id=task["id"])
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_delegates_only_update_fields(self, tmp_path):
        """task_id must not be passed as an update field to TaskManager."""
        tool, tm = _make_tool_with_mock_tm(TasksUpdateTool)
        detail = {"id": "task_x", "name": "New"}
        tm.update_task.return_value = detail
        result = tool.execute(task_id="task_x", name="New")
        call_kwargs = tm.update_task.call_args
        assert call_kwargs[0][0] == "task_x"
        assert "task_id" not in call_kwargs[1]
        assert call_kwargs[1]["name"] == "New"
        assert result == _ok({"task": detail})


# ---------------------------------------------------------------------------
# TasksAddNoteTool
# ---------------------------------------------------------------------------


class TestTasksAddNoteTool:
    def test_add_note(self, tmp_path):
        tool, tm = _make_tool(TasksAddNoteTool, tmp_path)
        task = tm.create_task("T", "DoD here")
        result = tool.execute(task_id=task["id"], note="Progress made.")
        assert result["status"] == "success"
        notes = result["data"]["task"]["notes"]
        assert any("Progress made." in n for n in notes)

    def test_notes_accumulate(self, tmp_path):
        tool, tm = _make_tool(TasksAddNoteTool, tmp_path)
        task = tm.create_task("T", "DoD here")
        tool.execute(task_id=task["id"], note="First note.")
        result = tool.execute(task_id=task["id"], note="Second note.")
        notes = result["data"]["task"]["notes"]
        assert len(notes) == 2

    def test_empty_note_returns_validation_error(self, tmp_path):
        tool, tm = _make_tool(TasksAddNoteTool, tmp_path)
        task = tm.create_task("T", "DoD here")
        result = tool.execute(task_id=task["id"], note="")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_empty_task_id_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksAddNoteTool, tmp_path)
        result = tool.execute(task_id="", note="Something.")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_unknown_task_returns_not_found(self, tmp_path):
        tool, _ = _make_tool(TasksAddNoteTool, tmp_path)
        result = tool.execute(task_id="task_ghost", note="A note.")
        assert result["status"] == "error"
        assert result["error"] == "not_found"

    def test_delegates_to_task_manager(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksAddNoteTool)
        detail = {"id": "task_x", "notes": ["[ts] note"]}
        tm.add_note.return_value = detail
        result = tool.execute(task_id="task_x", note="note")
        tm.add_note.assert_called_once_with("task_x", "note")
        assert result == _ok({"task": detail})


# ---------------------------------------------------------------------------
# TasksMarkDoneTool
# ---------------------------------------------------------------------------


class TestTasksMarkDoneTool:
    def test_mark_done_success(self, tmp_path):
        tool, tm = _make_tool(TasksMarkDoneTool, tmp_path)
        task = tm.create_task("T", "DoD here")
        result = tool.execute(task_id=task["id"])
        assert result["status"] == "success"
        assert result["data"]["task"]["status"] == "done"

    def test_subtask_not_done_returns_validation_error(self, tmp_path):
        tool, tm = _make_tool(TasksMarkDoneTool, tmp_path)
        parent = tm.create_task("Parent", "Parent DoD")
        tm.create_task("Child", "Child DoD", parent_id=parent["id"])
        result = tool.execute(task_id=parent["id"])
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_missing_dod_returns_validation_error(self, tmp_path):
        tool, tm = _make_tool(TasksMarkDoneTool, tmp_path)
        task = tm.create_task("T", "DoD here")
        # Corrupt the DoD on disk.
        import json

        data_path = tmp_path / "tasks.json"
        data = json.loads(data_path.read_text())
        data["tasks"][task["id"]]["definition_of_done"] = ""
        data_path.write_text(json.dumps(data))
        result = tool.execute(task_id=task["id"])
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_unknown_task_returns_not_found(self, tmp_path):
        tool, _ = _make_tool(TasksMarkDoneTool, tmp_path)
        result = tool.execute(task_id="task_ghost")
        assert result["status"] == "error"
        assert result["error"] == "not_found"

    def test_empty_task_id_returns_validation_error(self, tmp_path):
        tool, _ = _make_tool(TasksMarkDoneTool, tmp_path)
        result = tool.execute(task_id="")
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    def test_delegates_to_task_manager(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksMarkDoneTool)
        detail = {"id": "task_x", "status": "done"}
        tm.mark_done.return_value = detail
        result = tool.execute(task_id="task_x")
        tm.mark_done.assert_called_once_with("task_x")
        assert result == _ok({"task": detail})

    def test_storage_error_returns_error_response(self, tmp_path):
        tool, tm = _make_tool_with_mock_tm(TasksMarkDoneTool)
        tm.mark_done.side_effect = TaskStorageError("I/O error")
        result = tool.execute(task_id="task_x")
        assert result["status"] == "error"
        assert result["error"] == "storage_error"
        assert result["code"] == 500
