"""Tests for ai_cli.core.task_manager."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest

from ai_cli.core.task_manager import (
    TaskManager,
    TaskNotFoundError,
    TaskStorageError,
    TaskValidationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tm(tmp_path: Path) -> TaskManager:
    return TaskManager(tmp_path)


def _tasks_json(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "tasks.json").read_text())


# ---------------------------------------------------------------------------
# File lifecycle
# ---------------------------------------------------------------------------


class TestFileLifecycle:
    def test_file_not_created_on_init(self, tmp_path):
        _make_tm(tmp_path)
        assert not (tmp_path / "tasks.json").exists()

    def test_file_not_created_on_read(self, tmp_path):
        tm = _make_tm(tmp_path)
        tm.list_tasks()
        tm.find("not_started")
        tm.find_incomplete()
        tm.all_tasks()
        assert not (tmp_path / "tasks.json").exists()

    def test_file_created_on_first_write(self, tmp_path):
        tm = _make_tm(tmp_path)
        tm.create_task(name="T", definition_of_done="DoD here")
        assert (tmp_path / "tasks.json").exists()

    def test_file_created_on_set_goal(self, tmp_path):
        tm = _make_tm(tmp_path)
        tm.set_goal("My goal")
        assert (tmp_path / "tasks.json").exists()

    def test_clear_removes_file(self, tmp_path):
        tm = _make_tm(tmp_path)
        tm.create_task(name="T", definition_of_done="DoD here")
        tm.clear()
        assert not (tmp_path / "tasks.json").exists()

    def test_clear_on_missing_file_is_silent(self, tmp_path):
        tm = _make_tm(tmp_path)
        tm.clear()  # no error

    def test_load_invalid_json_raises_storage_error(self, tmp_path):
        (tmp_path / "tasks.json").write_text("not valid json")
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskStorageError, match="invalid JSON"):
            tm.list_tasks()

    def test_load_unexpected_structure_raises_storage_error(self, tmp_path):
        (tmp_path / "tasks.json").write_text(
            '{"tasks": []}'
        )  # tasks is a list, not dict
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskStorageError, match="unexpected structure"):
            tm.list_tasks()

    def test_load_unexpected_structure_quarantines_file(self, tmp_path):
        (tmp_path / "tasks.json").write_text('{"tasks": []}')
        tm = _make_tm(tmp_path)
        with contextlib.suppress(TaskStorageError):
            tm.list_tasks()
        assert not (tmp_path / "tasks.json").exists()
        # Quarantine file uses a timestamped name matching tasks.*.corrupt.
        assert list(tmp_path.glob("tasks.*.corrupt"))


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------


class TestGoal:
    def test_set_and_get_goal(self, tmp_path):
        tm = _make_tm(tmp_path)
        tm.set_goal("Implement X")
        assert tm.get_goal() == "Implement X"

    def test_get_goal_returns_none_when_absent(self, tmp_path):
        tm = _make_tm(tmp_path)
        assert tm.get_goal() is None

    def test_set_goal_is_idempotent(self, tmp_path):
        tm = _make_tm(tmp_path)
        tm.set_goal("First")
        tm.set_goal("Second")
        assert tm.get_goal() == "Second"

    def test_set_goal_non_string_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="goal"):
            tm.set_goal(42)  # type: ignore[arg-type]

    def test_set_goal_empty_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="goal"):
            tm.set_goal("   ")


# ---------------------------------------------------------------------------
# create_task
# ---------------------------------------------------------------------------


class TestCreateTask:
    def test_basic_create_returns_task_detail(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="Write_tests", definition_of_done="All tests pass")
        assert detail["name"] == "Write_tests"
        assert detail["definition_of_done"] == "All tests pass"
        assert detail["status"] == "not_started"
        assert detail["priority"] == "medium"
        assert detail["subtasks"] == []

    def test_id_format(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        assert detail["id"].startswith("task_")
        assert len(detail["id"]) == 11  # "task_" + 6 chars

    def test_default_fields(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        assert detail["description"] == ""
        assert detail["next_action"] == ""
        assert detail["blockers"] == []
        assert detail["notes"] == []

    def test_custom_priority(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(
            name="T", definition_of_done="DoD here", priority="high"
        )
        assert detail["priority"] == "high"

    def test_empty_name_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="name"):
            tm.create_task(name="", definition_of_done="DoD here")

    def test_dod_too_short_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="definition_of_done"):
            tm.create_task(name="T", definition_of_done="Hi")

    def test_dod_exactly_min_length_accepted(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="12345")
        assert detail["definition_of_done"] == "12345"

    def test_dod_whitespace_only_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="definition_of_done"):
            tm.create_task(name="T", definition_of_done="     ")

    def test_whitespace_only_name_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="name"):
            tm.create_task(name="   ", definition_of_done="DoD here")

    def test_invalid_priority_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="priority"):
            tm.create_task(name="T", definition_of_done="DoD here", priority="urgent")

    def test_non_string_priority_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="priority"):
            tm.create_task(name="T", definition_of_done="DoD here", priority=1)  # type: ignore[arg-type]

    def test_non_string_description_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="description"):
            tm.create_task(name="T", definition_of_done="DoD here", description=42)  # type: ignore[arg-type]

    def test_subtask_creation(self, tmp_path):
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="Parent done")
        child = tm.create_task(
            name="Child", definition_of_done="Child done", parent_id=parent["id"]
        )
        assert child["id"] != parent["id"]
        # Parent should now report has_subtasks
        parent_detail = tm.get_task(parent["id"])
        assert len(parent_detail["subtasks"]) == 1
        assert parent_detail["subtasks"][0]["id"] == child["id"]

    def test_unknown_parent_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskNotFoundError):
            tm.create_task(
                name="T", definition_of_done="DoD here", parent_id="task_xxxxxx"
            )

    def test_non_string_parent_id_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="parent_id"):
            tm.create_task(name="T", definition_of_done="DoD here", parent_id=123)  # type: ignore[arg-type]

    def test_persists_to_disk(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        on_disk = _tasks_json(tmp_path)
        assert detail["id"] in on_disk["tasks"]


# ---------------------------------------------------------------------------
# get_task / list_tasks
# ---------------------------------------------------------------------------


class TestGetAndList:
    def test_get_task_returns_full_detail(self, tmp_path):
        tm = _make_tm(tmp_path)
        created = tm.create_task(
            name="T", definition_of_done="DoD here", description="Desc"
        )
        fetched = tm.get_task(created["id"])
        assert fetched["id"] == created["id"]
        assert fetched["description"] == "Desc"

    def test_get_unknown_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskNotFoundError):
            tm.get_task("task_xxxxxx")

    def test_get_task_raises_on_missing_subtask_ref(self, tmp_path):
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="Parent done")
        data = _tasks_json(tmp_path)
        data["tasks"][parent["id"]]["subtask_ids"].append("task_ghost1")
        (tmp_path / "tasks.json").write_text(json.dumps(data))
        with pytest.raises(TaskValidationError, match="missing subtask"):
            tm.get_task(parent["id"])

    def test_list_root_tasks(self, tmp_path):
        tm = _make_tm(tmp_path)
        a = tm.create_task(name="A", definition_of_done="DoD A")
        b = tm.create_task(name="B", definition_of_done="DoD B")
        summaries = tm.list_tasks()
        ids = [s["id"] for s in summaries]
        assert a["id"] in ids
        assert b["id"] in ids

    def test_list_subtasks(self, tmp_path):
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="Parent done")
        child = tm.create_task(
            name="Child", definition_of_done="Child done", parent_id=parent["id"]
        )
        root_summaries = tm.list_tasks()
        root_ids = [s["id"] for s in root_summaries]
        assert parent["id"] in root_ids
        assert child["id"] not in root_ids

        child_summaries = tm.list_tasks(parent_id=parent["id"])
        assert len(child_summaries) == 1
        assert child_summaries[0]["id"] == child["id"]

    def test_list_empty_returns_empty_list(self, tmp_path):
        tm = _make_tm(tmp_path)
        assert tm.list_tasks() == []

    def test_summary_has_subtasks_flag(self, tmp_path):
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="DoD here")
        tm.create_task(
            name="Child", definition_of_done="DoD child", parent_id=parent["id"]
        )
        root_summaries = tm.list_tasks()
        parent_summary = next(s for s in root_summaries if s["id"] == parent["id"])
        assert parent_summary["has_subtasks"] is True

        child_summaries = tm.list_tasks(parent_id=parent["id"])
        assert child_summaries[0]["has_subtasks"] is False

    def test_list_stable_order(self, tmp_path):
        tm = _make_tm(tmp_path)
        ids = []
        for i in range(5):
            detail = tm.create_task(name=f"T{i}", definition_of_done="DoD here")
            ids.append(detail["id"])
        summaries = tm.list_tasks()
        returned_ids = [s["id"] for s in summaries]
        assert returned_ids == ids  # created_at ascending order


# ---------------------------------------------------------------------------
# update_task
# ---------------------------------------------------------------------------


class TestUpdateTask:
    def test_update_name(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="Old", definition_of_done="DoD here")
        updated = tm.update_task(detail["id"], name="New")
        assert updated["name"] == "New"

    def test_update_status_to_in_progress(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        updated = tm.update_task(detail["id"], status="in_progress")
        assert updated["status"] == "in_progress"

    def test_update_status_to_blocked_with_blockers(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        updated = tm.update_task(
            detail["id"], status="blocked", blockers=["Waiting for API key"]
        )
        assert updated["status"] == "blocked"
        assert updated["blockers"] == ["Waiting for API key"]

    def test_update_status_to_done_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="tasks_mark_done"):
            tm.update_task(detail["id"], status="done")

    def test_update_invalid_status_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="Invalid status"):
            tm.update_task(detail["id"], status="cancelled")

    def test_update_non_string_status_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="status"):
            tm.update_task(detail["id"], status=["in_progress"])  # type: ignore[arg-type]

    def test_update_non_string_priority_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="priority"):
            tm.update_task(detail["id"], priority=1)  # type: ignore[arg-type]

    def test_update_invalid_status_transition_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        # not_started → in_review is not a valid transition
        with pytest.raises(TaskValidationError, match="transition"):
            tm.update_task(detail["id"], status="in_review")

    def test_update_status_transitions_in_review_to_in_progress(self, tmp_path):
        # Reviewer rejection path: in_review → in_progress must be allowed.
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        tm.update_task(detail["id"], status="in_progress")
        tm.update_task(detail["id"], status="in_review")
        updated = tm.update_task(detail["id"], status="in_progress")
        assert updated["status"] == "in_progress"

    def test_update_status_blocked_to_in_progress(self, tmp_path):
        # Unblocking path: blocked → in_progress must be allowed.
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        tm.update_task(detail["id"], status="blocked")
        updated = tm.update_task(detail["id"], status="in_progress")
        assert updated["status"] == "in_progress"

    def test_update_invalid_priority_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="priority"):
            tm.update_task(detail["id"], priority="critical")

    def test_update_dod_too_short_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="definition_of_done"):
            tm.update_task(detail["id"], definition_of_done="Hi")

    def test_update_empty_name_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="name"):
            tm.update_task(detail["id"], name="")

    def test_update_whitespace_name_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="name"):
            tm.update_task(detail["id"], name="   ")

    def test_update_whitespace_dod_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="definition_of_done"):
            tm.update_task(detail["id"], definition_of_done="     ")

    def test_update_non_string_description_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="description"):
            tm.update_task(detail["id"], description=123)

    def test_update_blockers_must_be_list_of_strings(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="blockers"):
            tm.update_task(detail["id"], blockers=[1, 2, 3])

    def test_update_unknown_field_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="Unknown field"):
            tm.update_task(detail["id"], nonexistent_field="x")

    def test_update_unknown_task_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskNotFoundError):
            tm.update_task("task_xxxxxx", name="X")

    def test_update_persists(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        tm.update_task(detail["id"], name="Updated")
        on_disk = _tasks_json(tmp_path)
        assert on_disk["tasks"][detail["id"]]["name"] == "Updated"


# ---------------------------------------------------------------------------
# add_note
# ---------------------------------------------------------------------------


class TestAddNote:
    def test_note_appended(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        updated = tm.add_note(detail["id"], "First note")
        assert len(updated["notes"]) == 1
        assert "First note" in updated["notes"][0]

    def test_note_has_iso_timestamp_prefix(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        updated = tm.add_note(detail["id"], "Check this")
        note = updated["notes"][0]
        # Format: [2026-04-08T12:00:00.123456Z] Check this
        assert note.startswith("[20")
        assert "." in note  # microseconds always present
        assert "Z]" in note
        assert "Check this" in note

    def test_notes_accumulate(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        tm.add_note(detail["id"], "Note 1")
        updated = tm.add_note(detail["id"], "Note 2")
        assert len(updated["notes"]) == 2

    def test_add_note_unknown_task_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskNotFoundError):
            tm.add_note("task_xxxxxx", "A note")

    def test_add_note_empty_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="note"):
            tm.add_note(detail["id"], "   ")

    def test_add_note_non_string_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="note"):
            tm.add_note(detail["id"], 42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# mark_done
# ---------------------------------------------------------------------------


class TestMarkDone:
    def test_mark_done_succeeds(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        done = tm.mark_done(detail["id"])
        assert done["status"] == "done"

    def test_mark_done_requires_dod(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        # Manually corrupt the DoD on disk to simulate a task that slipped through
        data = _tasks_json(tmp_path)
        data["tasks"][detail["id"]]["definition_of_done"] = ""
        (tmp_path / "tasks.json").write_text(json.dumps(data))

        with pytest.raises(TaskValidationError, match="definition_of_done"):
            tm.mark_done(detail["id"])

    def test_mark_done_requires_all_subtasks_done(self, tmp_path):
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="Parent done")
        tm.create_task(
            name="Child", definition_of_done="Child done", parent_id=parent["id"]
        )
        with pytest.raises(TaskValidationError, match="not done"):
            tm.mark_done(parent["id"])

    def test_mark_done_succeeds_when_all_subtasks_done(self, tmp_path):
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="Parent done")
        child = tm.create_task(
            name="Child", definition_of_done="Child done", parent_id=parent["id"]
        )
        tm.mark_done(child["id"])
        done = tm.mark_done(parent["id"])
        assert done["status"] == "done"

    def test_mark_done_missing_subtask_reference_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="Parent done")
        # Manually corrupt subtask_ids to reference a non-existent task
        data = _tasks_json(tmp_path)
        data["tasks"][parent["id"]]["subtask_ids"].append("task_ghost1")
        (tmp_path / "tasks.json").write_text(json.dumps(data))
        with pytest.raises(TaskValidationError, match="missing"):
            tm.mark_done(parent["id"])

    def test_mark_done_unknown_task_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskNotFoundError):
            tm.mark_done("task_xxxxxx")

    def test_mark_done_persists(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        tm.mark_done(detail["id"])
        on_disk = _tasks_json(tmp_path)
        assert on_disk["tasks"][detail["id"]]["status"] == "done"


# ---------------------------------------------------------------------------
# find / find_incomplete / all_tasks
# ---------------------------------------------------------------------------


class TestQueries:
    def test_find_by_status(self, tmp_path):
        tm = _make_tm(tmp_path)
        a = tm.create_task(name="A", definition_of_done="DoD A")
        b = tm.create_task(name="B", definition_of_done="DoD B")
        tm.update_task(a["id"], status="in_progress")
        results = tm.find("in_progress")
        ids = [r["id"] for r in results]
        assert a["id"] in ids
        assert b["id"] not in ids

    def test_find_returns_summaries(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        tm.update_task(detail["id"], status="blocked")
        results = tm.find("blocked")
        assert len(results) == 1
        assert set(results[0].keys()) == {
            "id",
            "name",
            "status",
            "priority",
            "has_subtasks",
        }

    def test_find_empty(self, tmp_path):
        tm = _make_tm(tmp_path)
        assert tm.find("blocked") == []

    def test_find_incomplete_excludes_done(self, tmp_path):
        tm = _make_tm(tmp_path)
        a = tm.create_task(name="A", definition_of_done="DoD A")
        b = tm.create_task(name="B", definition_of_done="DoD B")
        tm.mark_done(b["id"])
        incomplete = tm.find_incomplete()
        ids = [r["id"] for r in incomplete]
        assert a["id"] in ids
        assert b["id"] not in ids

    def test_all_tasks_returns_raw_records(self, tmp_path):
        tm = _make_tm(tmp_path)
        a = tm.create_task(name="A", definition_of_done="DoD A")
        b = tm.create_task(name="B", definition_of_done="DoD B")
        all_t = tm.all_tasks()
        ids = [t["id"] for t in all_t]
        assert a["id"] in ids
        assert b["id"] in ids
        # Raw records include subtask_ids and created_at (needed by orchestrator)
        assert "subtask_ids" in all_t[0]
        assert "created_at" in all_t[0]

    def test_all_tasks_empty(self, tmp_path):
        tm = _make_tm(tmp_path)
        assert tm.all_tasks() == []

    def test_all_tasks_returns_copies(self, tmp_path):
        tm = _make_tm(tmp_path)
        tm.create_task(name="T", definition_of_done="DoD here")
        tasks = tm.all_tasks()
        tasks[0]["name"] = "mutated"
        # Re-fetch should be unaffected
        assert tm.all_tasks()[0]["name"] == "T"

    def test_task_detail_lists_are_copies(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        tm.update_task(detail["id"], status="blocked", blockers=["reason"])
        fetched = tm.get_task(detail["id"])
        fetched["blockers"].append("injected")
        # Re-fetch should be unaffected
        assert tm.get_task(detail["id"])["blockers"] == ["reason"]


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


class TestNameValidation:
    def test_valid_name_alphanumeric(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="abc123", definition_of_done="DoD here")
        assert t["name"] == "abc123"

    def test_valid_name_with_underscore(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="my_task_1", definition_of_done="DoD here")
        assert t["name"] == "my_task_1"

    def test_name_with_space_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="underscores"):
            tm.create_task(name="has space", definition_of_done="DoD here")

    def test_name_with_dot_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="underscores"):
            tm.create_task(name="a.b", definition_of_done="DoD here")

    def test_name_with_hyphen_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="underscores"):
            tm.create_task(name="a-b", definition_of_done="DoD here")

    def test_update_name_invalid_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="underscores"):
            tm.update_task(t["id"], name="bad name")

    def test_update_name_valid(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="T", definition_of_done="DoD here")
        updated = tm.update_task(t["id"], name="New_Name")
        assert updated["name"] == "New_Name"


# ---------------------------------------------------------------------------
# Sibling-scoped name uniqueness
# ---------------------------------------------------------------------------


class TestSiblingUniqueness:
    def test_duplicate_root_name_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        tm.create_task(name="Alpha", definition_of_done="DoD A")
        with pytest.raises(TaskValidationError, match="already exists"):
            tm.create_task(name="Alpha", definition_of_done="DoD B")

    def test_same_name_under_different_parents_allowed(self, tmp_path):
        tm = _make_tm(tmp_path)
        p1 = tm.create_task(name="Parent1", definition_of_done="DoD P1")
        p2 = tm.create_task(name="Parent2", definition_of_done="DoD P2")
        c1 = tm.create_task(
            name="Child", definition_of_done="DoD C1", parent_id=p1["id"]
        )
        c2 = tm.create_task(
            name="Child", definition_of_done="DoD C2", parent_id=p2["id"]
        )
        assert c1["name"] == "Child"
        assert c2["name"] == "Child"

    def test_duplicate_child_name_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="DoD P")
        tm.create_task(name="Sub", definition_of_done="DoD S1", parent_id=parent["id"])
        with pytest.raises(TaskValidationError, match="already exists"):
            tm.create_task(
                name="Sub", definition_of_done="DoD S2", parent_id=parent["id"]
            )

    def test_rename_to_existing_sibling_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        tm.create_task(name="Alpha", definition_of_done="DoD A")
        b = tm.create_task(name="Beta", definition_of_done="DoD B")
        with pytest.raises(TaskValidationError, match="already exists"):
            tm.update_task(b["id"], name="Alpha")

    def test_rename_to_own_name_is_noop(self, tmp_path):
        # Renaming to the same name should succeed (it is excluded from sibling check).
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="Alpha", definition_of_done="DoD A")
        updated = tm.update_task(t["id"], name="Alpha")
        assert updated["name"] == "Alpha"


# ---------------------------------------------------------------------------
# find_by_path
# ---------------------------------------------------------------------------


class TestFindByPath:
    def test_single_segment(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="Root", definition_of_done="DoD R")
        found = tm.find_by_path("Root")
        assert found["id"] == t["id"]
        assert found["name"] == "Root"

    def test_two_segments(self, tmp_path):
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=parent["id"]
        )
        found = tm.find_by_path("Root.Child")
        assert found["id"] == child["id"]

    def test_three_segments(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        mid = tm.create_task(
            name="Mid", definition_of_done="DoD M", parent_id=root["id"]
        )
        leaf = tm.create_task(
            name="Leaf", definition_of_done="DoD L", parent_id=mid["id"]
        )
        found = tm.find_by_path("Root.Mid.Leaf")
        assert found["id"] == leaf["id"]

    def test_missing_segment_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        tm.create_task(name="Root", definition_of_done="DoD R")
        with pytest.raises(TaskNotFoundError):
            tm.find_by_path("Root.Missing")

    def test_empty_path_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError):
            tm.find_by_path("")

    def test_invalid_segment_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskValidationError, match="Invalid path segment"):
            tm.find_by_path("bad segment")

    def test_path_not_found_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskNotFoundError):
            tm.find_by_path("Nonexistent")

    def test_corrupted_record_during_scan_raises_storage_error(self, tmp_path):
        """A malformed record encountered while scanning should raise TaskStorageError."""
        tm = _make_tm(tmp_path)
        tm.create_task(name="Good", definition_of_done="DoD here")
        # Inject a corrupted sibling record (missing required fields) directly into storage.
        raw = json.loads((tmp_path / "tasks.json").read_text())
        raw["tasks"]["corrupt-id"] = {
            "id": "corrupt-id"
        }  # missing name, status, priority
        (tmp_path / "tasks.json").write_text(json.dumps(raw))
        with pytest.raises(TaskStorageError):
            tm.find_by_path("Good")

    def test_non_string_parent_id_in_store_raises_storage_error(self, tmp_path):
        """A record with a non-string parent_id raises TaskStorageError up-front."""
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="Root", definition_of_done="DoD here")
        raw = json.loads((tmp_path / "tasks.json").read_text())
        raw["tasks"][t["id"]]["parent_id"] = 123
        (tmp_path / "tasks.json").write_text(json.dumps(raw))
        with pytest.raises(TaskStorageError, match="corrupted 'parent_id'"):
            tm.find_by_path("Root")

    def test_missing_parent_id_in_store_raises_storage_error(self, tmp_path):
        """A record whose parent_id references a missing task raises TaskStorageError."""
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="Root", definition_of_done="DoD here")
        raw = json.loads((tmp_path / "tasks.json").read_text())
        raw["tasks"][t["id"]]["parent_id"] = "nonexistent-parent"
        (tmp_path / "tasks.json").write_text(json.dumps(raw))
        with pytest.raises(TaskStorageError, match="references missing parent"):
            tm.find_by_path("Root")


# ---------------------------------------------------------------------------
# delete_task
# ---------------------------------------------------------------------------


class TestDeleteTask:
    def test_delete_leaf(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="T", definition_of_done="DoD here")
        tm.delete_task(t["id"])
        with pytest.raises(TaskNotFoundError):
            tm.get_task(t["id"])

    def test_delete_removes_from_parent_subtask_ids(self, tmp_path):
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="DoD P")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=parent["id"]
        )
        tm.delete_task(child["id"])
        updated_parent = tm.get_task(parent["id"])
        assert updated_parent["subtasks"] == []

    def test_delete_cascades_to_descendants(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=root["id"]
        )
        grandchild = tm.create_task(
            name="Grand", definition_of_done="DoD G", parent_id=child["id"]
        )
        tm.delete_task(root["id"])
        for tid in (root["id"], child["id"], grandchild["id"]):
            with pytest.raises(TaskNotFoundError):
                tm.get_task(tid)

    def test_delete_cascades_via_parent_id_when_subtask_ids_out_of_sync(self, tmp_path):
        """Descendants with correct parent_id but missing from parent's subtask_ids are deleted."""
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=root["id"]
        )
        # Corrupt the root's subtask_ids to remove the child reference, simulating
        # a desync between parent's subtask_ids and child's parent_id.
        raw = json.loads((tmp_path / "tasks.json").read_text())
        raw["tasks"][root["id"]]["subtask_ids"] = []
        (tmp_path / "tasks.json").write_text(json.dumps(raw))
        # Deleting root should still cascade to child via parent_id cross-check.
        tm.delete_task(root["id"])
        with pytest.raises(TaskNotFoundError):
            tm.get_task(child["id"])

    def test_delete_cascades_multi_level_orphans_via_parent_id(self, tmp_path):
        """Multi-level orphan chain is fully deleted even when all subtask_ids are cleared."""
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=root["id"]
        )
        grandchild = tm.create_task(
            name="Grand", definition_of_done="DoD G", parent_id=child["id"]
        )
        # Clear ALL subtask_ids at every level — only parent_id links remain.
        raw = json.loads((tmp_path / "tasks.json").read_text())
        raw["tasks"][root["id"]]["subtask_ids"] = []
        raw["tasks"][child["id"]]["subtask_ids"] = []
        (tmp_path / "tasks.json").write_text(json.dumps(raw))
        # BFS from root via parent_id index should reach child and grandchild.
        tm.delete_task(root["id"])
        for tid in (root["id"], child["id"], grandchild["id"]):
            with pytest.raises(TaskNotFoundError):
                tm.get_task(tid)

    def test_delete_nonexistent_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskNotFoundError):
            tm.delete_task("task_xxxxxx")

    def test_delete_allows_reuse_of_name(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="Alpha", definition_of_done="DoD A")
        tm.delete_task(t["id"])
        # Should be able to create a new task with the same name after deletion.
        t2 = tm.create_task(name="Alpha", definition_of_done="DoD A2")
        assert t2["name"] == "Alpha"

    def test_delete_corrupted_subtask_ids_raises_storage_error(self, tmp_path):
        """Malformed subtask_ids on a descendant must raise TaskStorageError."""
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=root["id"]
        )
        # Corrupt the child's subtask_ids directly in storage.
        raw = json.loads((tmp_path / "tasks.json").read_text())
        raw["tasks"][child["id"]]["subtask_ids"] = "not-a-list"
        (tmp_path / "tasks.json").write_text(json.dumps(raw))
        with pytest.raises(TaskStorageError):
            tm.delete_task(root["id"])

    def test_delete_corrupted_parent_subtask_ids_raises_storage_error(self, tmp_path):
        """Malformed subtask_ids on the parent record raises TaskStorageError."""
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="DoD P")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=parent["id"]
        )
        raw = json.loads((tmp_path / "tasks.json").read_text())
        raw["tasks"][parent["id"]]["subtask_ids"] = {"bad": "type"}
        (tmp_path / "tasks.json").write_text(json.dumps(raw))
        with pytest.raises(TaskStorageError, match="corrupted 'subtask_ids'"):
            tm.delete_task(child["id"])

    def test_delete_corrupted_parent_id_raises_storage_error(self, tmp_path):
        """Non-string parent_id on the task being deleted raises TaskStorageError."""
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="DoD P")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=parent["id"]
        )
        raw = json.loads((tmp_path / "tasks.json").read_text())
        raw["tasks"][child["id"]]["parent_id"] = 99
        (tmp_path / "tasks.json").write_text(json.dumps(raw))
        with pytest.raises(TaskStorageError, match="corrupted 'parent_id'"):
            tm.delete_task(child["id"])

    def test_delete_missing_parent_raises_storage_error(self, tmp_path):
        """A valid string parent_id that points to a missing task raises TaskStorageError."""
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="DoD P")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=parent["id"]
        )
        raw = json.loads((tmp_path / "tasks.json").read_text())
        # Remove the parent from storage to simulate a dangling reference.
        del raw["tasks"][parent["id"]]
        (tmp_path / "tasks.json").write_text(json.dumps(raw))
        with pytest.raises(TaskStorageError, match="missing"):
            tm.delete_task(child["id"])

    def test_delete_empty_string_parent_id_raises_storage_error(self, tmp_path):
        """An empty-string parent_id is not None so must also trigger the missing-parent check."""
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="Task", definition_of_done="DoD here")
        raw = json.loads((tmp_path / "tasks.json").read_text())
        raw["tasks"][t["id"]]["parent_id"] = ""
        (tmp_path / "tasks.json").write_text(json.dumps(raw))
        with pytest.raises(TaskStorageError, match="missing"):
            tm.delete_task(t["id"])


# ---------------------------------------------------------------------------
# _task_detail parent_id validation
# ---------------------------------------------------------------------------


class TestTaskDetailParentIdValidation:
    def test_non_string_parent_id_raises_storage_error(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="T", definition_of_done="Done criteria")
        raw = json.loads((tmp_path / "tasks.json").read_text())
        raw["tasks"][t["id"]]["parent_id"] = 42
        (tmp_path / "tasks.json").write_text(json.dumps(raw))
        with pytest.raises(TaskStorageError, match="corrupted 'parent_id'"):
            tm.get_task(t["id"])

    def test_missing_parent_raises_storage_error(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="T", definition_of_done="Done criteria")
        raw = json.loads((tmp_path / "tasks.json").read_text())
        raw["tasks"][t["id"]]["parent_id"] = "nonexistent-id"
        (tmp_path / "tasks.json").write_text(json.dumps(raw))
        with pytest.raises(TaskStorageError, match="missing parent task"):
            tm.get_task(t["id"])

    def test_none_parent_id_is_valid(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="Root", definition_of_done="Done criteria")
        detail = tm.get_task(t["id"])
        assert detail["parent_id"] is None
