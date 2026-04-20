"""Tests for ai_cli.core.task_manager."""

from __future__ import annotations

import contextlib
import json
import logging
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

    def test_empty_store_is_cached_without_file(self, tmp_path):
        """Repeated reads on a missing file must not re-check the filesystem."""
        tm = _make_tm(tmp_path)
        tm.list_tasks()  # first call — file absent, caches empty store
        assert tm._cache == {"tasks": {}}
        # A second read returns the same object from cache, no Path.exists() needed.
        data1 = tm._load()
        data2 = tm._load()
        assert data1 is data2

    def test_cache_invalidated_on_save_failure(self, tmp_path, monkeypatch):
        """If _save raises, the cache must be reset so the next read reloads from disk."""
        import os as _os

        tm = _make_tm(tmp_path)
        tm.create_task(name="T", definition_of_done="DoD here")
        assert tm._cache is not None

        # Simulate a disk-full failure by making os.fsync raise inside _save.
        def _fsync_fail(fd: int) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(_os, "fsync", _fsync_fail)
        # OSError from os.fsync must be wrapped into TaskStorageError by _save()
        # so callers get a consistent error type for all storage failures.
        with pytest.raises(TaskStorageError, match="could not be written"):
            tm.create_task(name="T2", definition_of_done="DoD here")
        assert tm._cache is None  # invalidated — next load will re-read disk

    def test_save_oserror_wrapped_as_storage_error(self, tmp_path, monkeypatch):
        """Any OSError raised during _save must surface as TaskStorageError."""
        import os as _os

        tm = _make_tm(tmp_path)

        def _fsync_fail(fd: int) -> None:
            raise OSError("permission denied")

        monkeypatch.setattr(_os, "fsync", _fsync_fail)
        exc = None
        try:
            tm.create_task(name="T", definition_of_done="DoD here")
        except TaskStorageError as e:
            exc = e
        assert exc is not None, "expected TaskStorageError"
        assert "could not be written" in str(exc)
        assert isinstance(exc.__cause__, OSError)

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
# Storage location & cross-instance persistence
# ---------------------------------------------------------------------------


class TestStorageLocation:
    """Locks in the contract that ``tasks.json`` is project-scoped.

    These tests do not care *which* directory the manager is given — they
    care that two managers pointed at the same directory share state, and
    two managers pointed at different directories do not.  Together with
    the ``__main__`` wiring, this guarantees that tasks survive across
    sessions in the same project and stay isolated between projects.
    """

    def test_two_managers_share_state_when_pointed_at_same_dir(self, tmp_path):
        # First manager creates a task and persists it.
        tm1 = TaskManager(tmp_path)
        created = tm1.create_task(name="Shared", definition_of_done="DoD here")

        # Second manager constructed against the same directory must see it.
        tm2 = TaskManager(tmp_path)
        ids = [t["id"] for t in tm2.list_tasks()]
        assert created["id"] in ids

    def test_managers_in_different_dirs_are_isolated(self, tmp_path):
        # Two distinct project directories.
        project_a = tmp_path / "project_a"
        project_b = tmp_path / "project_b"
        project_a.mkdir()
        project_b.mkdir()

        tm_a = TaskManager(project_a)
        tm_b = TaskManager(project_b)

        a_task = tm_a.create_task(name="OnlyA", definition_of_done="DoD A here")
        b_task = tm_b.create_task(name="OnlyB", definition_of_done="DoD B here")

        a_ids = [t["id"] for t in tm_a.list_tasks()]
        b_ids = [t["id"] for t in tm_b.list_tasks()]

        assert a_task["id"] in a_ids
        assert b_task["id"] not in a_ids
        assert b_task["id"] in b_ids
        assert a_task["id"] not in b_ids

    def test_missing_file_in_fresh_dir_returns_empty_without_creating_file(
        self, tmp_path
    ):
        # Brand-new directory with no prior task activity.
        tm = TaskManager(tmp_path)

        assert tm.list_tasks() == []
        # Reading must not create tasks.json.
        assert not (tmp_path / "tasks.json").exists()

    def test_first_write_creates_file_in_storage_dir(self, tmp_path):
        # The structural assertion: the file is written exactly under the
        # directory the manager was given, not in any subdirectory.
        tm = TaskManager(tmp_path)
        tm.create_task(name="T", definition_of_done="DoD here")
        assert (tmp_path / "tasks.json").exists()
        # No nested session-style directory was created.
        assert not (tmp_path / "sessions").exists()


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

    def test_set_goal_info_log_uses_truncated_preview(self, tmp_path, caplog):
        tm = _make_tm(tmp_path)
        long_goal = "x" * 300

        with caplog.at_level(logging.INFO, logger="ai_cli.core.task_manager"):
            tm.set_goal(long_goal)

        info_logs = [
            rec.getMessage()
            for rec in caplog.records
            if rec.name == "ai_cli.core.task_manager" and rec.levelno == logging.INFO
        ]
        assert any("Task goal updated:" in line for line in info_logs)
        # Full goal text should not appear in INFO logs.
        assert all(long_goal not in line for line in info_logs)


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
        tm._cache["tasks"][parent["id"]]["subtask_ids"].append("task_ghost1")
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

    def test_reparent_to_root_with_none(self, tmp_path):
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="DoD parent")
        child = tm.create_task(
            name="Child", definition_of_done="DoD child", parent_id=parent["id"]
        )

        updated = tm.update_task(child["id"], parent_id=None)

        assert updated["parent_id"] is None
        parent_detail = tm.get_task(parent["id"])
        assert all(sub["id"] != child["id"] for sub in parent_detail["subtasks"])
        root_tasks = tm.list_tasks(parent_id=None)
        assert any(t["id"] == child["id"] for t in root_tasks)

    def test_reparent_to_another_parent(self, tmp_path):
        tm = _make_tm(tmp_path)
        old_parent = tm.create_task(name="OldParent", definition_of_done="DoD old")
        new_parent = tm.create_task(name="NewParent", definition_of_done="DoD new")
        child = tm.create_task(
            name="Child", definition_of_done="DoD child", parent_id=old_parent["id"]
        )

        updated = tm.update_task(child["id"], parent_id=new_parent["id"])

        assert updated["parent_id"] == new_parent["id"]
        old_subtasks = tm.get_task(old_parent["id"])["subtasks"]
        new_subtasks = tm.get_task(new_parent["id"])["subtasks"]
        assert all(sub["id"] != child["id"] for sub in old_subtasks)
        assert any(sub["id"] == child["id"] for sub in new_subtasks)

    def test_reparent_to_self_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        task = tm.create_task(name="Root", definition_of_done="DoD root")

        with pytest.raises(TaskValidationError, match="own parent"):
            tm.update_task(task["id"], parent_id=task["id"])

    def test_reparent_to_descendant_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD root")
        child = tm.create_task(
            name="Child", definition_of_done="DoD child", parent_id=root["id"]
        )

        with pytest.raises(TaskValidationError, match="descendants"):
            tm.update_task(root["id"], parent_id=child["id"])

    def test_reparent_persists_subtask_link_when_new_parent_missing_subtask_ids(
        self, tmp_path
    ):
        tm = _make_tm(tmp_path)
        old_parent = tm.create_task(name="OldParent", definition_of_done="DoD old")
        new_parent = tm.create_task(name="NewParent", definition_of_done="DoD new")
        child = tm.create_task(
            name="Child", definition_of_done="DoD child", parent_id=old_parent["id"]
        )

        del tm._cache["tasks"][new_parent["id"]]["subtask_ids"]
        tm._save(tm._cache)

        tm.update_task(child["id"], parent_id=new_parent["id"])

        new_parent_detail = tm.get_task(new_parent["id"])
        assert any(sub["id"] == child["id"] for sub in new_parent_detail["subtasks"])


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
# obsolete_note
# ---------------------------------------------------------------------------


class TestObsoleteNote:
    def test_obsolete_note_removes_from_active_notes(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        tm.add_note(detail["id"], "temporary blocker")
        tm.add_note(detail["id"], "resolved")

        updated = tm.obsolete_note(detail["id"], 0, reason="Superseded")

        assert len(updated["notes"]) == 1
        assert "resolved" in updated["notes"][0]
        assert all("temporary blocker" not in n for n in updated["notes"])

    def test_obsolete_note_keeps_audit_history(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        tm.add_note(detail["id"], "first")
        tm.add_note(detail["id"], "second")
        tm.obsolete_note(detail["id"], 0, reason="no longer relevant")

        on_disk = _tasks_json(tmp_path)
        raw = on_disk["tasks"][detail["id"]]
        history = raw["note_history"]
        assert len(history) == 2
        obsolete = [n for n in history if n["status"] == "obsolete"]
        assert len(obsolete) == 1
        assert "first" in obsolete[0]["text"]
        assert obsolete[0]["obsolete_reason"] == "no longer relevant"

    def test_obsolete_note_unknown_task_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskNotFoundError):
            tm.obsolete_note("task_xxxxxx", 0)

    def test_obsolete_note_out_of_range_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        tm.add_note(detail["id"], "only")
        with pytest.raises(TaskValidationError, match="note_index"):
            tm.obsolete_note(detail["id"], 3)

    def test_obsolete_note_non_int_index_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        tm.add_note(detail["id"], "only")
        with pytest.raises(TaskValidationError, match="note_index"):
            tm.obsolete_note(detail["id"], "0")  # type: ignore[arg-type]

    def test_obsolete_note_supports_legacy_records_without_lifecycle_fields(
        self, tmp_path
    ):
        tm = _make_tm(tmp_path)
        detail = tm.create_task(name="T", definition_of_done="DoD here")
        task = tm._cache["tasks"][detail["id"]]
        task["notes"] = ["[2026-01-01T00:00:00.000000Z] old note"]
        task.pop("active_note_ids", None)
        task.pop("note_history", None)
        tm._save(tm._cache)

        updated = tm.obsolete_note(detail["id"], 0, reason="stale")

        assert updated["notes"] == []
        fetched = tm.get_task(detail["id"])
        assert any(
            isinstance(e, dict)
            and e.get("status") == "obsolete"
            and "old note" in str(e.get("text"))
            for e in fetched.get("note_history", [])
        )


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
        # Corrupt the DoD directly in the in-memory cache to simulate a task
        # whose DoD was cleared (e.g. by a storage migration or manual edit).
        tm._cache["tasks"][detail["id"]]["definition_of_done"] = ""
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
        tm._cache["tasks"][parent["id"]]["subtask_ids"].append("task_ghost1")
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
        # Inject a corrupted sibling record (missing required fields) into the cache.
        tm._cache["tasks"]["corrupt-id"] = {
            "id": "corrupt-id"
        }  # missing name, status, priority
        with pytest.raises(TaskStorageError):
            tm.find_by_path("Good")

    def test_non_string_parent_id_in_store_raises_storage_error(self, tmp_path):
        """A record with a non-string parent_id raises TaskStorageError up-front."""
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="Root", definition_of_done="DoD here")
        tm._cache["tasks"][t["id"]]["parent_id"] = 123
        with pytest.raises(TaskStorageError, match="corrupted 'parent_id'"):
            tm.find_by_path("Root")

    def test_missing_parent_id_in_store_raises_storage_error(self, tmp_path):
        """A record whose parent_id references a missing task raises TaskStorageError."""
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="Root", definition_of_done="DoD here")
        tm._cache["tasks"][t["id"]]["parent_id"] = "nonexistent-parent"
        with pytest.raises(TaskStorageError, match="references missing parent"):
            tm.find_by_path("Root")

    def test_invalid_stored_name_raises_storage_error(self, tmp_path):
        """A task whose stored name violates _NAME_RE must raise TaskStorageError."""
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="Root", definition_of_done="DoD R")
        tm._cache["tasks"][t["id"]]["name"] = "bad name!"  # spaces and ! are invalid
        with pytest.raises(TaskStorageError, match="invalid name"):
            tm.find_by_path("Root")

    def test_duplicate_sibling_name_raises_storage_error(self, tmp_path):
        """Two siblings with the same name in the store must raise TaskStorageError."""
        tm = _make_tm(tmp_path)
        tm.create_task(name="Root", definition_of_done="DoD R")
        # Inject a second root task with the same name directly into the cache
        # to simulate store corruption or a legacy-data scenario.
        import copy

        duplicate = copy.deepcopy(next(iter(tm._cache["tasks"].values())))
        duplicate["id"] = "task_dup001"
        tm._cache["tasks"]["task_dup001"] = duplicate
        with pytest.raises(TaskStorageError, match="Duplicate sibling"):
            tm.find_by_path("Root")

    def test_resolve_path_to_id_duplicate_sibling_raises_storage_error(self, tmp_path):
        """resolve_path_to_id also raises TaskStorageError on duplicate sibling names."""
        tm = _make_tm(tmp_path)
        tm.create_task(name="Root", definition_of_done="DoD R")
        import copy

        duplicate = copy.deepcopy(next(iter(tm._cache["tasks"].values())))
        duplicate["id"] = "task_dup002"
        tm._cache["tasks"]["task_dup002"] = duplicate
        with pytest.raises(TaskStorageError, match="Duplicate sibling"):
            tm.resolve_path_to_id("Root")


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
        tm._cache["tasks"][root["id"]]["subtask_ids"] = []
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
        tm._cache["tasks"][root["id"]]["subtask_ids"] = []
        tm._cache["tasks"][child["id"]]["subtask_ids"] = []
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
        tm._cache["tasks"][child["id"]]["subtask_ids"] = "not-a-list"
        with pytest.raises(TaskStorageError):
            tm.delete_task(root["id"])

    def test_delete_corrupted_parent_subtask_ids_raises_storage_error(self, tmp_path):
        """Malformed subtask_ids on the parent record raises TaskStorageError."""
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="DoD P")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=parent["id"]
        )
        tm._cache["tasks"][parent["id"]]["subtask_ids"] = {"bad": "type"}
        with pytest.raises(TaskStorageError, match="corrupted 'subtask_ids'"):
            tm.delete_task(child["id"])

    def test_delete_corrupted_parent_id_raises_storage_error(self, tmp_path):
        """Non-string parent_id on the task being deleted raises TaskStorageError."""
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="DoD P")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=parent["id"]
        )
        tm._cache["tasks"][child["id"]]["parent_id"] = 99
        with pytest.raises(TaskStorageError, match="corrupted 'parent_id'"):
            tm.delete_task(child["id"])

    def test_delete_missing_parent_raises_storage_error(self, tmp_path):
        """A valid string parent_id that points to a missing task raises TaskStorageError."""
        tm = _make_tm(tmp_path)
        parent = tm.create_task(name="Parent", definition_of_done="DoD P")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=parent["id"]
        )
        # Remove the parent from the cache to simulate a dangling reference.
        del tm._cache["tasks"][parent["id"]]
        with pytest.raises(TaskStorageError, match="missing"):
            tm.delete_task(child["id"])

    def test_delete_empty_string_parent_id_raises_storage_error(self, tmp_path):
        """An empty-string parent_id is not None so must also trigger the missing-parent check."""
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="Task", definition_of_done="DoD here")
        tm._cache["tasks"][t["id"]]["parent_id"] = ""
        with pytest.raises(TaskStorageError, match="missing"):
            tm.delete_task(t["id"])


# ---------------------------------------------------------------------------
# _task_detail parent_id validation
# ---------------------------------------------------------------------------


class TestTaskDetailParentIdValidation:
    def test_non_string_parent_id_raises_storage_error(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="T", definition_of_done="Done criteria")
        tm._cache["tasks"][t["id"]]["parent_id"] = 42
        with pytest.raises(TaskStorageError, match="corrupted 'parent_id'"):
            tm.get_task(t["id"])

    def test_missing_parent_raises_storage_error(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="T", definition_of_done="Done criteria")
        tm._cache["tasks"][t["id"]]["parent_id"] = "nonexistent-id"
        with pytest.raises(TaskStorageError, match="missing parent task"):
            tm.get_task(t["id"])

    def test_none_parent_id_is_valid(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="Root", definition_of_done="Done criteria")
        detail = tm.get_task(t["id"])
        assert detail["parent_id"] is None


# ---------------------------------------------------------------------------
# close_task
# ---------------------------------------------------------------------------


class TestCloseTask:
    def test_close_leaf_task(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="T", definition_of_done="DoD here")
        result = tm.close_task(t["id"])
        assert result["status"] == "done"

    def test_close_cascades_to_subtasks(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=root["id"]
        )
        grandchild = tm.create_task(
            name="Grand", definition_of_done="DoD G", parent_id=child["id"]
        )
        tm.close_task(root["id"])
        for tid in (root["id"], child["id"], grandchild["id"]):
            assert tm.get_task(tid)["status"] == "done"

    def test_close_does_not_affect_siblings(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        c1 = tm.create_task(
            name="C1", definition_of_done="DoD C1", parent_id=root["id"]
        )
        c2 = tm.create_task(
            name="C2", definition_of_done="DoD C2", parent_id=root["id"]
        )
        tm.close_task(c1["id"])
        assert tm.get_task(c1["id"])["status"] == "done"
        assert tm.get_task(c2["id"])["status"] == "not_started"

    def test_close_bypasses_dod_validation(self, tmp_path):
        """close_task succeeds even when DoD would fail mark_done checks."""
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=root["id"]
        )
        # child is not done — mark_done(root) would raise, but close_task must not
        result = tm.close_task(root["id"])
        assert result["status"] == "done"
        assert tm.get_task(child["id"])["status"] == "done"

    def test_close_nonexistent_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskNotFoundError):
            tm.close_task("nonexistent")

    def test_close_corrupted_subtask_ids_non_list_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        tm._cache["tasks"][root["id"]]["subtask_ids"] = "not-a-list"
        with pytest.raises(TaskStorageError, match="invalid subtask_ids"):
            tm.close_task(root["id"])

    def test_close_corrupted_subtask_ids_non_str_entry_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        tm._cache["tasks"][root["id"]]["subtask_ids"] = [42]
        with pytest.raises(TaskStorageError, match="invalid subtask_ids entry"):
            tm.close_task(root["id"])


# ---------------------------------------------------------------------------
# open_task
# ---------------------------------------------------------------------------


class TestOpenTask:
    def test_open_done_task(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="T", definition_of_done="DoD here")
        tm.close_task(t["id"])
        result = tm.open_task(t["id"])
        assert result["status"] == "not_started"

    def test_open_reopens_done_parent(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=root["id"]
        )
        tm.close_task(root["id"])
        tm.open_task(child["id"])
        assert tm.get_task(child["id"])["status"] == "not_started"
        assert tm.get_task(root["id"])["status"] == "not_started"

    def test_open_reopens_ancestor_chain(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=root["id"]
        )
        grandchild = tm.create_task(
            name="Grand", definition_of_done="DoD G", parent_id=child["id"]
        )
        tm.close_task(root["id"])
        tm.open_task(grandchild["id"])
        for tid in (root["id"], child["id"], grandchild["id"]):
            assert tm.get_task(tid)["status"] == "not_started"

    def test_open_stops_at_non_done_parent(self, tmp_path):
        """Ancestor chain walk stops when it reaches a parent that is not done."""
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=root["id"]
        )
        grandchild = tm.create_task(
            name="Grand", definition_of_done="DoD G", parent_id=child["id"]
        )
        # Only close child + grandchild; root stays not_started
        tm.close_task(child["id"])
        assert tm.get_task(root["id"])["status"] == "not_started"

        tm.open_task(grandchild["id"])
        assert tm.get_task(grandchild["id"])["status"] == "not_started"
        assert tm.get_task(child["id"])["status"] == "not_started"
        assert tm.get_task(root["id"])["status"] == "not_started"

    def test_open_does_not_affect_siblings(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        c1 = tm.create_task(
            name="C1", definition_of_done="DoD C1", parent_id=root["id"]
        )
        c2 = tm.create_task(
            name="C2", definition_of_done="DoD C2", parent_id=root["id"]
        )
        tm.close_task(root["id"])
        tm.open_task(c1["id"])
        assert tm.get_task(c1["id"])["status"] == "not_started"
        assert tm.get_task(c2["id"])["status"] == "done"  # sibling unchanged

    def test_open_nonexistent_raises(self, tmp_path):
        tm = _make_tm(tmp_path)
        with pytest.raises(TaskNotFoundError):
            tm.open_task("nonexistent")

    def test_open_not_done_raises_validation_error(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="T", definition_of_done="DoD here")
        with pytest.raises(TaskValidationError, match="not done"):
            tm.open_task(t["id"])

    def test_open_in_progress_raises_validation_error(self, tmp_path):
        tm = _make_tm(tmp_path)
        t = tm.create_task(name="T", definition_of_done="DoD here")
        tm.update_task(t["id"], status="in_progress")
        with pytest.raises(TaskValidationError, match="not done"):
            tm.open_task(t["id"])

    def test_open_corrupted_parent_id_raises_storage_error(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=root["id"]
        )
        tm.close_task(child["id"])
        tm._cache["tasks"][child["id"]]["parent_id"] = 99
        with pytest.raises(TaskStorageError, match="invalid parent_id"):
            tm.open_task(child["id"])

    def test_close_missing_descendant_raises_storage_error(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=root["id"]
        )
        # Remove child from the cache while leaving root's subtask_ids intact.
        del tm._cache["tasks"][child["id"]]
        with pytest.raises(TaskStorageError, match="missing from storage"):
            tm.close_task(root["id"])

    def test_open_empty_string_parent_id_raises_storage_error(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=root["id"]
        )
        tm.close_task(child["id"])
        tm._cache["tasks"][child["id"]]["parent_id"] = ""
        with pytest.raises(TaskStorageError, match="empty string"):
            tm.open_task(child["id"])

    def test_open_missing_parent_raises_storage_error(self, tmp_path):
        tm = _make_tm(tmp_path)
        root = tm.create_task(name="Root", definition_of_done="DoD R")
        child = tm.create_task(
            name="Child", definition_of_done="DoD C", parent_id=root["id"]
        )
        tm.close_task(child["id"])
        # Remove parent from the cache while child's parent_id still references it.
        del tm._cache["tasks"][root["id"]]
        with pytest.raises(TaskStorageError, match="missing parent"):
            tm.open_task(child["id"])
