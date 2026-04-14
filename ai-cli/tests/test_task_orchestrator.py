"""Tests for ai_cli.core.task_orchestrator.TaskOrchestrator."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from ai_cli.core.agent import AgentResult
from ai_cli.core.task_orchestrator import _CONTEXT_LIMIT_STRIKES, TaskOrchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task_summary(
    task_id: str = "t1",
    name: str = "Task",
    status: str = "not_started",
    priority: str = "medium",
    has_subtasks: bool = False,
    created_at: str = "2024-01-01T00:00:00",
    subtask_ids: list | None = None,
) -> dict:
    return {
        "id": task_id,
        "name": name,
        "status": status,
        "priority": priority,
        "has_subtasks": has_subtasks,
        "created_at": created_at,
        "subtask_ids": subtask_ids if subtask_ids is not None else [],
    }


def _make_task_detail(
    task_id: str = "t1",
    name: str = "Task",
    status: str = "not_started",
    description: str = "A task",
    definition_of_done: str = "Done criteria",
    notes: list | None = None,
    next_action: str = "",
) -> dict:
    return {
        "id": task_id,
        "name": name,
        "status": status,
        "description": description,
        "definition_of_done": definition_of_done,
        "notes": notes or [],
        "next_action": next_action,
    }


def _make_task_manager(
    *,
    root_tasks: list | None = None,
    all_tasks: list | None = None,
    find_results: dict | None = None,
    find_incomplete: list | None = None,
    goal: str | None = None,
    task_detail: dict | None = None,
) -> MagicMock:
    """Return a MagicMock TaskManager with configurable return values."""
    tm = MagicMock()
    tm.list_tasks.return_value = root_tasks or []
    tm.all_tasks.return_value = all_tasks or []
    tm.find_incomplete.return_value = find_incomplete or []
    tm.get_goal.return_value = goal
    tm.get_task.return_value = task_detail or _make_task_detail()

    # find(status=...) — route by status keyword via side_effect
    _find_map: dict[str, list] = find_results or {}

    def _find_side_effect(status: str) -> list:
        return _find_map.get(status, [])

    tm.find.side_effect = _find_side_effect
    return tm


def _make_agent_registry(*, has: dict[str, bool] | None = None) -> MagicMock:
    """Return a MagicMock AgentRegistry.

    *has* maps agent name → whether ``has(name)`` returns True.
    By default no agents exist.
    """
    registry = MagicMock()
    has_map = has or {}
    registry.has.side_effect = lambda name: has_map.get(name, False)
    return registry


def _make_orchestrator(
    task_manager=None,
    agent_registry=None,
    display=None,
) -> TaskOrchestrator:
    return TaskOrchestrator(
        task_manager=task_manager or _make_task_manager(),
        agent_registry=agent_registry or _make_agent_registry(),
        display=display or MagicMock(),
        workspace=MagicMock(),
        config=MagicMock(),
        coordinator_llm=MagicMock(),
        global_tool_registry=MagicMock(),
    )


def _ok_result(text: str = "done") -> AgentResult:
    return AgentResult(text=text, status="ok")


def _context_limit_result(text: str = "partial") -> AgentResult:
    return AgentResult(text=text, status="context_limit", partial=True)


# ---------------------------------------------------------------------------
# _needs_planning
# ---------------------------------------------------------------------------


class TestNeedsPlanning:
    def test_no_root_tasks_returns_true(self):
        tm = _make_task_manager(root_tasks=[])
        orch = _make_orchestrator(task_manager=tm)
        assert orch._needs_planning() is True

    def test_all_roots_done_no_blocked_returns_false(self):
        roots = [_make_task_summary("t1", status="done", has_subtasks=True)]
        tm = _make_task_manager(root_tasks=roots, find_results={"blocked": []})
        orch = _make_orchestrator(task_manager=tm)
        assert orch._needs_planning() is False

    def test_root_without_subtasks_not_done_returns_true(self):
        roots = [_make_task_summary("t1", status="not_started", has_subtasks=False)]
        tm = _make_task_manager(root_tasks=roots, find_results={"blocked": []})
        orch = _make_orchestrator(task_manager=tm)
        assert orch._needs_planning() is True

    def test_blocked_tasks_with_no_executable_leaf_returns_true(self):
        """Blocked tasks trigger planning only when there are no other executable leaves."""
        roots = [_make_task_summary("t1", status="done", has_subtasks=True)]
        blocked = [_make_task_summary("t2", status="blocked", subtask_ids=[])]
        # all_tasks returns only blocked tasks → _pick_next_task returns None
        tm = _make_task_manager(
            root_tasks=roots,
            all_tasks=blocked,
            find_results={"blocked": blocked},
        )
        orch = _make_orchestrator(task_manager=tm)
        assert orch._needs_planning() is True

    def test_blocked_tasks_with_executable_leaf_returns_false(self):
        """Blocked tasks do NOT force planning when other leaf tasks are executable."""
        roots = [_make_task_summary("t1", status="done", has_subtasks=True)]
        blocked = [_make_task_summary("t2", status="blocked", subtask_ids=[])]
        executable = [_make_task_summary("t3", status="not_started", subtask_ids=[])]
        tm = _make_task_manager(
            root_tasks=roots,
            all_tasks=blocked + executable,
            find_results={"blocked": blocked},
        )
        orch = _make_orchestrator(task_manager=tm)
        assert orch._needs_planning() is False


# ---------------------------------------------------------------------------
# _pick_next_task
# ---------------------------------------------------------------------------


class TestPickNextTask:
    def test_returns_none_when_no_candidates(self):
        tm = _make_task_manager(all_tasks=[])
        orch = _make_orchestrator(task_manager=tm)
        assert orch._pick_next_task() is None

    def test_skips_tasks_with_subtasks(self):
        tasks = [
            _make_task_summary("t1", status="not_started", subtask_ids=["child1"]),
            _make_task_summary("t2", status="not_started", subtask_ids=[]),
        ]
        tm = _make_task_manager(all_tasks=tasks)
        orch = _make_orchestrator(task_manager=tm)
        assert orch._pick_next_task()["id"] == "t2"

    def test_malformed_subtask_ids_skipped_with_warning(self):
        """Tasks with non-list subtask_ids are skipped rather than causing TypeError."""
        malformed = dict(_make_task_summary("t1", status="not_started"))
        malformed["subtask_ids"] = "not-a-list"
        good = _make_task_summary("t2", status="not_started", subtask_ids=[])
        tm = _make_task_manager(all_tasks=[malformed, good])
        orch = _make_orchestrator(task_manager=tm)
        assert orch._pick_next_task()["id"] == "t2"

    def test_in_progress_beats_not_started(self):
        tasks = [
            _make_task_summary("t1", status="not_started", created_at="2024-01-01"),
            _make_task_summary("t2", status="in_progress", created_at="2024-01-02"),
        ]
        tm = _make_task_manager(all_tasks=tasks)
        orch = _make_orchestrator(task_manager=tm)
        assert orch._pick_next_task()["id"] == "t2"

    def test_high_priority_beats_medium(self):
        tasks = [
            _make_task_summary(
                "t1", status="not_started", priority="medium", created_at="2024-01-01"
            ),
            _make_task_summary(
                "t2", status="not_started", priority="high", created_at="2024-01-02"
            ),
        ]
        tm = _make_task_manager(all_tasks=tasks)
        orch = _make_orchestrator(task_manager=tm)
        assert orch._pick_next_task()["id"] == "t2"

    def test_earlier_created_at_beats_later_when_same_priority(self):
        tasks = [
            _make_task_summary(
                "t1",
                status="not_started",
                priority="medium",
                created_at="2024-01-02",
            ),
            _make_task_summary(
                "t2",
                status="not_started",
                priority="medium",
                created_at="2024-01-01",
            ),
        ]
        tm = _make_task_manager(all_tasks=tasks)
        orch = _make_orchestrator(task_manager=tm)
        assert orch._pick_next_task()["id"] == "t2"

    def test_id_is_tiebreaker(self):
        tasks = [
            _make_task_summary(
                "tb",
                status="not_started",
                priority="medium",
                created_at="2024-01-01",
            ),
            _make_task_summary(
                "ta",
                status="not_started",
                priority="medium",
                created_at="2024-01-01",
            ),
        ]
        tm = _make_task_manager(all_tasks=tasks)
        orch = _make_orchestrator(task_manager=tm)
        assert orch._pick_next_task()["id"] == "ta"

    def test_skips_done_and_blocked_and_in_review(self):
        tasks = [
            _make_task_summary("t_done", status="done"),
            _make_task_summary("t_blocked", status="blocked"),
            _make_task_summary("t_review", status="in_review"),
            _make_task_summary("t_ok", status="not_started"),
        ]
        tm = _make_task_manager(all_tasks=tasks)
        orch = _make_orchestrator(task_manager=tm)
        assert orch._pick_next_task()["id"] == "t_ok"


# ---------------------------------------------------------------------------
# run() — main loop
# ---------------------------------------------------------------------------


class TestRunLoop:
    def _leaf_task(self, task_id="t1", status="not_started") -> dict:
        return _make_task_summary(task_id, status=status, subtask_ids=[])

    def test_run_stops_when_all_tasks_complete(self):
        """After execution the incomplete list is empty → 'All tasks complete.'"""
        tm = _make_task_manager(
            root_tasks=[_make_task_summary("t1", has_subtasks=False, status="done")],
            all_tasks=[],
            find_results={"blocked": [], "in_review": []},
            find_incomplete=[],
        )
        # _needs_planning returns False (root is done, no blocked)
        registry = _make_agent_registry(has={"executor": True, "reviewer": False})
        agent = MagicMock()
        agent.run.return_value = _ok_result()
        registry.get_or_create.return_value = agent

        display = MagicMock()
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )

        # No leaf tasks → immediately hits "no executable tasks" with empty incomplete
        orch.run("my goal")
        display.show_status.assert_any_call("All tasks complete.")

    def test_run_shows_executor_status_and_summary(self):
        tm = _make_task_manager(
            root_tasks=[_make_task_summary("t1", has_subtasks=True, status="done")],
            all_tasks=[_make_task_summary("t2", name="Leaf", status="not_started")],
            find_results={"blocked": [], "in_review": []},
            find_incomplete=[
                _make_task_summary("t2", name="Leaf", status="not_started")
            ],
            task_detail=_make_task_detail(task_id="t2", name="Leaf"),
        )
        registry = _make_agent_registry(has={"executor": True, "reviewer": False})
        executor = MagicMock()
        executor.run.return_value = AgentResult(
            text='{"success": true, "summary": "applied patch", "answer": "ok", "error_message": null}',
            status="ok",
        )
        registry.get_or_create.return_value = executor
        display = MagicMock()
        orch = _make_orchestrator(
            task_manager=tm,
            agent_registry=registry,
            display=display,
        )

        orch.run("goal", max_iterations=1, autonomous=True)

        status_messages = [
            call.args[0]
            for call in display.show_status.call_args_list
            if call.args and isinstance(call.args[0], str)
        ]
        assert any(
            msg.startswith("  → executor [ok]: applied patch")
            for msg in status_messages
        )

    def test_run_calls_planner_when_needs_planning(self):
        """When _needs_planning() is True the planner is invoked."""
        # No root tasks → _needs_planning returns True
        tm = _make_task_manager(
            root_tasks=[],
            all_tasks=[],
            find_results={"blocked": [], "in_review": []},
            find_incomplete=[],
        )
        registry = _make_agent_registry(has={"executor": True, "planner": True})
        planner = MagicMock()
        planner.run.return_value = _ok_result()

        def _get_or_create(name, **kwargs):
            return planner

        registry.get_or_create.side_effect = _get_or_create

        display = MagicMock()
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )
        orch.run("my goal", max_iterations=2)

        # Planner should have been called
        assert planner.run.called

    def test_run_sets_goal(self):
        tm = _make_task_manager(
            root_tasks=[_make_task_summary("t1", status="done", has_subtasks=False)],
            all_tasks=[],
            find_results={"blocked": [], "in_review": []},
            find_incomplete=[],
        )
        registry = _make_agent_registry(has={"executor": True})
        orch = _make_orchestrator(task_manager=tm, agent_registry=registry)
        orch.run("the goal")
        tm.set_goal.assert_called_once_with("the goal")

    def test_run_reaches_iteration_limit(self):
        """When max_iterations is exhausted the limit message is shown."""
        # Planner makes progress each round (a new root task appears) but the
        # tree never reaches a state where execution can take over, so the loop
        # only exits on max_iterations rather than the no-progress check.
        rounds = {"n": 0}

        def _list_tasks(parent_id=None):
            rounds["n"] += 1
            return [
                _make_task_summary(f"t{i}", status="not_started", has_subtasks=False)
                for i in range(rounds["n"])
            ]

        tm = _make_task_manager(
            root_tasks=[],
            all_tasks=[],
            find_results={"blocked": [], "in_review": []},
            find_incomplete=[_make_task_summary("t1")],
        )
        tm.list_tasks.side_effect = _list_tasks
        registry = _make_agent_registry(has={"executor": True, "planner": True})
        planner = MagicMock()
        planner.run.return_value = _ok_result()
        registry.get_or_create.return_value = planner

        display = MagicMock()
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )
        orch.run("goal", max_iterations=3)

        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        assert any("iteration limit" in m and "3" in m for m in status_msgs)

    def test_run_stops_on_interrupt(self):
        """SIGINT mid-loop causes the next iteration to exit with the interrupt message."""
        # Always needs planning — the loop would run forever without an interrupt.
        tm = _make_task_manager(
            root_tasks=[],
            all_tasks=[],
            find_results={"blocked": [], "in_review": []},
            find_incomplete=[_make_task_summary("t1")],
        )
        registry = _make_agent_registry(has={"executor": True, "planner": True})
        planner = MagicMock()

        # Simulate SIGINT firing during the planner call on the first step.
        def _planner_run(prompt):
            orch._interrupted = True
            return _ok_result()

        planner.run.side_effect = _planner_run
        registry.get_or_create.return_value = planner

        display = MagicMock()
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )
        orch.run("goal", max_iterations=10)

        # The interrupt message must appear somewhere in the status calls.
        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        assert any("Interrupted" in m for m in status_msgs)

    def test_run_invokes_reviewer_when_task_in_review(self):
        """Tasks with status 'in_review' are handled by the reviewer first."""
        in_review_task = _make_task_summary("t1", status="in_review")
        tm = _make_task_manager(
            root_tasks=[_make_task_summary("t1", has_subtasks=False, status="done")],
            all_tasks=[],
            find_results={
                "in_review": [in_review_task],
                "blocked": [],
            },
            find_incomplete=[],
        )
        registry = _make_agent_registry(has={"executor": True, "reviewer": True})
        reviewer = MagicMock()
        reviewer.run.return_value = _ok_result()

        # After the first review step the in_review list becomes empty so
        # the loop exits.
        call_count = [0]

        def _find_side(status):
            if status == "in_review":
                call_count[0] += 1
                if call_count[0] == 1:
                    return [in_review_task]
                return []
            return []

        tm.find.side_effect = _find_side
        registry.get_or_create.return_value = reviewer

        display = MagicMock()
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )
        orch.run("goal", max_iterations=5)

        assert reviewer.run.called

    def test_run_no_reviewer_skips_review_phase(self):
        """When no reviewer is configured, in_review tasks are not processed."""
        tm = _make_task_manager(
            root_tasks=[_make_task_summary("t1", has_subtasks=False, status="done")],
            all_tasks=[],
            find_results={
                "blocked": [],
                "in_review": [_make_task_summary("t2", status="in_review")],
            },
            find_incomplete=[],
        )
        registry = _make_agent_registry(has={"executor": True, "reviewer": False})
        agent = MagicMock()
        agent.run.return_value = _ok_result()
        registry.get_or_create.return_value = agent

        display = MagicMock()
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )
        orch.run("goal", max_iterations=2)

        # registry.has("reviewer") returns False — get_or_create("reviewer") never called
        for call_args in registry.get_or_create.call_args_list:
            assert call_args[0][0] != "reviewer"

    def test_run_shows_incomplete_when_no_executable_tasks(self):
        """When there are incomplete tasks but none are executable, show status."""
        # Root has subtask (has_subtasks=True, done=True) so _needs_planning=False.
        # But all_tasks returns tasks that have subtask_ids → no leaves.
        tm = _make_task_manager(
            root_tasks=[_make_task_summary("t1", status="done", has_subtasks=True)],
            all_tasks=[_make_task_summary("t2", status="blocked", subtask_ids=[])],
            find_results={"blocked": [], "in_review": []},
            find_incomplete=[_make_task_summary("t2", status="blocked")],
        )
        registry = _make_agent_registry(has={"executor": True})
        display = MagicMock()
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )
        orch.run("goal")

        msg = display.show_status.call_args_list[-1][0][0]
        assert "No executable tasks" in msg
        assert "1" in msg


# ---------------------------------------------------------------------------
# Context limit handling
# ---------------------------------------------------------------------------


class TestContextLimitHandling:
    def _make_executor_registry(self, results: list[AgentResult]) -> MagicMock:
        registry = _make_agent_registry(has={"executor": True})
        agent = MagicMock()
        agent.run.side_effect = list(results)
        registry.get_or_create.return_value = agent
        return registry

    def _make_single_task_tm(
        self, task_id: str = "t1", executor_calls: int = 2
    ) -> MagicMock:
        """Make a TaskManager mock that serves the task exactly *executor_calls* times.

        After that, all_tasks() returns [] so _pick_next_task returns None and
        the loop exits cleanly without exhausting the agent result list.
        """
        task = _make_task_summary(task_id, status="not_started")
        detail = _make_task_detail(task_id)
        call_count = [0]

        def _all_tasks() -> list:
            call_count[0] += 1
            if call_count[0] <= executor_calls:
                return [task]
            return []

        tm = _make_task_manager(
            root_tasks=[_make_task_summary(task_id, has_subtasks=False, status="done")],
            find_results={"blocked": [], "in_review": []},
            find_incomplete=[],
            task_detail=detail,
        )
        tm.all_tasks.side_effect = _all_tasks
        return tm

    def test_context_limit_adds_note(self):
        tm = self._make_single_task_tm()
        results = [_context_limit_result(), _ok_result()]
        registry = self._make_executor_registry(results)

        orch = _make_orchestrator(task_manager=tm, agent_registry=registry)
        orch.run("goal", max_iterations=5)

        # add_note called at least once
        assert tm.add_note.called
        note_text = tm.add_note.call_args[0][1]
        assert "context limit" in note_text.lower()

    def test_context_limit_resets_executor_session(self):
        tm = self._make_single_task_tm()
        results = [_context_limit_result(), _ok_result()]
        registry = self._make_executor_registry(results)

        orch = _make_orchestrator(task_manager=tm, agent_registry=registry)
        orch.run("goal", max_iterations=5)

        registry.reset.assert_called_with("executor")

    def test_three_context_limits_block_task(self):
        """After _CONTEXT_LIMIT_STRIKES consecutive context limits the task is blocked."""
        tm = self._make_single_task_tm("t1", executor_calls=_CONTEXT_LIMIT_STRIKES)
        results = [_context_limit_result()] * _CONTEXT_LIMIT_STRIKES
        registry = self._make_executor_registry(results)

        display = MagicMock()
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )
        orch.run("goal", max_iterations=_CONTEXT_LIMIT_STRIKES + 2)

        tm.update_task.assert_called_once()
        call_kwargs = tm.update_task.call_args[1]
        assert call_kwargs["status"] == "blocked"
        assert "context limit" in call_kwargs["blockers"][0].lower()

    def test_tool_limit_adds_note_and_clears_strikes(self):
        """tool_limit adds a note, does not block, and clears context-limit strikes."""
        tm = self._make_single_task_tm("t1", executor_calls=2)
        # One context_limit then tool_limit — counter should be cleared after tool_limit
        results = [
            _context_limit_result(),
            AgentResult(text="partial", status="tool_limit", partial=True),
        ]
        registry = self._make_executor_registry(results)

        orch = _make_orchestrator(task_manager=tm, agent_registry=registry)
        orch.run("goal", max_iterations=5)

        assert tm.add_note.called
        notes = [c[0][1] for c in tm.add_note.call_args_list]
        assert any("tool" in n.lower() for n in notes)
        tm.update_task.assert_not_called()
        assert "t1" not in orch._context_limit_counts

    def test_error_result_adds_note_with_error_message_and_clears_strikes(self):
        """error result includes error_message in note, clears strikes, does not block."""
        tm = self._make_single_task_tm("t1", executor_calls=2)
        results = [
            _context_limit_result(),
            AgentResult(text="", status="error", error_message="LLM down"),
        ]
        registry = self._make_executor_registry(results)

        orch = _make_orchestrator(task_manager=tm, agent_registry=registry)
        orch.run("goal", max_iterations=5)

        assert tm.add_note.called
        notes = [c[0][1] for c in tm.add_note.call_args_list]
        assert any("LLM down" in n for n in notes)
        tm.update_task.assert_not_called()
        assert "t1" not in orch._context_limit_counts

    def test_context_limit_counter_cleared_on_success(self):
        """A successful run clears the strike count for the task."""
        tm = self._make_single_task_tm("t1", executor_calls=3)
        # Two failures then success
        results = [
            _context_limit_result(),
            _context_limit_result(),
            _ok_result(),
        ]
        registry = self._make_executor_registry(results)

        orch = _make_orchestrator(task_manager=tm, agent_registry=registry)
        orch.run("goal", max_iterations=10)

        # Task was NOT blocked (only 2 strikes before a success)
        for c in tm.update_task.call_args_list:
            assert c[1].get("status") != "blocked"

        # Counter cleared after success
        assert "t1" not in orch._context_limit_counts


class TestExecutorAnswerFallback:
    def test_structured_null_answer_does_not_fallback_to_raw_json(self):
        tm = _make_task_manager(task_detail=_make_task_detail("t1", name="Task"))
        registry = _make_agent_registry(has={"executor": True})
        executor = MagicMock()
        executor.run.return_value = AgentResult(
            text='{"success": false, "summary": "failed", "answer": null, "error_message": "boom"}',
            status="ok",
        )
        registry.get_or_create.return_value = executor
        orch = _make_orchestrator(task_manager=tm, agent_registry=registry)

        result = orch._run_executor({"id": "t1", "name": "Task"})

        assert result.status == "error"
        assert result.text == ""
        assert tm.add_note.called
        note_text = tm.add_note.call_args[0][1]
        assert "Partial progress:" in note_text
        assert '{"success"' not in note_text

    def test_unstructured_answer_falls_back_to_raw_text(self):
        tm = _make_task_manager(task_detail=_make_task_detail("t1", name="Task"))
        registry = _make_agent_registry(has={"executor": True})
        executor = MagicMock()
        executor.run.return_value = AgentResult(
            text="plain partial output",
            status="error",
            error_message="boom",
        )
        registry.get_or_create.return_value = executor
        orch = _make_orchestrator(task_manager=tm, agent_registry=registry)

        result = orch._run_executor({"id": "t1", "name": "Task"})

        assert result.status == "error"
        assert tm.add_note.called
        note_text = tm.add_note.call_args[0][1]
        assert "plain partial output" in note_text

    def test_ok_partial_without_structured_report_is_not_rewritten_to_error(self):
        tm = _make_task_manager(task_detail=_make_task_detail("t1", name="Task"))
        registry = _make_agent_registry(has={"executor": True})
        executor = MagicMock()
        executor.run.return_value = AgentResult(
            text="partial stream output",
            status="ok",
            partial=True,
        )
        registry.get_or_create.return_value = executor
        orch = _make_orchestrator(task_manager=tm, agent_registry=registry)

        result = orch._run_executor({"id": "t1", "name": "Task"})

        assert result.status == "ok"
        assert result.partial is True
        tm.add_note.assert_not_called()


# ---------------------------------------------------------------------------
# AgentRegistry.has() and AgentRegistry.reset()
# ---------------------------------------------------------------------------


class TestAgentRegistryExtensions:
    def test_has_returns_true_when_spec_exists(self):
        from ai_cli.core.agent import AgentSpec
        from ai_cli.core.agent_registry import AgentRegistry

        spec = AgentSpec(
            name="planner",
            system_message="Plan things.",
            tools=[],
            model="llama3",
        )
        registry = AgentRegistry({"planner": spec})
        assert registry.has("planner") is True

    def test_has_returns_false_for_unknown_name(self):
        from ai_cli.core.agent_registry import AgentRegistry

        registry = AgentRegistry({})
        assert registry.has("executor") is False

    def test_reset_removes_cached_instance(self):
        from ai_cli.core.agent import AgentSpec
        from ai_cli.core.agent_registry import AgentRegistry

        spec = AgentSpec(
            name="executor",
            system_message="Execute.",
            tools=[],
            model="llama3",
            persistence="session",
        )
        registry = AgentRegistry({"executor": spec})
        fake_agent = MagicMock()
        fake_agent.spec = spec
        # Manually inject a cached instance
        registry._instances["executor"] = fake_agent

        registry.reset("executor")
        assert "executor" not in registry._instances

    def test_reset_unknown_name_is_noop(self):
        from ai_cli.core.agent_registry import AgentRegistry

        registry = AgentRegistry({})
        # Should not raise
        registry.reset("nonexistent")


# ---------------------------------------------------------------------------
# /plan command in REPL
# ---------------------------------------------------------------------------


class TestPlanCommand:
    def _make_repl_with_plan(self, task_manager=None, agent_registry=None, config=None):
        from ai_cli.cli.repl import REPL

        repl = REPL(
            session=MagicMock(),
            tool_registry=MagicMock(),
            llm_client=MagicMock(),
            display=MagicMock(),
            workspace=MagicMock(),
            config=config or MagicMock(),
            agent_registry=agent_registry,
            task_manager=task_manager,
        )
        return repl

    def test_no_task_manager_shows_error(self):
        repl = self._make_repl_with_plan(task_manager=None)
        repl._handle_slash_command("plan")
        repl._display.show_error.assert_called_once()
        assert "task manager" in repl._display.show_error.call_args[0][0].lower()

    def test_no_config_shows_error(self):
        """REPL constructed without a config shows a clear error instead of AssertionError."""
        from ai_cli.cli.repl import REPL

        repl = REPL(
            session=MagicMock(),
            tool_registry=MagicMock(),
            llm_client=MagicMock(),
            display=MagicMock(),
            workspace=MagicMock(),
            config=None,
            agent_registry=self._make_full_registry(),
            task_manager=MagicMock(),
        )
        repl._task_manager.get_goal.return_value = "goal"
        repl._handle_slash_command("plan")
        repl._display.show_error.assert_called_once()
        assert "config" in repl._display.show_error.call_args[0][0].lower()

    def test_no_agent_registry_shows_error(self):
        repl = self._make_repl_with_plan(task_manager=MagicMock(), agent_registry=None)
        repl._handle_slash_command("plan")
        repl._display.show_error.assert_called_once()

    def test_empty_agent_registry_shows_error(self):
        from ai_cli.core.agent_registry import AgentRegistry

        repl = self._make_repl_with_plan(
            task_manager=MagicMock(),
            agent_registry=AgentRegistry({}),
        )
        repl._handle_slash_command("plan")
        repl._display.show_error.assert_called_once()

    def test_no_executor_agent_shows_error(self):
        from ai_cli.core.agent import AgentSpec
        from ai_cli.core.agent_registry import AgentRegistry

        spec = AgentSpec(
            name="planner", system_message="Plan.", tools=[], model="llama3"
        )
        repl = self._make_repl_with_plan(
            task_manager=MagicMock(),
            agent_registry=AgentRegistry({"planner": spec}),
        )
        repl._handle_slash_command("plan")
        repl._display.show_error.assert_called_once()
        assert "executor" in repl._display.show_error.call_args[0][0].lower()

    def test_no_planner_agent_shows_error(self):
        from ai_cli.core.agent import AgentSpec
        from ai_cli.core.agent_registry import AgentRegistry

        spec = AgentSpec(
            name="executor", system_message="Execute.", tools=[], model="llama3"
        )
        repl = self._make_repl_with_plan(
            task_manager=MagicMock(),
            agent_registry=AgentRegistry({"executor": spec}),
        )
        repl._handle_slash_command("plan")
        repl._display.show_error.assert_called_once()
        assert "planner" in repl._display.show_error.call_args[0][0].lower()

    def test_no_goal_and_none_stored_shows_error(self):
        from ai_cli.core.agent import AgentSpec
        from ai_cli.core.agent_registry import AgentRegistry

        specs = {
            name: AgentSpec(name=name, system_message=".", tools=[], model="llama3")
            for name in ("executor", "planner")
        }
        tm = MagicMock()
        tm.get_goal.return_value = None

        repl = self._make_repl_with_plan(
            task_manager=tm,
            agent_registry=AgentRegistry(specs),
        )
        repl._handle_slash_command("plan")
        repl._display.show_error.assert_called_once()
        assert "goal" in repl._display.show_error.call_args[0][0].lower()

    def _make_full_registry(self) -> object:
        from ai_cli.core.agent import AgentSpec
        from ai_cli.core.agent_registry import AgentRegistry

        return AgentRegistry(
            {
                name: AgentSpec(name=name, system_message=".", tools=[], model="llama3")
                for name in ("executor", "planner")
            }
        )

    def test_plan_with_goal_creates_orchestrator_and_runs(self):
        tm = MagicMock()

        repl = self._make_repl_with_plan(
            task_manager=tm,
            agent_registry=self._make_full_registry(),
        )

        with patch(
            "ai_cli.core.task_orchestrator.TaskOrchestrator"
        ) as MockOrchestrator:
            mock_orch = MagicMock()
            MockOrchestrator.return_value = mock_orch

            repl._handle_slash_command('plan "implement the feature"')

            MockOrchestrator.assert_called_once()
            mock_orch.run.assert_called_once_with(
                "implement the feature", autonomous=False
            )

    def test_plan_without_goal_uses_stored_goal(self):
        tm = MagicMock()
        tm.get_goal.return_value = "stored goal"

        repl = self._make_repl_with_plan(
            task_manager=tm,
            agent_registry=self._make_full_registry(),
        )

        with patch(
            "ai_cli.core.task_orchestrator.TaskOrchestrator"
        ) as MockOrchestrator:
            mock_orch = MagicMock()
            MockOrchestrator.return_value = mock_orch

            repl._handle_slash_command("plan")

            mock_orch.run.assert_called_once_with("stored goal", autonomous=False)

    def test_plan_reuses_existing_orchestrator(self):
        """Second /plan call reuses the same orchestrator instance."""
        tm = MagicMock()
        tm.get_goal.return_value = "stored goal"

        repl = self._make_repl_with_plan(
            task_manager=tm,
            agent_registry=self._make_full_registry(),
        )

        with patch(
            "ai_cli.core.task_orchestrator.TaskOrchestrator"
        ) as MockOrchestrator:
            mock_orch = MagicMock()
            MockOrchestrator.return_value = mock_orch

            repl._handle_slash_command("plan")
            repl._handle_slash_command("plan")

            # Constructor called only once
            assert MockOrchestrator.call_count == 1
            assert mock_orch.run.call_count == 2

    def test_task_storage_error_is_caught_and_shown(self):
        """TaskStorageError from orchestrator.run is caught and shown, not re-raised."""
        from ai_cli.core.task_manager import TaskStorageError

        tm = MagicMock()
        tm.get_goal.return_value = "goal"

        repl = self._make_repl_with_plan(
            task_manager=tm,
            agent_registry=self._make_full_registry(),
        )

        with patch(
            "ai_cli.core.task_orchestrator.TaskOrchestrator"
        ) as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.run.side_effect = TaskStorageError("corrupt tasks.json")
            MockOrchestrator.return_value = mock_orch

            repl._handle_slash_command("plan")

            repl._display.show_error.assert_called_once()
            assert "storage" in repl._display.show_error.call_args[0][0].lower()

    def test_key_error_from_agent_registry_is_caught_and_shown(self):
        """KeyError from get_or_create (missing agent) is caught, not re-raised."""
        tm = MagicMock()
        tm.get_goal.return_value = "goal"

        repl = self._make_repl_with_plan(
            task_manager=tm,
            agent_registry=self._make_full_registry(),
        )

        with patch(
            "ai_cli.core.task_orchestrator.TaskOrchestrator"
        ) as MockOrchestrator:
            mock_orch = MagicMock()
            mock_orch.run.side_effect = KeyError("reviewer")
            MockOrchestrator.return_value = mock_orch

            repl._handle_slash_command("plan")

            repl._display.show_error.assert_called_once()
            assert "agent" in repl._display.show_error.call_args[0][0].lower()

    def test_plan_in_slash_commands_list(self):
        from ai_cli.cli.repl import _SLASH_COMMANDS

        names = [cmd.lstrip("/").split()[0] for cmd, _ in _SLASH_COMMANDS]
        assert "plan" in names

    def test_autonomous_flag_passed_to_run(self):
        """``/plan --autonomous`` passes autonomous=True to orchestrator.run()."""
        tm = MagicMock()
        tm.get_goal.return_value = "goal"

        repl = self._make_repl_with_plan(
            task_manager=tm,
            agent_registry=self._make_full_registry(),
        )

        with patch(
            "ai_cli.core.task_orchestrator.TaskOrchestrator"
        ) as MockOrchestrator:
            mock_orch = MagicMock()
            MockOrchestrator.return_value = mock_orch

            repl._handle_slash_command("plan --autonomous")

            mock_orch.run.assert_called_once_with("goal", autonomous=True)

    def test_autonomous_flag_with_goal(self):
        """``/plan "goal" --autonomous`` strips the flag and passes both correctly."""
        tm = MagicMock()

        repl = self._make_repl_with_plan(
            task_manager=tm,
            agent_registry=self._make_full_registry(),
        )

        with patch(
            "ai_cli.core.task_orchestrator.TaskOrchestrator"
        ) as MockOrchestrator:
            mock_orch = MagicMock()
            MockOrchestrator.return_value = mock_orch

            repl._handle_slash_command('plan "build the thing" --autonomous')

            mock_orch.run.assert_called_once_with("build the thing", autonomous=True)


# ---------------------------------------------------------------------------
# Plan checkpoint
# ---------------------------------------------------------------------------


def _make_single_executable_tm() -> MagicMock:
    """TaskManager with a root task containing one executable leaf — no planning needed.

    Structure:
      root  (has_subtasks=True, subtask_ids=["leaf"]) — not a candidate for execution
      └── leaf  (has_subtasks=False, subtask_ids=[])  — picked by _pick_next_task
    """
    root = _make_task_summary(
        "root", status="not_started", has_subtasks=True, subtask_ids=["leaf"]
    )
    leaf = _make_task_summary(
        "leaf", status="not_started", has_subtasks=False, subtask_ids=[]
    )
    tm = _make_task_manager(
        root_tasks=[root],
        all_tasks=[root, leaf],
        find_results={"blocked": [], "in_review": []},
        find_incomplete=[root, leaf],
        task_detail=_make_task_detail("leaf"),
        goal="test goal",
    )
    tm.get_all_task_details_map.return_value = {
        "root": {
            "id": "root",
            "name": "Root Task",
            "status": "not_started",
            "priority": "medium",
            "description": "Root",
            "parent_id": None,
            "subtasks": [{"id": "leaf", "status": "not_started"}],
        },
        "leaf": {
            "id": "leaf",
            "name": "Leaf Task",
            "status": "not_started",
            "priority": "medium",
            "description": "A task",
            "parent_id": "root",
            "subtasks": [],
        },
    }
    return tm


class TestPlanCheckpoint:
    def _make_orch_with_executor(self, tm, display):
        registry = _make_agent_registry(has={"executor": True})
        executor = MagicMock()
        executor.run.return_value = _ok_result()
        registry.get_or_create.return_value = executor
        return _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )

    def test_checkpoint_shown_before_first_execution(self):
        """confirm_plan is called before the first executor call when not autonomous."""
        tm = _make_single_executable_tm()
        display = MagicMock()
        display.confirm_plan.return_value = True
        orch = self._make_orch_with_executor(tm, display)

        orch.run("goal")

        display.confirm_plan.assert_called_once()

    def test_checkpoint_skipped_when_autonomous(self):
        """confirm_plan is never called when autonomous=True."""
        tm = _make_single_executable_tm()
        display = MagicMock()
        orch = self._make_orch_with_executor(tm, display)

        orch.run("goal", autonomous=True)

        display.confirm_plan.assert_not_called()

    def test_user_decline_stops_loop_before_execution(self):
        """When the user declines the checkpoint, no executor call is made."""
        tm = _make_single_executable_tm()
        display = MagicMock()
        display.confirm_plan.return_value = False
        registry = _make_agent_registry(has={"executor": True})
        executor = MagicMock()
        registry.get_or_create.return_value = executor
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )

        orch.run("goal")

        executor.run.assert_not_called()

    def test_user_decline_shows_cancel_message(self):
        """Declining the checkpoint shows a status message with 'Cancelled'."""
        tm = _make_single_executable_tm()
        display = MagicMock()
        display.confirm_plan.return_value = False
        orch = self._make_orch_with_executor(tm, display)

        orch.run("goal")

        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        assert any("Cancelled" in m for m in status_msgs)

    def test_user_confirm_proceeds_to_execution(self):
        """Confirming the checkpoint allows the executor to run."""
        root = _make_task_summary(
            "root", status="not_started", has_subtasks=True, subtask_ids=["leaf"]
        )
        leaf = _make_task_summary(
            "leaf", status="not_started", has_subtasks=False, subtask_ids=[]
        )
        tm = _make_single_executable_tm()
        # Return root+leaf once, then nothing — stops the loop after one execution.
        tm.all_tasks.side_effect = [
            [root, leaf],  # first _pick_next_task call
            [],  # second call → no more tasks → loop exits
        ]

        display = MagicMock()
        display.confirm_plan.return_value = True
        registry = _make_agent_registry(has={"executor": True})
        executor = MagicMock()
        executor.run.return_value = _ok_result()
        registry.get_or_create.return_value = executor
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )

        orch.run("goal")

        executor.run.assert_called_once()

    def test_checkpoint_not_shown_again_after_approval(self):
        """Once approved, confirm_plan is not called again for subsequent tasks."""
        # Consistent structure: one root with two leaf children.
        root = _make_task_summary(
            "root", status="not_started", has_subtasks=True, subtask_ids=["t1", "t2"]
        )
        leaf1 = _make_task_summary(
            "t1", status="not_started", has_subtasks=False, subtask_ids=[]
        )
        leaf2 = _make_task_summary(
            "t2", status="not_started", has_subtasks=False, subtask_ids=[]
        )

        call_count = {"n": 0}

        def _all_tasks_side_effect():
            call_count["n"] += 1
            # First two calls: all three tasks (root + both leaves).
            # Subsequent calls: only root + leaf2 (simulates leaf1 being done).
            if call_count["n"] <= 2:
                return [root, leaf1, leaf2]
            return [root, leaf2]

        tm = _make_task_manager(
            root_tasks=[root],
            find_results={"blocked": [], "in_review": []},
            find_incomplete=[root, leaf1, leaf2],
            task_detail=_make_task_detail("t1"),
            goal="test goal",
        )
        tm.all_tasks.side_effect = _all_tasks_side_effect
        tm.get_all_task_details_map.return_value = {
            "root": {
                "id": "root",
                "name": "Root",
                "status": "not_started",
                "priority": "medium",
                "description": "",
                "parent_id": None,
                "subtasks": [
                    {"id": "t1", "status": "not_started"},
                    {"id": "t2", "status": "not_started"},
                ],
            },
            "t1": {
                "id": "t1",
                "name": "Task1",
                "status": "not_started",
                "priority": "medium",
                "description": "",
                "parent_id": "root",
                "subtasks": [],
            },
            "t2": {
                "id": "t2",
                "name": "Task2",
                "status": "not_started",
                "priority": "medium",
                "description": "",
                "parent_id": "root",
                "subtasks": [],
            },
        }

        display = MagicMock()
        display.confirm_plan.return_value = True
        registry = _make_agent_registry(has={"executor": True})
        executor = MagicMock()
        executor.run.return_value = _ok_result()
        registry.get_or_create.return_value = executor
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )

        orch.run("goal", max_iterations=10)

        # confirm_plan called exactly once despite multiple executor invocations
        assert display.confirm_plan.call_count == 1

    def test_plan_approved_reset_between_run_calls(self):
        """_plan_approved is reset at the start of each run() call."""
        tm = _make_single_executable_tm()
        display = MagicMock()
        display.confirm_plan.return_value = True
        orch = self._make_orch_with_executor(tm, display)

        orch.run("goal")
        orch.run("goal")

        # checkpoint shown once per run() call
        assert display.confirm_plan.call_count == 2


# ---------------------------------------------------------------------------
# Planner progress detection / status summary / error surfacing / logging
# ---------------------------------------------------------------------------


def _planner_progress_orch(
    *,
    root_tasks_sequence: list[list[dict]] | None = None,
    all_tasks_sequence: list[list[dict]] | None = None,
    blocked_sequence: list[list[dict]] | None = None,
    planner_result: AgentResult | None = None,
    planner_side_effect=None,
) -> tuple[TaskOrchestrator, MagicMock, MagicMock, MagicMock]:
    """Build an orchestrator wired for planner-progress tests.

    Each ``*_sequence`` argument supplies the value returned by the
    corresponding ``TaskManager`` call on successive invocations.  The first
    entry models the state seen by the planning heuristic before the planner
    runs; the second entry models the state seen by the no-progress check
    after the planner returns.  When the sequence is shorter than the number
    of calls, the final entry is reused (mirrors how ``side_effect`` raises
    ``StopIteration`` only when fully exhausted).

    Returns ``(orchestrator, task_manager, planner, display)``.
    """
    if root_tasks_sequence is None:
        root_tasks_sequence = [[], []]
    if all_tasks_sequence is None:
        all_tasks_sequence = [[], []]
    if blocked_sequence is None:
        blocked_sequence = [[], []]

    def _make_iter(seq: list[list[dict]]) -> object:
        last = seq[-1] if seq else []

        def _next(*args, **kwargs):
            if seq:
                return seq.pop(0)
            return last

        return _next

    tm = _make_task_manager()
    tm.list_tasks.side_effect = _make_iter(list(root_tasks_sequence))
    tm.all_tasks.side_effect = _make_iter(list(all_tasks_sequence))

    blocked_iter = _make_iter(list(blocked_sequence))

    def _find(status):
        if status == "blocked":
            return blocked_iter()
        return []

    tm.find.side_effect = _find
    tm.find_incomplete.return_value = []

    registry = _make_agent_registry(has={"executor": True, "planner": True})
    planner = MagicMock()
    if planner_side_effect is not None:
        planner.run.side_effect = planner_side_effect
    else:
        planner.run.return_value = planner_result or _ok_result()
    registry.get_or_create.return_value = planner

    display = MagicMock()
    orch = _make_orchestrator(task_manager=tm, agent_registry=registry, display=display)
    return orch, tm, planner, display


class TestPlannerProgressDetection:
    """Step 1+2: detect when the planner failed to mutate the task tree.

    A planner that produces no task-tree changes must not be retried at the
    same snapshot — but the rest of the loop (execution, review) must still
    run.  Aborting the whole loop on a no-op planner regression the user
    reported in alkanen/lms-cli#91 — well-formed task trees can legitimately
    leave the planner with nothing to do.
    """

    def test_planner_no_progress_does_not_loop(self):
        """A planner returning ok with no mutations must not be retried forever.

        With no executable tasks AND no planner progress, the loop terminates
        via the execution phase's existing "all tasks complete" / "no
        executable tasks" messages — not by aborting from the planning branch.
        """
        empty = []
        orch, _tm, planner, display = _planner_progress_orch(
            root_tasks_sequence=[empty, empty, empty, empty],
            all_tasks_sequence=[empty, empty, empty, empty],
            blocked_sequence=[empty, empty, empty, empty],
            planner_result=_ok_result(),
        )

        orch.run("goal", max_iterations=10)

        # Planner called exactly once — the snapshot guard suppresses the
        # second call at the same tree state.
        assert planner.run.call_count == 1

        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        # User sees the per-step "no task changes" summary.
        assert any(
            "no" in m.lower() and ("change" in m.lower() or "progress" in m.lower())
            for m in status_msgs
        )

    def test_planner_creating_root_tasks_continues_loop(self):
        """When the planner mutates the heuristic-relevant fields, the loop continues.

        Sequences the snapshot so iteration 0 sees an empty tree (planner
        runs, "creates" a root, snapshot changes — counted as progress) and
        iteration 1 sees the new root unchanged (planner runs again at the
        new state, this time produces no changes — the snapshot guard then
        kicks in and the loop falls through to execution).  Verifies the
        planner is called once per *distinct* tree state, not once per
        iteration.
        """
        from unittest.mock import MagicMock

        new_root = _make_task_summary("r1", status="not_started", has_subtasks=False)
        leaf = _make_task_summary(
            "r1", status="not_started", has_subtasks=False, subtask_ids=[]
        )

        # State the orchestrator sees grows from "empty" to "{new_root}" on
        # the first planner call, then stays put.
        list_tasks_returns = [[], [new_root]]
        all_tasks_returns = [[], [leaf]]

        def _next_or_last(seq: list[list[dict]]) -> object:
            def _impl(*_args, **_kwargs):
                if len(seq) > 1:
                    return seq.pop(0)
                return seq[0]

            return _impl

        tm = _make_task_manager()
        tm.list_tasks.side_effect = _next_or_last(list_tasks_returns)
        tm.all_tasks.side_effect = _next_or_last(all_tasks_returns)
        tm.find.side_effect = lambda status: []
        tm.find_incomplete.return_value = []
        tm.get_task.return_value = _make_task_detail("r1")

        registry = _make_agent_registry(has={"executor": True, "planner": True})
        planner = MagicMock()
        planner.run.return_value = _ok_result()
        executor = MagicMock()
        executor.run.return_value = _ok_result()

        def _get_or_create(name, **_kwargs):
            return planner if name == "planner" else executor

        registry.get_or_create.side_effect = _get_or_create

        display = MagicMock()
        display.confirm_plan.return_value = True
        tm.get_all_task_details_map.return_value = {
            "r1": {
                "id": "r1",
                "name": "Root r1",
                "status": "not_started",
                "priority": "medium",
                "description": "",
                "parent_id": None,
                "subtasks": [],
            }
        }

        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )

        orch.run("goal", max_iterations=10)

        # Planner called at least twice — once at the empty snapshot
        # (progress) and once at the {new_root} snapshot (no progress).  May
        # be exactly 2 depending on how many distinct snapshots the loop
        # observes; any further calls would mean the snapshot guard failed.
        assert planner.run.call_count == 2

    def test_planner_only_adds_notes_counts_as_no_progress(self):
        """Step 2: the snapshot ignores fields the heuristic does not consult.

        A planner that mutates only notes / descriptions / priorities has not
        changed any of the inputs to ``_needs_planning`` — it would loop
        forever without the snapshot guard.
        """
        empty = []
        orch, tm, planner, display = _planner_progress_orch(
            root_tasks_sequence=[empty, empty, empty, empty],
            all_tasks_sequence=[empty, empty, empty, empty],
            blocked_sequence=[empty, empty, empty, empty],
            planner_result=_ok_result(),
        )

        # The planner "adds notes" via a side effect — this should NOT count
        # as progress because the heuristic does not read the notes field.
        def _planner_run(_prompt):
            tm.add_note("anything", "I tried but did not change the tree")
            return _ok_result()

        planner.run.side_effect = _planner_run

        orch.run("goal", max_iterations=10)

        # The snapshot guard suppresses retries at the same tree state, so
        # the planner is called exactly once despite max_iterations=10.
        assert planner.run.call_count == 1
        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        assert any(
            "no" in m.lower() and ("change" in m.lower() or "progress" in m.lower())
            for m in status_msgs
        )

    def test_planner_no_progress_runs_executor_when_tasks_executable(self):
        """The user's bug from alkanen/lms-cli#91 follow-up.

        When the planner correctly has nothing to do but executable leaf
        tasks already exist in the tree, the orchestrator must defer to the
        executor instead of aborting.  Reproduces the scenario where a fully
        manual task tree is presented to /plan: the planning heuristic still
        fires (a leaf-only root task triggers it), the planner makes no
        changes, and we expect the executor to pick up the leaf and run it.
        """
        # A leaf-only root task: also a leaf, ready to execute.  The planning
        # heuristic returns True (root has no subtasks and is not done) but
        # the planner can't decompose it any further — execution should take
        # over.
        leaf_root = _make_task_summary(
            "leaf_root",
            status="not_started",
            has_subtasks=False,
            subtask_ids=[],
        )

        # _planner_progress_orch shares one mock between planner and
        # executor (same registry, same get_or_create stub), so we build
        # the orchestrator by hand to use distinct planner/executor mocks
        # and assert on each independently.
        tm = _make_task_manager()
        # The tree never changes: same leaf-only root every call.
        tm.list_tasks.return_value = [leaf_root]
        tm.all_tasks.return_value = [leaf_root]
        tm.find.side_effect = lambda status: []
        tm.find_incomplete.return_value = [leaf_root]
        tm.get_task.return_value = _make_task_detail("leaf_root")

        registry = _make_agent_registry(has={"executor": True, "planner": True})
        planner = MagicMock()
        planner.run.return_value = _ok_result()
        executor = MagicMock()
        executor.run.return_value = _ok_result()

        def _get_or_create(name, **_kwargs):
            return planner if name == "planner" else executor

        registry.get_or_create.side_effect = _get_or_create

        display = MagicMock()
        # Auto-approve the plan checkpoint so execution proceeds.
        display.confirm_plan.return_value = True
        # Provide the detail map the checkpoint needs.
        tm.get_all_task_details_map.return_value = {
            "leaf_root": {
                "id": "leaf_root",
                "name": "Leaf Root",
                "status": "not_started",
                "priority": "medium",
                "description": "",
                "parent_id": None,
                "subtasks": [],
            }
        }

        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )

        orch.run("goal", max_iterations=5)

        # The critical assertion: the executor actually ran.
        assert executor.run.called, (
            "executor should run when planner has nothing to do but "
            "executable leaf tasks exist"
        )
        # Planner was tried once and then short-circuited by the snapshot
        # guard, not retried in a loop.
        assert planner.run.call_count == 1

    def test_planner_no_progress_falls_through_same_iteration(self):
        """No-progress planning still allows execution when max_iterations is 1."""
        leaf_root = _make_task_summary(
            "leaf_root",
            status="not_started",
            has_subtasks=False,
            subtask_ids=[],
        )

        tm = _make_task_manager()
        tm.list_tasks.return_value = [leaf_root]
        tm.all_tasks.return_value = [leaf_root]
        tm.find.side_effect = lambda status: []
        tm.find_incomplete.return_value = [leaf_root]
        tm.get_task.return_value = _make_task_detail("leaf_root")
        tm.get_all_task_details_map.return_value = {
            "leaf_root": {
                "id": "leaf_root",
                "name": "Leaf Root",
                "status": "not_started",
                "priority": "medium",
                "description": "",
                "parent_id": None,
                "subtasks": [],
            }
        }

        registry = _make_agent_registry(has={"executor": True, "planner": True})
        planner = MagicMock()
        planner.run.return_value = _ok_result()
        executor = MagicMock()
        executor.run.return_value = _ok_result()

        def _get_or_create(name, **_kwargs):
            return planner if name == "planner" else executor

        registry.get_or_create.side_effect = _get_or_create

        display = MagicMock()
        display.confirm_plan.return_value = True

        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )

        orch.run("goal", max_iterations=1)

        assert planner.run.call_count == 1
        assert executor.run.called

    def test_planner_progress_clears_failed_snapshot_memo(self):
        """A successful planning round clears the failed-snapshot memo.

        Otherwise, if a future planning round at a different state produced
        no changes, the loop would mistakenly skip it because the memo from
        an earlier failure was still set.
        """
        # Sequence: empty → no progress → set memo → snapshot still empty
        # next iteration → would loop without guard, so guard kicks in →
        # defer to execution.  This test mainly verifies the memo is *set*
        # so subsequent iterations skip the planner.
        empty = []
        orch, _tm, planner, _display = _planner_progress_orch(
            root_tasks_sequence=[empty, empty, empty, empty, empty],
            all_tasks_sequence=[empty, empty, empty, empty, empty],
            blocked_sequence=[empty, empty, empty, empty, empty],
            planner_result=_ok_result(),
        )

        orch.run("goal", max_iterations=10)

        # The memo prevents repeat planner calls at the same snapshot.
        assert planner.run.call_count == 1
        assert orch._last_failed_planning_snapshot is not None


class TestPlanningStatusSummary:
    """Step 3: per-step status message includes a one-line diff summary."""

    def test_status_message_includes_no_changes_summary(self):
        empty = []
        orch, _tm, _planner, display = _planner_progress_orch(
            root_tasks_sequence=[empty, empty, empty],
            all_tasks_sequence=[empty, empty, empty],
            blocked_sequence=[empty, empty, empty],
            planner_result=_ok_result(),
        )

        orch.run("goal", max_iterations=5)

        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        # The opaque "Step 0: planning" message should be replaced or augmented
        # with a summary saying nothing changed.
        assert any(
            "step 0" in m.lower() and "planning" in m.lower() and "no" in m.lower()
            for m in status_msgs
        )

    def test_status_message_includes_created_count(self):
        new_root = _make_task_summary("r1", status="not_started", has_subtasks=False)
        leaf = _make_task_summary(
            "r1", status="not_started", has_subtasks=False, subtask_ids=[]
        )

        orch, _tm, _planner, display = _planner_progress_orch(
            root_tasks_sequence=[[], [new_root], [new_root], [new_root]],
            all_tasks_sequence=[[], [leaf], [leaf], [leaf]],
            planner_result=_ok_result(),
        )

        orch.run("goal", max_iterations=5)

        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        # The first planning step's status message mentions a created task.
        assert any(
            "step 0" in m.lower()
            and "planning" in m.lower()
            and "1" in m
            and "creat" in m.lower()
            for m in status_msgs
        )


class TestPlannerErrorSurfacing:
    """Step 4: a non-ok planner result is surfaced to the user.

    The loop is *not* aborted on a planner error — executable tasks should
    still get a chance to run via the execution branch.  The snapshot guard
    suppresses retries of the failed planner call at the same tree state.
    """

    def test_planner_error_does_not_loop(self):
        empty = []
        orch, _tm, planner, display = _planner_progress_orch(
            root_tasks_sequence=[empty, empty, empty],
            all_tasks_sequence=[empty, empty, empty],
            blocked_sequence=[empty, empty, empty],
            planner_result=AgentResult(
                text="", status="error", error_message="LLM down"
            ),
        )

        orch.run("goal", max_iterations=10)

        # Planner only called once — no silent retries at the same snapshot.
        assert planner.run.call_count == 1

        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        # Status surfaces the failure mode and the underlying error message.
        assert any("error" in m.lower() for m in status_msgs)
        assert any("LLM down" in m for m in status_msgs)

    def test_planner_context_limit_does_not_loop(self):
        empty = []
        orch, _tm, planner, display = _planner_progress_orch(
            root_tasks_sequence=[empty, empty, empty],
            all_tasks_sequence=[empty, empty, empty],
            blocked_sequence=[empty, empty, empty],
            planner_result=AgentResult(
                text="partial", status="context_limit", partial=True
            ),
        )

        orch.run("goal", max_iterations=10)

        assert planner.run.call_count == 1
        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        assert any("context" in m.lower() for m in status_msgs)

    def test_planner_status_line_uses_normalized_error_summary(self):
        """When planner report says success=false, status and summary stay consistent.

        The orchestrator rewrites the planner AgentResult from ok->error in this
        case. The displayed "→ planner [...]" summary should be derived from the
        normalized error result, not stale success text from the original report.
        """
        empty = []
        orch, _tm, planner, display = _planner_progress_orch(
            root_tasks_sequence=[empty, empty, empty],
            all_tasks_sequence=[empty, empty, empty],
            blocked_sequence=[empty, empty, empty],
            planner_result=AgentResult(
                text='{"success": false, "summary": "all good", "answer": null, "error_message": "boom"}',
                status="ok",
            ),
        )

        orch.run("goal", max_iterations=5)

        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        planner_lines = [m for m in status_msgs if m.startswith("  → planner")]
        assert planner_lines
        assert any("[error]" in m for m in planner_lines)
        assert all("all good" not in m for m in planner_lines)

    def test_planner_ok_partial_without_structured_report_is_not_normalized(self):
        empty = []
        orch, _tm, planner, display = _planner_progress_orch(
            root_tasks_sequence=[empty, empty, empty],
            all_tasks_sequence=[empty, empty, empty],
            blocked_sequence=[empty, empty, empty],
            planner_result=AgentResult(
                text="interrupted before structured report",
                status="ok",
                partial=True,
            ),
        )

        orch.run("goal", max_iterations=5)

        assert planner.run.call_count == 1
        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        planner_lines = [m for m in status_msgs if m.startswith("  → planner")]
        assert planner_lines
        assert any("[ok]" in m for m in planner_lines)

    def test_planner_tool_limit_does_not_loop(self):
        empty = []
        orch, _tm, planner, display = _planner_progress_orch(
            root_tasks_sequence=[empty, empty, empty],
            all_tasks_sequence=[empty, empty, empty],
            blocked_sequence=[empty, empty, empty],
            planner_result=AgentResult(
                text="partial", status="tool_limit", partial=True
            ),
        )

        orch.run("goal", max_iterations=10)

        assert planner.run.call_count == 1
        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        assert any("tool" in m.lower() for m in status_msgs)

    def test_planner_error_falls_through_same_iteration(self):
        """Planner errors still allow execution when max_iterations is 1."""
        leaf_root = _make_task_summary(
            "leaf_root",
            status="not_started",
            has_subtasks=False,
            subtask_ids=[],
        )

        tm = _make_task_manager()
        tm.list_tasks.return_value = [leaf_root]
        tm.all_tasks.return_value = [leaf_root]
        tm.find.side_effect = lambda status: []
        tm.find_incomplete.return_value = [leaf_root]
        tm.get_task.return_value = _make_task_detail("leaf_root")
        tm.get_all_task_details_map.return_value = {
            "leaf_root": {
                "id": "leaf_root",
                "name": "Leaf Root",
                "status": "not_started",
                "priority": "medium",
                "description": "",
                "parent_id": None,
                "subtasks": [],
            }
        }

        registry = _make_agent_registry(has={"executor": True, "planner": True})
        planner = MagicMock()
        planner.run.return_value = AgentResult(
            text="",
            status="error",
            error_message="planner unavailable",
        )
        executor = MagicMock()
        executor.run.return_value = _ok_result()

        def _get_or_create(name, **_kwargs):
            return planner if name == "planner" else executor

        registry.get_or_create.side_effect = _get_or_create

        display = MagicMock()
        display.confirm_plan.return_value = True

        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )

        orch.run("goal", max_iterations=1)

        assert planner.run.call_count == 1
        assert executor.run.called


class TestResultSummary:
    """_result_summary returns a one-line description of an AgentResult."""

    def test_ok_no_text_returns_done(self):
        result = AgentResult(text="", status="ok")
        assert TaskOrchestrator._result_summary(result) == "done"

    def test_ok_with_text_returns_first_line(self):
        result = AgentResult(text="Created foo.py\nand also bar.py", status="ok")
        assert TaskOrchestrator._result_summary(result) == "Created foo.py"

    def test_ok_text_truncated_at_120(self):
        result = AgentResult(text="x" * 200, status="ok")
        assert len(TaskOrchestrator._result_summary(result)) == 120

    def test_ok_partial_returns_partial_label(self):
        result = AgentResult(text="partial", status="ok", partial=True)
        summary = TaskOrchestrator._result_summary(result)
        assert "partial" in summary

    def test_context_limit_label(self):
        result = AgentResult(text="", status="context_limit", partial=True)
        assert "context limit" in TaskOrchestrator._result_summary(result)

    def test_tool_limit_label(self):
        result = AgentResult(text="", status="tool_limit", partial=True)
        assert "tool" in TaskOrchestrator._result_summary(result)

    def test_error_with_message(self):
        result = AgentResult(text="", status="error", error_message="LLM down")
        summary = TaskOrchestrator._result_summary(result)
        assert "error" in summary
        assert "LLM down" in summary

    def test_error_without_message_uses_text(self):
        result = AgentResult(text="something bad", status="error")
        summary = TaskOrchestrator._result_summary(result)
        assert "error" in summary
        assert "something bad" in summary


class TestAgentReportParsing:
    def test_agent_report_parses_valid_json(self):
        orch = _make_orchestrator()
        result = AgentResult(
            text='{"success": true, "summary": "done", "answer": "artifact", "error_message": null}',
            status="ok",
        )

        report = orch._agent_report("executor", result)

        assert report["success"] is True
        assert report["summary"] == "done"
        assert report["answer"] == "artifact"
        assert report["error_message"] is None

    def test_agent_report_handles_markdown_json_block(self):
        orch = _make_orchestrator()
        result = AgentResult(
            text="""```json
{"success": false, "summary": "failed", "answer": null, "error_message": "boom"}
```""",
            status="ok",
        )

        report = orch._agent_report("planner", result)

        assert report["success"] is False
        assert report["summary"] == "failed"
        assert report["answer"] == ""
        assert report["error_message"] == "boom"


class TestExecutorReviewerStatusSummary:
    """Executor and reviewer results are followed by a one-line → summary."""

    def _single_task_tm(self) -> MagicMock:
        """TaskManager that serves one leaf task once then returns nothing."""
        leaf = _make_task_summary("t1", status="not_started", subtask_ids=[])
        call_count = {"n": 0}

        def _all_tasks():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [leaf]
            return []

        root = _make_task_summary("t1", status="done", has_subtasks=False)
        tm = _make_task_manager(
            root_tasks=[root],
            find_results={"blocked": [], "in_review": []},
            find_incomplete=[],
            task_detail=_make_task_detail("t1"),
        )
        tm.all_tasks.side_effect = _all_tasks
        return tm

    def test_executor_ok_shows_arrow_summary(self):
        tm = self._single_task_tm()
        registry = _make_agent_registry(has={"executor": True})
        executor = MagicMock()
        executor.run.return_value = AgentResult(text="Wrote foo.py", status="ok")
        registry.get_or_create.return_value = executor

        display = MagicMock()
        display.confirm_plan.return_value = True
        tm.get_all_task_details_map.return_value = {
            "t1": {
                "id": "t1",
                "name": "Task 1",
                "status": "not_started",
                "priority": "medium",
                "description": "",
                "parent_id": None,
                "subtasks": [],
            }
        }
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )
        orch.run("goal")

        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        assert any(m.startswith("  →") for m in status_msgs)
        assert any("executor [" in m for m in status_msgs if m.startswith("  →"))
        assert any(
            m.startswith("  → executor [ok]:") and m.endswith("s)") for m in status_msgs
        )

    def test_executor_error_shows_error_in_summary(self):
        tm = self._single_task_tm()
        registry = _make_agent_registry(has={"executor": True})
        executor = MagicMock()
        executor.run.return_value = AgentResult(
            text="", status="error", error_message="LLM unavailable"
        )
        registry.get_or_create.return_value = executor

        display = MagicMock()
        display.confirm_plan.return_value = True
        tm.get_all_task_details_map.return_value = {
            "t1": {
                "id": "t1",
                "name": "Task 1",
                "status": "not_started",
                "priority": "medium",
                "description": "",
                "parent_id": None,
                "subtasks": [],
            }
        }
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )
        orch.run("goal")

        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        arrow_msgs = [m for m in status_msgs if m.startswith("  →")]
        assert any("error" in m.lower() for m in arrow_msgs)

    def test_reviewer_shows_arrow_summary(self):
        in_review_task = _make_task_summary("t1", status="in_review")
        root = _make_task_summary("t1", status="done", has_subtasks=False)
        call_count = {"n": 0}

        def _find_side(status):
            if status == "in_review":
                call_count["n"] += 1
                return [in_review_task] if call_count["n"] == 1 else []
            return []

        tm = _make_task_manager(
            root_tasks=[root],
            all_tasks=[],
            find_results={"blocked": [], "in_review": [in_review_task]},
            find_incomplete=[],
            task_detail=_make_task_detail("t1"),
        )
        tm.find.side_effect = _find_side

        registry = _make_agent_registry(has={"executor": True, "reviewer": True})
        reviewer = MagicMock()
        reviewer.run.return_value = AgentResult(text="Task is complete.", status="ok")
        registry.get_or_create.return_value = reviewer

        display = MagicMock()
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )
        orch.run("goal")

        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        assert any(m.startswith("  →") for m in status_msgs)
        assert any("reviewer [" in m for m in status_msgs if m.startswith("  →"))
        assert any(
            m.startswith("  → reviewer [ok]:") and m.endswith("s)") for m in status_msgs
        )

    def test_reviewer_uses_structured_failure_status(self):
        in_review_task = _make_task_summary("t1", status="in_review")
        root = _make_task_summary("t1", status="done", has_subtasks=False)
        call_count = {"n": 0}

        def _find_side(status):
            if status == "in_review":
                call_count["n"] += 1
                return [in_review_task] if call_count["n"] == 1 else []
            return []

        tm = _make_task_manager(
            root_tasks=[root],
            all_tasks=[],
            find_results={"blocked": [], "in_review": [in_review_task]},
            find_incomplete=[],
            task_detail=_make_task_detail("t1"),
        )
        tm.find.side_effect = _find_side

        registry = _make_agent_registry(has={"executor": True, "reviewer": True})
        reviewer = MagicMock()
        reviewer.run.return_value = AgentResult(
            text='{"success": false, "summary": "missing checks", "answer": null, "error_message": "DoD not met"}',
            status="ok",
        )
        registry.get_or_create.return_value = reviewer

        display = MagicMock()
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )
        orch.run("goal")

        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        assert any(
            m.startswith("  → reviewer [error]: missing checks") for m in status_msgs
        )


class TestPlannerLogging:
    """Step 5: structured INFO logging around each planning round."""

    def test_planner_round_emits_info_log_records(self, caplog):
        empty = []
        orch, _tm, _planner, _display = _planner_progress_orch(
            root_tasks_sequence=[empty, empty, empty],
            all_tasks_sequence=[empty, empty, empty],
            blocked_sequence=[empty, empty, empty],
            planner_result=_ok_result(),
        )

        with caplog.at_level(logging.INFO, logger="ai_cli.core.task_orchestrator"):
            orch.run("goal", max_iterations=5)

        records = [
            r
            for r in caplog.records
            if r.name == "ai_cli.core.task_orchestrator" and r.levelno >= logging.INFO
        ]
        assert records, (
            "expected at least one INFO-level log record from the orchestrator"
        )

        log_text = "\n".join(r.getMessage() for r in records)
        # The log records describe the planner round's status and outcome.
        assert "planner" in log_text.lower() or "planning" in log_text.lower()
        assert "ok" in log_text.lower()

    def test_planner_failure_logged_at_info_or_higher(self, caplog):
        empty = []
        orch, _tm, _planner, _display = _planner_progress_orch(
            root_tasks_sequence=[empty, empty, empty],
            all_tasks_sequence=[empty, empty, empty],
            blocked_sequence=[empty, empty, empty],
            planner_result=AgentResult(
                text="", status="error", error_message="LLM down"
            ),
        )

        with caplog.at_level(logging.INFO, logger="ai_cli.core.task_orchestrator"):
            orch.run("goal", max_iterations=5)

        records = [
            r
            for r in caplog.records
            if r.name == "ai_cli.core.task_orchestrator" and r.levelno >= logging.INFO
        ]
        log_text = "\n".join(r.getMessage() for r in records)
        assert "error" in log_text.lower()
