"""Tests for ai_cli.core.task_orchestrator.TaskOrchestrator."""

from __future__ import annotations

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
        # Always has planning to do so the loop never exits naturally.
        tm = _make_task_manager(
            root_tasks=[],
            all_tasks=[],
            find_results={"blocked": [], "in_review": []},
            find_incomplete=[_make_task_summary("t1")],
        )
        registry = _make_agent_registry(has={"executor": True, "planner": True})
        planner = MagicMock()
        planner.run.return_value = _ok_result()
        registry.get_or_create.return_value = planner

        display = MagicMock()
        orch = _make_orchestrator(
            task_manager=tm, agent_registry=registry, display=display
        )
        orch.run("goal", max_iterations=3)

        msg = display.show_status.call_args_list[-1][0][0]
        assert "iteration limit" in msg
        assert "3" in msg

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
            mock_orch.run.assert_called_once_with("implement the feature")

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

            mock_orch.run.assert_called_once_with("stored goal")

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
