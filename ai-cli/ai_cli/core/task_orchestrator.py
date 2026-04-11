"""
task_orchestrator.py — Deterministic plan → execute → review loop for /plan mode.

``TaskOrchestrator.run(goal)`` drives the full loop without any LLM-driven
routing: every routing decision is pure Python; only sub-agent work consumes
the GPU.  The loop is resumable in the sense that the task tree and durable
workflow state are stored in ``tasks.json`` via ``TaskManager``; some
transient in-memory bookkeeping (e.g. the context-limit strike counter) is
recreated when the process restarts.
"""

from __future__ import annotations

import logging
import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_cli.cli.display import Display
    from ai_cli.core.agent import Agent
    from ai_cli.core.agent_registry import AgentRegistry
    from ai_cli.core.config_manager import ConfigManager
    from ai_cli.core.llm_client import LLMClient
    from ai_cli.core.task_manager import TaskManager
    from ai_cli.core.tool_registry import ToolRegistry
    from ai_cli.core.workspace import Workspace

logger = logging.getLogger(__name__)

# Consecutive context-limit hits before a task is force-blocked.
_CONTEXT_LIMIT_STRIKES = 3


class TaskOrchestrator:
    """Deterministic plan → execute → review loop for /plan mode.

    Routing decisions are pure Python (no LLM calls).  Each step dispatches
    exactly one sub-agent via ``Agent.run()``.

    Parameters
    ----------
    task_manager:
        Manages the persistent task tree (``tasks.json``).
    agent_registry:
        Registry used to get/create sub-agents by role name.
    display:
        Display surface for status messages.
    workspace, config, coordinator_llm, global_tool_registry:
        Passed through to :meth:`AgentRegistry.get_or_create` when building
        agents; the orchestrator stores them so callers don't need to pass them
        at each step.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        agent_registry: AgentRegistry,
        display: Display,
        *,
        workspace: Workspace,
        config: ConfigManager,
        coordinator_llm: LLMClient,
        global_tool_registry: ToolRegistry,
    ) -> None:
        self.tm = task_manager
        self.agents = agent_registry
        self.display = display
        self._workspace = workspace
        self._config = config
        self._coordinator_llm = coordinator_llm
        self._global_tool_registry = global_tool_registry
        self._interrupted = False
        # Per-task consecutive context-limit strike counter: {task_id: count}
        self._context_limit_counts: dict[str, int] = {}
        # Set to True once the user approves the plan checkpoint in the current run().
        self._plan_approved: bool = False

    # ------------------------------------------------------------------
    # Routing heuristics (pure Python — no LLM calls)
    # ------------------------------------------------------------------

    def _needs_planning(self) -> bool:
        """Return ``True`` when the planner should be invoked."""
        roots = self.tm.list_tasks(parent_id=None)
        if not roots:
            return True
        # Any root task without subtasks that is not done needs planning
        for t in roots:
            if t["status"] != "done" and not t.get("has_subtasks", False):
                return True
        # Blocked tasks may need decomposition or re-routing, but only trigger
        # planning when there are no other executable leaf tasks — otherwise a
        # single blocked task would starve progress on all remaining work.
        if self.tm.find(status="blocked"):
            return self._pick_next_task() is None
        return False

    def _pick_next_task(self) -> dict | None:
        """Pick an executable leaf task using a stable, deterministic ranking.

        Prefers ``in_progress`` over ``not_started``, high priority over low,
        earlier ``created_at`` over later, and task id as a final tie-breaker.
        Returns ``None`` when no leaf task is executable.
        """
        candidates = []
        for t in self.tm.all_tasks():
            if t["status"] not in ("not_started", "in_progress"):
                continue
            subtask_ids = t.get("subtask_ids", [])
            if not isinstance(subtask_ids, list):
                logger.warning(
                    "Skipping task %r with malformed subtask_ids: expected list, got %s",
                    t.get("id"),
                    type(subtask_ids).__name__,
                )
                continue
            if len(subtask_ids) == 0:
                candidates.append(t)
        if not candidates:
            return None

        priority_rank = {"high": 0, "medium": 1, "low": 2}

        def _sort_key(t: dict) -> tuple:
            status_rank = 0 if t["status"] == "in_progress" else 1
            prio = priority_rank.get(t.get("priority", "medium"), 1)
            timestamp = t.get("created_at", "")
            return (status_rank, prio, timestamp, t["id"])

        return sorted(candidates, key=_sort_key)[0]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self, goal: str, max_iterations: int = 50, *, autonomous: bool = False
    ) -> None:
        """Drive the plan → execute → review loop for *goal*.

        Stores *goal* in ``tasks.json`` (idempotent) so ``/plan`` (no args)
        can resume later.  Installs a ``SIGINT`` handler so Ctrl+C interrupts
        cleanly after the current step, restoring the original handler on exit.

        Parameters
        ----------
        goal:
            The high-level goal string stored in ``tasks.json`` and passed to
            the planner agent.
        max_iterations:
            Hard upper bound on loop iterations.  Prevents runaway loops;
            the user can resume with ``/plan`` after the limit is reached.
        autonomous:
            When ``False`` (default), the orchestrator pauses after the first
            planning round and asks the user to confirm before executing
            anything.  Pass ``True`` (``/plan --autonomous``) to skip the
            checkpoint and run the full loop unattended.
        """
        self.tm.set_goal(goal)
        self._interrupted = False
        self._plan_approved = False

        original_handler = signal.getsignal(signal.SIGINT)

        def _sigint_handler(signum: int, frame: object) -> None:
            self._interrupted = True

        signal.signal(signal.SIGINT, _sigint_handler)

        try:
            for step in range(max_iterations):
                if self._interrupted:
                    self.display.show_status("Interrupted — task tree saved.")
                    return

                # 1. Review phase — handle tasks awaiting review first.
                if self.agents.has("reviewer"):
                    in_review = self.tm.find(status="in_review")
                    if in_review:
                        task = in_review[0]
                        self.display.show_status(
                            f"Step {step}: reviewing '{task['name']}'"
                        )
                        self._run_reviewer(task)
                        continue

                # 2. Planning phase.
                if self._needs_planning():
                    self.display.show_status(f"Step {step}: planning")
                    self._run_planner(goal)
                    continue

                # 3. Execution phase.
                next_task: dict | None = self._pick_next_task()
                if next_task is None:
                    incomplete = self.tm.find_incomplete()
                    if not incomplete:
                        self.display.show_status("All tasks complete.")
                    else:
                        self.display.show_status(
                            f"No executable tasks. {len(incomplete)} task(s) remain "
                            f"incomplete (blocked or awaiting subtask completion). "
                            f"Inspect with /tasks, adjust if needed, then run /plan."
                        )
                    return

                if not autonomous and not self._plan_approved:
                    # Plan checkpoint: pause before the first execution so the
                    # user can review the task tree and confirm before any files
                    # are written.  Subsequent planning rounds (e.g. triggered
                    # by blocked tasks) skip this check because _plan_approved
                    # is already True.
                    depth = self._get_tree_depth()
                    nodes = self._build_tree_nodes(self.tm, max_depth=depth)
                    if self.display.confirm_plan(
                        nodes, self.tm.get_goal(), depth=depth
                    ):
                        self._plan_approved = True
                    else:
                        self.display.show_status(
                            "Cancelled — task tree saved. "
                            "Edit with /tasks, then run /plan to try again."
                        )
                        return

                self.display.show_status(
                    f"Step {step}: executing '{next_task['name']}'"
                )
                self._run_executor(next_task)

            self.display.show_status(
                f"Reached iteration limit ({max_iterations}). Run /plan to continue."
            )
        finally:
            signal.signal(signal.SIGINT, original_handler)

    # ------------------------------------------------------------------
    # Agent dispatch helpers
    # ------------------------------------------------------------------

    def _get_agent(self, name: str) -> Agent:
        """Return an agent by role name, building it if necessary."""
        return self.agents.get_or_create(
            name,
            workspace=self._workspace,
            config=self._config,
            coordinator_llm=self._coordinator_llm,
            global_tool_registry=self._global_tool_registry,
        )

    def _run_planner(self, goal: str) -> None:
        roots = self.tm.list_tasks(parent_id=None)
        blocked = self.tm.find(status="blocked")
        prompt = (
            f"Goal: {goal}\n\nCurrent root tasks:\n{self._format_summaries(roots)}\n\n"
        )
        if blocked:
            prompt += (
                f"Blocked tasks (may need decomposition or re-planning):\n"
                f"{self._format_summaries(blocked)}\n\n"
            )
        prompt += (
            "Break the goal into clear, actionable tasks. "
            "Ensure each task has a meaningful Definition of Done."
        )
        self._get_agent("planner").run(prompt)

    def _run_executor(self, task: dict) -> None:
        detail = self.tm.get_task(task["id"])
        prompt = (
            f"Execute the following task.\n\n"
            f"Task: {detail['name']}\n"
            f"Description: {detail.get('description', '')}\n"
            f"Definition of Done: {detail.get('definition_of_done', '')}\n"
        )
        if detail.get("next_action"):
            prompt += f"Suggested next action: {detail['next_action']}\n"
        if detail.get("notes"):
            prompt += "\nNotes from prior work:\n"
            for note in detail["notes"][-5:]:
                prompt += f"  - {note}\n"

        result = self._get_agent("executor").run(prompt)

        if result.status == "context_limit":
            count = self._context_limit_counts.get(task["id"], 0) + 1
            self._context_limit_counts[task["id"]] = count
            self.tm.add_note(
                task["id"],
                f"Executor hit context limit (attempt {count}/{_CONTEXT_LIMIT_STRIKES}). "
                f"Partial progress: {result.text[:500]}",
            )
            # Always reset the executor session after a context overflow so the
            # next call starts fresh, seeded by the notes above.
            self.agents.reset("executor")
            if count >= _CONTEXT_LIMIT_STRIKES:
                self.tm.update_task(
                    task["id"],
                    status="blocked",
                    blockers=[
                        "Repeated context limit — may need decomposition into "
                        "smaller subtasks."
                    ],
                )
                self._context_limit_counts.pop(task["id"], None)
        elif result.status == "tool_limit":
            self._context_limit_counts.pop(task["id"], None)
            self.tm.add_note(
                task["id"],
                f"Executor hit tool-call round limit. Partial progress: {result.text[:500]}",
            )
        elif result.status == "error":
            self._context_limit_counts.pop(task["id"], None)
            error_detail = (
                f"Error: {result.error_message}. " if result.error_message else ""
            )
            self.tm.add_note(
                task["id"],
                f"Executor encountered an error. {error_detail}"
                f"Partial progress: {result.text[:500]}",
            )
        else:
            # ok — clear any accumulated context-limit strikes.
            self._context_limit_counts.pop(task["id"], None)

    def _run_reviewer(self, task: dict) -> None:
        detail = self.tm.get_task(task["id"])
        prompt = (
            f"Review the following task.\n\n"
            f"Task: {detail['name']}\n"
            f"Definition of Done: {detail.get('definition_of_done', '')}\n"
            f"Notes:\n"
        )
        for note in detail.get("notes", []):
            prompt += f"  - {note}\n"
        prompt += (
            "\nVerify whether the Definition of Done is satisfied. "
            "If yes, mark the task as done. "
            "If not, set it back to in_progress with a note explaining what is missing."
        )
        self._get_agent("reviewer").run(prompt)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tree_nodes(tm: TaskManager, max_depth: int = 3) -> list[dict]:
        """Build display-ready tree node dicts from the current task tree.

        Mirrors the REPL's ``_build_task_tree`` helper but operates directly on
        a :class:`TaskManager` instance.  Used by the plan checkpoint to render
        the task tree before asking for user confirmation.
        """
        detail_map = tm.get_all_task_details_map()

        def _build(task_id: str, current: int) -> dict:
            full = detail_map.get(task_id, {})
            subtasks = full.get("subtasks", [])
            done_count = sum(1 for s in subtasks if s["status"] == "done")
            node: dict = {
                "id": task_id,
                "name": full.get("name", "?"),
                "status": full.get("status", "?"),
                "priority": full.get("priority", "?"),
                "description": full.get("description", ""),
                "subtask_count": len(subtasks),
                "done_subtask_count": done_count,
                "children": None,
            }
            if current < max_depth and subtasks:
                node["children"] = [_build(s["id"], current + 1) for s in subtasks]
            elif not subtasks:
                node["children"] = []
            return node

        return [
            _build(task_id, 1)
            for task_id, detail in detail_map.items()
            if detail.get("parent_id") is None
        ]

    def _get_tree_depth(self) -> int:
        """Return the configured ``tasks.tree_depth`` value (default 3).

        Mirrors :meth:`~ai_cli.cli.repl.REPL._get_tree_depth` so the plan
        checkpoint renders at the same depth as ``/tasks tree``.
        """
        default = 3
        tasks_cfg = self._config.get("tasks")
        if not isinstance(tasks_cfg, dict):
            return default
        raw = tasks_cfg.get("tree_depth", default)
        try:
            depth = int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid tasks.tree_depth value %r; using default %d.", raw, default
            )
            return default
        if depth < 1:
            logger.warning(
                "Invalid tasks.tree_depth value %r; using default %d.", raw, default
            )
            return default
        return depth

    @staticmethod
    def _format_summaries(tasks: list[dict]) -> str:
        if not tasks:
            return "  (none)"
        lines = []
        for t in tasks:
            subtask_marker = " [has subtasks]" if t.get("has_subtasks") else ""
            lines.append(f"  {t['id']}: [{t['status']}] {t['name']}{subtask_marker}")
        return "\n".join(lines)
