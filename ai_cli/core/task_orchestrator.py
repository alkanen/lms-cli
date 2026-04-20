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

import json
import logging
import signal
import time
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from ai_cli.cli.display import Display
    from ai_cli.core.agent import Agent, AgentResult
    from ai_cli.core.agent_registry import AgentRegistry
    from ai_cli.core.config_manager import ConfigManager
    from ai_cli.core.llm_client import LLMClient
    from ai_cli.core.task_manager import TaskManager
    from ai_cli.core.tool_registry import ToolRegistry
    from ai_cli.core.workspace import Workspace

logger = logging.getLogger(__name__)

# Consecutive context-limit hits before a task is force-blocked.
_CONTEXT_LIMIT_STRIKES = 3


class AgentReport(TypedDict):
    success: bool
    summary: str
    answer: str
    error_message: str | None


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
        # Snapshot of the task tree at the most recent failed planning round
        # (planner returned non-ok or did not mutate the heuristic-relevant
        # fields).  Used as a guard so the loop does not call the planner
        # again at the same tree state — the planner-progress check would
        # produce the same outcome and the user would get nothing new.  Reset
        # to ``None`` whenever planning succeeds OR the snapshot changes (so
        # the loop retries planning after the executor mutates the tree).
        self._last_failed_planning_snapshot: (
            tuple[tuple[tuple, ...], tuple[tuple, ...], tuple[str, ...]] | None
        ) = None

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
    # Planner-progress detection
    # ------------------------------------------------------------------

    def _planning_snapshot(
        self,
    ) -> tuple[tuple[tuple, ...], tuple[tuple, ...], tuple[str, ...]]:
        """Capture the task-tree fields the planning heuristic reads.

        Returns a ``(roots, all_tasks, blocked)`` triple where each component
        is a sorted tuple of hashable summaries.  Two snapshots compare equal
        exactly when the heuristic would make the same routing decision —
        which means a planner that returned ok without mutating any of these
        fields is by definition not making progress and the loop should
        terminate rather than re-enter the same branch.

        Fields included:

        * **roots** — ``(id, status, has_subtasks)`` per root task.
          These are the inputs to the "any non-done root without subtasks"
          check.
        * **all_tasks** — ``(id, status, num_subtasks)`` per task.
          These are the inputs to ``_pick_next_task``: status drives
          executability and ``num_subtasks == 0`` identifies leaves.
        * **blocked** — sorted tuple of blocked task ids.
          Drives the "blocked tasks may need re-planning" branch.

        Fields *deliberately excluded*: notes, descriptions, priorities,
        timestamps, definitions of done.  A planner that only updates these
        fields has not changed any of the heuristic's inputs and would cause
        the loop to re-enter the planning branch on the next iteration.
        """

        def _subtask_count(task: dict) -> int:
            ids = task.get("subtask_ids", [])
            return len(ids) if isinstance(ids, list) else 0

        roots = tuple(
            sorted(
                (
                    t.get("id", ""),
                    t.get("status", ""),
                    bool(t.get("has_subtasks", False)),
                )
                for t in self.tm.list_tasks(parent_id=None)
            )
        )
        all_tasks = tuple(
            sorted(
                (
                    t.get("id", ""),
                    t.get("status", ""),
                    _subtask_count(t),
                )
                for t in self.tm.all_tasks()
            )
        )
        blocked = tuple(
            sorted(t.get("id", "") for t in (self.tm.find(status="blocked") or []))
        )
        return roots, all_tasks, blocked

    @staticmethod
    def _planning_diff_summary(
        before: tuple[tuple[tuple, ...], tuple[tuple, ...], tuple[str, ...]],
        after: tuple[tuple[tuple, ...], tuple[tuple, ...], tuple[str, ...]],
    ) -> str:
        """Return a one-line summary of what changed between two snapshots.

        Examples: ``"created 3 task(s)"``, ``"updated status of 2 task(s)"``,
        ``"created 1 task(s), removed 2 task(s)"``, ``"no task changes"``.
        """
        before_tasks = {entry[0]: entry for entry in before[1]}
        after_tasks = {entry[0]: entry for entry in after[1]}

        created_ids = set(after_tasks) - set(before_tasks)
        removed_ids = set(before_tasks) - set(after_tasks)
        common_ids = set(before_tasks) & set(after_tasks)
        status_changed = sum(
            1 for tid in common_ids if before_tasks[tid][1] != after_tasks[tid][1]
        )
        structure_changed = sum(
            1 for tid in common_ids if before_tasks[tid][2] != after_tasks[tid][2]
        )

        parts: list[str] = []
        if created_ids:
            parts.append(f"created {len(created_ids)} task(s)")
        if removed_ids:
            parts.append(f"removed {len(removed_ids)} task(s)")
        if status_changed:
            parts.append(f"updated status of {status_changed} task(s)")
        if structure_changed:
            parts.append(f"restructured {structure_changed} task(s)")
        if not parts:
            return "no task changes"
        return ", ".join(parts)

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
        self._last_failed_planning_snapshot = None

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
                        logger.info(
                            "Step %d: reviewing task id=%s name=%r",
                            step,
                            task.get("id"),
                            task.get("name"),
                        )
                        review_t0 = time.monotonic()
                        review_result = self._run_reviewer(task)
                        review_elapsed = time.monotonic() - review_t0
                        review_report = self._agent_report("reviewer", review_result)
                        review_summary = review_report["summary"]
                        review_status: str = review_result.status
                        if review_report.get("success") is True:
                            review_status = "ok"
                        elif review_report.get("success") is False:
                            review_status = (
                                "error"
                                if review_report.get("error_message")
                                else "failed"
                            )
                        self.display.show_status(
                            "  → reviewer "
                            f"[{review_status}]: {review_summary} "
                            f"({review_elapsed:.2f}s)"
                        )
                        logger.info(
                            "Step %d: reviewer returned status=%s summary=%r",
                            step,
                            review_status,
                            review_summary,
                        )
                        continue

                # 2. Planning phase.
                #
                # The planning heuristic can fire even on well-formed task
                # trees (e.g. when a leaf-only root task exists), so the
                # planner may legitimately have nothing to do.  We use the
                # snapshot guard below to make sure we only call the planner
                # once per distinct tree state — if the planner already
                # returned no progress at this exact snapshot, skip it and
                # let the execution branch take over.  Once the executor
                # mutates the tree, the snapshot will differ and planning is
                # retried automatically.
                if self._needs_planning():
                    current_snapshot = self._planning_snapshot()
                    if current_snapshot == self._last_failed_planning_snapshot:
                        logger.info(
                            "Step %d: skipping planning — snapshot unchanged "
                            "since last failed planning round; deferring to "
                            "execution",
                            step,
                        )
                        # Fall through to the execution phase below.
                    else:
                        before = current_snapshot
                        logger.info(
                            "Step %d: planning round starting "
                            "(roots=%d, all_tasks=%d, blocked=%d)",
                            step,
                            len(before[0]),
                            len(before[1]),
                            len(before[2]),
                        )
                        planner_t0 = time.monotonic()
                        result = self._run_planner(goal)
                        planner_elapsed = time.monotonic() - planner_t0
                        planner_report = self._agent_report("planner", result)
                        parsed_planner_report = self._parse_json_object(result.text)
                        planner_structured_failure = (
                            isinstance(parsed_planner_report, dict)
                            and parsed_planner_report.get("success") is False
                        )
                        planner_error = planner_report.get("error_message")
                        if result.status == "ok" and planner_structured_failure:
                            planner_error_text = (
                                planner_error
                                if isinstance(planner_error, str)
                                and planner_error.strip()
                                else "planner reported unsuccessful completion"
                            )
                            result = result.__class__(
                                text=str(planner_report.get("answer") or ""),
                                status="error",
                                partial=result.partial,
                                error_message=planner_error_text,
                            )
                            # Keep displayed status/summary consistent with the
                            # normalized AgentResult above.
                            planner_report = self._agent_report("planner", result)
                        self.display.show_status(
                            "  → planner "
                            f"[{result.status}]: {planner_report['summary']} "
                            f"({planner_elapsed:.2f}s)"
                        )

                        # Honour an interrupt that fired *during* the planner
                        # call before we apply any post-planner logic.
                        if self._interrupted:
                            self.display.show_status("Interrupted — task tree saved.")
                            return

                        after = self._planning_snapshot()
                        summary = self._planning_diff_summary(before, after)
                        logger.info(
                            "Step %d: planner returned status=%s summary=%s",
                            step,
                            result.status,
                            summary,
                        )

                        # 2a. Surface non-ok planner results to the user, but
                        # do NOT abort the loop — executable tasks should
                        # still get a chance to run via the execution phase.
                        # The snapshot memo below ensures we don't re-call
                        # the planner at this same state.
                        if result.status != "ok":
                            msg = f"Step {step}: planning — {result.status}"
                            if result.error_message:
                                msg += f" — {result.error_message}"
                            self.display.show_status(msg)
                            logger.info(
                                "Step %d: planner failed (status=%s, "
                                "error=%r) — falling through to execution",
                                step,
                                result.status,
                                result.error_message,
                            )
                            self._last_failed_planning_snapshot = after
                        else:
                            # Show the per-step summary so the user can see what
                            # (if anything) the planner just did.
                            self.display.show_status(
                                f"Step {step}: planning — {summary}"
                            )

                            if before == after:
                                # 2b. No-progress: the planner did not mutate any
                                # of the heuristic-relevant fields, so a retry at
                                # this same snapshot would produce the same
                                # outcome.  Memo the snapshot to suppress the
                                # retry and fall through to execution in this
                                # same iteration — execution will either run a
                                # task (mutating the tree, which clears the memo)
                                # or report "no executable tasks" / "all tasks
                                # complete" with the existing messages.
                                self._last_failed_planning_snapshot = after
                                logger.info(
                                    "Step %d: planner made no task-tree changes "
                                    "— falling through to execution. Planning will "
                                    "retry once the tree changes.",
                                    step,
                                )
                            else:
                                # Successful planning round — clear any stale memo
                                # so a future planner failure at a different state
                                # is not masked by this one.
                                self._last_failed_planning_snapshot = None
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
                logger.info(
                    "Step %d: executing task id=%s name=%r",
                    step,
                    next_task.get("id"),
                    next_task.get("name"),
                )
                exec_t0 = time.monotonic()
                exec_result = self._run_executor(next_task)
                exec_elapsed = time.monotonic() - exec_t0
                exec_report = self._agent_report("executor", exec_result)
                exec_summary = exec_report["summary"]
                self.display.show_status(
                    "  → executor "
                    f"[{exec_result.status}]: {exec_summary} "
                    f"({exec_elapsed:.2f}s)"
                )
                logger.info(
                    "Step %d: executor returned status=%s summary=%r",
                    step,
                    exec_result.status,
                    exec_summary,
                )

            self.display.show_status(
                f"Reached iteration limit ({max_iterations}). Run /plan to continue."
            )
        finally:
            signal.signal(signal.SIGINT, original_handler)

    # ------------------------------------------------------------------
    # Agent dispatch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _active_notes(task_detail: dict) -> list[str]:
        """Return active notes from task detail, preferring lifecycle metadata."""
        history = task_detail.get("note_history")
        if isinstance(history, list):
            active: list[str] = []
            for entry in history:
                if not isinstance(entry, dict):
                    continue
                if entry.get("status") != "active":
                    continue
                text = entry.get("text")
                if isinstance(text, str):
                    active.append(text)
            if active:
                return active
        notes = task_detail.get("notes", [])
        if isinstance(notes, list):
            return [n for n in notes if isinstance(n, str)]
        return []

    @staticmethod
    def _report_schema_instruction(extra: str = "") -> str:
        """Return the shared structured-report instruction for agent prompts."""
        instruction = (
            "\n\nReturn ONLY a JSON object with this shape: "
            '{"success": <bool>, "summary": <short string>, '
            '"answer": <string or null>, "error_message": <string or null>}. '
        )
        if extra:
            instruction += extra
        return instruction

    def _get_agent(self, name: str) -> Agent:
        """Return an agent by role name, building it if necessary."""
        return self.agents.get_or_create(
            name,
            workspace=self._workspace,
            config=self._config,
            coordinator_llm=self._coordinator_llm,
            global_tool_registry=self._global_tool_registry,
        )

    def _run_planner(self, goal: str) -> AgentResult:
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
        prompt += self._report_schema_instruction(
            "Set answer to null if not applicable."
        )
        logger.debug("Planner prompt: %s", prompt)
        return self._get_agent("planner").run(prompt)

    def _run_executor(self, task: dict) -> AgentResult:
        detail = self.tm.get_task(task["id"])
        active_notes = self._active_notes(detail)
        prompt = (
            f"Execute the following task.\n\n"
            f"Task: {detail['name']}\n"
            f"Description: {detail.get('description', '')}\n"
            f"Definition of Done: {detail.get('definition_of_done', '')}\n"
        )
        if detail.get("next_action"):
            prompt += f"Suggested next action: {detail['next_action']}\n"
        if active_notes:
            prompt += "\nNotes from prior work:\n"
            for note in active_notes[-5:]:
                prompt += f"  - {note}\n"

        prompt += self._report_schema_instruction(
            "Use answer for concrete execution output; use error_message when success is false."
        )

        result = self._get_agent("executor").run(prompt)
        report = self._agent_report("executor", result)
        parsed_report = self._parse_json_object(result.text)
        structured_failure = (
            isinstance(parsed_report, dict) and parsed_report.get("success") is False
        )
        answer = report["answer"] if isinstance(parsed_report, dict) else result.text
        error_message = report["error_message"] or result.error_message

        if result.status == "ok" and structured_failure:
            self._context_limit_counts.pop(task["id"], None)
            error_detail = (
                f"Error: {error_message}. "
                if isinstance(error_message, str) and error_message
                else ""
            )
            self.tm.add_note(
                task["id"],
                f"Executor reported failure. {error_detail}"
                f"Partial progress: {answer[:500]}",
            )
            return result.__class__(
                text=str(answer),
                status="error",
                partial=result.partial,
                error_message=(
                    str(error_message)
                    if isinstance(error_message, str) and error_message
                    else "executor reported unsuccessful completion"
                ),
            )

        if result.status == "context_limit":
            count = self._context_limit_counts.get(task["id"], 0) + 1
            self._context_limit_counts[task["id"]] = count
            self.tm.add_note(
                task["id"],
                f"Executor hit context limit (attempt {count}/{_CONTEXT_LIMIT_STRIKES}). "
                f"Partial progress: {answer[:500]}",
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
                f"Executor hit tool-call round limit. Partial progress: {answer[:500]}",
            )
        elif result.status == "error":
            self._context_limit_counts.pop(task["id"], None)
            error_detail = f"Error: {error_message}. " if error_message else ""
            self.tm.add_note(
                task["id"],
                f"Executor encountered an error. {error_detail}"
                f"Partial progress: {answer[:500]}",
            )
        else:
            # ok — clear any accumulated context-limit strikes.
            self._context_limit_counts.pop(task["id"], None)

        return result

    def _run_reviewer(self, task: dict) -> AgentResult:
        detail = self.tm.get_task(task["id"])
        active_notes = self._active_notes(detail)
        prompt = (
            f"Review the following task.\n\n"
            f"Task: {detail['name']}\n"
            f"Definition of Done: {detail.get('definition_of_done', '')}\n"
            f"Notes:\n"
        )
        for note in active_notes:
            prompt += f"  - {note}\n"
        prompt += (
            "\nVerify whether the Definition of Done is satisfied. "
            "If yes, mark the task as done. "
            "If not, set it back to in_progress with a note explaining what is missing."
        )
        prompt += self._report_schema_instruction(
            "Use answer for reviewer rationale and suggested next action if applicable."
        )
        logger.debug("Reviewer prompt: %s", prompt)
        return self._get_agent("reviewer").run(prompt)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _agent_report(self, agent_name: str, result: AgentResult) -> AgentReport:
        """Parse a structured JSON report from agent output with safe fallbacks.

        Expected schema:
          {
            "success": bool,
            "summary": str,
            "answer": str | null,
            "error_message": str | null,
          }
        """
        default_success = result.status == "ok" and not result.partial
        report: AgentReport = {
            "success": default_success,
            "summary": self._result_summary(result),
            "answer": result.text or "",
            "error_message": result.error_message or None,
        }

        parsed = self._parse_json_object(result.text)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("success"), bool):
                report["success"] = parsed["success"]
            if isinstance(parsed.get("summary"), str) and parsed["summary"].strip():
                report["summary"] = parsed["summary"].strip()
            answer = parsed.get("answer")
            if isinstance(answer, str):
                report["answer"] = answer
            elif answer is None:
                report["answer"] = ""

            err_val = parsed.get("error_message")
            if err_val is None and isinstance(parsed.get("error"), str):
                err_val = parsed.get("error")
            if isinstance(err_val, str):
                report["error_message"] = err_val
            elif err_val is None:
                report["error_message"] = None
        elif result.text.strip().startswith("{"):
            logger.warning(
                "Agent '%s' returned malformed JSON report; falling back to default summary",
                agent_name,
            )

        success = bool(report.get("success"))
        if not success and result.status == "ok":
            msg = report.get("error_message")
            report["error_message"] = (
                msg
                if isinstance(msg, str) and msg.strip()
                else "Agent reported unsuccessful completion."
            )

        logger.debug(
            "Parsed report from agent '%s': success=%s summary=%r",
            agent_name,
            report.get("success"),
            report.get("summary"),
        )
        return report

    @staticmethod
    def _parse_json_object(text: str) -> dict | None:
        """Best-effort parse of a JSON object from raw agent text."""
        candidate = text.strip()
        if not candidate:
            return None

        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                candidate = "\n".join(lines[1:-1]).strip()

        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

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
    def _result_summary(result: AgentResult) -> str:
        """Return a one-line human-readable summary of an ``AgentResult``.

        Used to append a short outcome note to the per-step status line so
        users can see at a glance what each step did without trawling the log.
        """
        if result.status == "ok" and not result.partial:
            first_line = (result.text or "").strip().split("\n")[0]
            if first_line:
                return first_line[:120]
            return "done"
        if result.status == "ok" and result.partial:
            return "done (partial — aborted or stopped early)"
        if result.status == "context_limit":
            return "context limit reached"
        if result.status == "tool_limit":
            return "tool-call round limit reached"
        if result.status == "error":
            msg = result.error_message or (result.text or "").strip()
            return f"error — {msg[:100]}" if msg else "error"
        return result.status

    @staticmethod
    def _format_summaries(tasks: list[dict]) -> str:
        if not tasks:
            return "  (none)"
        lines = []
        for t in tasks:
            subtask_marker = " [has subtasks]" if t.get("has_subtasks") else ""
            lines.append(f"  {t['id']}: [{t['status']}] {t['name']}{subtask_marker}")
        return "\n".join(lines)
