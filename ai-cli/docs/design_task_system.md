# Design: Task System

## Purpose

The task system provides persistent, structured memory for multi-step work.
It consists of a **task tree** (hierarchical JSON stored on disk) and a set of
**task tools** that any agent can use to read and mutate it.  Two modes of
operation share the same tree and tools:

- **Interactive mode** — the coordinator LLM uses task tools alongside
  `call_agent` during normal conversation.
- **Autonomous mode** (`/plan`) — a deterministic Python orchestrator drives a
  plan → execute → review loop, calling sub-agents sequentially without
  spending LLM calls on routing decisions.

Both modes can coexist within a single session.  The user might start with
`/plan "implement X"`, pause, inspect the tree with `/tasks`, adjust a task
interactively, then `/plan` again to resume.

---

## Relationship to Agent Design

This document depends on and extends `design_agents.md`.  Key touchpoints:

| Agent design concept | Task system usage |
|---|---|
| `AgentSpec` | Planner, executor, and reviewer are agent types in config |
| `Agent.run()` | The orchestrator calls `agent.run(prompt)` per step |
| `ToolRegistry` isolation | Each agent gets only the task tools its role requires |
| `CallAgentTool` | In interactive mode the coordinator dispatches via `call_agent` — no orchestrator needed |
| `AgentResult` | The orchestrator inspects `status` to handle context limits and errors |
| Sequential default | The orchestrator runs agents one at a time (single GPU) |

The task tools are ordinary `Tool` subclasses registered on agent tool
registries like any other tool.  No special framework support is needed — the
task system is built entirely on top of the existing agent and tool
infrastructure.

---

## Hardware Context

The primary deployment target is a single consumer GPU with approximately
24 GiB of VRAM, running local models via LM Studio or Ollama.  Depending on
model selection and quantisation, this supports context windows of roughly
90,000 to 262,000 tokens.

**Implications for the task system:**

- **Context is ample for focused work.** A task detail (~200–500 tokens), a
  handful of source files (~2,000–8,000 tokens each), and tool result history
  fit comfortably in 90K+ context.  Progressive disclosure is valuable for
  keeping agents focused, not for fitting under a tight token budget.

- **One inference at a time.** The GPU cannot serve concurrent requests
  (without severe throughput degradation), so the orchestrator runs agents
  sequentially.  Every LLM call saved by code-driven routing is time saved.

- **Models are capable but not infallible.** 14B–32B parameter models handle
  structured tool calling reliably on well-scoped tasks.  The task tree gives
  them external structure so they do not need to hold the full plan in context.

- **Small models (< 7B) may struggle** with multi-tool orchestration,
  particularly in interactive mode where the coordinator must read the task
  tree, reason about what to do, and dispatch to the right agent.  Autonomous
  mode with code-driven routing is more robust in this case because routing
  decisions are deterministic Python, not LLM output.

---

## Core Concepts

### Task Tree

Tasks are stored as a flat map (`task_id → Task`) with parent–child
relationships encoded via `parent_id` and `subtask_ids`.  The tree supports
arbitrary depth and stable identifiers.

Each task contains:

| Field | Purpose |
|---|---|
| `name` | Short description (what to do) |
| `description` | Detailed context (how to approach it) |
| `definition_of_done` | Explicit completion criteria |
| `status` | Lifecycle state |
| `priority` | Low / medium / high |
| `next_action` | Hint for the next agent that picks this up |
| `blockers` | Reasons the task cannot proceed |
| `notes` | Timestamped observations, discoveries, and partial results |
| `subtask_ids` | Children in the task hierarchy |

Tasks act as the shared working memory across all agents.  No agent needs to
hold the full tree in context — it queries what it needs via the task tools.

### Status Model

```
not_started ──→ in_progress ──→ in_review ──→ done
     │               │               │
     └───→ blocked ←──┘               │
                │                     │
                └───→ (unblocked) ────┘
                         ↓
                    in_progress
```

| Status | Meaning |
|---|---|
| `not_started` | Created but no work begun |
| `in_progress` | Actively being worked on |
| `blocked` | Cannot proceed; `blockers` array explains why |
| `in_review` | Executor considers it complete; awaiting reviewer validation |
| `done` | Structurally complete (all subtasks done, DoD field present and non-empty) |

**Constraints enforced by the task manager:**

- `done` requires all subtasks to also be `done`.
- `done` requires `definition_of_done` to be present and at least 5 characters.
- Moving to `done` only succeeds through `tasks_mark_done`, which enforces
  both structural constraints.
- Whether the DoD criteria are actually *satisfied* is a reviewer/LLM
  judgement — `tasks_mark_done` does not evaluate the content of the DoD field.
- `in_review` is meaningful only when a reviewer agent is configured.  Without
  one, the executor uses `tasks_mark_done` directly.

### Progressive Disclosure

Agents receive **summaries by default** and **detail on demand**:

- `tasks_list` returns ~10 tokens per task (id, name, status, priority, has_subtasks).
  200 tasks = ~2,000 tokens.
- `tasks_get` returns the full record for one task — typically 200–500 tokens.

Even with 90K+ context available, flooding an agent with irrelevant task
detail degrades output quality.  Progressive disclosure keeps agents focused
on the task at hand, not the entire project plan.

---

## Two Modes of Operation

### Interactive Mode

The coordinator LLM has task tools and `call_agent`.  It reads the task tree,
reasons about what to do, and dispatches to sub-agents conversationally.  The
user can intervene at any point — ask questions, override decisions, add tasks
manually.

This is the natural way to use tasks during a normal conversation.  The
coordinator might create a few tasks to track a multi-step request, delegate
each to a sub-agent, and report the results — all within one turn.

No Python orchestrator is involved.  The LLM makes all routing decisions.

**Best for:** shorter multi-step tasks (3–10 steps), tasks where user input
is needed frequently, exploratory work where the plan evolves through
conversation.

### Autonomous Mode (`/plan`)

A Python `TaskOrchestrator` drives the plan → execute → review loop.  Routing
decisions (should I plan? which task next? is review needed?) are deterministic
Python — no LLM calls consumed.  Only the actual planning, execution, and
review steps invoke sub-agents.

```
/plan "implement the agent system"
```

The orchestrator:

1. Creates an initial root task from the goal (if no tasks exist yet).
2. Enters a loop: plan if needed → pick a task → execute → review.
3. Shows progress via `Display.show_status()` after each step.
4. Stops when all root tasks are `done`, when no executable tasks remain,
   or when interrupted by the user (Ctrl+C).
5. Persists all state to `tasks.json` — resumable at any time with `/plan`.

**Best for:** larger tasks (10+ steps), autonomous operation, situations where
the user wants to walk away and check results later, and smaller models that
benefit from deterministic routing.

### Mode Interaction

Both modes read and write the same `tasks.json`.  A session might look like:

1. User runs `/plan "add logging to all tools"` — autonomous mode creates
   tasks, executes several, hits a design question and blocks.
2. User sees the blocked task via `/tasks`, answers the question interactively.
3. User runs `/plan` again — autonomous mode resumes from where it stopped.
4. User notices a task result they want to adjust — switches to interactive
   conversation with the coordinator, modifies files, updates the task.
5. User runs `/plan` to finish the remaining tasks.

---

## Task Storage

### File Location

```
<session_dir>/tasks.json
```

The task file is tied to the session and persists across CLI restarts via
session resume.  It is created on first use (first `tasks_create` call or
first `/plan` invocation).

### File Schema

```json
{
  "goal": "Implement the multi-agent system",
  "tasks": {
    "task_a1b2c3": {
      "id": "task_a1b2c3",
      "parent_id": null,
      "name": "Design the Agent class",
      "description": "Create the Agent class that holds...",
      "definition_of_done": "Agent class exists, has run() method, passes unit tests",
      "status": "in_progress",
      "priority": "high",
      "next_action": "Read design_agents.md for the spec",
      "blockers": [],
      "notes": [
        "[2026-03-26T14:30:00Z] Planner: broken into 3 subtasks",
        "[2026-03-26T14:35:00Z] Executor: AgentSpec dataclass implemented"
      ],
      "subtask_ids": ["task_d4e5f6", "task_g7h8i9", "task_j0k1l2"],
      "created_at": "2026-03-26T14:30:00Z",
      "updated_at": "2026-03-26T14:35:00Z"
    }
  }
}
```

The `goal` field is optional.  It is set by `/plan` and displayed by `/tasks`.
In interactive mode, tasks may be created without a top-level goal.

### Canonical JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Task System Storage",
  "type": "object",
  "required": ["tasks"],
  "properties": {
    "goal": {
      "type": ["string", "null"],
      "description": "Top-level goal set by /plan. Null or absent in interactive mode."
    },
    "tasks": {
      "type": "object",
      "description": "Flat map of task_id → Task object.",
      "additionalProperties": { "$ref": "#/definitions/task" }
    }
  },

  "definitions": {
    "task": {
      "type": "object",
      "required": [
        "id", "parent_id", "name", "description", "definition_of_done",
        "status", "priority", "next_action", "blockers", "notes",
        "subtask_ids", "created_at", "updated_at"
      ],
      "properties": {
        "id":                 { "type": "string", "pattern": "^task_[a-zA-Z0-9]+$" },
        "parent_id":          { "type": ["string", "null"] },
        "name":               { "type": "string", "minLength": 1 },
        "description":        { "type": "string" },
        "definition_of_done": { "type": "string", "minLength": 5 },
        "status":             { "type": "string", "enum": ["not_started", "in_progress", "blocked", "in_review", "done"] },
        "priority":           { "type": "string", "enum": ["low", "medium", "high"] },
        "next_action":        { "type": "string" },
        "blockers":           { "type": "array", "items": { "type": "string" } },
        "notes":              { "type": "array", "items": { "type": "string" } },
        "subtask_ids":        { "type": "array", "items": { "type": "string", "pattern": "^task_[a-zA-Z0-9]+$" } },
        "created_at":         { "type": "string", "format": "date-time" },
        "updated_at":         { "type": "string", "format": "date-time" }
      }
    },

    "task_summary": {
      "type": "object",
      "description": "Lightweight shape returned by tasks_list.",
      "required": ["id", "name", "status", "priority", "has_subtasks"],
      "properties": {
        "id":           { "type": "string" },
        "name":         { "type": "string" },
        "status":       { "type": "string" },
        "priority":     { "type": "string" },
        "has_subtasks": { "type": "boolean" }
      }
    },

    "task_detail": {
      "type": "object",
      "description": "Full shape returned by tasks_get.",
      "required": [
        "id", "name", "description", "definition_of_done", "status",
        "priority", "next_action", "blockers", "notes", "subtasks"
      ],
      "properties": {
        "id":                 { "type": "string" },
        "name":               { "type": "string" },
        "description":        { "type": "string" },
        "definition_of_done": { "type": "string" },
        "status":             { "type": "string" },
        "priority":           { "type": "string" },
        "next_action":        { "type": "string" },
        "blockers":           { "type": "array", "items": { "type": "string" } },
        "notes":              { "type": "array", "items": { "type": "string" } },
        "subtasks":           { "type": "array", "items": { "$ref": "#/definitions/task_summary" } }
      }
    }
  }
}
```

---

## Task Tools

Six tools, consolidated from the original eight.  The `tasks_` prefix groups
them visually; the verb makes the action clear.

All task tools use the canonical tool response schema (see
`docs/technical_requirements.md` — Canonical Tool Response Schema):

- **Success:** `{"status": "success", "data": {...}}`
- **Error:** `{"status": "error", "error": "<code>", "message": "...", "code": 400}`

The `data` payload for each tool is described below.

### `tasks_list`

List tasks under a given parent as lightweight summaries.

```
Arguments:
  parent_id   string|null   optional   Parent task ID, or null/omitted for root tasks.
Returns (data):
  tasks       array                    List of task_summary objects.
```

### `tasks_get`

Retrieve full details of a specific task, including subtask summaries.

```
Arguments:
  task_id     string        required   The task ID.
Returns (data):
  task        task_detail              Full task record with subtask summaries.
```

### `tasks_create`

Create a new task.  If `parent_id` is provided, the task is attached as a
subtask (replaces the separate `add_subtask` tool from the original design).

`definition_of_done` is **required** and must be at least 5 characters.
`TaskManager` enforces this at create time and rejects the call with a
`validation_error` if the field is missing or too short.  This prevents tasks
from entering the tree without actionable completion criteria.

Optional fields that are omitted default to empty strings on the stored task
(`description: ""`, `next_action: ""`), satisfying the storage schema's
`required` constraint.

```
Arguments:
  name                 string        required   Short task name.
  definition_of_done   string        required   Completion criteria (min 5 chars).
  description          string        optional   Detailed description (default: "").
  parent_id            string|null   optional   Parent task ID (null = root task).
  priority             string        optional   "low" | "medium" | "high" (default: "medium").
Returns (data):
  task                 task_detail              The newly created task.
Error (missing/short DoD):
  {"status": "error", "error": "validation_error", "message": "definition_of_done is required and must be at least 5 characters", "code": 400}
```

### `tasks_update`

Update one or more fields of an existing task.  Only include fields that need
to change.  This also covers the `set_blocked` action from the original design
— set `status: "blocked"` and `blockers: [...]` in one call.

```
Arguments:
  task_id              string        required   The task ID.
  name                 string        optional
  description          string        optional
  definition_of_done   string        optional
  status               string        optional   Must be a valid non-"done" status enum value; use `tasks_mark_done` to mark a task as done.
  priority             string        optional
  next_action          string        optional
  blockers             array         optional   Replaces the current blockers list.
Returns (data):
  task                 task_detail              The updated task.
```

### `tasks_add_note`

Append a timestamped note to a task.  Notes accumulate — they are never
replaced, only appended.  This is intentionally separate from `tasks_update`
because notes are append-only (an update would require the agent to GET the
current notes array and SET the full replacement, costing an extra round trip).

```
Arguments:
  task_id     string        required   The task ID.
  note        string        required   The note content.
Returns (data):
  task        task_detail              The task with the new note appended.
```

The tool prepends a timestamp: `[2026-03-26T14:30:00Z] <note>`.

### `tasks_mark_done`

Attempt to mark a task as `done`.  The task manager validates two conditions
before allowing the transition:

1. All subtasks must already be `done`.
2. `definition_of_done` must be non-empty (≥ 5 chars) — a task with no DoD
   cannot be marked done, even if it has no subtasks.

Returns a canonical error response if either check fails.

```
Arguments:
  task_id     string        required   The task ID.
Returns (data):
  task        task_detail              The task with status "done".
Error (subtask not done):
  {"status": "error", "error": "validation_error", "message": "subtask task_xyz is not done", "code": 400}
Error (missing/short DoD):
  {"status": "error", "error": "validation_error", "message": "definition_of_done is required and must be at least 5 characters", "code": 400}
```

### OpenAI Function-Call Schema

```json
[
  {
    "type": "function",
    "function": {
      "name": "tasks_list",
      "description": "List tasks under a given parent as lightweight summaries. Omit parent_id or pass null for root-level tasks.",
      "parameters": {
        "type": "object",
        "properties": {
          "parent_id": {
            "type": ["string", "null"],
            "description": "Parent task ID, or null for root tasks."
          }
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "tasks_get",
      "description": "Retrieve full details of a specific task, including its subtask summaries.",
      "parameters": {
        "type": "object",
        "properties": {
          "task_id": {
            "type": "string",
            "description": "The task ID."
          }
        },
        "required": ["task_id"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "tasks_create",
      "description": "Create a new task. Provide parent_id to create a subtask; omit or pass null for a root-level task.",
      "parameters": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "Short name of the task."
          },
          "description": {
            "type": "string",
            "description": "Detailed description of the task."
          },
          "definition_of_done": {
            "type": "string",
            "minLength": 5,
            "description": "Criteria that must be met for the task to be considered complete. Required; must be at least 5 characters."
          },
          "parent_id": {
            "type": ["string", "null"],
            "description": "Parent task ID. Null or omitted for root-level tasks."
          },
          "priority": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Priority level (default: medium)."
          }
        },
        "required": ["name", "definition_of_done"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "tasks_update",
      "description": "Update fields of an existing task. Only include fields that need to change. Use status 'blocked' with blockers to mark a task as blocked.",
      "parameters": {
        "type": "object",
        "properties": {
          "task_id": { "type": "string", "description": "The task ID." },
          "name": { "type": "string" },
          "description": { "type": "string" },
          "definition_of_done": { "type": "string" },
          "status": { "type": "string", "enum": ["not_started", "in_progress", "blocked", "in_review"] },
          "priority": { "type": "string", "enum": ["low", "medium", "high"] },
          "next_action": { "type": "string" },
          "blockers": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["task_id"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "tasks_add_note",
      "description": "Append a timestamped note to a task. Notes accumulate and are never replaced.",
      "parameters": {
        "type": "object",
        "properties": {
          "task_id": { "type": "string", "description": "The task ID." },
          "note": { "type": "string", "description": "The note content." }
        },
        "required": ["task_id", "note"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "tasks_mark_done",
      "description": "Mark a task as done. Fails if any subtask is not done, or if the task's definition_of_done is missing or shorter than 5 characters.",
      "parameters": {
        "type": "object",
        "properties": {
          "task_id": { "type": "string", "description": "The task ID." }
        },
        "required": ["task_id"]
      }
    }
  }
]
```

Note: `tasks_update` intentionally excludes `"done"` from the `status` enum —
transitioning to `done` must go through `tasks_mark_done` so validation is
enforced.

---

## Agent Roles

Three agent types are defined for the task workflow.  All are configured via
the `agents:` section in `config.yaml` per `design_agents.md`.

### Tool Access by Role

| Tool | Coordinator | Planner | Executor | Reviewer |
|---|---|---|---|---|
| `tasks_list` | yes | yes | yes | yes |
| `tasks_get` | yes | yes | yes | yes |
| `tasks_create` | yes | yes | yes | no |
| `tasks_update` | yes | yes | yes | yes |
| `tasks_add_note` | yes | yes | yes | yes |
| `tasks_mark_done` | yes | no | (configurable) | yes |
| `call_agent` | yes | no | no | no |
| `read_file` | optional | yes | yes | yes |
| `write_file` | no | no | yes | no |
| `find_files` | optional | yes | yes | yes |

When no reviewer agent is configured, the executor receives `tasks_mark_done`
so it can close tasks after completing them.

### Planner

Decomposes high-level goals into structured tasks.  Reads files to understand
the codebase but does not modify them.  Operates on the task tree only.

**Persistence:** `ephemeral` — the planner is called with the current goal and
task tree state.  It does not need memory of prior planning rounds because the
task tree itself is the persistent record of all planning decisions.

**Model choice:** the planner does not generate code, so a general-purpose
model (not necessarily a code-specialist) works well.  Reasoning capability
matters more than code fluency.

### Executor

Carries out a single task: reads files, writes code, runs tests, updates
the task status.  Focused on one task at a time.

**Persistence:** `session` — the executor benefits from remembering what it
already tried, especially for multi-file changes or debugging cycles.  When
context fills up, the orchestrator handles it (see Context Overflow below).

**Model choice:** a code-specialist model (e.g. Qwen-Coder, DeepSeek-Coder)
is ideal.  This is where the bulk of GPU time is spent.

### Reviewer (optional)

Validates that a completed task actually satisfies its Definition of Done.
Reads files and may run tests, but does not write code.  Either accepts
(marks `done`) or rejects (moves back to `in_progress` with a note explaining
what is missing).

**Persistence:** `ephemeral` — each review is a self-contained evaluation.

**Model choice:** can be a smaller model than the executor since it only
needs to read and evaluate, not generate code.  A 7B model is sufficient for
reviewing outputs from a 14B executor.

---

## Autonomous Orchestrator

The `TaskOrchestrator` is a Python class that implements the plan → execute →
review loop without LLM-driven routing.  Each step calls exactly one sub-agent
via `Agent.run()`.

```python
class TaskOrchestrator:
    """Deterministic plan → execute → review loop for /plan mode.

    Note: the sample below calls ``self.agents.has(name)`` and
    ``self.agents.get(name)``.  ``AgentRegistry`` currently exposes the
    ``has_agents`` property and ``get_or_create(...)``.  Thin ``has(name)`` /
    ``get(name)`` convenience wrappers will be added to ``AgentRegistry``
    in PR 4 to match this interface.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        agent_registry: AgentRegistry,
        display: Display,
    ) -> None:
        self.tm = task_manager
        self.agents = agent_registry
        self.display = display
        self._interrupted = False

    # ----------------------------------------------------------------
    # Routing heuristics (pure Python — no LLM calls)
    # ----------------------------------------------------------------

    def _needs_planning(self) -> bool:
        roots = self.tm.list_tasks(parent_id=None)
        if not roots:
            return True
        # Any root task without subtasks that is not done
        for t in roots:
            if t["status"] != "done" and not t["has_subtasks"]:
                return True
        # Any blocked tasks — planner may need to decompose or reroute
        if self.tm.find(status="blocked"):
            return True
        return False

    def _pick_next_task(self) -> dict | None:
        """Pick an executable leaf task using a stable, deterministic ranking."""
        candidates = [
            t for t in self.tm.all_tasks()
            if t["status"] in ("not_started", "in_progress")
            and len(t["subtask_ids"]) == 0
        ]
        if not candidates:
            return None

        priority_rank = {"high": 0, "medium": 1, "low": 2}

        def sort_key(t: dict) -> tuple:
            # in_progress (0) before not_started (1)
            status_rank = 0 if t["status"] == "in_progress" else 1
            # high < medium < low; default to medium if somehow missing
            prio = priority_rank.get(t.get("priority", "medium"), 1)
            # created_at as a stable timestamp tie-breaker (earlier = higher priority)
            timestamp = t.get("created_at", "")
            # task id as final tie-breaker (lexicographic, always unique)
            return (status_rank, prio, timestamp, t["id"])

        return sorted(candidates, key=sort_key)[0]

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------

    def run(self, goal: str, max_iterations: int = 50) -> None:
        # Set the goal in the task file (idempotent)
        self.tm.set_goal(goal)

        for step in range(max_iterations):
            if self._interrupted:
                self.display.show_status("Interrupted — task tree saved.")
                return

            # 1. Review phase — handle tasks awaiting review first
            if self.agents.has("reviewer"):
                in_review = self.tm.find(status="in_review")
                if in_review:
                    task = in_review[0]
                    self.display.show_status(
                        f"Step {step}: reviewing '{task['name']}'"
                    )
                    self._run_reviewer(task)
                    continue

            # 2. Planning phase
            if self._needs_planning():
                self.display.show_status(f"Step {step}: planning")
                self._run_planner(goal)
                continue

            # 3. Execution phase
            task = self._pick_next_task()
            if task is None:
                # Check if everything is done or if we are stuck
                incomplete = self.tm.find_incomplete()
                if not incomplete:
                    self.display.show_status("All tasks complete.")
                else:
                    self.display.show_status(
                        f"No executable tasks. {len(incomplete)} task(s) "
                        f"remain incomplete (blocked or awaiting subtask completion)."
                    )
                return

            self.display.show_status(
                f"Step {step}: executing '{task['name']}'"
            )
            self._run_executor(task)

        self.display.show_status(
            f"Reached iteration limit ({max_iterations}). "
            f"Run /plan to continue."
        )

    # ----------------------------------------------------------------
    # Agent dispatch helpers
    # ----------------------------------------------------------------

    def _run_planner(self, goal: str) -> None:
        roots = self.tm.list_tasks(parent_id=None)
        blocked = self.tm.find(status="blocked")
        prompt = (
            f"Goal: {goal}\n\n"
            f"Current root tasks:\n{self._format_summaries(roots)}\n\n"
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
        self.agents.get("planner").run(prompt)

    def _run_executor(self, task: dict) -> None:
        detail = self.tm.get_task(task["id"])
        prompt = (
            f"Execute the following task.\n\n"
            f"Task: {detail['name']}\n"
            f"Description: {detail['description']}\n"
            f"Definition of Done: {detail['definition_of_done']}\n"
        )
        if detail.get("next_action"):
            prompt += f"Suggested next action: {detail['next_action']}\n"
        if detail.get("notes"):
            prompt += f"\nNotes from prior work:\n"
            for note in detail["notes"][-5:]:  # last 5 notes to limit size
                prompt += f"  - {note}\n"

        result = self.agents.get("executor").run(prompt)

        if result.status == "context_limit":
            self.tm.add_note(
                task["id"],
                f"Executor: hit context limit. Partial progress: "
                f"{result.text[:500]}"
            )

    def _run_reviewer(self, task: dict) -> None:
        detail = self.tm.get_task(task["id"])
        prompt = (
            f"Review the following task.\n\n"
            f"Task: {detail['name']}\n"
            f"Definition of Done: {detail['definition_of_done']}\n"
            f"Notes:\n"
        )
        for note in detail.get("notes", []):
            prompt += f"  - {note}\n"
        prompt += (
            "\nVerify whether the Definition of Done is satisfied. "
            "If yes, mark the task as done. "
            "If not, set it back to in_progress with a note explaining what is missing."
        )
        self.agents.get("reviewer").run(prompt)

    # ----------------------------------------------------------------

    @staticmethod
    def _format_summaries(tasks: list[dict]) -> str:
        if not tasks:
            return "  (none)"
        lines = []
        for t in tasks:
            subtask_marker = " [has subtasks]" if t.get("has_subtasks") else ""
            lines.append(
                f"  {t['id']}: [{t['status']}] {t['name']}{subtask_marker}"
            )
        return "\n".join(lines)
```

### Context Overflow in the Orchestrator

When a sub-agent returns `AgentResult(status="context_limit")`:

1. The orchestrator adds a note to the task with the partial result.
2. For a session-persistent executor, the orchestrator resets its session
   (clears history) so the next call starts fresh, seeded by the notes.
3. The loop continues — the executor will be called again on the same task,
   now with the partial-progress note as context.
4. If the same task hits `context_limit` three times, the orchestrator marks
   it as `blocked` with the reason "Repeated context limit — may need
   decomposition into smaller subtasks" and continues to the planning phase.

### Blocker Handling and User Input

When the orchestrator encounters only blocked tasks and `_needs_planning()`
is true (because blocked tasks exist), it calls the planner with the blocked
task summaries.  The planner may:

- Decompose the blocked task into smaller subtasks and unblock the parent.
- Determine that user input is genuinely needed and leave the task blocked
  with a clarifying note.

If after a planning round blocked tasks remain and no new executable tasks
were created, the orchestrator stops:

> "No executable tasks. 3 task(s) remain incomplete (blocked or awaiting
> subtask completion)."

The user can inspect with `/tasks`, answer questions or adjust tasks, and
resume with `/plan`.

---

## Slash Commands

### `/plan [goal]`

Start or resume autonomous orchestration.

- `/plan "implement the agent system"` — sets the goal and starts the loop.
- `/plan` (no argument) — resumes from the current task tree state.  The
  goal from the previous `/plan` invocation is reused.
- Ctrl+C interrupts cleanly after the current step finishes.  The task tree
  is always in a consistent state on disk.

### `/tasks [task_id]`

View the task tree without consuming an LLM call.

- `/tasks` — shows all root tasks as a summary table.
- `/tasks task_abc123` — shows full detail for one task including subtasks.
- `/tasks --all` — shows the full tree, indented by depth.

Rendered through `Display.show_task_tree()` / `Display.show_task_detail()`.

### `/tasks-clear`

Delete all tasks and the goal.  Requires confirmation.

---

## Agent System Prompts

### Planner

```markdown
You are a planning agent.

Your role is to break down a high-level goal into structured tasks using the
task management tools.

You may:
- Create new tasks and subtasks
- Update descriptions and Definitions of Done
- Reorganise the task hierarchy
- Read files to understand the codebase before planning

You should:
- Ensure tasks are clear, actionable, and testable
- Break large tasks into smaller subtasks
- Define a clear Definition of Done for each task
- Prefer multiple small tasks over one large vague task

You should NOT:
- Execute tasks (write code, run commands)
- Mark tasks as done
- Create tasks with vague or missing Definitions of Done

Use the task tools to modify the task structure.
Do not produce free-form text unless you encounter a design decision
that is too large-impact to make unilaterally — in that case, add a
note to the relevant task explaining the decision needed and mark it
as blocked.
```

### Executor

```markdown
You are an execution agent.

You will be given a single task to complete. Focus exclusively on that task.

You may:
- Read and write files
- Update the task status and add notes
- Add subtasks if you discover the task needs decomposition
- Mark the task as blocked if progress is not possible

You should:
- Follow the Definition of Done exactly
- Follow the "next_action" hint if one is provided
- Add notes recording important discoveries or decisions
- Work incrementally — commit logical units of progress

You should NOT:
- Modify unrelated tasks or files
- Perform global planning or reorganisation
- Mark a task as "done" directly — set it to "in_review" when you
  believe the Definition of Done is satisfied

If blocked:
- Set status to "blocked" with clear blockers
- Add a note explaining what you tried

Always prefer tool usage over free-form text.
```

When no reviewer agent is configured, the executor's prompt is adjusted:
replace the "do not mark as done" instruction with "use `tasks_mark_done`
when the Definition of Done is satisfied."

### Reviewer

```markdown
You are a review agent.

Your role is to verify whether a task's Definition of Done is satisfied.

You will be given a task with its description, DoD, notes, and context.

You must:
1. Read any files referenced in the task or notes to verify the work.
2. Decide: accept or reject.

If the Definition of Done is too vague to evaluate:
- Set status to "in_progress"
- Add a note explaining that the DoD must be clarified before review

If accepting:
- Mark the task as done using tasks_mark_done

If rejecting:
- Set status to "in_progress"
- Add a note explaining specifically what is missing or incorrect

Guidelines:
- Be strict but fair
- Do not assume work is correct without reading the relevant files
- Prefer rejecting incomplete work over accepting low-quality work

Do not perform execution or planning. Only validate.
```

---

## Configuration Example

```yaml
agents:
  planner:
    model: qwen2.5:14b
    system_message: |
      # (planner prompt from above)
    tools:
      - tasks_list
      - tasks_get
      - tasks_create
      - tasks_update
      - tasks_add_note
      - read_file
      - find_files
    max_response_tokens: 4096
    persistence: ephemeral

  executor:
    model: qwen2.5-coder:14b
    system_message: |
      # (executor prompt from above)
    tools:
      - tasks_list
      - tasks_get
      - tasks_create
      - tasks_update
      - tasks_add_note
      - tasks_mark_done    # only when reviewer is not configured
      - read_file
      - write_file
      - find_files
    tool_permission_overrides:
      write_file: false    # orchestrator pre-vetted; no interactive prompts
    max_response_tokens: 8192
    persistence: session

  reviewer:
    model: qwen2.5-coder:7b
    system_message: |
      # (reviewer prompt from above)
    tools:
      - tasks_list
      - tasks_get
      - tasks_update
      - tasks_add_note
      - tasks_mark_done
      - read_file
      - find_files
    max_response_tokens: 4096
    persistence: ephemeral
```

---

## Module Layout

```
ai_cli/
├── core/
│   ├── task_manager.py      # TaskManager — file I/O, validation, queries
│   └── task_orchestrator.py  # TaskOrchestrator — autonomous /plan loop
├── tools/
│   └── tasks.py              # TaskListTool, TaskGetTool, TaskCreateTool,
│                             # TaskUpdateTool, TaskAddNoteTool, TaskMarkDoneTool
```

All six tool classes live in one file because they share the `TaskManager`
instance.  The `TaskManager` owns the JSON file handle and all validation
logic; the tools are thin wrappers that parse arguments, delegate to
`TaskManager`, and format the response.

Dependency additions (one-way, no cycles):

```
task_manager → session_manager   (for session_dir path)
task_orchestrator → task_manager
task_orchestrator → agent_registry
task_orchestrator → display
tasks (tools) → task_manager
repl → task_orchestrator         (for /plan command)
```

---

## Design Principles

### 1. Structured State Over Free Text

All important state is stored in the task tree, not in any agent's
conversation history.  An agent's context can be discarded and rebuilt from
the task tree at any time.

### 2. Progressive Disclosure

Agents receive summaries by default and request detail on demand.  This is
primarily for focus, not for fitting under a token budget — even with 262K
context, flooding an agent with the full tree degrades output quality.

### 3. Strict Validation

The task manager enforces status transition rules and completion constraints.
Agents cannot bypass validation by setting fields directly — `tasks_mark_done`
is the only way to reach `done` status.

### 4. Code-Driven Routing, LLM-Driven Work

In autonomous mode, the orchestrator makes routing decisions in Python (which
agent to call, which task to work on).  Only the actual planning, execution,
and review consume LLM inference.  On a single GPU that processes one request
at a time, this means every second of GPU time is spent on productive work.

### 5. Graceful Degradation

If a sub-agent fails (context overflow, tool error, incoherent output), the
task tree remains consistent.  The orchestrator records the failure as a note,
and the loop continues.  No single agent failure can corrupt the task tree or
halt the system permanently.

### 6. Separation of Planning and Execution

The planner reads the codebase and creates structured tasks but never writes
code.  The executor writes code but does not reorganise the plan.  The
reviewer evaluates but does neither.  This prevents role confusion in the LLM
and keeps each agent's context focused on its specific job.

---

## Prior Art

This design integrates and supersedes the original
`tool_design__task_planner.md`, which described the task tree schema,
orchestrator pseudocode, and agent prompts as a standalone concept.  The key
changes are:

- The Python `Orchestrator` class from the original is now the **autonomous
  mode** of a hybrid system, not the only mode of operation.
- The original's 8 tools are consolidated to 6 (`add_subtask` merged into
  `tasks_create`; `set_blocked` merged into `tasks_update`).
- Agent roles are implemented as `AgentSpec` entries from `design_agents.md`,
  not as bespoke classes.
- Hardware constraints (single GPU, 90K–262K context) are explicitly
  accounted for in design decisions.

---

## Implementation Plan

Four PRs, each independently reviewable and mergeable.  Later PRs depend on
earlier ones but no PR mixes concerns across the boundary below.

### PR 1 — `TaskManager` (zero wiring risk)

**Files:** `ai_cli/core/task_manager.py` (new), `tests/test_task_manager.py` (new)

**Scope:** Pure data layer.  No REPL changes, no tool registration, no agent
integration — just the class that owns `tasks.json`.

`TaskManager` implements:

- `__init__(session_dir: Path)` — resolves `tasks.json` path; does not create
  the file until first write.
- `set_goal(goal: str)` — writes or updates the top-level `goal` field.
- `get_goal() -> str | None`
- `create_task(name, definition_of_done, description="", parent_id=None, priority="medium") -> dict` — generates a `task_<random6>` ID, validates DoD ≥ 5 chars, appends to `subtask_ids` of parent if given, persists, returns `task_detail`.
- `get_task(task_id) -> dict` — returns `task_detail` (full record + subtask summaries).
- `list_tasks(parent_id=None) -> list[dict]` — returns `task_summary` list for
  direct children of `parent_id` (or root tasks if `None`).
- `update_task(task_id, **fields) -> dict` — updates allowed fields, enforces
  that `"done"` cannot be set via this method, persists, returns `task_detail`.
- `add_note(task_id, note: str) -> dict` — prepends ISO timestamp, appends to
  `notes`, persists, returns `task_detail`.
- `mark_done(task_id) -> dict` — validates all subtasks done and DoD ≥ 5 chars;
  raises `TaskValidationError` on failure. Tool wrappers translate that into canonical error responses.
- `find(status: str) -> list[dict]` — returns `task_summary` list filtered by
  status.  Used by the orchestrator to find blocked/in_review tasks.
- `find_incomplete() -> list[dict]` — returns all tasks not in `done` status.
- `all_tasks() -> list[dict]` — returns all tasks as full records (used by
  `_pick_next_task()`).

**Tests** cover: create/get/list round-trip; DoD validation at create and
mark_done; subtask completion gate; status transition guards; add_note
timestamp format; find/find_incomplete filtering; file not created until first
write; concurrent-write safety is out of scope (single-process CLI).

---

### PR 2 — Task Tools (low risk)

**Files:** `ai_cli/tools/tasks.py` (new), `tests/test_tasks_tools.py` (new)

**Scope:** Six `Tool` subclasses that wrap `TaskManager`.  No REPL changes.
The tools are not wired into any registry in this PR — they exist and are
tested in isolation.

Each tool class:

| Class | `NAME` | Arguments | Returns |
|---|---|---|---|
| `TasksListTool` | `tasks_list` | `parent_id?` | `{"tasks": [task_summary, ...]}` |
| `TasksGetTool` | `tasks_get` | `task_id` | `{"task": task_detail}` |
| `TasksCreateTool` | `tasks_create` | `name`, `definition_of_done`, `description?`, `parent_id?`, `priority?` | `{"task": task_detail}` |
| `TasksUpdateTool` | `tasks_update` | `task_id`, optional fields (no `"done"` in status enum) | `{"task": task_detail}` |
| `TasksAddNoteTool` | `tasks_add_note` | `task_id`, `note` | `{"task": task_detail}` |
| `TasksMarkDoneTool` | `tasks_mark_done` | `task_id` | `{"task": task_detail}` |

All six classes live in one file and share a `TaskManager` instance passed to
`__init__`.  They have non-standard constructors and must use
`REGISTER_VIA_INSTANCE = True` so the three-tier file loader skips them.
They are wired in PR 3.

**Tests** cover: argument validation; each tool delegates correctly to
`TaskManager`; error paths (unknown task_id, DoD too short, subtask not done)
produce canonical error responses.

---

### PR 3 — Wire tools into REPL + slash commands (medium risk, touches existing files)

**Files:** `ai_cli/__main__.py`, `ai_cli/core/repl.py`, `ai_cli/core/display.py`

**Scope:** Makes the task tools available in a running session and adds the
`/tasks` and `/tasks-clear` slash commands.  No orchestrator yet.

Changes:

- `__main__.py` — instantiate `TaskManager(session_dir)` after session is
  resolved; instantiate all six task tool objects; register them via
  `tool_registry.register_instance()` on both the coordinator registry and any
  per-agent registries that list task tool names in their spec.

- `repl.py` — add slash command handlers:
  - `/tasks` → `display.show_task_tree(tm.list_tasks())`
  - `/tasks <task_id>` → `display.show_task_detail(tm.get_task(task_id))`
  - `/tasks --all` → `display.show_task_tree(tm.all_tasks(), indent=True)`
  - `/tasks-clear` → confirm prompt → `tm.clear()` (already implemented in PR 1; PR 3 only wires the REPL command and confirmation prompt).

- `display.py` — add two display methods:
  - `show_task_tree(tasks, indent=False)` — renders a Rich table or indented tree of `task_summary` objects; columns: ID, name, status (coloured), priority, subtasks indicator.
  - `show_task_detail(task)` — renders a Rich panel with all fields; notes shown newest-first (last 10); subtasks as a mini-table.

**Tests** — integration test that wires a `TaskManager` to the tool instances,
calls each tool's `execute()`, and verifies output shape.  Display methods are
tested visually / not unit-tested.

---

### PR 4 — `TaskOrchestrator` + `/plan` command (medium risk, new file + REPL change)

**Files:** `ai_cli/core/task_orchestrator.py` (new), `ai_cli/core/repl.py`

**Scope:** Autonomous orchestration loop and the `/plan` slash command.  Builds
on all three prior PRs.

`TaskOrchestrator.__init__(task_manager, agent_registry, display)` — no LLM
client; routing is pure Python.

`TaskOrchestrator.run(goal, max_iterations=50)` — synchronous (matches
`Agent.run()`); the full loop as specified in the Autonomous Orchestrator
section above.  Key implementation notes:

- Context overflow handling: after `result.status == "context_limit"`, call
  `agent_registry.reset("executor")` (adds a `reset()` method to
  `AgentRegistry` that clears cached session state for a named agent) and
  increment a per-task counter; after 3 consecutive context limits on the same
  task, `task_manager.update_task(task_id, status="blocked", blockers=[...])`.
- Ctrl+C: `signal.signal(SIGINT, handler)` sets `self._interrupted = True`;
  the loop checks it at the top of each iteration.

`repl.py` changes:
- `/plan [goal]` — instantiate or reuse `TaskOrchestrator`; call `orchestrator.run(goal or tm.get_goal())`.
- Argument-less `/plan` resumes using the stored goal; errors clearly if no goal is set and none is provided.

**Tests** cover: `_needs_planning()` logic; `_pick_next_task()` ordering
(in_progress before not_started, high before low, created_at tie-breaker);
context limit counter and blocked transition; iteration limit exit; interrupt
flag exit.  Agent calls are mocked via a fake `AgentRegistry`.
