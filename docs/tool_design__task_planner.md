# Task-Oriented Multi-Agent System (Planning Document)

> **Note — Historical Prior Art.**  This document is the original task system
> design and uses older tool names (`tasks_read_*`, `tasks_write_*`,
> `tasks_control_*`, `add_subtask`, `set_blocked`).  The canonical design is
> in `design_task_system.md`, which consolidates the tools to 6
> (`tasks_list`, `tasks_get`, `tasks_create`, `tasks_update`, `tasks_add_note`,
> `tasks_mark_done`) and integrates with the multi-agent system from
> `design_agents.md`.  Refer to that document for implementation.

## Overview

This document describes a task-oriented multi-agent system built around a structured task manager. The system is designed to improve long-horizon reasoning, reduce LLM context load, and increase reliability by separating planning and execution responsibilities across specialized agents.

The task manager serves as a persistent, structured memory layer shared between agents.

---

## Goals

* Enable scalable task decomposition for complex objectives
* Reduce LLM context size via structured external memory
* Improve reliability through separation of concerns
* Support iterative refinement of tasks and requirements
* Provide clear, enforceable task completion criteria

---

## Core Concepts

### 1. Task Tree

Tasks are stored as a hierarchical structure with:

* Parent-child relationships
* Arbitrary depth
* Stable identifiers

Each task contains:

* Name (short description)
* Description (detailed context)
* Definition of Done (DoD)
* Status (enum)
* Subtasks
* Notes and blockers

Tasks act as the shared working memory across agents.

---

### 2. Definition of Done (DoD)

Each task includes explicit completion criteria.

Purpose:

* Prevent premature task completion
* Provide evaluation criteria for agents
* Enable validation by reviewer agents or system rules

---

### 3. Status Model

Tasks move through a strict lifecycle:

* `not_started`
* `in_progress`
* `in_review`
* `blocked`
* `done`

Constraints:

* A task cannot be marked as `done` unless:

  * All subtasks are `done`
  * The Definition of Done is satisfied

---

### 4. Progressive Disclosure

To optimize token usage:

* Agents receive only high-level task summaries by default
* Detailed task data is retrieved on demand

This ensures focus and scalability.

---

## System Architecture

### Components

#### 1. Task Manager (Core System)

Responsible for:

* Persistent storage of tasks
* Enforcing constraints
* Providing structured APIs for agents

Acts as:

* Source of truth
* Coordination layer between agents

---

#### 2. Planner Agent

Responsibilities:

* Decompose high-level goals into tasks
* Define and refine task structure
* Write and update Definitions of Done
* Add or reorganize subtasks as needed

Behavior:

* Operates primarily on high-level tasks
* Expands tasks when execution reveals missing detail

---

#### 3. Executor Agent

Responsibilities:

* Select tasks to work on
* Perform actual work (code, API calls, etc.)
* Update task status and notes
* Report blockers when progress is not possible

Behavior:

* Focused on one task at a time
* Avoids planning unless necessary for execution

---

#### 4. Reviewer Agent (Optional)

Responsibilities:

* Validate task completion
* Compare outputs against Definition of Done
* Reject or accept completion

Purpose:

* Increase correctness
* Reduce false positives on “done” status

---

## Agent Interaction Model

### Typical Workflow

1. Planner Agent:

   * Creates or updates task structure
   * Defines subtasks and DoD

2. Executor Agent:

   * Selects a task
   * Performs work
   * Updates status, notes, or blockers

3. (Optional) Reviewer Agent:

   * Validates completed tasks
   * Confirms or rejects completion

4. Loop continues until all top-level tasks are complete

---

## Task Manager Responsibilities

The task manager must enforce:

* Valid status transitions
* Completion constraints
* Data integrity (e.g., valid parent-child relationships)

It must reject invalid operations and return structured errors.

---

## Agent-Facing API (Conceptual)

### Task Retrieval

#### List Tasks

Returns lightweight summaries:

* id
* name
* status
* has_subtasks

#### Get Task

Returns full task details including:

* description
* Definition of Done
* subtasks (summarized)

---

### Task Mutation

#### Create Task

* Create a new task (optionally with parent)

#### Add Subtask

* Attach a new child task to an existing task

#### Update Task Fields

* Modify description, DoD, priority, or next action

#### Add Note

* Append observations or discoveries

#### Set Blocked

* Mark task as blocked with reasons

#### Mark Done

* Attempt to complete a task (validated by system)

---

### Error Handling

All invalid operations must return structured errors, e.g.:

* Attempting to complete a task with incomplete subtasks
* Invalid status transitions
* Missing required fields

Agents are expected to react and adjust behavior accordingly.

---

## Design Principles

### 1. Separation of Concerns

* Planning and execution are handled by different agents

### 2. Structured State Over Free Text

* All important state is stored in structured form
* Avoid relying on LLM memory

### 3. Incremental Expansion

* Tasks start abstract and become more detailed over time

### 4. Strict Validation

* The system enforces rules instead of trusting agents

### 5. Token Efficiency

* Only necessary data is exposed to agents at each step

---

## Future Extensions

Potential enhancements include:

* Automatic task prioritization
* Progress estimation based on subtasks
* Search and filtering capabilities
* Task summarization for large trees
* Multi-user or collaborative workflows

---

## Summary

This system combines:

* Structured task management
* Multi-agent collaboration
* Strict validation rules

to enable reliable execution of complex, long-running tasks with LLMs.

The task manager acts as the backbone, ensuring consistency and enabling agents to operate efficiently with limited context.

## Legacy Data Schema for Task Storage and API Responses (Historical Prior Art)

This legacy schema from the original task system design describes how the task data is stored in a JSON file by the tool.
For the current canonical schema used in production, refer to `design_task_system.md`.

When an agent requests a task list, the `task_summary` section should be used
for each entry in the list.  When a specific task is requested, it instead
returns the `task_detail` data.  Any error generates an `error`-conforming
object.

It is up to the tool implementation itself to ensure that `parent_id`
references a task that actually exists in the task list, and the same thing
for `subtask_ids`.

When an agent tries to mark a task as done, it is up to the tool implementation
to check that all its subtasks are already marked as done or return an error.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Task Management System Schema",
  "type": "object",
  "required": ["tasks"],
  "properties": {
    "tasks": {
      "type": "object",
      "description": "Flat map of task_id -> Task object",
      "additionalProperties": {
        "$ref": "#/definitions/task"
      }
    }
  },

  "definitions": {
    "task": {
      "type": "object",
      "required": [
        "id",
        "parent_id",
        "name",
        "description",
        "definition_of_done",
        "status",
        "priority",
        "next_action",
        "blockers",
        "notes",
        "subtask_ids",
        "created_at",
        "updated_at"
      ],
      "properties": {
        "id": {
          "type": "string",
          "pattern": "^task_[a-zA-Z0-9]+$"
        },
        "parent_id": {
          "type": ["string", "null"],
          "pattern": "^task_[a-zA-Z0-9]+$"
        },
        "name": {
          "type": "string",
          "minLength": 1
        },
        "description": {
          "type": "string"
        },
        "definition_of_done": {
          "type": "string"
        },
        "status": {
          "type": "string",
          "enum": ["not_started", "in_progress", "blocked", "in_review", "done"]
        },
        "priority": {
          "type": "string",
          "enum": ["low", "medium", "high"]
        },
        "next_action": {
          "type": "string"
        },
        "blockers": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "notes": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "subtask_ids": {
          "type": "array",
          "items": {
            "type": "string",
            "pattern": "^task_[a-zA-Z0-9]+$"
          }
        },
        "created_at": {
          "type": "string",
          "format": "date-time"
        },
        "updated_at": {
          "type": "string",
          "format": "date-time"
        }
      }
    },

    "task_summary": {
      "type": "object",
      "required": ["id", "name", "status", "has_subtasks"],
      "properties": {
        "id": {
          "type": "string"
        },
        "name": {
          "type": "string"
        },
        "status": {
          "$ref": "#/definitions/task/properties/status"
        },
        "has_subtasks": {
          "type": "boolean"
        }
      }
    },

    "task_detail": {
      "type": "object",
      "required": [
        "id",
        "name",
        "description",
        "definition_of_done",
        "status",
        "priority",
        "next_action",
        "blockers",
        "notes",
        "subtasks"
      ],
      "properties": {
        "id": { "type": "string" },
        "name": { "type": "string" },
        "description": { "type": "string" },
        "definition_of_done": { "type": "string" },
        "status": {
          "$ref": "#/definitions/task/properties/status"
        },
        "priority": {
          "$ref": "#/definitions/task/properties/priority"
        },
        "next_action": { "type": "string" },
        "blockers": {
          "type": "array",
          "items": { "type": "string" }
        },
        "notes": {
          "type": "array",
          "items": { "type": "string" }
        },
        "subtasks": {
          "type": "array",
          "items": {
            "$ref": "#/definitions/task_summary"
          }
        }
      }
    },

    "error": {
      "type": "object",
      "required": ["error"],
      "properties": {
        "error": {
          "type": "string"
        }
      }
    }
  }
}
```

## OpenAI API Schema Suggestion

The tool API towards the AI agent might look something like the following. Note
that it is a list of several smaller tools for the different functionalities,
but with a shared prefix to show that they belong together, and with a secondary
prefix showing their intent category ("read", "write" and "control").

```
[
  {
    "type": "function",
    "function": {
      "name": "tasks_read_list_tasks",
      "description": "List tasks under a given parent. Use this to explore the task tree at a high level.",
      "parameters": {
        "type": "object",
        "properties": {
          "parent_id": {
            "type": ["string", "null"],
            "description": "ID of the parent task. Use null to list root-level tasks."
          }
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "tasks_read_get_task",
      "description": "Retrieve full details of a specific task, including its subtasks.",
      "parameters": {
        "type": "object",
        "properties": {
          "task_id": {
            "type": "string",
            "description": "The unique task ID."
          }
        },
        "required": ["task_id"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "tasks_write_create_task",
      "description": "Create a new task. Use for top-level tasks or subtasks.",
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
            "description": "Criteria that must be met for the task to be considered complete."
          },
          "parent_id": {
            "type": ["string", "null"],
            "description": "Optional parent task ID. Null for root-level tasks."
          },
          "priority": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Priority level of the task."
          }
        },
        "required": ["name"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "tasks_write_add_subtask",
      "description": "Create and attach a subtask to an existing task.",
      "parameters": {
        "type": "object",
        "properties": {
          "parent_id": {
            "type": "string",
            "description": "ID of the parent task."
          },
          "name": {
            "type": "string",
            "description": "Short name of the subtask."
          },
          "description": {
            "type": "string",
            "description": "Detailed description of the subtask."
          },
          "definition_of_done": {
            "type": "string",
            "description": "Completion criteria for the subtask."
          }
        },
        "required": ["parent_id", "name"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "tasks_write_update_task",
      "description": "Update fields of an existing task. Only include fields that need to be changed.",
      "parameters": {
        "type": "object",
        "properties": {
          "task_id": {
            "type": "string",
            "description": "ID of the task to update."
          },
          "name": {
            "type": "string"
          },
          "description": {
            "type": "string"
          },
          "definition_of_done": {
            "type": "string"
          },
          "status": {
            "type": "string",
            "enum": ["not_started", "in_progress", "blocked", "in_review", "done"]
          },
          "priority": {
            "type": "string",
            "enum": ["low", "medium", "high"]
          },
          "next_action": {
            "type": "string"
          },
          "blockers": {
            "type": "array",
            "items": {
              "type": "string"
            }
          }
        },
        "required": ["task_id"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "tasks_write_add_note",
      "description": "Add a note or observation to a task. Use this to record discoveries or important context.",
      "parameters": {
        "type": "object",
        "properties": {
          "task_id": {
            "type": "string",
            "description": "ID of the task."
          },
          "note": {
            "type": "string",
            "description": "The note content."
          }
        },
        "required": ["task_id", "note"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "tasks_control_set_blocked",
      "description": "Mark a task as blocked and specify the reasons.",
      "parameters": {
        "type": "object",
        "properties": {
          "task_id": {
            "type": "string",
            "description": "ID of the task."
          },
          "blockers": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "List of blocking issues preventing progress."
          }
        },
        "required": ["task_id", "blockers"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "tasks_control_mark_done",
      "description": "Mark a task as completed. This will fail if subtasks are not completed or Definition of Done is not satisfied.",
      "parameters": {
        "type": "object",
        "properties": {
          "task_id": {
            "type": "string",
            "description": "ID of the task."
          }
        },
        "required": ["task_id"]
      }
    }
  }
]
```

## Orchestrator Pseudocode

This Python code gives some hints on the structure of an orchestrator that takes
a task manager built on the above task tool as well as three separate LLM agents
(planner, executor and reviewer) and makes them work on a task list.

It is not meant to be implemented as is in this project, but to act as an inspiration
for coming up with a workflow that fits into the AI CLI application.

Each agent has its own system prompt and message history, as well as its own
set of tools to use that may be limited depending on the type of agent in
question. E.g. it's likely that the planner is allowed to read files, but maybe
not write to them since it's only supposed to refine the task list with new
subtasks.

```python
import random


class Orchestrator:
    def __init__(self, task_manager, planner, executor, reviewer):
        self.tm = task_manager
        self.planner = planner
        self.executor = executor
        self.reviewer = reviewer

    # ------------------------
    # Decision helpers
    # ------------------------

    def _get_root_tasks(self):
        return self.tm.list_tasks(parent_id=None)

    def _get_all_tasks(self):
        return list(self.tm.data["tasks"].values())

    def _find_incomplete_tasks(self):
        return [
            t for t in self._get_all_tasks()
            if t["status"] != "done"
        ]

    def _find_blocked_tasks(self):
        return [
            t for t in self._get_all_tasks()
            if t["status"] == "blocked"
        ]

    def _find_unreviewed(self):
        return [
            t for t in self._get_all_tasks()
            if t["status"] == "in_review"
        ]

    def _find_executable_tasks(self):
        tasks = []
        for t in self._get_all_tasks():
            if t["status"] in ("not_started", "in_progress"):
                # Only leaf tasks (no subtasks)
                if len(t["subtask_ids"]) == 0:
                    tasks.append(t)
        return tasks

    def _needs_planning(self):
        roots = self._get_root_tasks()

        # No tasks at all
        if not roots:
            return True

        # Any vague tasks (no subtasks but still high-level)
        for t in roots:
            if t["status"] != "done" and len(t["subtask_ids"]) == 0:
                return True

        # Blocked tasks exist
        if self._find_blocked_tasks():
            return True

        return False

    def _pick_next_task(self):
        candidates = self._find_executable_tasks()

        if not candidates:
            return None

        # Simple heuristic: prefer in_progress, then not_started
        in_progress = [t for t in candidates if t["status"] == "in_progress"]
        if in_progress:
            return random.choice(in_progress)

        return random.choice(candidates)

    # ------------------------
    # Agent runners
    # ------------------------

    def run_planner(self, goal):
        tasks = self._get_root_tasks()

        prompt = {
            "goal": goal,
            "tasks": tasks
        }

        self.planner.run(prompt)

    def run_executor(self, task):
        task_detail = self.tm.get_task(task["id"])

        prompt = {
            "task": task_detail
        }

        self.executor.run(prompt)

    def run_reviewer(self, task):
        task_detail = self.tm.get_task(task["id"])

        prompt = {
            "task": task_detail
        }

        self.reviewer.run(prompt)

    # ------------------------
    # Main loop
    # ------------------------

    def run(self, goal, max_iterations=50):
        for step in range(max_iterations):
            print(f"\n--- Step {step} ---")

            # 1. Planning phase
            if self._needs_planning():
                print("Running planner...")
                self.run_planner(goal)
                continue

            # 2. Execution phase
            task = self._pick_next_task()

            if not task:
                print("No executable tasks found.")
                break

            print(f"Executing task: {task['name']}")
            self.run_executor(task)

            # Refresh task state
            updated_task = self.tm.data["tasks"][task["id"]]

            # 3. Review phase
            if updated_task["status"] == "in_review":
                print(f"Reviewing task: {updated_task['name']}")
                self.run_reviewer(updated_task)

        print("\nOrchestration finished.")
```

## Agent Prompts

Each agent needs its own prompt since they will all work on very different
levels and with different types of tasks.

### Planner

```markdown
You are a planning agent.

Your role is to break down a high-level goal into structured tasks using the task management tools.

You may:
- Create new tasks
- Add subtasks
- Update descriptions and Definitions of Done
- Ask the user for design decisions

You should:
- Ensure tasks are clear, actionable, and testable
- Break large tasks into smaller subtasks
- Define a clear Definition of Done for each task
- Make your own decisions on small details that are unlikely
  to have a large effect on the end results

You should NOT:
- Execute tasks
- Mark tasks as done unless explicitly trivial
- Make your own decisions on large-impact issues

Guidelines:
- Prefer multiple small tasks over one large vague task
- Ensure each task has a meaningful Definition of Done
- Avoid redundant or duplicate tasks

Input:
- Goal
- Current task list (high-level)

Output:
- Use tools to modify the task structure
- Do not produce free-form text unless necessary to ask the user for clarifying input or large-impact decisions
```

### Executor

```markdown
You are an execution agent.

Your role is to carry out tasks and make progress.

You will be given a task with:
- Description
- Definition of Done
- Current status
- Subtasks (if any)

You may:
- Update task status
- Add notes
- Add subtasks if necessary
- Mark task as blocked if progress is not possible

You should:
- Focus on completing the task
- Follow the "next_action" if provided
- Work incrementally

You should NOT:
- Perform global planning
- Modify unrelated tasks
- Mark a task as "done", that is for the review agent.  If you are finished with a task you should mark it as "in_review"

If the task is unclear:
- Add a note
- Optionally add subtasks to clarify

If blocked:
- Set status to "blocked"
- Provide clear blockers

If complete:
- Mark the task as "in_review" ONLY if Definition of Done is satisfied

Always prefer tool usage over free-form text.
```

### Reviewer

```markdown
You are a reviewer agent.

Your role is to verify whether a completed task truly satisfies its Definition of Done.

You will be given:
- Task details
- Definition of Done
- Notes and context

You must decide:
- If the Definition of Done is sufficiently defined
- Accept completion
- Reject completion

If DoD not well-defined:
- Update the task:
  - Set status to "in_progress"
  - Add a note explaining that the DoD needs to be properly defined in order to make a decision

If accepting:
- Update the task:
  - Set status to "done"

If rejecting:
- Update the task:
  - Set status back to "in_progress"
  - Add a note explaining what is missing or incorrect

Guidelines:
- Be strict but fair
- Do not assume work is correct without evidence
- Prefer rejecting incomplete work over accepting low-quality work

Do not perform execution or planning.
Only validate.
```
