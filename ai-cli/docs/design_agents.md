# Design: Multi-Agent System

## Purpose

The agent system allows the main LLM (the *coordinator*) to offload focused
sub-tasks to specialised *sub-agents* — independent LLM instances each with
their own session history, system message, tool set, and optionally a different
model or backend.  Sub-agents keep their work out of the coordinator's context
window and report back only the result.

The system is entirely additive.  When no agents are configured, the CLI behaves
exactly as it does today.

---

## Backward Compatibility

Today's REPL already *is* a coordinator — it just has no sub-agents available.
The refactor is purely structural:

```
Current:                      With agents:
REPL                          REPL
  └─ LLMClient                  └─ Agent  (main / coordinator)
  └─ Session                        └─ LLMClient
  └─ ToolRegistry                   └─ Session
  └─ Display                        └─ ToolRegistry
                                    └─ Display
                                  [+ CallAgentTool if agents configured]
                                  └─ sub-agents: dict[str, Agent]
```

The REPL's inner "send → handle tool calls → repeat" loop is extracted into
`Agent.run()`.  The REPL constructs a main `Agent` (the coordinator) and calls
it in a loop with terminal I/O wrapping.  If `agents:` is absent or empty in
the config, `CallAgentTool` is never registered and there is no behavioural
difference.

---

## Core Concepts

### `AgentSpec`

An `AgentSpec` is a declarative description of an agent type read from the
config file.  It specifies everything needed to construct and run an `Agent`.

```python
@dataclass
class BackendConfig:
    base_url: str
    api_key_env: str | None = None  # env var name; resolved at instantiation time

@dataclass
class AgentSpec:
    name: str
    system_message: str
    tools: list[str]              # tool names from the global registry
    model: str                    # may differ from the coordinator's model
    max_response_tokens: int
    persistence: Literal["ephemeral", "session"]
    backend: BackendConfig | None = None   # None → inherit coordinator's backend
    tool_permission_overrides: dict[str, bool] = field(default_factory=dict)
    max_tool_rounds: int = 10
    context_limit_threshold: float = 0.90  # fraction of context window at which to return early
```

### `Agent`

An `Agent` holds live runtime state.  For ephemeral agents a new `Agent` is
constructed on each call; for session-persistent agents the same instance is
reused for the lifetime of the CLI session.

```python
class Agent:
    spec: AgentSpec
    session: Session           # isolated from the coordinator and all other agents
    llm_client: LLMClient
    tool_registry: ToolRegistry
    display: Display           # SubAgentDisplay for sub-agents; terminal Display for coordinator

    async def run(self, prompt: str) -> AgentResult:
        """
        Run the full send → tool-call → repeat loop for one prompt.
        Returns when the LLM issues end_turn or when a safety limit is hit.
        Never writes to the terminal directly — output goes through self.display.
        """
```

### `AgentResult`

```python
@dataclass
class AgentResult:
    text: str                         # final assistant text (may be empty)
    status: Literal["ok", "context_limit", "tool_limit", "error"]
    partial: bool = False             # set explicitly by caller; typically True when status != "ok"
    error_message: str = ""
```

### `CallAgentTool`

The tool the coordinator uses to invoke a sub-agent.  Only registered on the
coordinator's `ToolRegistry`; sub-agents do not receive it (preventing
unbounded recursion unless explicitly configured otherwise).

```
Tool name:    call_agent
Arguments:
  agent_type  string  required   Name of the agent type (must match a key in the
                                 agents: config section)
  prompt      string  required   The task or question to give to the agent
Returns (canonical tool wrapper):
  status       string             "success" | "error" (tool invocation outcome)
  data         object             Agent result payload (on success):
    result        string             The agent's final text response
    agent_status  string             "ok" | "context_limit" | "tool_limit" | "error"
    partial       bool               true when the agent returned early without completing
    error_message string             Human-readable error description when agent_status == "error" (empty otherwise)
```

The outer `status` field follows the canonical tool response schema.
`agent_status` carries the `AgentResult.status` value so it does not conflict
with the tool-level `"success"`/`"error"` convention.

---

## REPL Refactor: Extracting `Agent.run()`

The inner loop currently in `REPL._send_to_llm()` moves verbatim to
`Agent.run()`.  The REPL retains ownership of:

- The `PromptSession` and all terminal interaction
- Slash commands
- Session creation/resume (via `__main__.py` as today)
- Compaction thresholds

`Agent.run()` owns:

- Calling `session.add_message("user", prompt)`
- The `LLMClient.send()` streaming loop
- Tool execution and result injection
- Returning an `AgentResult` when the loop ends

The REPL's main loop becomes:

```python
while True:
    user_input = await prompt_session.prompt_async(...)
    # slash command handling ...
    result = await self._main_agent.run(user_input)
    self._display.end_assistant_turn()
    if result.partial:
        self._display.show_status(f"Response incomplete: {result.status}")
    self._check_compaction()
```

For the coordinator/REPL case, `self._display` is the terminal Display so
streaming output still appears in real time via `SubAgentDisplay` passthrough
(see below).

---

## Configuration

```yaml
# ~/.ai-cli/config.yaml or .ai-cli/config.yaml

agents:
  explore:
    model: llama3.2:3b
    system_message: |
      You are a focused file-exploration assistant.
      Read and search files to answer the question you are given.
      Report findings concisely. Do not modify any files.
    tools:
      - read_file
      - find_files
    max_response_tokens: 2048
    persistence: ephemeral
    # backend: inherits coordinator's backend when omitted

  coder:
    model: qwen2.5-coder:14b
    system_message: |
      You are a precise coding assistant.
      Implement exactly what is asked. Do not refactor unrelated code.
      Confirm what you changed in a brief summary.
    tools:
      - read_file
      - write_file
      - find_files
    tool_permission_overrides:
      write_file: false   # coordinator has pre-vetted the task; skip interactive prompts
    max_response_tokens: 8192
    persistence: session

  validator:
    model: qwen2.5-coder:7b
    system_message: |
      You are a code review assistant. Read the files you are asked to
      review and report any bugs, style violations, or logical errors.
    tools:
      - read_file
      - find_files
      - run_tests          # example future tool not available to the coordinator
    max_response_tokens: 4096
    persistence: ephemeral

    backend:
      base_url: http://localhost:11435/v1   # separate Ollama instance
      # api_key_env: OLLAMA_KEY            # optional; resolved from env at runtime
```

If the `agents:` key is absent or is an empty mapping, no sub-agent
infrastructure is initialised and no `CallAgentTool` is registered.

### Global agent defaults

An `agent_defaults:` section provides fallback values for any key not set on
an individual agent spec:

```yaml
agent_defaults:
  persistence: ephemeral
  max_response_tokens: 4096
  max_tool_rounds: 10
  context_limit_threshold: 0.90
```

---

## Tool Registry Isolation

Each `Agent` receives its own `ToolRegistry` **and its own `PermissionManager`**,
both scoped to that agent's `Display`.  Sharing a `PermissionManager` across
agents would leak "always allow" grants between the coordinator and sub-agents,
potentially bypassing `SubAgentDisplay`'s default-deny behaviour.  A helper
factory function builds the per-agent pair:

```python
def build_agent_tool_registry(
    spec: AgentSpec,
    workspace: Workspace,
    config: ConfigManager,
    display: Display,                 # agent's own Display (SubAgentDisplay for sub-agents)
    global_tool_registry: ToolRegistry,
) -> ToolRegistry:
    # IMPORTANT: create a fresh PermissionManager for each agent so that
    # "always allow" grants are scoped to this agent and its Display.
    # Sharing the coordinator's PermissionManager would leak grants between agents.
    permission_manager = PermissionManager(prompt_fn=display.show_permission_prompt)
    registry = ToolRegistry(workspace, config, permission_manager)
    for name in spec.tools:
        tool_instance = global_tool_registry.get(name)
        if tool_instance is None:
            logger.warning("Agent '%s': unknown tool '%s' — skipped", spec.name, name)
            continue
        # Register the same tool class in the new registry, then apply any
        # per-agent permission override directly on the freshly created instance.
        # This is an in-memory-only change — ToolRegistry.set_permission_required()
        # persists to the project config and must NOT be used here.
        tool_cls = type(tool_instance)
        registry.register(tool_cls)
        override = spec.tool_permission_overrides.get(name)
        if override is not None:
            new_instance = registry.get(name)
            if new_instance is not None:
                new_instance.permission_required = override
    return registry
```

This requires `ToolRegistry` to expose:
- `get(name) -> Tool | None` — returns the live tool instance by name.

The override is applied by setting `tool_instance.permission_required` directly
on the newly registered instance.  This is **in-memory only** — it does not
touch the project config.

> **Design note:** The current `ToolRegistry.set_permission_required()` always
> persists to the project config, which is the wrong default for temporary
> per-agent or per-session overrides.  A planned refactor (tracked in the project
> plan) will invert this: mutations are temporary by default; `--persist` saves
> to project config; `--global` (with confirmation) saves to the global config.
> The per-agent factory above must always use the non-persistent path.

The result:
- A `read_file` tool on the coordinator and on an explore agent are
  **separate instances** with independent state.
- Per-tool permission settings can differ between agents.
- An agent that should never prompt (e.g. a pure read-only agent) sets
  `permission_required: false` for all its tools via `tool_permission_overrides`.
- `CallAgentTool` is added to the coordinator's registry *after* the above
  loop, not derived from `AgentSpec.tools`.

---

## Display in Sub-Agents

Sub-agents do not have a terminal.  They use a `SubAgentDisplay` that:

- Captures streaming text in a buffer (returned as `AgentResult.text`).
- Routes `show_tool_call` / `show_tool_result` to `logger.debug()`.
- Routes `show_status` / `show_error` to `logger.info()` / `logger.warning()`.
- **Handles permission prompts by defaulting to "no"** — sub-agents should
  not require interactive approval at runtime.  If a task genuinely requires
  write access, the agent spec should set `tool_permission_overrides` to
  disable prompting for the relevant tools (the coordinator is responsible for
  scoping the task appropriately before dispatching).

`SubAgentDisplay` implements the full `Display` ABC so it can be passed
anywhere a `Display` is expected.

```python
class SubAgentDisplay(Display):
    def begin_assistant_turn(self) -> None: ...        # no-op
    def stream_text(self, delta: str) -> None:
        self._buffer.append(delta)
    def end_assistant_turn(self) -> None: ...          # no-op
    def show_permission_prompt(self, question, extra_options):
        logger.warning("Sub-agent permission prompt denied (non-interactive): %s", question)
        return ("no", "")
    # ... all other methods log or buffer as appropriate

    @property
    def captured_text(self) -> str:
        return "".join(self._buffer)
```

---

## Persistence Modes

### `ephemeral` (default)

A new `Agent` instance — including a fresh `Session` — is created for every
`call_agent` invocation.  The sub-agent has no memory of prior calls.

Best for: focused one-off lookups, file searches, single-function code
generation, validation passes.

### `session`

The `Agent` instance is created once (at first call) and cached for the
lifetime of the CLI session.  Subsequent calls append to the existing session
history, allowing the sub-agent to accumulate context across multiple
coordinator prompts.

Best for: multi-step coding tasks, ongoing planning agents, any task where
the sub-agent benefits from remembering what it already did.

**Context overflow risk:** a session-persistent agent will eventually fill its
context window.  See the next section.

---

## Context Overflow Handling

No automatic compaction is performed on sub-agent sessions — silent context
loss is worse than a clear failure signal.  Instead:

1. **Token monitoring.** After each LLM turn, `Agent.run()` checks the
   token usage from the `done` chunk against the model's context window.
   When usage exceeds `context_limit_threshold` (default 90 %), the loop
   breaks immediately after the current turn completes.

2. **Structured early return.** `AgentResult.status` is set to
   `"context_limit"` and `AgentResult.partial` is `True`.  The coordinator
   receives this signal in the tool call result.

3. **Coordinator's responsibility.** The coordinator LLM decides what to do:
   - Summarise what the sub-agent returned so far and call it again with a
     fresh ephemeral instance seeded by that summary.
   - Escalate by asking the user for guidance.
   - Accept the partial result if it is sufficient.

4. **Manual compaction.** A session-persistent agent can be compacted the
   same way the main session can — either automatically at threshold or via
   a future `/compact-agent <name>` slash command.  Unlike the main session,
   this is not done automatically because the coordinator may need to inspect
   the partial result before deciding whether to continue.

This design puts context management decisions where they belong: in the LLM
that is orchestrating the overall task.

---

## Concurrency

Sub-agents run **sequentially by default**.  This is the correct default
for a single-GPU local backend that can only process one request at a time.

Parallel execution is available as a separate tool, opt-in only:

```
Tool name:    call_agents_parallel
Arguments:
  calls        array   required   List of {agent_type, prompt} objects
Returns (canonical tool wrapper):
  status       string             "success" | "error" (tool invocation outcome)
  data         object             On success:
    results    array              List of per-agent result objects in input order:
      agent_type    string
      result        string             The agent's final text response
      agent_status  string             "ok" | "context_limit" | "tool_limit" | "error"
      partial       bool
      error_message string
```

`call_agents_parallel` uses `asyncio.gather()` internally.  It is only
registered on the coordinator if `agent_settings.allow_parallel: true` is set
in config (default `false`).

```yaml
agent_settings:
  allow_parallel: true   # opt-in; only enable if the backend supports concurrent requests
```

The `agent_settings:` section is separate from `agents:` (which holds only
agent type definitions) to avoid ambiguity — a key inside `agents:` would
otherwise be indistinguishable from an agent type name.

---

## Coordinator Dispatch

From the coordinator's perspective, calling a sub-agent is no different from
calling any other tool.  The coordinator's system message should describe the
available agents and their intended use, just as it describes available file
tools.  The main system message or per-agent descriptions in config can be
surfaced as part of `CallAgentTool`'s tool description, giving the coordinator
LLM enough information to choose the right agent for each task.

`CallAgentTool.definition()` builds its description dynamically at startup,
including the name, purpose snippet (first line of `system_message`), and
available tools for each configured agent type:

```
call_agent — Delegate a focused task to a specialised sub-agent.

Available agent types:
  explore    Read files and search the workspace to answer questions.
             Tools: read_file, find_files
  coder      Write or modify files to implement a specific change.
             Tools: read_file, write_file, find_files
  validator  Review code and run tests.
             Tools: read_file, find_files, run_tests
```

---

## Module Layout

```
ai_cli/
├── core/
│   ├── agent.py          # Agent, AgentSpec, AgentResult, SubAgentDisplay
│   └── agent_registry.py # Parses AgentSpecs from config (load()); instantiates agents lazily via get()
├── tools/
│   └── call_agent.py     # CallAgentTool, CallAgentsParallelTool
```

Dependency additions (one-way, no cycles introduced):

```
agent → llm_client
agent → session_manager
agent → tool_registry
agent → display
call_agent → agent_registry
repl → agent_registry   (to construct the coordinator Agent)
```

---

## What the Agent System Does Not Own

- Terminal I/O — belongs to `Display` / `REPL`
- Session file persistence — belongs to `SessionManager`
- Workspace file access — belongs to `Workspace` / individual tools
- Permission UI — belongs to `Display`; sub-agents bypass via `SubAgentDisplay`
- MCP server lifecycle — belongs to `MCPManager` (future); MCP tools can be
  listed in `AgentSpec.tools` once `MCPManager` is implemented
- Model selection validation — `LLMClient` handles unknown model names at
  connection time, same as today

---

## Implementation Plan

The agent system is split into five PRs.  Each one is independently mergeable
and testable; later PRs build on earlier ones.  PRs 1–2 are purely additive
with zero risk to existing behaviour.  PR 3 is the structural pivot.  PR 4
lights up the feature.  PR 5 adds advanced capabilities.

### Design Decisions (resolved)

1. **Abort / cancellation:** `Agent.run()` accepts an optional
   `abort: threading.Event | None` parameter.  The REPL creates the event
   and `_AbortMonitor` as today; sub-agents pass `None` (non-interactive).
   If coordinator-initiated sub-agent cancellation is needed later, this
   can be refactored to an agent-owned event with a `cancel()` method —
   the interface change is minimal.

2. **Compaction:** The REPL calls `_check_compaction()` after
   `agent.run()` returns — compaction remains the REPL's responsibility.
   `Agent.run()` never triggers compaction.  Sub-agents return
   `AgentResult(status="context_limit")` instead, letting the coordinator
   decide how to proceed.

3. **Coordinator display:** The coordinator's `Agent` receives the real
   terminal `Display` (RichDisplay or PlainDisplay) directly.  Only
   sub-agents use `SubAgentDisplay`.  If coordinator-specific display
   behaviour is ever needed, any `Display` subclass is a drop-in
   replacement.

---

### PR 1 — Data Structures and Config Parsing ✅

**Goal:** Introduce `AgentSpec`, `AgentResult`, `BackendConfig` dataclasses
and the config-parsing layer.  No runtime behaviour change.

#### Files to create

**`ai_cli/core/agent.py`**

```python
@dataclass
class BackendConfig:
    base_url: str
    api_key_env: str | None = None  # env var name; resolved at instantiation time

@dataclass
class AgentSpec:
    name: str
    system_message: str
    tools: list[str]
    model: str
    max_response_tokens: int = 4096
    persistence: Literal["ephemeral", "session"] = "ephemeral"
    backend: BackendConfig | None = None
    tool_permission_overrides: dict[str, bool] = field(default_factory=dict)
    max_tool_rounds: int = 10
    context_limit_threshold: float = 0.90

@dataclass
class AgentResult:
    text: str
    status: Literal["ok", "context_limit", "tool_limit", "error"]
    partial: bool = False       # set explicitly by caller
    error_message: str = ""
```

**`ai_cli/core/agent_registry.py`**

Provides:

- `load_agent_specs(config: ConfigManager) -> dict[str, AgentSpec]`
  Reads the `agents:` section from config, merges each entry with
  `agent_defaults:`, validates, and returns a name → spec mapping.
  Returns an empty dict when `agents:` is absent or empty.

- `AgentRegistry` class (instantiated with the spec dict):
  - `specs` property → `dict[str, AgentSpec]` (read-only copy).
  - `has_agents` property → `bool` (True when at least one spec loaded).
  - `get_or_create(name, *, build_fn)` → for PR 4 (lazy instantiation).
    In this PR, declare the class with `specs` and `has_agents` only;
    `get_or_create` is added in PR 4.

#### Changes to existing files

**`ai_cli/core/config_manager.py`**

Add two methods:

```python
def get_agents_config(self) -> dict:
    """Return the ``agents:`` mapping from config, or empty dict."""
    return self.get("agents") or {}

def get_agent_defaults(self) -> dict:
    """Return the ``agent_defaults:`` mapping, or empty dict."""
    return self.get("agent_defaults") or {}
```

#### Tests

**`tests/test_agent.py`** — unit tests for the dataclasses:

- `AgentSpec` defaults: verify `persistence == "ephemeral"`,
  `max_tool_rounds == 10`, `context_limit_threshold == 0.90`,
  `tool_permission_overrides == {}`, `backend is None`.
- `AgentResult` defaults: verify `partial == False`,
  `error_message == ""`.
- `AgentResult` preserves explicitly provided `partial` and `status`
  values (no auto-derivation — both are set by the caller).

**`tests/test_agent_registry.py`** — parsing and validation:

- Empty/absent `agents:` → empty spec dict, `has_agents == False`.
- Valid single-agent config → correct `AgentSpec` fields.
- Valid multi-agent config → all specs present.
- `agent_defaults:` merge: agent-level keys override defaults;
  defaults fill in missing keys.
- `backend:` section → `BackendConfig` when present, `None` when absent.
- `tool_permission_overrides:` parsed correctly.
- Unknown top-level keys in an agent entry → logged warning, ignored.
- Missing required fields (`name`, `system_message`, `tools`, `model`) →
  raise `ValueError` (or skip with warning — choose one; document the
  choice in a comment).

**`tests/test_config_manager.py`** — add tests for the two new methods:

- `get_agents_config()` returns the mapping when present.
- `get_agents_config()` returns `{}` when absent.
- Same pair for `get_agent_defaults()`.

---

### PR 2 — SubAgentDisplay

**Goal:** Implement a `Display` subclass that captures output in a buffer
instead of writing to a terminal.  No runtime behaviour change.

#### Files to create / modify

**`ai_cli/cli/display.py`** — add `SubAgentDisplay` at the bottom of the
file, after `RichDisplay`.

`SubAgentDisplay` extends `Display` and implements every abstract method:

| Method | Behaviour |
|---|---|
| `begin_assistant_turn` | No-op. |
| `stream_text(delta)` | Append `delta` to `self._buffer: list[str]`. |
| `stream_reasoning(delta)` | `logger.debug("reasoning: %s", delta)` |
| `end_assistant_turn` | No-op. |
| `update_usage(usage, ctx)` | Store as `self._last_usage` for inspection. |
| `show_tool_call(name, args)` | `logger.debug(...)` |
| `show_tool_result(name, result, display_str)` | `logger.debug(...)` |
| `show_status(message)` | `logger.info(...)` |
| `show_error(message)` | `logger.warning(...)` |
| `show_help(commands)` | No-op (sub-agents don't need help). |
| `show_tool_list(tools)` | No-op. |
| `show_session_info(session)` | No-op. |
| `show_tool_list_all(tools_info)` | No-op. |
| `show_tool_info(tool_info)` | No-op. |
| `show_history(messages)` | No-op. |
| `show_permission_prompt(q, extras)` | Log warning, return `("no", "")`. |
| `show_session_list(sessions)` | Return `None`. |

Public API:

```python
@property
def captured_text(self) -> str:
    """Return all streamed text as a single string."""
    return "".join(self._buffer)

def reset(self) -> None:
    """Clear the buffer for reuse (session-persistent agents)."""
    self._buffer.clear()
    self._last_usage = {}
```

Constructor: inherits `Display.__init__` defaults (`verbose=False`,
`markdown_enabled=True`).  These flags have no visible effect since
`SubAgentDisplay` never renders to a terminal, but the defaults match the
ABC contract.

#### Tests

**`tests/test_display.py`** — add a `TestSubAgentDisplay` class (or a new
file `tests/test_sub_agent_display.py` if `test_display.py` is large):

- `stream_text` accumulates deltas; `captured_text` joins them.
- `reset()` clears the buffer.
- `show_permission_prompt` always returns `("no", "")`.
- `show_permission_prompt` logs a warning (use `caplog`).
- `show_error` logs a warning.
- `show_status` logs an info message.
- `update_usage` stores the dict in `_last_usage`.
- All no-op methods are callable without error (parametrize the list).

---

### PR 3 — REPL Refactor: Extract `Agent.run()`

**Goal:** Move the `_send_rounds` loop into `Agent.run()` so the REPL
delegates to an `Agent` instance.  **Zero behavioural change** — all
existing tests must pass without modification.

This is the riskiest PR.  It should contain *only* the extraction, no new
features.

#### Detailed steps

##### 1. Add `Agent.run()` to `ai_cli/core/agent.py`

`Agent` is a new class in the same file as `AgentSpec` / `AgentResult`:

```python
class Agent:
    def __init__(
        self,
        spec: AgentSpec,
        session: Session,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        display: Display,
    ) -> None:
        self.spec = spec
        self._session = session
        self._llm = llm_client
        self._tool_registry = tool_registry
        self._display = display
        self._pending_transients: dict[str, dict] = {}

    def run(
        self,
        prompt: str | list[dict],
        *,
        abort: threading.Event | None = None,
    ) -> AgentResult:
        """Drive the send → tool-call → repeat loop for one prompt.

        Returns an ``AgentResult`` when the LLM issues end_turn, the
        tool-round limit is hit, or the abort event is set.
        """
```

The body of `run()` is the **verbatim** content of the current
`REPL._send_rounds()` method (lines ~1338–1567 of `repl.py`), with these
mechanical substitutions:

| In `_send_rounds` | In `Agent.run()` |
|---|---|
| `self._session` | `self._session` (same) |
| `self._llm` | `self._llm` (same) |
| `self._tool_registry` | `self._tool_registry` (same) |
| `self._display` | `self._display` (same) |
| `self._max_tool_rounds` | `self.spec.max_tool_rounds` |
| `self._pending_transients` | `self._pending_transients` (same) |
| `self._check_compaction()` call at end | **Remove.** Return `AgentResult` instead. |
| `abort.is_set()` checks | Guard with `if abort is not None and abort.is_set()` |
| bare `return` on error/abort | `return AgentResult(text=..., status="error"|"ok", ...)` |
| tool-round limit warning | `return AgentResult(text=..., status="tool_limit", partial=True)` |

The method collects `text_parts` across all rounds and returns
`AgentResult(text="".join(all_text_parts), status="ok")` on normal
completion.

**Key invariant:** `Agent.run()` must not reference any REPL-specific
state (`_AbortMonitor`, `PromptSession`, `_handle_slash_command`, etc.).
It depends only on `Session`, `LLMClient`, `ToolRegistry`, and `Display`.

##### 2. Create a coordinator `AgentSpec` in the REPL

In `REPL.__init__`, construct a coordinator `AgentSpec` and `Agent`:

```python
coordinator_spec = AgentSpec(
    name="coordinator",
    system_message="",          # not used — system message is in the session
    tools=[],                   # not used — registry already built
    model="",                   # not used — llm_client already configured
    max_response_tokens=0,      # not used
    max_tool_rounds=self._max_tool_rounds,
)
self._main_agent = Agent(
    spec=coordinator_spec,
    session=self._session,
    llm_client=self._llm,
    tool_registry=self._tool_registry,
    display=self._display,
)
```

The coordinator spec's `tools`, `model`, `system_message`, and
`max_response_tokens` fields are unused because the REPL already owns the
fully-configured `LLMClient`, `ToolRegistry`, and `Session`.  Only
`max_tool_rounds` is read by `Agent.run()`.

##### 3. Replace `_send_rounds` call in `_send_to_llm`

Before (in `_send_to_llm`):
```python
self._send_rounds(user_input, abort)
```

After:
```python
result = self._main_agent.run(user_input, abort=abort)
```

Handle the result after the monitor stops:

```python
if result.status == "tool_limit":
    self._display.show_error(
        f"Tool call limit ({self._max_tool_rounds} rounds) reached. Stopping."
    )
self._check_compaction()
```

The tool-limit warning and compaction check move *out* of `Agent.run()`
and into `_send_to_llm`, after the monitor's `try/finally`.

##### 4. Delete `_send_rounds`

The method is fully replaced by `Agent.run()`.  Remove it from `repl.py`.

##### 5. Update `/rounds` slash command

The `/rounds` command currently sets `self._max_tool_rounds`.  It must now
*also* update `self._main_agent.spec.max_tool_rounds` so the new value
takes effect.  One line:

```python
self._main_agent.spec.max_tool_rounds = new_value
```

#### Tests

- **All existing REPL tests must pass unchanged.** This is the primary
  acceptance criterion.  If a test breaks, the extraction was not clean.
- **`tests/test_agent.py`** — add unit tests for `Agent.run()` with a
  mock `LLMClient` that returns canned streaming chunks:
  - Simple text-only response → `AgentResult(status="ok", text=...)`.
  - Single tool call + text follow-up → tool executed, result injected,
    second LLM round produces text, returns `"ok"`.
  - Abort event set before first round → returns immediately.
  - Abort event set mid-tool-execution → stub results injected for
    remaining calls, returns with status `"ok"` (abort is a soft stop,
    not an error).
  - Tool-round limit hit → `AgentResult(status="tool_limit", partial=True)`.
  - LLM error → `AgentResult(status="error", error_message=...)`.

---

### PR 4 — `CallAgentTool` and End-to-End Agent Dispatch

**Goal:** Wire up sub-agent creation and dispatch.  After this PR, users
can configure agents in their config file and the coordinator LLM can
delegate tasks to them.

#### Prerequisites

PRs 1–3 merged.

#### Files to create

**`ai_cli/tools/call_agent.py`** — `CallAgentTool`

Must follow the existing `Tool` base class conventions:

- Class attributes: `NAME`, `DESCRIPTION`, `PERMISSION_REQUIRED` (uppercase).
- `definition()` returns a `ToolSchema` (not a raw dict).
- `execute(self, **kwargs)` (not `execute(self, args: dict)`).

```python
class CallAgentTool(Tool):
    NAME = "call_agent"
    DESCRIPTION = "Delegate a focused task to a specialised sub-agent."
    PERMISSION_REQUIRED = False

    def __init__(self, workspace, permission_manager, agent_registry: AgentRegistry):
        super().__init__(workspace, permission_manager,
                         self.PERMISSION_REQUIRED, self.NAME, self.DESCRIPTION)
        self._agent_registry = agent_registry
        self._dynamic_description = self._build_description()

    def _build_description(self) -> str:
        """Build tool description from available agent specs.

        Format:
            Delegate a focused task to a specialised sub-agent.

            Available agent types:
              explore    Read files and search the workspace.
                         Tools: read_file, find_files
        """

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.NAME,
            description=self._dynamic_description,
            arguments=[
                ToolArgument("agent_type", "Name of the agent type.",
                             "string", required=True,
                             enum=sorted(self._agent_registry.specs)),
                ToolArgument("prompt", "The task or question for the agent.",
                             "string", required=True),
            ],
        )

    def execute(self, **kwargs) -> dict:
        agent_type = kwargs["agent_type"]
        prompt = kwargs["prompt"]
        # 1. Validate agent_type exists in registry
        # 2. Get or create Agent instance via registry
        # 3. Call agent.run(prompt)
        # 4. Return canonical tool response with AgentResult fields
```

#### Changes to existing files

**`ai_cli/core/agent_registry.py`** — add `get_or_create()`:

```python
def get_or_create(
    self,
    name: str,
    *,
    workspace: Workspace,
    config: ConfigManager,
    coordinator_llm: LLMClient,
    coordinator_display: Display,
    global_tool_registry: ToolRegistry,
) -> Agent:
    """Return a cached Agent for session-persistent specs, or build a
    new one for ephemeral specs."""
```

Logic:
1. Look up `spec = self._specs[name]` (raise `KeyError` if missing).
2. For `persistence == "session"`: check `self._instances[name]`.  If
   present, call `display.reset()` on its `SubAgentDisplay` and return
   it.  If absent, build and cache.
3. For `persistence == "ephemeral"`: always build a new `Agent`.
4. Building an agent:
   a. Create an `LLMClient` — if `spec.backend` is set, construct a new
      client pointing at that backend; otherwise reuse
      `coordinator_llm` (or construct a new one with the same backend
      but `spec.model`).
   b. Create a `SubAgentDisplay`.
   c. Create a fresh `Session` (in-memory only, no file persistence) with
      `spec.system_message` as the system message.
   d. Call `build_agent_tool_registry()` (from the design doc's "Tool
      Registry Isolation" section) to build a scoped `ToolRegistry`.
   e. Return `Agent(spec, session, llm_client, registry, display)`.

**`ai_cli/core/agent.py`** — add `build_agent_tool_registry()`:

Implement the function exactly as shown in the "Tool Registry Isolation"
section of this design document.

**`ai_cli/cli/repl.py`** — wire up on startup:

In `REPL.__init__` (or in `__main__.py` before constructing the REPL):

`CallAgentTool` has a non-standard constructor (it needs `agent_registry`).
The existing `ToolRegistry.register()` instantiates tool classes with
`(workspace, permission_manager)`, which won't work here.  Add a
`register_instance(tool: Tool)` method to `ToolRegistry` that stores a
pre-built tool instance directly:

```python
agent_registry = AgentRegistry(load_agent_specs(config))
if agent_registry.has_agents:
    call_agent_tool = CallAgentTool(workspace, permission_manager, agent_registry)
    tool_registry.register_instance(call_agent_tool)
```

**`ai_cli/core/session_manager.py`** (if needed) — sub-agents need an
in-memory `Session` that is not persisted to disk.  If the current
`Session` class always writes to a file, add a flag or a subclass:

```python
class InMemorySession(Session):
    """Session that only lives in memory — for ephemeral sub-agents."""
```

Or add `persist=False` to `Session.__init__`.  Check the existing code
to determine which approach is less invasive.

#### Tests

**`tests/test_call_agent.py`**:

- `build_description` includes all configured agent names and their tools.
- `definition()` returns a `ToolSchema` with the correct `enum` of names.
- Execute with valid agent_type + prompt → mock `Agent.run()`, verify
  canonical response shape with `status`, `data.result`,
  `data.agent_status`, `data.partial`, `data.error_message`.
- Execute with unknown agent_type → tool-level `"error"` response.
- Session-persistent agent: two calls with same type → same `Agent`
  instance (mock `get_or_create` and verify call count).
- Ephemeral agent: two calls → different `Agent` instances.

**`tests/test_agent_registry.py`** — extend with `get_or_create` tests:

- Ephemeral spec → new agent every call.
- Session spec → cached agent on second call.
- Unknown name → `KeyError`.
- `SubAgentDisplay.reset()` called on cached session agents.

**Integration test** (can be in `test_call_agent.py`):

- Build a real `AgentRegistry` from a config dict, create a
  `CallAgentTool`, mock only the `LLMClient.send()` to return canned
  chunks.  Call `execute({"agent_type": "explore", "prompt": "..."})`.
  Verify the sub-agent ran, tools were called, and the result came back
  through the canonical wrapper.

---

### PR 5 — Parallel Dispatch, Context Overflow, and Polish

**Goal:** Add `CallAgentsParallelTool`, context-overflow detection in
`Agent.run()`, and any remaining UX polish.

#### Prerequisites

PR 4 merged.

#### `CallAgentsParallelTool`

**`ai_cli/tools/call_agent.py`** — add alongside `CallAgentTool`:

Following the same `Tool` conventions (`NAME`, `DESCRIPTION`,
`PERMISSION_REQUIRED`, `ToolSchema`, `execute(**kwargs)`):

```python
class CallAgentsParallelTool(Tool):
    NAME = "call_agents_parallel"
    DESCRIPTION = "Run multiple sub-agent calls in parallel."
    PERMISSION_REQUIRED = False

    def definition(self) -> ToolSchema:
        # "calls" argument: array of {agent_type, prompt} objects
        ...

    def execute(self, **kwargs) -> dict:
        calls = kwargs["calls"]
        # Use asyncio.gather or concurrent.futures.ThreadPoolExecutor
        # Return list of per-agent results in input order
```

**Gating:** Only registered when `agent_settings.allow_parallel: true` in
config.  Add `get_agent_settings()` to `ConfigManager`:

```python
def get_agent_settings(self) -> dict:
    return self.get("agent_settings") or {}
```

#### Context overflow detection

In `Agent.run()`, after processing the `"done"` chunk:

```python
if chunk["type"] == "done":
    usage = chunk.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    context_window = self._llm.get_model_metadata().get("context_window", 0)
    if (
        context_window > 0
        and prompt_tokens / context_window >= self.spec.context_limit_threshold
    ):
        # Return early — let the caller decide what to do.
        return AgentResult(
            text="".join(all_text_parts),
            status="context_limit",
            partial=True,
        )
```

This check only fires for sub-agents in practice.  The coordinator's
threshold is handled by the REPL's existing `_check_compaction()`.

#### Polish items

- **`/agents` slash command** — list configured agents, their models,
  tools, and persistence mode.  Display-only, no LLM call.
- **Completer additions** — `/agents` completion.
- **Config validation on startup** — warn if `agents:` references tools
  that don't exist in the global registry.
- **Logging** — structured log lines for agent dispatch (agent name,
  prompt length, result status, elapsed time).

#### Tests

**`tests/test_call_agent.py`** — extend:

- `CallAgentsParallelTool`: two concurrent calls → both results returned
  in order.  Mock `Agent.run()` with a small sleep to verify true
  concurrency (or verify `gather`/executor was used).
- Parallel tool not registered when `allow_parallel` is false/absent.

**`tests/test_agent.py`** — extend:

- Context overflow: mock `done` chunk with `prompt_tokens` at 91% of
  context window → `AgentResult(status="context_limit", partial=True)`.
- Context overflow: mock at 89% → normal `"ok"` return.
- Context overflow: `context_window == 0` (unknown) → never triggers.
