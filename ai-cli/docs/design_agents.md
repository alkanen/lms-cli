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
    api_key: str = "not-required"

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
    partial: bool = False             # True when status != "ok"
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
      api_key: not-required
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
