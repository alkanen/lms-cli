# New Project Plan: AI CLI Tool with Enhanced Functionality

## Overview
This project aims to replace the existing `lms_cli` with a more robust, flexible, and feature-rich AI CLI tool. The new implementation will address limitations in the current design, such as:
- **Workspace Handling**: Current implementation requires execution within the parent directory of `lms_cli`.
- **Tool Management**: Improve tool registration, discovery, and execution.
- **MCP (Model Context Protocol) Support**: Add support for connecting to external MCP tool servers.
- **Structured Output**: Ensure tools return structured data for better integration.
- **Configuration Flexibility**: Decouple configuration from the project root directory.

## Key Features
1. **Dynamic Workspace Handling**
   - Run scripts in any folder while maintaining proper functionality.
   - Use relative paths and environment variables to locate resources.
   - Workspace root is resolved by walking up the directory tree from the starting directory, looking for a `.ai-cli/` folder.
     - The `.ai-cli/` folder in the user's home directory (`~/.ai-cli/`) is reserved for global user settings and sessions. It is skipped during project traversal and never treated as a project workspace root.
     - If no project `.ai-cli/` is found, prompt the user asking whether to initialise a new project in the current directory.
   - Support `--workspace <path>` CLI argument to use a specified path as the point of origin for traversal instead of `cwd`.
   - Support `--init [--workspace <path>]` to skip traversal and create a `.ai-cli/` folder with default contents in `cwd` or the specified path. If a `.ai-cli/` already exists there, ask the user whether to use it or replace it with a fresh default.

2. **Enhanced Tool Management**
   - Improved tool registration with better metadata handling.
   - Dynamic loading of tools from any directory, not just a fixed `tools` folder.
   - Tools can access the entire filesystem but must request permission for every action where `permission_required` is `True` for that tool.
   - Universal permission options: "Yes" (once), "No" (reject), "Always" (always allow for this tool), or custom rejection with a user-provided suggestion sent back to the LLM. Tools may offer additional variants (e.g., "Always in this folder").

3. **Tool Manager Tool**
   - A bundled tool (`tool_manager`) that acts as a gatekeeper for the tool list, reducing context usage and preventing information overload for the LLM.
   - At startup, most tools are disabled. This is a convention of the bundled tools — each declares its own default enabled state in code, and `ToolRegistry` respects those defaults (and any config overrides). It is not a global "disabled unless whitelisted" rule enforced by `ToolRegistry` itself. In the bundled distribution, only `tool_manager` and a small set of essentials (e.g., `read_file`) declare themselves enabled by default.
   - The LLM interacts with `tool_manager` via two actions:
     - `list` — returns each available tool's name, a one-line description, and whether it is currently enabled, so the LLM can make informed enable requests without seeing full schemas.
     - `enable` — requests one or more tools for a single API call by passing a `tool_names` array. The REPL injects those tools' schemas into the immediately following LLM call only; the tools are not added to the permanent or session-level tool list. No state change is made to `ToolRegistry`.
   - This is implemented via `ToolRegistry.enable_transient(name)`, called once per entry in `tool_names`, returning each schema for injection without modifying enabled state.
   - The LLM workflow is: call `list` → call `enable` with a `tool_names` array of all needed tools → the REPL automatically appends the schemas to the next API call → the LLM uses the tools → all injected tools disappear on the subsequent call. No cleanup required from the LLM.
   - **Three distinct enable modes** (in increasing permanence):
     - **Transient** (`tool_manager` enable): injected for one API call only, no state change.
     - **Session** (`/tools enable <name> --session`): in-memory for the current session, reset on exit/resume.
     - **Persistent** (`/tools enable <name>`): written to the project-level `.ai-cli/config.yaml`, survives across sessions.

4. **MCP (Model Context Protocol) Support**
   - Integrate Anthropic's Model Context Protocol for connecting the LLM to external tool servers.
   - MCP servers expose tools via stdio or SSE transports, discovered and invoked by the CLI.
   - Potential use cases:
     - External tool servers (e.g., filesystem, database, API integrations) running as MCP servers.
     - Context switching (e.g., toggling between different MCP server configurations).
     - External integration (e.g., APIs, databases) via dedicated MCP server processes.

5. **Structured Output**
   - Tools export schemas for LLM compatibility.
   - Tool responses MUST be JSON objects conforming to the canonical tool response schema defined in `docs/technical_requirements.md` (success: `{status, data}`; error: `{status, error, message, code}`). Plain-text output is represented as a string field inside `data`.
   - Enforce structured data formats (e.g., JSON, YAML) for fields within `data` where applicable.

6. **Configuration Flexibility**
   - Decouple configuration from the project root directory.
   - Configuration stored in YAML files (`config.yaml`) at global and project level.
   - CLI overrides for critical parameters (e.g., config file paths, server addresses, working directories).
   - Small, localized configurations with minimal overhead.
   - **Configuration resolution** follows a layered model — each level overrides the previous (more specific = higher priority):
     1. Hardcoded defaults (built-in tool attributes, bundled system prompt).
     2. User-global config: `~/.ai-cli/config.yaml` (or `AI_CLI_GLOBAL_DIR/config.yaml`).
     3. Project-specific config: `<project>/.ai-cli/config.yaml`.
     4. CLI flag overrides (highest priority).
     - Project config always wins over global config. Global config acts as default values for projects that do not override them.
     - **Intended behavior (planned refactor):** Runtime mutations to tool settings (via `/tools allow`, `set_permission_required`, etc.) should be **in-memory only by default** — applying for the current session without being written to disk. To persist a change, the user must explicitly opt in: `--persist` writes to the project config (scoped to the current project); `--global` writes to `~/.ai-cli/config.yaml` (affects all projects and requires a confirmation step). This is a planned change; the current implementation persists to the project config by default. See the Permission System section below.
     - If no model configuration is found at any level, the CLI exits with a clear error message guiding the user to set one up.

7. **Improved Permission Handling**
   - Permissions are held in-memory only, scoped to the current process lifetime.
   - All permissions reset on exit or session resume — no persistence to disk.
   - Universal permission options (available for every tool action):
     - **Yes**: Allow this once.
     - **No**: Reject this once.
     - **Always**: Allow all future requests from this tool for the rest of the session.
     - **Custom rejection**: Reject with a user-provided message/suggestion sent back to the LLM.
   - Tools may propose additional permission variants beyond the universal set (e.g., `read_file` and `write_file` offer "Always in this folder"). These tool-specific options are presented alongside the universal ones but are not guaranteed to be available for every tool.

8. **Error Handling and Logging**
   - Comprehensive error handling for tool execution.
   - Structured logging with severity levels using Python's `logging` module.
   - JSONL format for error logs (one entry per line) to facilitate structured data handling while remaining human-readable.
   - Session-specific folders in the user's home directory (`~/.ai-cli/`) to store metadata, session history, and error logs in JSONL format.

9. **Testing and Maintainability**
   - Unit test-friendly design with clear interfaces.
   - Dependency injection for easier mocking in tests.

## Resolved Topics
1. **Session History, Compaction, Resuming, and Editing** — see Session Management section and technical_requirements.md.
2. **CLI Experience** — see Phase 3 and Key Decisions.
3. **MCP Requirements** — MCP = Anthropic's Model Context Protocol (stdio/SSE transports).

## Project Structure

Legend: ✅ implemented and tested · 🔲 planned · ⚠️ partial

```
ai-cli/
├── ai_cli/                         # Python package root
│   ├── __main__.py                 # ✅ Entry point — --workspace, --init, --resume, --continue
│   ├── core/                       # Core functionality
│   │   ├── config_manager.py       # ✅ Layered YAML config loading
│   │   ├── workspace.py            # ✅ Workspace root resolution, file ops, ignore rules
│   │   ├── permission_manager.py   # ✅ In-memory permission state
│   │   ├── tool_registry.py        # ✅ Three-tier tool discovery, loading, argument validation; apply_config(); register_instance(); is_allowed()
│   │   ├── llm_client.py           # ✅ LLMClient ABC + OpenAIClient (REST/streaming); LMStudio WebSocket 🔲
│   │   ├── session_manager.py      # ✅ Session create/resume/compact/persist; InMemorySession; SessionProtocol
│   │   ├── mcp_manager.py          # 🔲 MCP server connections, tool exposure
│   │   ├── agent.py               # ✅ Agent, AgentSpec, AgentResult, BackendConfig, build_agent_tool_registry()
│   │   ├── agent_registry.py      # ✅ AgentSpec loading from config, lazy instance caching (get_or_create)
│   │   ├── task_manager.py        # ✅ Task tree persistence, validation, queries, CRUD
│   │   ├── task_orchestrator.py   # ✅ Deterministic plan→execute→review loop (/plan)
│   │   ├── embedding_provider.py  # ✅ EmbeddingProvider ABC + OpenAIEmbeddingProvider
│   │   ├── vector_store.py        # ✅ VectorStore ABC + SQLiteVectorStore
│   │   ├── chunker.py             # ✅ Chunk, ChunkStrategy ABC, all chunker impls, make_chunker()
│   │   └── embedding_index.py     # ✅ IndexRoot, EmbeddingIndex (orchestration + access control)
│   ├── tools/                      # Bundled tools
│   │   ├── base.py                 # ✅ Tool ABC; ToolArgument (with min/max bounds); ToolSchema
│   │   ├── read_file.py            # ✅ Read a file or line range from the workspace
│   │   ├── write_file.py           # ✅ Write or partially replace a file in the workspace
│   │   ├── find_files.py           # ✅ Glob-pattern file search with ignore-rule enforcement
│   │   ├── tool_manager.py         # ✅ Context-saving tool gatekeeper
│   │   ├── search_files.py         # ✅ search_files tool — semantic search over indexed corpus
│   │   ├── call_agent.py          # ✅ CallAgentTool, CallAgentsParallelTool (coordinator → sub-agent dispatch)
│   │   └── tasks.py               # ✅ Task tools (tasks_list, tasks_get, tasks_create, tasks_update, tasks_add_note, tasks_mark_done)
│   ├── cli/                        # CLI interface and user-facing components
│   │   ├── repl.py                 # ✅ REPL loop; all slash commands; keyboard shortcuts; streaming abort
│   │   ├── display.py              # ✅ Display ABC + PlainDisplay + RichDisplay
│   │   └── completer.py            # ✅ Tab completion for slash commands, tool names, and @path references
│   └── utils/                      # Utility functions and helpers
│       ├── ignore_filter.py        # ✅ .gitignore-style pattern matching
│       └── logging_utils.py        # ✅ JSONL structured logging
├── tests/                          # ✅ Unit tests mirroring ai_cli/ structure (1534 tests)
│   ├── test_workspace.py
│   ├── test_ignore_filter.py
│   ├── test_config_manager.py
│   ├── test_permission_manager.py
│   ├── test_tool_registry.py
│   ├── test_tool_base.py
│   ├── test_read_file.py
│   ├── test_write_file.py
│   ├── test_find_files.py
│   ├── test_tool_manager.py
│   ├── test_llm_client.py
│   ├── test_session_manager.py
│   ├── test_repl.py
│   ├── test_display.py
│   ├── test_completer.py
│   ├── test_main.py
│   ├── test_task_manager.py       # ✅
│   ├── test_task_orchestrator.py  # ✅
│   ├── test_agent.py              # ✅
│   ├── test_agent_registry.py     # ✅
│   ├── test_call_agent.py         # ✅
│   ├── test_chunker.py            # ✅
│   ├── test_vector_store.py       # ✅
│   ├── test_embedding_provider.py # ✅
│   ├── test_embedding_index.py    # ✅
│   └── test_search_files.py       # ✅
└── docs/                           # Documentation
    ├── project_plan.md             # This file
    ├── design_embeddings.md        # ✅ Embedding index + semantic search design
    └── HOWTO_custom_tools.md       # ✅ Guide for writing custom tools
```

## Implementation Plan

Legend: ✅ done · 🔲 planned · ⚠️ partial · → next

### Phase 1: Core Infrastructure
1. **Workspace Handling** ✅
   - Implement a `Workspace` class to manage relative paths and resource resolution.
   - Support for `.ai-cli/` directory in the project root for configuration.
   - The `.ai-cli/` directory structure (created by `--init`):
     ```
     .ai-cli/
     ├── config.yaml          # Model/backend config (YAML, template with placeholders on init)
     ├── system_prompt.md     # Project-specific system prompt override (optional)
     ├── mcp_servers.yaml     # MCP server definitions for this project (optional)
     ├── .ignore              # Files/paths the LLM and tools should not read or modify
     └── tools/               # Project-specific tool implementations (optional)
     ```
   - `~/.ai-cli/` (global user folder) mirrors the same structure including a `tools/` subfolder for globally available user tools.

2. **Configuration Management** ✅
   - `ConfigManager` loads layered config (global → project → CLI overrides).
   - `get_project(key)` exposes the project-only layer for security checks in `ToolRegistry`.
   - `get_model_config()` resolves `api_key_env` to the actual key from the environment.

3. **Tool Registry Enhancements** ✅
   - Redesign the `ToolRegistry` to support dynamic tool discovery from three tiers, loaded in order:
     1. **Bundled tools**: Packaged with `ai-cli` itself (e.g., file read/write, basic shell). Always available.
     2. **Global user tools**: `~/.ai-cli/tools/` — available in all projects for that user.
     3. **Project tools**: `<project>/.ai-cli/tools/` — available only within that project.
   - Tools discovered later in the load order can override earlier ones by name. The user is warned at startup when an override occurs, but it is allowed.
   - Add metadata validation for tools at load time.
   - Per-tool settings are read from a `tools` mapping keyed by tool name in both `~/.ai-cli/config.yaml` (global) and `<project>/.ai-cli/config.yaml` (project), merged in that order (global → project → CLI flags). If a tool is not mentioned, or a key is absent, the tool's own declared defaults apply. Each tool declares `NAME`, `DESCRIPTION`, and `PERMISSION_REQUIRED` as class attributes (e.g., `read_file` sets `PERMISSION_REQUIRED = False`, `write_file` sets it to `True`). **Trust distinction**: global config is treated as trusted (the user's own file); project config is untrusted (a cloned repo could contain it). Lowering `permission_required` from `true` to `false` is therefore allowed unconditionally from global config, but requires an explicit `user_confirmed: true` marker in the project config entry (written by `ToolRegistry.set_permission_required()`; the `/tools allow` REPL command that calls it is 🔲 planned). Example:
     ```yaml
     tools:
       write_file:
         permission_required: true
       bash:
         permission_required: true
         disabled: false
       file_search:
         disabled: true
       read_file:
         allow_outside_workspace: true
     ```
   - Currently the registry applies only `permission_required` and `disabled` from the config. Other keys (e.g., `allow_outside_workspace`) are reserved for future extension; the `Tool` base class would need to accept per-tool settings before they can be applied.
   - **Security**: Project-level `config.yaml` is treated as untrusted. For `permission_required`, `ToolRegistry._apply_config()` ignores any attempt to lower it unless the entry carries `user_confirmed: true` (written by `ToolRegistry.set_permission_required()`; the 🔲 planned `/tools allow` REPL command will call this), and logs a warning otherwise. A startup confirmation prompt for untrusted settings is 🔲 planned for when the REPL exists, but is not yet implemented. Other security-weakening keys such as `allow_outside_workspace` are reserved for future extension and are currently ignored.
   - **`ToolSchema` as return type**: `Tool.definition()` returns a `ToolSchema` object (not a raw dict). The registry calls `.schema()` on it at registration time to produce the OpenAI function-calling dict. This gives full type-checking on tool definitions at load time.
   - **`ToolArgument` bounds**: `ToolArgument` accepts optional `minimum` and `maximum` keyword arguments for `"integer"` and `"number"` types. Validation in `__init__` rejects non-numeric bounds, bounds on non-numeric argument types, and `minimum > maximum` with a `ValueError` (causing the tool to be skipped at registration). The registry enforces bounds at call time via `_check_bounds()`.
   - **Pre-call argument validation** (`_validate_args()`): Before `execute()` is called, the registry validates the incoming kwargs against the tool's declared `ToolSchema`:
     - Required arguments must be present (returns `invalid_arguments` error if missing).
     - Unknown arguments are stripped with a warning log (not an error — the model can send extra keys).
     - Each argument's JSON Schema primitive type is checked via `_check_type()`; mismatches return `invalid_arguments`.
     - `bool` is rejected for `"integer"` and `"number"` types (Python `bool` is a subclass of `int`).
     - `int` is accepted for `"number"` type (widening conversion).
     - Numeric bounds (`minimum`, `maximum`) are enforced via `_check_bounds()` with defensive handling of post-construction mutation.
   - All `invalid_arguments` errors are returned to the LLM so the model can self-correct — values are never silently coerced.

4. **MCP Support** 🔲
   - Implement a `MCPManager` class to discover, connect to, and communicate with MCP servers.
   - Support stdio and SSE transports as defined by the Model Context Protocol.
   - Expose MCP server tools through the same tool registry as built-in tools.

### Phase 2: Tooling and Execution
1. **Tool Execution Improvements** ✅
   - Canonical `{"status": "success"/"error", ...}` response shape standardised via `_ok()`/`_err()` helpers and followed by built-in tools by convention — nothing enforces that third-party tools use them.
   - `ToolRegistry.execute()` handles unknown tool, disabled tool, permission denied, and execution errors — all return canonical error dicts.
   - `allow_transient=True` parameter lets the REPL execute transiently-injected tools that aren't in the persistent enabled set. This intentionally bypasses the *disabled* gate (soft) but must never bypass the *disallowed* gate (hard) — see the two-level permission design under `/tools` subcommands below.
   - Pre-call argument validation (`_validate_args`) runs before `execute()` — see Tool Registry Enhancements above.

2. **Permission System** ✅
   - `PermissionManager` handles in-memory grants (yes/no/always/custom rejection).
   - `always` grants are stored per tool name for the lifetime of the process. The startup sequence explicitly calls `PermissionManager.reset()` and `ToolRegistry.reset_session_overrides()` on every resume, as required by `technical_requirements.md`. Because `PermissionManager` and `ToolRegistry` are created fresh at process start these calls are no-ops in practice, but they satisfy the contract and ensure correctness for any future in-process session-switching path.
   - File tools (`read_file`, `write_file`) additionally manage session-scoped file/dir allow-lists at the tool level via `extra_permission_options()` / `on_permission_granted()` / `reset_session_state()`. These are cleared by `ToolRegistry.reset_session_overrides()`, which iterates over all registered tools and calls their `reset_session_state()` hook.
   - The universal four options (yes/no/always/custom) are always rendered by the prompt implementation. `PermissionManager` passes only tool-specific extras to `prompt_fn`; the prompt handles the universal set itself.
   - **Planned refactor — invert mutation default:** Currently `ToolRegistry.set_permission_required()` persists to the project config by default, with `--session` as the opt-out for temporary changes. The intended design inverts this: mutations are in-memory only by default; `--persist` writes to the project config; `--global` (with a confirmation step, because it affects all projects) writes to `~/.ai-cli/config.yaml`. The underlying `ToolRegistry` API should adopt the same default so callers that need in-memory-only overrides (e.g. `build_agent_tool_registry()`) work naturally without workarounds.

3. **Bundled Tools** ✅
   - `read_file` ✅ — workspace-scoped, no permission by default, disabled by default, session allow-list, line-range support.
   - `write_file` ✅ — workspace-scoped, permission required by default, disabled by default, session allow-list, full and partial writes.
   - `find_files` ✅ — glob-pattern search across the workspace, disabled by default. Supports `*`, `**`, `?`, `[ranges]`, `{alternation}`. Respects all ignore rules (global `.ignore`, project `.gitignore`, project `.ai-cli/.ignore`). Prunes ignored directories during traversal for performance (matching standard Git walk behaviour).
   - `tool_manager` ✅ — context-saving tool gatekeeper; `list` and `enable` actions; transient one-call schema injection via `ToolRegistry.enable_transient()`.

4. **Error Handling** ✅
   - Structured error dicts returned by all tool calls. ✅
   - JSONL logging (`logging_utils.py`) ✅ — `JsonlFormatter`, `setup_logging()`, per-module level overrides, idempotent re-call with handler dedup.
   - Session-specific log file (`<session_dir>/session.log`) ✅ — `setup_logging` called from `__main__.py` after session selection.

### Phase 3: CLI and User Experience ⚠️ (partial)
1. **LLMClient** ✅
   - `LLMClient` abstract base class with `send()`, `get_model_metadata()`, and `count_tokens()`.
   - `OpenAIClient` — OpenAI-compatible REST backend with streaming and tool-call support.
   - LM Studio WebSocket backend 🔲 — planned; LM Studio can be used in OpenAI-compatible mode in the meantime.
   - `create_llm_client(config_manager)` factory function.

2. **Session Management** ✅
   - `Session` and `SessionManager` classes.
   - Two history files: `history_full.jsonl` (append-only) and `history_current.jsonl` (system prompt + summary + recent messages).
   - Compaction: LLM-generated summary replaces older messages; full history preserved in `history_full.jsonl`.
   - `SessionManager.list()`, `load()`, `most_recent()` — used by `--resume` and `--continue`.
   - `Session.set_name()` — used by `/session name`.
   - `Session.should_compact()` / `Session.record_usage()` — automatic compaction trigger.

3. **REPL** ✅
   - Interactive loop using `prompt_toolkit`.
   - `@path` inline file reference expansion. `@path` respects ignore rules; `@!path` bypasses them.
     - **Text files**: the `@ref` token is replaced with a `[file: …]\ncontent\n[/file]` block inline; message stays a plain string.
     - **Image files** (`.png`, `.jpg`/`.jpeg`, `.gif`, `.webp`) 🔲: the file is base64-encoded and the user message is converted to a content block array (`{"type": "text", …}` + `{"type": "image_url", …}`). `_preprocess_at_references()` returns `str | list[dict]`; the REPL calls `add_raw_message` when the result is a list. The backend adapter is responsible for translating canonical `image_url` blocks to the wire format required by its endpoint (e.g. `input_image` for the OpenAI Responses API). See `docs/technical_requirements.md` — Multimodal Messages.
   - Implemented slash commands: `/help`, `/exit`, `/clear`, `/verbose`, `/compact`, `/markdown`, `/tools`, `/session`, `/history`, `/agents`, `/rounds`, `/index`.
   - `/tasks` ✅ — see design_task_system.md § Slash Commands for full subcommand spec (replaces old `/tasks-clear`).
   - RichDisplay prerequisite REPL changes ✅:
     - Route `{"type": "reasoning", "delta": str}` chunks to `display.stream_reasoning()`.
     - Capture `usage` from `"done"` chunk and call `display.update_usage(usage, context_window)`.
     - `/history` command: call `display.show_history(session.get_messages())`.
     - After each tool execute, call `tool.format_display(args, result)` and pass result to `display.show_tool_result(..., display_str=...)`.  See [design_display.md](design_display.md) — Tool call display.

4. **Display** ✅
   - `Display` ABC with full interface defined.
   - `PlainDisplay` ✅ — `print()`-based output, `prompt_toolkit` for interactive prompts.
   - `RichDisplay` ✅ — fully implemented (see [design_display.md](design_display.md)). Key design points:
     - Scrolling output model (no fixed TUI panels); TUI layout deferred as future `TUIDisplay`.
     - `_LeftBorderRenderable` helper: custom `__rich_console__` adds `│ ` prefix to each rendered line.
     - `Live(transient=True)` with `_LiveRenderable` (re-calls build function on every refresh tick) during streaming → formatted Markdown on turn end.
     - Animated `Spinner("dots", " Thinking…")` shown before first text/reasoning arrives.
     - Reasoning preview (dim, truncated) live during stream; full block in verbose mode only.
     - Turn separators: coloured `Rule` + left-bordered `Panel`; user=green, assistant=cyan, tool=yellow.
     - Bottom toolbar via `prompt_toolkit` `refresh_interval=1`: context bar + two elapsed timers + last status.
     - Permission prompt: `prompt_toolkit` radio-list style prompt (arrow-key navigation with hotkeys is 🔲 planned).
     - `/history`: `Console.pager()` for screen/tmux-safe scrollable history view.
     - `tool.format_display(args, result) -> str | None` on Tool base class for ANSI-capable custom tool output.
     - `Console(highlight=False)` — no auto syntax highlighting in streamed text.
   - ABC methods added (all non-abstract with defaults except `show_history`):
     - `stream_reasoning(delta)` — reasoning/thinking content; no-op default
     - `update_usage(usage, context_window)` — token counts for toolbar; no-op default
     - `show_history(messages)` — pageable history viewer; abstract
     - `prompt_session_kwargs()` — concrete method returning `{}` by default; `RichDisplay` returns `{"bottom_toolbar": ..., "refresh_interval": 1}`

5. **Remaining CLI completions** ✅ (all implemented and tested)
   - **`--resume` / `--resume <id>` / `--continue` CLI flags** ✅ in `__main__.py`.
   - **`/tools` subcommands** ✅ — `/tools list`, `/tools info <name>`, `/tools enable|disable|allow|disallow [--session] <name>`.
   - **Two-level tool visibility design** ✅:
     - **enabled / disabled** (soft gate): controls whether the tool appears in the normal per-turn tool list. A `disabled` tool is hidden from the LLM's tool list but `tool_manager` can still transiently enable it for a single turn. Changeable at runtime via `/tools enable`/`disable`.
     - **allowed / disallowed** (hard gate): controls whether the tool is visible to the agent at all. A `disallowed` tool is not listed by `tool_manager` and cannot be transiently enabled — the agent has no way to know it exists. Only changeable via `/tools allow`/`disallow` or by editing config and restarting. `allow_transient=True` in `ToolRegistry.execute()` must respect this gate (i.e. must not execute a `disallowed` tool).
   - **`/compact [instructions]`** ✅ — instructions string forwarded to `session.compact()`.
   - **`/session name "<name>"`** ✅ — calls `session.set_name()`.
   - **`completer.py`** ✅ — tab completion for slash commands (with subcommand/flag completion), tool names, and `@path` / `@!path` file references. Completions are workspace-aware, respect ignore rules, cap at `repl_behavior.completion_max_results` (default 200). Interactive `@` popup/picker and image-file attachments are 🔲 planned.

6. **UX and keyboard improvements** ✅
   - **Streaming abort** ✅ — `_AbortMonitor` background thread watches stdin for lone ESC or Ctrl+C and sets a `threading.Event`; the streaming loop checks it between chunks. Uses `tty.setcbreak` + `select.select` with a 20 ms ESC-disambiguation peek (to distinguish bare ESC from arrow-key sequences). Only active when `_HAS_TTY` and `sys.stdin.isatty()`.
   - **Keyboard shortcuts** ✅ — injected via `prompt_toolkit.KeyBindings`:
     - **Ctrl+C / ESC**: abort current LLM response (or abort tool execution mid-round).
     - **Ctrl+Z**: suspend to background (Unix/TTY only; controlled by `repl_behavior.enable_suspend`, default `true`).
     - **Ctrl+L**: clear the terminal screen.
     - **Ctrl+G**: open the current prompt buffer in `$VISUAL` / `$EDITOR` (uses `shlex.split` to support arguments in the env var; catches `OSError` and prints to stderr if the editor is not found).
   - **`repl_behavior` config section** ✅:
     - `complete_while_typing` (default `false`) — enable prompt_toolkit auto-complete as you type.
     - `enable_suspend` (default `true`) — include Ctrl+Z binding and show it in `/help`.
     - `completion_max_results` (default `200`) — cap on `@path` tab completions per keystroke; must be ≥ 1.

7. **Logging** ✅
   - `logging_utils.py` — JSONL structured logging to `<session_dir>/session.log`. Integrated into `__main__.py`.

### Phase 4: Advanced Features

1. **Multi-Agent System** ✅
   - See [design_agents.md](design_agents.md) for full design.
   - `agent.py` — `Agent` class (send→tool→repeat loop), `AgentSpec`, `AgentResult`, `BackendConfig`, `build_agent_tool_registry()`. `Agent.run()` extracted from what was `REPL._send_rounds()`.
   - `agent_registry.py` — loads `AgentSpec` entries from the `agents:` config section; lazy `get_or_create()` with double-checked locking for thread-safe parallel dispatch; caches session-persistent agent instances.
   - `call_agent.py` — `CallAgentTool` registered on the coordinator when agents are configured. Dynamically builds its tool description from available agent types using `is_allowed()` (cheap dict lookup, no schema computation). `CallAgentsParallelTool` runs multiple sub-agents concurrently via `ThreadPoolExecutor`; gated by `agent_settings.allow_parallel: true`; max batch size configurable via `agent_settings.max_parallel_calls` (default 10); rejects duplicate session-persistent agents.
   - `session_manager.py` — `SessionProtocol` (`@runtime_checkable`) and `InMemorySession` added for sub-agent use (isolated in-memory history with deep-copy isolation).
   - `tool_registry.py` — `register_instance()` for non-standard constructors (used by `CallAgentTool`/`CallAgentsParallelTool`); `is_allowed(name)` public method (dict-lookup only); `DISALLOWED_BY_DEFAULT` honoured in both `_register()` and `register_instance()`.
   - `__main__.py` — `_wire_agents()` extracted function: registers `CallAgentTool` (always) and optionally `CallAgentsParallelTool` when `agent_settings.allow_parallel: true`; validates tool names in agent specs at startup and logs warnings for unknown tools.
   - `SubAgentDisplay` (in `display.py`) captures streaming output in a buffer; permission prompts default to "no"; `reset()` clears buffer and supports agent reuse.
   - Per-agent `ToolRegistry` instances with independent tool sets and permission overrides (`build_agent_tool_registry()` in `agent.py`).
   - Sequential execution by default (single-GPU constraint). Parallel opt-in via `call_agents_parallel` tool gated by `agent_settings.allow_parallel: true`.
   - Persistence modes: `ephemeral` (fresh context per call) and `session` (context accumulates across calls within the CLI session).
   - Context overflow: token monitoring → stream loop breaks (assistant message persisted first) → dangling tool_call stubs injected → `AgentResult(status="context_limit", partial=True, error_message="Context limit reached (x/y tokens).")` returned → coordinator decides how to proceed.
   - `/agents` slash command: lists configured agent types with model, persistence, tools, and max_tool_rounds.

2. **Task System** ✅
   - See [design_task_system.md](design_task_system.md) for full design.
   - `task_manager.py` ✅ — CRUD, status transitions, parent–child integrity, completion validation, `delete_task()`, `find_by_path()`, `close_task()`, `open_task()`, name constraint (`^[A-Za-z0-9_]+$`), sibling uniqueness enforcement.
   - Six task tools in `tasks.py` ✅: `tasks_list`, `tasks_get`, `tasks_create`, `tasks_update`, `tasks_add_note`, `tasks_mark_done`.  Always registered via `_wire_tasks()` in `__main__.py` ✅.
   - `/tasks` slash command ✅ design, ✅ implementation — full subcommand set: `list`, `list <path>`, `tree`, `tree <path>`, `info`, `add`, `add <path>`, `edit`, `delete`, `close`, `open`. Tasks addressed by dot-path of names (e.g., `root.child.leaf`). See design_task_system.md § Slash Commands.
   - `/tasks delete [<path>]` handles both cases — omitting the path deletes everything; always confirms.
   - `/tasks close <path>` — force-closes target and all descendants regardless of DoD validation.
   - `/tasks open <path>` — re-opens a `done` task and walks the ancestor chain re-opening any `done` ancestors to preserve tree consistency.
   - `tasks.tree_depth` config key ✅ — controls `/tasks tree` render depth (default 3, overridable with `--depth <n>`).
   - Hybrid orchestration:
     - **Interactive mode** — coordinator LLM uses task tools + `call_agent` during normal conversation. No Python orchestrator involved.
     - **Autonomous mode** (`/plan <goal>`) — `task_orchestrator.py` ✅ drives a deterministic plan→execute→review loop. Routing decisions are pure Python; only sub-agent work consumes the GPU. Ctrl+C interrupts cleanly; `/plan` resumes from task tree state.
     - **Plan checkpoint** ✅ — after the first planning round, `/plan` pauses by default, renders the task tree, and asks the user to confirm before executing anything. `/plan --autonomous` bypasses the checkpoint for unattended runs. See design_task_system.md § Plan Checkpoint.
   - Progressive disclosure: `tasks_list` returns ~10 tokens per task; `tasks_get` returns full detail on demand. Keeps agents focused even when context windows are large (90K–262K).
   - Three agent roles: planner (read-only, creates task structure), executor (reads/writes files, updates tasks), reviewer (optional, validates DoD and marks done).
   - Reviewer is optional — when not configured, the executor marks tasks done directly via `tasks_mark_done`.

3. **Embedding Index and Semantic Search** ✅
   - See [design_embeddings.md](design_embeddings.md) for the full design.
   - Entirely additive. When `embeddings.enabled` is absent or `false`, nothing is initialised and `search_files` is not registered.
   - Four core modules implemented:
     1. `chunker.py` — `Chunk` dataclass, `ChunkStrategy` ABC, `FixedSizeChunker`, `TreeSitterChunker` (optional dep), domain chunkers for YAML/TOML/config formats, `make_chunker()` factory.
     2. `vector_store.py` — `VectorStore` ABC + `SQLiteVectorStore`. WAL-mode SQLite at `.ai-cli/embeddings/index.db`; thread-safe with `threading.Lock`; numpy vectorised cosine similarity; clean swap-in boundary for future vector databases.
     3. `embedding_provider.py` — `EmbeddingProvider` ABC + `OpenAIEmbeddingProvider`. Config resolved via `ConfigManager.get_embedding_config()` which falls back to LLM backend values when embedding-specific values are absent. Thread-safe: `threading.Lock` guards lazy client construction and `_dimension` writes.
     4. `embedding_index.py` — `IndexRoot`, `EmbeddingIndex`. Orchestrates chunking → embedding → storage. Incremental re-index via xxhash64 change detection. Manages external index roots (user-added paths). `is_indexed_path()` access control used by tools.
   - One new tool: `search_files.py` — `search_files` tool. `DISABLED_BY_DEFAULT = True`. Parameters: `query`, `k`, `level` ("chunk"/"document"/"both"), `path_glob`. Returns ranked results with symbol names, line ranges, live snippets.
   - Access control: indexing a path grants read access. `read_file` checks `workspace.embedding_index.is_indexed_path()` for non-workspace paths; `find_files` access control for its optional `path` parameter is 🔲 planned.
   - `/index [path] [--label] [--file] [--full] [--remove]` slash command added to `repl.py`; tab completion via `completer.py`.
   - Startup background indexing: not in phase 1, but `index()` is `async` from the start so adding `asyncio.create_task(index.index())` later requires zero changes to `EmbeddingIndex`.
   - Multi-granularity: both chunk-level and document-level vectors stored. Document vectors: `average` strategy (L2-normalised mean of chunk vectors) for code; `summary` strategy (LLM-generated thematic summary → embed that text) for prose. Input text is truncated to `summary_max_tokens * 4` chars before the LLM call. The summary prompt includes a word-count hint derived from `summary_response_tokens` (default `chunk_size // 4`); no per-call API `max_tokens` cap is set to preserve reasoning-model token budgets. Failed summary calls fall back to `average`. `auto` strategy routes per file extension via `_resolve_doc_strategy()`. Summary embedding during indexing is dispatched via `asyncio.to_thread` to avoid blocking the event loop.
   - Optional dependencies in `pyproject.toml` under `[embeddings]` and `[semantic]` extras.

4. **MCP Server Support** 🔲
   - `mcp_manager.py` — discover, connect to, and proxy MCP server tools.

5. **LM Studio WebSocket backend** 🔲
   - Optional; LM Studio currently works via its OpenAI-compatible HTTP endpoint.

---

## Key Decisions
- **MCP**: Anthropic's Model Context Protocol. External tool servers connected via stdio/SSE, exposed through the unified tool registry.
- **LLM backend**: OpenAI-compatible REST API is primary. LM Studio WebSocket is optional, selected via config or `--backend lmstudio`. No silent fallback between backends.
- **Configuration format**: YAML throughout. Layered: bundled defaults → `~/.ai-cli/config.yaml` → `<project>/.ai-cli/config.yaml` → CLI flags.
- **Workspace**: Nearest `.ai-cli/` ancestor when walking up from start directory, skipping `~/.ai-cli/`. Initialised via `--init`.
- **Permissions**: In-memory only, reset on exit or session resume. Universal options (Yes/No/Always/Custom rejection); tools may add their own variants. Universal four are always rendered by the prompt; `PermissionManager` passes only tool-specific extras to `prompt_fn`.
- **Tool discovery**: Three tiers — bundled → global (`~/.ai-cli/tools/`) → project (`.ai-cli/tools/`). Later tiers override earlier ones with a warning.
- **Session compaction**: LLM-generated summary. Two history files: `history_full.jsonl` (append-only) and `history_current.jsonl` (system + summary + recent messages).
- **Session resume**: `--resume` (pick from list), `--resume <id>` (direct), `--continue` (most recent or new). Flag routing wired into `__main__.py` via `_pick_session()` ✅. Remaining planned behaviours 🔲: (a) session list displaying name and last-message preview with role indicator (currently shows `first_user_message` only); (b) on resume, prompt to resend if the last message was from the user, or display the last assistant message in full so the user can respond.
- **Output modes**: Summary (default) and verbose, toggled via `/verbose` slash command (keyboard shortcut binding TBD).
- **`find_files` directory pruning**: Ignored directories are pruned from `os.walk` for performance. Files inside an ignored directory are never returned even if a negation rule would re-include them — this matches standard Git walk behaviour and is essential for avoiding traversal of `env/`, `.git/`, `node_modules/`, etc.
- **Multi-agent system**: Sub-agents are independent `Agent` instances with isolated sessions, tool registries, and optionally different models/backends. The coordinator dispatches via `CallAgentTool`. When `agents:` is absent/empty in config, no agent infrastructure is initialised and the CLI behaves identically to today. Sequential by default (single-GPU); parallel is opt-in.
- **Task system — hybrid orchestration**: Interactive mode (coordinator LLM drives tasks via tools) and autonomous mode (`/plan` command, deterministic Python orchestrator). Both share the same `tasks.json` file and task tools. The orchestrator spends zero LLM calls on routing — only sub-agent work uses the GPU.
- **Task system — reviewer is optional**: When no reviewer agent is configured, the executor marks tasks done directly. The `in_review` status is only meaningful when a reviewer is active.
- **Hardware target**: Primary deployment is a single consumer GPU (~24 GiB VRAM, 90K–262K token context depending on model). Sequential agent calls are the norm. Progressive disclosure is for focus, not token budget.
- **Multimodal `@` references — canonical content block format**: Image files attached via `@ref` are stored in session history using the OpenAI `chat/completions` content block shape (`"type": "text"` / `"type": "image_url"`) as the canonical in-memory and on-disk representation. Backend adapters that target other endpoints (e.g. the OpenAI `responses` API) translate to the required wire format before sending. This keeps the session layer backend-agnostic. See `docs/technical_requirements.md` — Multimodal Messages.
- **Embedding storage — SQLite over numpy+jsonl**: A single WAL-mode SQLite database (`.ai-cli/embeddings/index.db`) is used for all index data. Incremental upsert, per-file deletion, and atomic writes are trivially correct in SQLite; the equivalent with flat files requires compaction passes and file-replace tricks. Search still uses numpy vectorised cosine similarity (all vectors loaded into a float32 matrix). Human-readable metadata is accessible via the `sqlite3` CLI.
- **Indexed roots as access control**: Running `/index /path/to/external` records the path in the persistent SQLite `index_roots` table and grants `read_file` and `find_files` access to that path in all sessions until explicitly removed. Indexed roots are the only persisted access-control list in the system — all other ad-hoc permission grants (yes/always) remain in-memory and reset on process exit or session resume. No symlinks, no separate permission config. Removing a root with `/index --remove /path` revokes access immediately and permanently.
- **Chunking strategy — auto selection**: `make_chunker()` checks domain patterns first (Helm, Kubernetes, Ansible, Compose, TOML), then tree-sitter for source languages, then falls back to fixed-size. tree-sitter is optional; its absence is handled gracefully at construction time with no user-visible error.
- **Document-level embeddings — auto strategy**: Code files → average of chunk vectors (no LLM call, fast). Prose files (configurable by extension) → LLM-generated thematic summary → embed that text. Input is truncated to `summary_max_tokens * 4` chars (default 1600 chars). The summary prompt includes a word-count hint (`summary_response_tokens`, default `chunk_size // 4`) so the response fits one embedding chunk; no per-call API `max_tokens` cap is applied (reasoning models keep their full budget). Failed LLM calls fall back to `average` with a warning. LLM summary calls happen at `/index` time (dispatched via `asyncio.to_thread`), not at query time. `prose_extensions` are normalised to lowercase for case-insensitive matching.
- **Embedding backend config inheritance**: `embeddings.base_url` and `embeddings.api_key_env` inherit from the LLM backend config when absent or null. This allows Ollama (single port, different model names) to serve both the generation and embedding model with zero extra config.
- **Startup indexing deferred to phase 2**: Phase 1 is on-demand only (`/index` command). `EmbeddingIndex.index()` is `async` from day one so startup background indexing (`asyncio.create_task(...)`) requires no changes to `EmbeddingIndex` when added later.

## Assumptions
- The existing `lms_cli` is a starting point but not a strict requirement.
- The new project should prioritize flexibility and maintainability over backward compatibility.
- Tool execution should be decoupled from the workspace root directory.
- This is a single-user application; no multi-user isolation is required.

---

## Session Management
- Sessions are stored in `~/.ai-cli/sessions/<session-id>/` and are associated with a project by workspace path.
- Each session folder contains a `metadata.yaml` file with: session ID, workspace path, start time, message count, and an optional user-defined name.
- Sessions can be named from within the CLI using a slash command, e.g. `/session name "My feature work"`. The name is stored in `metadata.yaml` and shown in the `--resume` list.
- Three resume modes:
  - `--resume`: Show a list of sessions for the current project to pick from. Each entry shows:
    - Session ID
    - Date/time started
    - Message count
    - First user message (truncated)
    - Last message (truncated, with role indicated) to show where the session ended
  - `--resume <session-id>`: Resume a specific session directly by ID.
  - `--continue`: Automatically resume the most recent session for the current project. If none exists, start a new session silently.
- On resuming (any mode), the last message is handled based on its role:
  - If the last message is from the **user**: ask whether to resend it to the LLM (the message is already in history and must not be added again — only the API call is repeated).
  - If the last message is from the **assistant**: display it in full so the user knows what to respond to.
- All permissions reset when resuming a session regardless of resume mode.

---

## Next Steps (priority order)
1. **MCP support** — `mcp_manager.py` and integration with `ToolRegistry` (stdio + SSE transports).
2. **Multi-agent system** — `agent.py`, `agent_registry.py`, `call_agent.py`, `SubAgentDisplay`. REPL refactor to extract `Agent.run()`. See [design_agents.md](design_agents.md).
3. **Task system** — `task_manager.py`, `task_orchestrator.py`, `tasks.py`. `/plan` and `/tasks` slash commands. See [design_task_system.md](design_task_system.md).
4. **LM Studio WebSocket backend** — optional; LM Studio already works via its OpenAI-compatible HTTP endpoint.
5. **Interactive `@` file picker** — full popup/picker UX for `@path` references; image-file attachments as base64 content blocks.
6. **Session resume UX polish** — on resume, display last assistant message in full; prompt to resend if last message was from the user.
7. **`/tools allow` REPL command + permission mutation refactor** — Invert the default persistence of `ToolRegistry.set_permission_required()`: bare invocation is in-memory only for the current session; `--persist` writes to the project config (`user_confirmed: true`); `--global` writes to `~/.ai-cli/config.yaml` (requires a confirmation step because it affects all projects). Update the underlying `ToolRegistry` API to match — temporary by default — so callers that need in-memory overrides (e.g. per-agent `ToolRegistry` factories) do not need to work around the API.
8. **`find_files` access control for external roots** — call `workspace.embedding_index.is_indexed_path()` to validate any optional `path` parameter, consistent with how `read_file` handles external paths.

## Dependencies
- Python 3.10+
- `openai` — primary LLM backend
- `websockets` — optional LM Studio WebSocket backend
- `rich` — CLI formatting and output
- `pyyaml` — configuration file parsing
- `prompt_toolkit` — REPL input, tab completion, and `@` file picker
- `pydantic` — data validation (optional)
