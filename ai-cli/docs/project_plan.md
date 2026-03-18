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
   - **Configuration resolution** follows a layered model — each level overrides the previous:
     - **System prompt**:
       1. Built-in default bundled with the tool.
       2. User-global override: `~/.ai-cli/system_prompt.md` (if present).
       3. Project-specific override: `<project>/.ai-cli/system_prompt.md` (if present).
     - **Model/backend configuration** (no built-in default — must be explicitly provided):
       1. User-global config: `~/.ai-cli/config.yaml`.
       2. Project-specific override: `<project>/.ai-cli/config.yaml`.
       3. CLI flag overrides (highest priority).
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
│   ├── __main__.py                 # ✅ Entry point — --workspace, --init; --resume/--continue 🔲
│   ├── core/                       # Core functionality
│   │   ├── config_manager.py       # ✅ Layered YAML config loading
│   │   ├── workspace.py            # ✅ Workspace root resolution, file ops, ignore rules
│   │   ├── permission_manager.py   # ✅ In-memory permission state
│   │   ├── tool_registry.py        # ✅ Three-tier tool discovery, loading, settings
│   │   ├── llm_client.py           # ✅ LLMClient ABC + OpenAIClient (REST/streaming); LMStudio WebSocket 🔲
│   │   ├── session_manager.py      # ✅ Session create/resume/compact/persist
│   │   └── mcp_manager.py          # 🔲 MCP server connections, tool exposure
│   ├── tools/                      # Bundled tools
│   │   ├── base.py                 # ✅ Tool abstract base class
│   │   ├── read_file.py            # ✅ Read a file or line range from the workspace
│   │   ├── write_file.py           # ✅ Write or partially replace a file in the workspace
│   │   ├── find_files.py           # ✅ Glob-pattern file search with ignore-rule enforcement
│   │   └── tool_manager.py         # 🔲 Context-saving tool gatekeeper
│   ├── cli/                        # CLI interface and user-facing components
│   │   ├── repl.py                 # ✅ REPL loop; slash commands ⚠️ (subset implemented — see Phase 3)
│   │   ├── display.py              # ⚠️ Display ABC + PlainDisplay ✅; RichDisplay 🔲
│   │   └── completer.py            # 🔲 Tab completion + interactive @ file picker
│   └── utils/                      # Utility functions and helpers
│       ├── ignore_filter.py        # ✅ .gitignore-style pattern matching
│       └── logging_utils.py        # 🔲 JSONL structured logging
├── tests/                          # ✅ Unit tests mirroring ai_cli/ structure
│   ├── test_workspace.py
│   ├── test_ignore_filter.py
│   ├── test_config_manager.py
│   ├── test_permission_manager.py
│   ├── test_tool_registry.py
│   ├── test_tool_base.py
│   ├── test_read_file.py
│   ├── test_write_file.py
│   ├── test_find_files.py
│   └── test_main.py
└── docs/                           # Documentation
    └── project_plan.md             # This file
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

4. **MCP Support** 🔲
   - Implement a `MCPManager` class to discover, connect to, and communicate with MCP servers.
   - Support stdio and SSE transports as defined by the Model Context Protocol.
   - Expose MCP server tools through the same tool registry as built-in tools.

### Phase 2: Tooling and Execution
1. **Tool Execution Improvements** ✅
   - Canonical `{"status": "success"/"error", ...}` response shape standardised via `_ok()`/`_err()` helpers and followed by built-in tools by convention — nothing enforces that third-party tools use them.
   - `ToolRegistry.execute()` handles unknown tool, disabled tool, permission denied, and execution errors — all return canonical error dicts.
   - `allow_transient=True` parameter lets the REPL execute transiently-injected tools that aren't in the persistent enabled set.

2. **Permission System** ✅
   - `PermissionManager` handles in-memory grants (yes/no/always/custom rejection).
   - `always` grants are stored per tool name. They persist unless `PermissionManager.reset()` is explicitly called — the session manager must call both `PermissionManager.reset()` and `ToolRegistry.reset_session_overrides()` on session resume to clear all session-scoped state.
   - File tools (`read_file`, `write_file`) additionally manage session-scoped file/dir allow-lists at the tool level via `extra_permission_options()` / `on_permission_granted()` / `reset_session_state()`, which are cleared by `ToolRegistry.reset_session_overrides()`.
   - The universal four options (yes/no/always/custom) are always rendered by the prompt implementation. `PermissionManager` passes only tool-specific extras to `prompt_fn`; the prompt handles the universal set itself.

3. **Bundled Tools** ⚠️ (partial)
   - `read_file` ✅ — workspace-scoped, no permission by default, session allow-list, line-range support.
   - `write_file` ✅ — workspace-scoped, permission required by default, session allow-list, full and partial writes.
   - `find_files` ✅ — glob-pattern search across the workspace. Supports `*`, `**`, `?`, `[ranges]`, `{alternation}`. Respects all ignore rules (global `.ignore`, project `.gitignore`, project `.ai-cli/.ignore`). Prunes ignored directories during traversal for performance (matching standard Git walk behaviour).
   - `tool_manager` 🔲 — **next priority** now that the REPL exists.

4. **Error Handling** ⚠️ (partial)
   - Structured error dicts returned by all tool calls. ✅
   - JSONL logging (`logging_utils.py`) 🔲
   - Session-specific log folders in `~/.ai-cli/` 🔲

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
   - `@path` inline file reference expansion (text substitution). `@path` respects ignore rules; `@!path` bypasses them.
   - Implemented slash commands: `/help`, `/exit`, `/clear`, `/verbose`, `/compact`, `/markdown`, `/tools`, `/session`.

4. **Display** ⚠️ (partial)
   - `Display` ABC with full interface defined.
   - `PlainDisplay` ✅ — `print()`-based output, `prompt_toolkit` for interactive prompts.
   - `RichDisplay` 🔲 — Rich-formatted output; currently falls back to `PlainDisplay`.

5. **Remaining CLI completions** 🔲
   - **`--resume` / `--resume <id>` / `--continue` CLI flags** in `__main__.py` — session resume at startup.
   - **`/tools` subcommands** — currently `/tools` only lists enabled tools. Planned subcommands:
     - `/tools list` — list all tools (enabled and disabled) with tier and status.
     - `/tools info <name>` — full details: description, parameters, current settings.
     - `/tools enable <name> [--session]` / `/tools disable <name> [--session]`
     - `/tools allow <name>` / `/tools disallow <name>` — toggle `permission_required`.
   - **`/compact [instructions]`** — optional instructions argument (currently ignored).
   - **`/session name "<name>"`** — currently `/session` only shows info; naming not yet wired up.
   - **`completer.py`** — tab completion for slash commands, tool names, and file paths. Interactive `@` popup/picker (vs. the current text-substitution approach).

6. **Logging** 🔲
   - `logging_utils.py` — JSONL structured logging to session-specific folders.

### Phase 4: Advanced Features 🔲
1. **`tool_manager` tool** → **next priority**
   - Context-saving tool gatekeeper; `list` and `enable` actions.
   - Requires `ToolRegistry.enable_transient()`.

2. **MCP Server Support**
   - `mcp_manager.py` — discover, connect to, and proxy MCP server tools.

3. **LM Studio WebSocket backend**
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
- **Session resume**: `--resume` (pick from list), `--resume <id>` (direct), `--continue` (most recent or new). 🔲 Not yet wired into `__main__.py`.
- **Output modes**: Summary (default) and verbose, toggled via `/verbose` slash command (keyboard shortcut binding TBD).
- **`find_files` directory pruning**: Ignored directories are pruned from `os.walk` for performance. Files inside an ignored directory are never returned even if a negation rule would re-include them — this matches standard Git walk behaviour and is essential for avoiding traversal of `env/`, `.git/`, `node_modules/`, etc.

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
1. **`tool_manager` tool** — implement `list` and `enable` actions, `ToolRegistry.enable_transient()`, and REPL integration for transient tool injection.
2. **`--resume` / `--continue` CLI flags** — wire `SessionManager` into `__main__.py` startup.
3. **`/tools` subcommands** — expand `/tools` beyond the current simple list.
4. **`/session name`** and **`/compact [instructions]`** — complete the remaining slash commands.
5. **`RichDisplay`** — replace `PlainDisplay` with Rich-formatted output.
6. **`completer.py`** — tab completion and interactive `@` file picker.
7. **`logging_utils.py`** — JSONL structured logging.
8. **MCP support** — `mcp_manager.py` and integration with `ToolRegistry`.

## Dependencies
- Python 3.10+
- `openai` — primary LLM backend
- `websockets` — optional LM Studio WebSocket backend
- `rich` — CLI formatting and output
- `pyyaml` — configuration file parsing
- `prompt_toolkit` — REPL input, tab completion, and `@` file picker
- `pydantic` — data validation (optional)
