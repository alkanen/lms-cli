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

Files not yet implemented are marked *(planned)*.

```
ai-cli/
├── ai_cli/                         # Python package root
│   ├── __main__.py                 # Entry point (python -m ai_cli)
│   ├── core/                       # Core functionality
│   │   ├── config_manager.py       # Layered YAML config loading
│   │   ├── workspace.py            # Workspace root resolution, file ops, ignore rules
│   │   ├── permission_manager.py   # In-memory permission state
│   │   ├── tool_registry.py        # Three-tier tool discovery, loading, settings *(planned)*
│   │   ├── llm_client.py           # Abstract LLMClient + OpenAI/LMStudio implementations *(planned)*
│   │   ├── mcp_manager.py          # MCP server connections, tool exposure *(planned)*
│   │   └── session_manager.py      # Session create/resume/compact/persist *(planned)*
│   ├── tools/                      # Bundled tools *(planned)*
│   │   └── tool_manager.py         # Context-saving tool gatekeeper *(planned)*
│   ├── cli/                        # CLI interface and user-facing components *(planned)*
│   │   ├── repl.py                 # Main REPL loop, input handling, slash commands *(planned)*
│   │   ├── completer.py            # Tab completion + @ file picker *(planned)*
│   │   └── display.py              # Rich output, summary/verbose modes *(planned)*
│   └── utils/                      # Utility functions and helpers
│       ├── ignore_filter.py        # .gitignore-style pattern matching
│       └── logging_utils.py        # JSONL structured logging *(planned)*
├── tests/                          # Unit and integration tests
│   ├── test_workspace.py
│   ├── test_ignore_filter.py
│   ├── test_config_manager.py
│   ├── test_permission_manager.py
│   └── ...                         # Additional test files
└── docs/                           # Documentation
    └── project_plan.md             # This file
```

## Implementation Plan
### Phase 1: Core Infrastructure
1. **Workspace Handling**
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

2. **Configuration Management**
   - Create a `ConfigManager` class to load layered configurations from YAML files.
   - Implement fallback mechanisms for missing configurations.
   - Support CLI overrides for critical parameters (e.g., config file paths, server addresses).

3. **Tool Registry Enhancements**
   - Redesign the `ToolRegistry` to support dynamic tool discovery from three tiers, loaded in order:
     1. **Bundled tools**: Packaged with `ai-cli` itself (e.g., file read/write, basic shell). Always available.
     2. **Global user tools**: `~/.ai-cli/tools/` — available in all projects for that user.
     3. **Project tools**: `<project>/.ai-cli/tools/` — available only within that project.
   - Tools discovered later in the load order can override earlier ones by name. The user is warned at startup when an override occurs, but it is allowed.
   - Add metadata validation for tools at load time.
   - Per-tool settings are read from a `tools` mapping keyed by tool name in both `~/.ai-cli/config.yaml` (global) and `<project>/.ai-cli/config.yaml` (project), merged in that order (global → project → CLI flags). If a tool is not mentioned, or a key is absent, the tool's own declared defaults apply. Each tool declares `NAME`, `DESCRIPTION`, and `PERMISSION_REQUIRED` as class attributes (e.g., `read_file` sets `PERMISSION_REQUIRED = False`, `write_file` sets it to `True`). **Trust distinction**: global config is treated as trusted (the user's own file); project config is untrusted (a cloned repo could contain it). Lowering `permission_required` from `true` to `false` is therefore allowed unconditionally from global config, but requires an explicit `user_confirmed: true` marker in the project config entry (written automatically by the `/tools allow` REPL command). Example:
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
   - **Security**: Project-level `config.yaml` is treated as untrusted. Settings that weaken security — such as disabling `permission_required` or enabling `allow_outside_workspace` — must trigger an explicit user confirmation prompt at startup before taking effect. They cannot silently override the safe baseline.

4. **MCP Support**
   - Implement a `MCPManager` class to discover, connect to, and communicate with MCP servers.
   - Support stdio and SSE transports as defined by the Model Context Protocol.
   - Expose MCP server tools through the same tool registry as built-in tools.

### Phase 2: Tooling and Execution
1. **Tool Execution Improvements**
   - Enforce structured output (e.g., JSON) for all tools where applicable.
   - Add schema validation for tool responses.
   - Implement retry logic for transient failures.
   - Tools with `permission_required: True` must request permission for every action, with granular options (Yes/No/Always/Custom rejection or tool-specific suggestions).

2. **Permission System**
   - Introduce a `PermissionManager` to handle tool permissions in-memory for the current process only.
   - Permissions are never written to disk. When the CLI exits or a session is resumed, all permissions reset and must be granted anew.
   - This ensures the tool never silently retains permissions across changed circumstances.

3. **Error Handling**
   - Standardize error formats (e.g., JSON with `error`, `message`, `code`).
   - Add logging integration using Python's `logging` module.
   - Session-specific folders in `~/.ai-cli/` to store metadata, session history, and error logs in JSONL format.

### Phase 3: CLI and User Experience
1. **CLI Interface**
   - Redesign the CLI interface to be more intuitive and user-friendly.
   - Support for interactive mode with:
     - Tab completion for commands, tools, slash commands, and file paths.
     - Help system (e.g., `--help`, inline help).
     - Rich output formatting (e.g., colors, tables).
     - REPL-like interface for iterative tool execution.
   - Popup-like mechanism triggered by `@` for file path completion in the current directory and its children.
     - `@` — filtered mode: only shows files not matching `.ignore` patterns.
     - `@!` — explicit mode: shows all files regardless of ignore rules, for intentional override.
     - Abort on escape or backspacing far enough to delete the trigger (`@` or `@!`).
     - Include file data as attachments with clear references for the LLM.
   - **Slash commands** available from within the REPL:
     - `/help [topic]` — show available slash commands and keyboard shortcuts. Optional topic argument gives detailed help on a specific command or subject.
     - `/exit` — exit the CLI cleanly.
     - `/clear` — clear the terminal display (does not affect session history).
     - `/tools` — alias for `/tools list`.
       - `/tools list` — list all loaded tools with their source tier (bundled/global/project) and current enabled/permission status.
       - `/tools info <name>` — show full details for a specific tool: description, parameters, and current settings.
       - `/tools enable <name> [--session]` / `/tools disable <name> [--session]` — toggle a tool's enabled state.
         - Without `--session`: change is written to project-level `.ai-cli/config.yaml` and persists across sessions.
         - With `--session`: change is in-memory only for the current session, reset on exit or resume. No config write.
       - `/tools allow <name>` / `/tools disallow <name>` — toggle `permission_required` for a tool.
       - Persistent changes (without `--session`) are written to the project-level `.ai-cli/config.yaml`. If the tool is not yet listed there, it is added with only the changed key; all other settings remain at their defaults.
     - `/compact [instructions]` — manually trigger session compaction. Optional instructions guide the LLM on what to emphasise in the summary.
     - `/session name "<name>"` — assign a human-readable name to the current session, stored in `metadata.yaml`.

   - **Output verbosity modes** (toggleable during a session via keyboard shortcut):
     - **Summary mode** (default): Show condensed activity — e.g., tool name being called, one-line status. LLM response text is always shown in full.
     - **Verbose mode**: Show full tool inputs, outputs, LLM reasoning, and all intermediate steps.
     - Toggle between modes with a keyboard shortcut (e.g., Ctrl+O or Ctrl+E — exact binding TBD).

2. **Testing and Validation**
   - Write unit tests for core components (e.g., `ToolRegistry`, `Workspace`).
   - Integrate integration tests for tool execution and configuration.

3. **Documentation**
   - Update documentation to reflect new features and improvements.
   - Add examples for tool development and usage.

## Key Decisions
- **MCP**: Anthropic's Model Context Protocol. External tool servers connected via stdio/SSE, exposed through the unified tool registry.
- **LLM backend**: OpenAI-compatible REST API is primary. LM Studio WebSocket is optional, selected via config or `--backend lmstudio`. No silent fallback between backends.
- **Configuration format**: YAML throughout. Layered: bundled defaults → `~/.ai-cli/config.yaml` → `<project>/.ai-cli/config.yaml` → CLI flags.
- **Workspace**: Nearest `.ai-cli/` ancestor when walking up from start directory, skipping `~/.ai-cli/`. Initialised via `--init`.
- **Permissions**: In-memory only, reset on exit or session resume. Universal options (Yes/No/Always/Custom rejection); tools may add their own variants.
- **Tool discovery**: Three tiers — bundled → global (`~/.ai-cli/tools/`) → project (`.ai-cli/tools/`). Later tiers override earlier ones with a warning.
- **Session compaction**: LLM-generated summary. Two history files: `history_full.jsonl` (append-only) and `history_current.jsonl` (system + summary + recent messages).
- **Session resume**: `--resume` (pick from list), `--resume <id>` (direct), `--continue` (most recent or new).
- **Output modes**: Summary (default) and verbose, toggled by keyboard shortcut during session.

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

## Next Steps
1. Draft initial implementations for core components (`Workspace`, `ConfigManager`, `LLMClient`).
2. Design tool base class interface and metadata standards, including permission handling.
3. Implement the tool registry with three-tier discovery and per-tool config loading.
4. Implement MCP server connection manager.
5. Build the REPL interface with slash commands, verbosity toggle, and `@` file picker.
6. Write unit tests for core components.

## Dependencies
- Python 3.10+
- `openai` — primary LLM backend
- `websockets` — optional LM Studio WebSocket backend
- `rich` — CLI formatting and output
- `pyyaml` — configuration file parsing
- `prompt_toolkit` — REPL input, tab completion, and `@` file picker
- `pydantic` — data validation (optional)
