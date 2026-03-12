# Technical Requirements for AI CLI Tool

## Overview
This document outlines the technical requirements for the AI CLI tool project. It covers details that are essential for planning, implementation, and testing but may not fit neatly into the main project plan.

---

## Tool-LLM Communication
- **Schema Export**:
  - Tools must export schemas in the OpenAI function-calling format via a `definition()` method.
  - The name, description, and parameters are defined by each tool individually.
  - Each parameter includes `type` and `description`. Required parameters are listed in a top-level `required` array on the `parameters` object, per the OpenAI function-calling / JSON Schema specification.
  - Example schema (from the implemented `read_file` tool):
    ```json
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a file (or line range) from the workspace. Returns start_line, end_line, lines_returned, and total_lines (1-based, inclusive). For an empty file, start_line and end_line are both 0.",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "Path to the file, relative to the workspace root (e.g. './src/main.py')."
            },
            "start_line": {
              "type": "integer",
              "description": "1-based first line to read (inclusive). Omit to start from the beginning of the file."
            },
            "end_line": {
              "type": "integer",
              "description": "1-based last line to read (inclusive). Omit to read to the end of the file."
            }
          },
          "required": ["path"]
        }
      }
    }
    ```

- **Implemented tools** (as of current state):
  - `read_file` — `path` (required), `start_line`, `end_line` (optional, 1-based inclusive)
  - `write_file` — `path`, `content` (required), `start_line`, `end_line` (optional, must be provided together)

- **LLM Tool Calls**:
  - The LLM should call tools using function calls or API endpoints.
  - The OpenAI API enforces the JSON Schema before the call reaches the tool, so structurally invalid calls (missing required args, wrong types) are rejected at the API boundary and never reach `execute()`. Tools therefore only need to validate semantic constraints (e.g. `start_line > total_lines`), which they return as canonical 4xx error dicts.

---

## Serialization Formats
- **Tool Inputs/Outputs**:
  - Use JSON for tool inputs and outputs.
  - Tool input example (`read_file` with an optional line range):
    ```json
    {
      "path": "./src/main.py",
      "start_line": 10,
      "end_line": 30
    }
    ```
  - Tool output example (success, `read_file`):
    ```json
    {
      "status": "success",
      "data": {
        "content": "def main():\n    ...\n",
        "path": "./src/main.py",
        "start_line": 10,
        "end_line": 30,
        "lines_returned": 21,
        "total_lines": 120
      }
    }
    ```
  - Tool output example (error, `read_file` on a missing or ignored file):
    ```json
    {
      "status": "error",
      "error": "read_error",
      "message": "File not found: './src/missing.py'",
      "code": 400
    }
    ```

- **LLM Messages**:
  - Use JSON for LLM messages, including user inputs and tool responses.
  - Example:
    ```json
    {
      "role": "user",
      "content": "List the files in the current directory."
    }
    ```

- **Canonical Tool Response Schema**:
  - All tools return a JSON object with a consistent shape. The `error`, `message`, `code`, and `details` fields are only present on error responses and omitted on success.
  - Success response:
    ```json
    {
      "status": "success",
      "data": {}
    }
    ```
  - Error response:
    ```json
    {
      "status": "error",
      "error": "error_code",
      "message": "Human-readable description.",
      "code": 400,
      "details": {}
    }
    ```
  - The `data`, `details` fields are optional. All other fields are required. The error format in the **Error Handling** section below uses the same shape.

---

## LLM Backend

### Abstraction Layer
- All LLM communication goes through a unified `LLMClient` interface.
- The active backend is selected via configuration file, with a CLI flag override (e.g., `--backend openai` or `--backend lmstudio`).
- The interface must support: sending messages, receiving streamed responses, and querying model metadata (context window, token limits).

### OpenAI-Compatible REST API (Primary)
- **Authentication**:
  - Prefer environment variables for API keys (e.g., `OPENAI_API_KEY`); never commit API keys to source control.
  - If config-file storage is used, the file must live only under `~/.ai-cli/` (never inside a project or repo workspace) and be created with strict, user-only file permissions (e.g., `chmod 600`).
  - Document secure setup steps (e.g., example `.env` usage) in user-facing configuration instructions.

- **Rate Limiting**:
  - Handle 429 errors gracefully with retries.
  - Implement exponential backoff for retries.

- **Error Handling**:
  - Log failed requests with details (e.g., status code, error message).
  - Retry transient errors (e.g., network issues) automatically.

- **Model Metadata**:
  - Context window and token limits must be provided via configuration when using this backend, as the OpenAI API does not expose them reliably.

### LM Studio WebSocket (Optional)
- Selected via config (`backend: lmstudio`) or CLI override (`--backend lmstudio`).
- Preferred when available because it exposes richer model metadata directly.

- **Connection Management**:
  - Establish and maintain WebSocket connections for real-time communication.
  - Handle connection drops gracefully with reconnection logic.

- **Model-Specific Configurations**:
  - Read token limits and context windows from model metadata returned by LM Studio.
  - These values override any config-file defaults when the LM Studio backend is active.
  - Example:
    ```json
    {
      "max_response_tokens": 60000,
      "context_window": 60000
    }
    ```

- **Error Handling**:
  - Log connection issues or invalid responses.
  - Notify the user if the LLM returns unexpected data.
  - Fall back gracefully with a clear error if LM Studio is unreachable (do not silently fall back to OpenAI).

---

## Performance Considerations
- **Session History Management**:
  - Each session maintains two files in its session folder under `~/.ai-cli/sessions/<session-id>/`:
    - `history_full.jsonl`: Append-only, complete record of every message including compaction responses. Never modified, only appended to.
    - `history_current.jsonl`: The active context sent to the LLM. Structure:
      1. System message (always first).
      2. A single compaction summary message (if compaction has occurred), with role `system` or `assistant` and a note that it is a summary.
      3. All subsequent messages since the last compaction.
  - Use JSONL format (one message object per line) for both files.

- **Token Limit Management**:
  - Compact sessions when approaching the token limit (default: 10% remaining).
  - Monitor token usage in real-time and warn the user when nearing limits.
  - **Compaction process**:
    1. Send the current history to the LLM with a prompt requesting a concise summary.
    2. Append the summary response to `history_full.jsonl`.
    3. Rewrite `history_current.jsonl` with: system message + summary message + any messages received after the compaction request.
    4. Notify the user that compaction occurred.

- **Long-Running Tasks**:
  - All tool execution runs synchronously within the session. The user sees streamed output and can interrupt at any time (e.g., Ctrl+C).
  - No background task or subprocess system is required. Progress is communicated through the normal streamed output in verbose mode.

---

## Testing Strategies
- **Unit Tests**:
  - Test core components (e.g., `ToolRegistry`, `Workspace`, `ConfigManager`).
  - Mock external dependencies (e.g., LLM, tools) for isolated testing.

- **Integration Tests**:
  - Test tool execution and LLM interactions end-to-end.
  - Verify CLI commands and their outputs.

- **Edge Cases**:
  - Test invalid inputs, network errors, and unexpected LLM responses.
  - Ensure graceful degradation when external systems fail.

---

## Error Handling
- **Standardized Errors**:
  - Use the canonical tool response schema defined in the Serialization Formats section above (`status: "error"` with `error`, `message`, `code`, and optional `details`).

- **Logging**:
  - Log errors in JSONL format (one entry per line) for structured data handling.
  - Example:
    ```jsonl
    {"timestamp": "2023-10-25T14:30:22Z", "level": "error", "message": "Failed to read file", "details": { ... }}
    {"timestamp": "2023-10-25T14:30:23Z", "level": "warning", "message": "Token limit approaching", "details": { ... }}
    ```

- **User Notifications**:
  - Notify the user of errors in a user-friendly manner (e.g., color-coded messages).
  - Provide actionable suggestions for recovery.

---

## Additional Considerations

- **Tool class attributes**:
  - `NAME: str` — canonical name used in schemas, config, and slash commands.
  - `DESCRIPTION: str` — one-line description shown in `tool_manager list` and the LLM schema.
  - `PERMISSION_REQUIRED: bool` — tool's own default; overridable via config.
  - `DISABLED_BY_DEFAULT: bool` (optional, default `False`) — set `True` to start the tool disabled in the registry. Config and session overrides can still enable it. This is how most bundled tools will opt out of the default tool list until the user or LLM enables them.

- **Tool session state**:
  - Tools with session-scoped state (e.g. per-path permission allow-lists) implement `reset_session_state()`.
  - `ToolRegistry.reset_session_overrides()` calls this hook on every tool when a session is resumed, clearing tool-level session state (e.g. per-path allow-lists). Each call is guarded individually — a bug in one tool's hook does not prevent other tools from being reset.
  - `PermissionManager` "always" grants are **not** cleared by `reset_session_overrides()`. They must be reset separately by calling `PermissionManager.reset()`. The startup sequence is responsible for calling both.

- **Ignore File**:
  - `.ai-cli/.ignore` uses `.gitignore` syntax (glob patterns, negation with `!`, comments with `#`).
  - Paths matching the ignore rules are excluded from LLM context and tool access.
  - Both project-level (`.ai-cli/.ignore`) and global (`~/.ai-cli/.ignore`) ignore files are applied, with project-level patterns taking precedence.

- **Security**:
  - Sanitize user inputs to prevent injection attacks.
  - Validate file paths and tool parameters strictly.

- **Extensibility**:
  - Design the system to support future additions (e.g., new tools, LLM models).
  - Use dependency injection for modularity.

- **Documentation**:
  - Document all public APIs, CLI commands, and tool schemas.
  - Include examples for common use cases.
