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
  - Text-only example:
    ```json
    {
      "role": "user",
      "content": "List the files in the current directory."
    }
    ```
  - Multimodal example (text + image — see Multimodal Messages section below):
    ```json
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "What does this diagram show?" },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,<b64data>",
            "detail": "auto"
          }
        }
      ]
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

## Multimodal Messages

When the user attaches an image file via `@path/to/image.png`, the user message
must be sent to the LLM as a **content block array** rather than a plain string.
The exact wire format depends on the API endpoint in use.

### Canonical in-memory / on-disk format

The session layer and all internal code use the OpenAI `chat/completions` content
block shape as the canonical representation. This is what gets stored in
`history_current.jsonl` and `history_full.jsonl`:

```json
{
  "role": "user",
  "content": [
    { "type": "text",      "text": "Explain this architecture diagram." },
    { "type": "image_url", "image_url": { "url": "data:image/png;base64,<b64>", "detail": "auto" } }
  ]
}
```

Pure-text messages continue to use a plain `"content": "string"` value.
Mixed text+image messages always use the array form even if only one block is
present.

### OpenAI `chat/completions` API

The current backend (`OpenAIClient`) uses `client.chat.completions.create()`.
This endpoint accepts content blocks natively:

| Content type | Block shape |
|---|---|
| Text | `{"type": "text", "text": "..."}` |
| Image (base64) | `{"type": "image_url", "image_url": {"url": "data:<mime>;base64,<b64>", "detail": "auto\|low\|high"}}` |

The canonical format is sent as-is — no translation needed.

### OpenAI `responses` API (future)

The `/v1/responses` endpoint uses different block type names. An `LLMClient`
adapter targeting this endpoint must rewrite blocks before sending:

| Canonical type | Responses API type |
|---|---|
| `"text"` | `"input_text"` (field: `"text"`) |
| `"image_url"` | `"input_image"` (field: `"image_url"`, data URI unchanged) |

Example translated block:
```json
{ "type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,<b64>" }
```

The translation is the adapter's responsibility. Session history is always
stored in canonical form so it can be replayed through any future adapter.

### LM Studio (OpenAI-compatible REST)

LM Studio's REST endpoint accepts the same `image_url` block shape as the
OpenAI `chat/completions` API. No special handling is required when using LM
Studio via the `openai` backend with a custom `base_url`. Vision capability
depends on the loaded model; if the model lacks vision support the API returns
an error, which surfaces to the user as an `LLMError`.

### Image encoding

Images are always inlined as base64 data URIs. URL-based references (remote
images) are not supported — all content must be workspace-local.

```python
import base64
data = path.read_bytes()
b64 = base64.b64encode(data).decode()
url = f"data:{mime_type};base64,{b64}"
```

### Supported image formats

| Extension | MIME type | Notes |
|---|---|---|
| `.png` | `image/png` | |
| `.jpg`, `.jpeg` | `image/jpeg` | |
| `.gif` | `image/gif` | OpenAI treats animated GIFs as a single frame |
| `.webp` | `image/webp` | |

Files with any other extension are treated as text. Future work: MIME-sniff the
leading bytes for extensionless or misnamed files.

### Token counting

`count_tokens()` uses tiktoken, which only counts text tokens. Image token cost
is model- and resolution-dependent. OpenAI's tile-based pricing:

- `"detail": "low"` — fixed 85 tokens per image.
- `"detail": "high"` — 85 base + 170 per 512×512 tile (capped at the image
  size). A 1024×1024 image costs 85 + 4×170 = 765 tokens.
- `"detail": "auto"` — the model decides; treat as `"high"` for worst-case
  estimates.

The local tiktoken estimate will undercount when images are present. Compaction
threshold decisions should prefer the actual `usage` figures returned by the API
in the `"done"` chunk over the local estimate.

---

## LLM Backend

### Abstraction Layer
- All LLM communication goes through a unified `LLMClient` interface.
- The active backend is selected via configuration file, with a CLI flag override (e.g., `--backend openai` or `--backend lmstudio`).
- The interface must support: sending messages, receiving streamed responses, and querying model metadata (context window, token limits).

### Chunk types

All backends yield the same normalised chunk dict format:

| Type | Fields | Notes |
|---|---|---|
| `text` | `delta: str` | LLM response text delta; yield immediately as received |
| `reasoning` | `delta: str` | Reasoning/thinking content delta (see below) |
| `tool_call` | `name`, `call_id`, `arguments: dict` | Buffered internally until complete; emitted as one chunk |
| `done` | `stop_reason: str`, `usage: dict` | Always the last chunk; `usage` has `prompt_tokens`, `completion_tokens`, `total_tokens` |

### Reasoning content

Several model families expose internal chain-of-thought reasoning separately
from the visible response.  The LLMClient normalises all sources into
`{"type": "reasoning", "delta": str}` chunks so the Display layer handles
them uniformly.

**Source 1 — `reasoning_content` delta field (OpenAI `o1`, `o3`, compatible
models):** The streaming delta may carry a `reasoning_content` field alongside
`content`.  The backend extracts it and emits a `reasoning` chunk in parallel
with any `text` chunk from the same delta.

**Source 2 — `<think>…</think>` tags embedded in the text stream (DeepSeek R1,
QwQ, and similar open-source reasoning models):** The content between the tags
is reasoning; text outside the tags is the visible response.  A stateful
`_ThinkTagParser` in the LLMClient intercepts the text stream:

```
State machine:
  OUTSIDE  →  text before <think>   → emit as "text" chunks
  INSIDE   →  text between tags     → emit as "reasoning" chunks
  OUTSIDE  ←  text after </think>   → emit as "text" chunks

Tag boundaries may fall mid-chunk; the parser buffers until the full tag is
confirmed or ruled out.
```

The parser is implemented as a helper class on the backend, not in the Display
or REPL.  This keeps the two layers decoupled: adding a new reasoning source
(e.g. Anthropic thinking blocks, a future model convention) requires only a
new extraction path in the LLMClient, with no changes to Display or REPL.

The REPL routes `reasoning` chunks to `display.stream_reasoning(delta)` in the
same streaming loop as `text` chunks.

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

## Multi-Agent Processing

See `design_agents.md` for the full architecture.  This section covers the
technical constraints that affect implementation.

- **Sequential execution (default)**:
  - A single consumer GPU can process one LLM request at a time.  Sub-agent
    calls block the coordinator until the sub-agent's `Agent.run()` returns.
  - Parallel execution (`call_agents_parallel`) is opt-in via
    `agent_settings.allow_parallel: true` and uses `asyncio.gather()`.  Only enable
    when the backend supports concurrent requests (e.g. remote API, multiple
    GPUs).

- **Agent isolation**:
  - Each agent has its own `Session`, `ToolRegistry`, and `Display` instance.
    Agents share state only through the task file (see Task Storage below) and
    through the `AgentResult` returned to the coordinator.
  - Sub-agents use `SubAgentDisplay`, which captures streaming output in a
    buffer and defaults permission prompts to "no".  Sub-agents must never
    write to the terminal directly.

- **Context overflow**:
  - After each LLM turn, `Agent.run()` checks token usage from the `done`
    chunk against the model's context window.  When usage exceeds
    `context_limit_threshold` (default 90 %), the loop breaks and returns
    `AgentResult(status="context_limit", partial=True)`.
  - No automatic compaction is performed on sub-agent sessions — the
    coordinator receives the partial result and decides whether to retry with
    a fresh ephemeral instance, escalate, or accept the partial result.

- **Tool round limits**:
  - `max_tool_rounds` (default 10) caps the number of tool-call rounds per
    `Agent.run()` invocation.  Exceeding this limit returns
    `AgentResult(status="tool_limit", partial=True)`.

---

## Task Storage

See `design_task_system.md` for the full schema and tool definitions.

- **File location**: `<session_dir>/tasks.json`.  One file per CLI session,
  co-located with session history files.

- **Format**: JSON object with an optional top-level `goal` field and a
  top-level `tasks` map keyed by task IDs of the form `task_<...>`.  Each
  entry maps a task ID to a task object with `name`, optional `parent_id`,
  `subtask_ids`, `status`, `priority`, `next_action`, `description`,
  `definition_of_done` (string), `notes`, timestamps, and `blockers`.

- **Integrity rules enforced by `TaskManager`**:
  - `parent_id` must reference an existing task (or be `null`).
  - Valid status values: `not_started`, `in_progress`, `in_review`, `blocked`,
    and `done`.  Tasks start in `not_started`, move to `in_progress`, and may
    transition to `in_review` when ready for validation.  `blocked` can be set
    from any non-`done` state and transitions back to `in_progress` once
    blockers are resolved.  Only `tasks_mark_done` may set status to `"done"`.
  - `tasks_mark_done` validates that all subtasks (if any) are `done` and may
    require the task to have a non-empty `definition_of_done` string.  The LLM
    and/or human reviewer is responsible for determining that the
    natural-language `definition_of_done` criteria are actually satisfied
    before calling `tasks_mark_done`.
  - `tasks_update` cannot set `status` to `"done"` — the `tasks_mark_done`
    tool must be used instead.

- **Concurrent access**: Only one agent runs at a time in sequential mode, so
  file locking is unnecessary.  If parallel execution is enabled, `TaskManager`
  must use OS-appropriate file locking (e.g. `fcntl.flock()` on Unix-like
  systems, or a cross-platform file-locking library / Windows-specific locking
  API) to prevent write races.  Note that `fcntl` is Unix-only.

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

- **Agent Tests**:
  - Test `Agent.run()` with mock `LLMClient` and `ToolRegistry` in isolation.
  - Verify tool registry isolation: tools on different agents are independent
    instances with separate state.
  - Test context overflow detection (mock a `done` chunk near the threshold).
  - Test tool round limit enforcement.
  - Test `SubAgentDisplay` captures output and denies permission prompts.

- **Task Tests**:
  - Test `TaskManager` CRUD operations and status transition validation.
  - Test integrity rules: invalid `parent_id`, illegal transitions,
    `tasks_mark_done` without meeting DoD criteria.
  - Test concurrent access guards when parallel execution is enabled.

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
