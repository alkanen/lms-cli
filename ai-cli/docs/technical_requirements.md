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

- **Tool name constraints**:
  - All tool names (built-in and MCP) must satisfy the enforced pattern `^[a-zA-Z0-9_][a-zA-Z0-9_-]{0,63}$`. In other words, a tool name must start with an ASCII letter, digit, or underscore; hyphens are allowed after the first character; the maximum length is 64 characters. Slashes, dots, and other characters are not permitted.
  - MCP tool names are namespaced as `<server>__<tool>` (double underscore) to avoid collisions. The combined name must still satisfy the same pattern and 64-character limit; names that exceed it or otherwise fail validation are skipped at registration with a warning.

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
    `agent_settings.allow_parallel: true` and uses
    `concurrent.futures.ThreadPoolExecutor` (not `asyncio`).  Only enable
    when the backend supports concurrent requests (e.g. remote API, multiple
    GPUs).  Maximum batch size defaults to 10; configurable via
    `agent_settings.max_parallel_calls`.  Session-persistent agent types may
    not appear more than once in a single parallel call (shared state).

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
    `context_limit_threshold` (default 90 %), the stream loop breaks (not an
    early return — the assistant message is persisted first), any dangling
    tool_calls receive stub error responses so session history stays consistent,
    and then returns
    `AgentResult(status="context_limit", partial=True, error_message="Context limit reached (x/y tokens).")`.
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
  - Test context overflow detection (mock a `done` chunk at/near the threshold);
    verify assistant message is persisted before returning, dangling tool_calls
    get stub responses, and `AgentResult.error_message` is populated.
  - Test tool round limit enforcement (`status="tool_limit"`).
  - Test `SubAgentDisplay` captures output and denies permission prompts.
  - Test `CallAgentsParallelTool`: concurrent calls return results in input
    order; duplicate session-persistent agents rejected; max batch size enforced.
  - Test `_wire_agents()`: `CallAgentTool` registered when agents present;
    `CallAgentsParallelTool` registered only when `allow_parallel: true`; startup
    warnings for unknown tool names in agent specs.

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

## Embedding Index and Semantic Search

See `design_embeddings.md` for the full design. This section covers the technical
constraints that affect implementation choices.

### Runtime Dependencies

The embedding subsystem is optional. Its dependencies are declared under
`[project.optional-dependencies]` in `pyproject.toml`, not as core
dependencies, so users who do not enable embeddings incur no additional
install cost:

```
pip install ai-cli[embeddings]           # numpy, xxhash, tomli (Python <3.11)
pip install ai-cli[embeddings,semantic]  # [embeddings] + tree-sitter grammars
```

- **`numpy>=1.24`** — vectorised float32 dot-product search in
  `SQLiteVectorStore.search()` and element-wise mean in `EmbeddingIndex`.
- **`xxhash>=3.0`** — fast non-cryptographic hashing for per-file change
  detection (`xxhash.xxh64(content).hexdigest()`).
- **`tomli>=2.0; python_version < '3.11'`** — TOML chunking on Python < 3.11
  (`tomllib` is stdlib on 3.11+). Included as a conditional marker inside the
  `[embeddings]` extra so it is ignored when running on 3.11+.

Additionally, `pyyaml` (already a core dependency) is used by the YAML-based
domain chunkers.

### Storage

- All index data is stored in a single WAL-mode SQLite database at
  `<project>/.ai-cli/embeddings/index.db`.
- Vectors are stored as raw `float32` little-endian bytes (numpy `ndarray.tobytes()`).
  Dimension is stored in the `meta` table and validated on every load.
- Schema version is stored in `meta`; a version mismatch triggers a full re-index
  with a clear user-facing message rather than a silent corruption.

### Change Detection

- File identity is tracked by `xxhash64(file_content)` stored as `char_hash`.
  This is correct across all filesystems and git checkout/merge scenarios
  where mtime is reset. `xxhash` is a fast non-cryptographic hash with a
  well-maintained PyPI package.
- `EmbeddingIndex.index()` calls `VectorStore.all_file_hashes()` once at the
  start of each run to build the full known-hash map, then scans the filesystem
  to detect new, changed, and deleted files in a single pass.

### Embedding API

- The `/v1/embeddings` endpoint is called via the `openai` Python client
  (already a dependency), pointing at whatever `base_url` is configured.
- Batch size defaults to 32 texts per request — a conservative default for
  local servers (LM Studio, Ollama) that can stall on large batches. Cloud APIs
  (OpenAI) handle 96+ comfortably; users can increase `batch_size` in config.
  A warning is logged when `batch_size` exceeds 512. Larger lists are split and
  results concatenated in order.
- The model's declared dimension is retrieved from the first successful response
  and stored in `meta.dimension`. Subsequent loads validate that the stored
  dimension matches the current model config; a mismatch requires a full re-index.

### Tree-Sitter

- `tree-sitter>=0.21` is required; the 0.20→0.21 API is a breaking change.
- Per-language grammar packages (`tree-sitter-python`, etc.) are separate PyPI
  packages grouped under the `[semantic]` optional extra in `pyproject.toml`.
- `TreeSitterChunker.__init__` raises `ImportError` immediately if `tree-sitter`
  or the requested grammar package is not installed. `make_chunker()` catches
  this and falls back to `FixedSizeChunker` without logging at warning level
  (absence of the optional dep is not an error).

### Thread Safety

- `SQLiteVectorStore` is constructed with `check_same_thread=False` and
  protected by a `threading.Lock`. All DB operations acquire the lock before
  executing, making the store safe to use across threads, including the
  background indexing thread and the main thread (search / tool calls).
- `OpenAIEmbeddingProvider` uses a `threading.Lock` to guard lazy
  construction of `_sync_client` and writes to `_dimension`, since
  `embed_sync()` may be called concurrently from the main thread (search)
  and from `asyncio.to_thread` (summary embedding during indexing).

### Document-Level Embeddings

- `strategy: average` — element-wise mean of all chunk vectors, L2-normalised.
  No additional API calls. Default for code files.
- `strategy: summary` — calls `llm_client.send()` with the (truncated) file
  content and asks for a thematic summary, then embeds that summary text.
  Input is truncated to `summary_max_tokens * 4` characters (~4 chars per token)
  before sending. `summary_max_tokens` controls input size, not output length.
  The summary prompt includes a word-count hint derived from
  `summary_response_tokens` (default: `chunk_size // 4`) so the LLM targets a
  length that fits within one embedding chunk. No per-call `max_tokens` cap is
  set at the API level — the LLM client's configured `max_response_tokens`
  still applies — to avoid starving reasoning models of their token budget.
  If the LLM call fails or returns empty text, the strategy falls back to
  `average` with a warning. If `llm_client` is `None`, a clear warning is
  logged and the strategy falls back to `average` immediately.
- `strategy: auto` — routes to `summary` for extensions in `prose_extensions`
  (default: `.md .markdown .txt .rst .adoc .asciidoc .tex .org`), `average`
  for everything else. Strategy resolution is handled by
  `_resolve_doc_strategy()`, shared by both `_compute_doc_vector()` and
  `_embed_and_upsert_file()` (which uses it to decide whether to dispatch to
  `asyncio.to_thread`). `prose_extensions` entries are normalised to lowercase
  for case-insensitive matching; a bare string (YAML scalar) is treated as a
  single-item list.
- `summary_model` is logged as a warning and ignored — per-call model switching
  is not supported by the `LLMClient` interface.
- Invalid or null `strategy` values are normalised to `"auto"` with a warning.

### Access Control

- `read_file` calls `workspace.embedding_index.is_indexed_path(path)` before
  falling through to the normal permission-prompt flow. `find_files` access
  control for its optional `path` parameter is planned but not yet implemented.
- `is_indexed_path()` returns `True` only for paths strictly under a user-added
  external root (not the workspace root, which is always allowed by `Workspace.contains()`).
  Uses `Path.resolve().relative_to(root)` — not string prefix matching — so
  sibling paths with a common prefix are correctly rejected.
- External roots are persisted in the SQLite `index_roots` table and loaded at
  `EmbeddingIndex` construction time, so access grants survive process restarts.

### Performance Constraints

- Target: <200 ms search latency for corpora up to 200 k chunks on a
  mid-range CPU (numpy vectorised cosine similarity; no GPU required).
- Indexing throughput target: ≥ 50 files/minute on an average SSD, excluding
  LLM summary API calls. LLM summary calls are dispatched via
  `asyncio.to_thread` so they do not block the event loop; they are made
  per-file only for prose files under the `summary` or `auto` strategy.
  The `average` strategy (pure numpy) runs inline without thread handoff.
- `SQLiteVectorStore` L2-normalises stored vectors row-wise on upsert so
  cosine similarity reduces to a dot product; no separate norms cache is needed.

### Testing

- All tree-sitter tests must use `pytest.importorskip("tree_sitter")` so they
  are automatically skipped when the optional dep is absent.
- SQLite tests must not share a database path across tests — use `tmp_path`
  fixtures to create an isolated `index.db` per test.
- Embedding API calls must be mocked in unit tests — no real HTTP requests.
  Use `unittest.mock.AsyncMock` for `embed()`.

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
