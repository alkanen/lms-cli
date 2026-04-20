# Design: REPL

## Purpose

The REPL is the main interaction loop. It owns the prompt, routes user input,
drives the LLM streaming loop (including the agentic tool-call cycle), and
coordinates `Session`, `ToolRegistry`, `LLMClient`, and `Display`. It has no
knowledge of file I/O, config, or network details — those live in the layers
below it.

---

## Relationship to `__main__.py`

Session resolution (resume / new) happens in `__main__.py` before the REPL is
constructed. The REPL receives a fully initialised `Session` and never calls
`SessionManager` directly. This keeps the REPL focused on the conversation loop
and makes it easier to test.

```
__main__.py  →  resolve_session()  →  Session
                                          ↓
                                        REPL.run()
```

---

## prompt_toolkit Setup

The REPL creates one `PromptSession` for its lifetime:

```python
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

session = PromptSession(
    history=FileHistory(get_global_dir() / "history"),
    # completer injected here once Completer is implemented
)
```

The history file lives in `~/.ai-cli/history` (global, shared across projects)
so the user gets command recall across sessions.

`KeyboardInterrupt` (Ctrl+C) during input cancels the current line and re-prompts
without exiting. `EOFError` (Ctrl+D) exits the REPL cleanly.

**Relationship to `Display` prompts:** `PlainDisplay` (and eventually
`RichDisplay`) uses `prompt_toolkit.prompt()` standalone calls for permission
prompts and the session picker — not the REPL's `PromptSession`. This is
intentional: those prompts are modal interruptions, not part of the ongoing
conversation history.

---

## Input Routing

```
raw input
    │
    ├─ starts with "/"  →  _handle_slash_command()
    │
    ├─ empty / whitespace only  →  re-prompt silently
    │
    └─ anything else  →  _preprocess_at_references()  →  _send_to_llm()
```

### `@` file references

`@path/to/file` anywhere in the user's message is replaced inline before the
message is sent to the LLM.

If resolution fails (file not found, ignored, outside workspace), the `@ref`
is left in place and `display.show_error()` is called with an explanation.
The user can then correct or send anyway.

`@!path/to/file` bypasses the ignore filter (explicit override, consistent with
the Completer design). Behaviour otherwise identical.

#### Text files

For non-image files, resolution:

1. `workspace.resolve(path)` — enforces workspace bounds.
2. `workspace.read_file(path)` — reads the file as UTF-8 text.
3. The `@ref` token is replaced inline with:
   ```
   [file: path/to/file]
   <content>
   [/file]
   ```
4. The overall message content remains a plain string and is saved to session
   history with `session.add_message("user", text)` as normal.

#### Image files

Files with a recognised image extension (`.png`, `.jpg`/`.jpeg`, `.gif`,
`.webp`) are handled differently because the LLM cannot parse them as text.

1. `workspace.resolve(path)` — enforces workspace bounds (same as text).
2. The file is **not** ignore-filtered by default when referenced with `@`
   (the user is explicitly attaching it). `@!` is still accepted for clarity
   or consistency but has no additional effect for binary files.
3. The file is read as raw bytes and base64-encoded.
4. The overall user message is converted from a plain string into a **content
   block array** — the OpenAI multimodal message format — containing:
   - One `text` block with the rest of the user's message text (the `@ref`
     token itself is removed; no `[file: …]` wrapper is inserted).
   - One image block per image reference.
5. The assembled raw message is saved with `session.add_raw_message({"role":
   "user", "content": [...]})` instead of `add_message()`.

**Return type of `_preprocess_at_references()`:** the method currently returns
`str`. With image support it must return `str | list[dict]` (a content block
array when at least one image is present). The REPL dispatches on the type to
choose the correct `Session` write call.

**Content block format — OpenAI `chat/completions` (current backend):**

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

The `detail` field controls tile-based processing cost (`"auto"`, `"low"`,
`"high"`). Default to `"auto"` so the model decides.

**Content block format — OpenAI `responses` API (future):**

The `/v1/responses` endpoint uses different content block types. Text blocks
become `"input_text"` and image blocks become `"input_image"`:

```json
{
  "role": "user",
  "content": [
    { "type": "input_text", "text": "What does this diagram show?" },
    {
      "type": "input_image",
      "detail": "auto",
      "image_url": "data:image/png;base64,<b64data>"
    }
  ]
}
```

The `LLMClient` backend is responsible for translating the internal content
block representation into whichever wire format its endpoint requires. The REPL
and session layer always use the `chat/completions` content block shape
(`"text"` / `"image_url"`) as the canonical in-memory and on-disk format;
adapters that target the `responses` endpoint rewrite the blocks before sending.

**LM Studio compatibility:**

LM Studio's OpenAI-compatible REST endpoint accepts the same `image_url`
content block format as the OpenAI `chat/completions` API. No special handling
is needed when using LM Studio via the `openai` backend with a custom
`base_url`. Vision capability depends on the loaded model; if the model does
not support vision the API will return an error, which surfaces as an
`LLMError` to the user.

**Supported MIME types and extensions:**

| Extension | MIME type |
|---|---|
| `.png` | `image/png` |
| `.jpg`, `.jpeg` | `image/jpeg` |
| `.gif` | `image/gif` |
| `.webp` | `image/webp` |

Unrecognised extensions are treated as text files. Future: MIME-sniff the first
few bytes as a fallback for extensionless files.

**Token counting caveat:**

`count_tokens()` uses tiktoken, which can only count text tokens. Image tokens
are model- and resolution-dependent (OpenAI charges roughly 85–1105 tokens per
image tile depending on `detail` level). The local estimate will therefore
undercount when images are present. The actual token usage returned by the API
in the `"done"` chunk should be preferred for compaction threshold decisions
whenever available.

---

## LLM Streaming and the Agentic Tool Loop

`_send_to_llm()` runs the full exchange including any tool-call cycles:

```python
def _send_to_llm(self, user_input: str) -> None:
    self._session.add_message("user", user_input)

    while True:
        tool_calls: list[dict] = []
        text_parts: list[str] = []

        self._display.begin_assistant_turn()
        for chunk in self._llm.send(
            self._session.get_messages(),
            tools=self._tool_registry.definitions(),
        ):
            if chunk["type"] == "text":
                self._display.stream_text(chunk["delta"])
                text_parts.append(chunk["delta"])
            elif chunk["type"] == "tool_call":
                tool_calls.append(chunk)
            elif chunk["type"] == "done":
                ...  # capture usage if needed
        self._display.end_assistant_turn()

        # Record the assistant's reply using OpenAI tool-calling protocol.
        # When tool calls are present the assistant message MUST include the
        # tool_calls array so the LLM can associate each result with its request.
        full_text = "".join(text_parts)
        if tool_calls:
            self._session.add_raw_message({
                "role": "assistant",
                "content": full_text or None,
                "tool_calls": [
                    {
                        "id": call["call_id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"]),
                        },
                    }
                    for call in tool_calls
                ],
            })
        elif full_text:
            self._session.add_message("assistant", full_text)

        if not tool_calls:
            break  # no tools called — exchange complete

        # Execute each tool and feed results back.
        # Tool result messages must carry tool_call_id to match the assistant's request.
        for call in tool_calls:
            self._display.show_tool_call(call["name"], call["arguments"])
            result = self._tool_registry.execute(call["name"], call["arguments"])
            self._display.show_tool_result(call["name"], result)
            self._session.add_raw_message({
                "role": "tool",
                "tool_call_id": call["call_id"],
                "content": json.dumps(result),
            })

        # Loop: LLM sees tool results and continues

    self._check_compaction()
```

**Tool call depth limit:** to prevent runaway agentic loops, a configurable
`max_tool_rounds` (default `10`) caps the number of tool-call cycles per user
turn. When the limit is reached the loop breaks and `display.show_error()` is
called with a notice that the limit was hit.

**Errors mid-stream:**
- `LLMError` during streaming → `display.show_error()`, the partial assistant
  text is discarded (not added to session history), re-prompt.
- `SessionError` on `add_message` → `display.show_error()`, same recovery.
- Tool execution errors are returned as error-result dicts and fed back to the
  LLM as normal tool results — the LLM decides how to handle them.

**Note on tool call display order:** `show_tool_call` and `show_tool_result` are
called *after* the LLM's streaming turn ends (so they appear below the
assistant's text, not interspersed with it). `end_assistant_turn()` is called
once before the first tool call display.

---

## Slash Commands

All slash commands are handled synchronously before re-prompting.

| Command | Behaviour |
|---|---|
| `/help` | Print list of commands and their descriptions via `display.show_help()` |
| `/exit` | Exit the REPL cleanly (same as Ctrl+D) |
| `/clear` | Clear session history (`session.clear()` — see note below) |
| `/verbose` | `display.toggle_verbose()` + status message |
| `/markdown` | `display.toggle_markdown()` + status message |
| `/compact` | Force compaction now, show status |
| `/tools` | Show enabled tools via `display.show_tool_list()` |
| `/session` | Show current session info via `display.show_session_info()` |

**`/clear`** removes all messages from `history_current.jsonl` (the working
history) but leaves `history_full.jsonl` intact. This requires a new
`Session.clear()` method.

**Unknown command:** `display.show_error(f"Unknown command: {command!r}")`,
then suggest `/help`.

### Impact on Display interface

`show_help`, `show_tool_list`, and `show_session_info` are defined in the
`Display` ABC and implemented in `PlainDisplay`.

---

## Compaction

`_check_compaction()` is called after every completed exchange:

```python
def _check_compaction(self) -> None:
    if self._session.should_compact():
        self._display.show_status("Context window nearing limit — compacting…")
        try:
            self._session.compact()
            self._display.show_status("Compaction complete.")
        except SessionError as exc:
            self._display.show_error(f"Compaction failed: {exc}")
```

Manual `/compact` bypasses the threshold check and calls `compact()` directly.

---

## `Session.clear()` — new method needed

`/clear` needs a way to wipe `history_current.jsonl` without touching
`history_full.jsonl`. Add to `Session`:

```python
def clear(self) -> None:
    """Delete history_current.jsonl and reset message_count in metadata."""
```

---

## What REPL Does Not Own

- Session creation / resume logic → `__main__.py`
- File I/O → `Workspace` / `ToolRegistry`
- Permission decisions → `PermissionManager` (injected into `ToolRegistry`)
- `@` completion / file picker UI → `Completer` (future)
- MCP server lifecycle → `MCPManager` (future)
