# Design: Display Layer

## Purpose

`Display` is responsible for all user-facing output and interactive prompts.
The REPL calls into Display; Display knows nothing about sessions, tools, or LLM
internals. This strict boundary means the entire UI can be swapped by changing
one factory call at startup.

---

## Implementation Order

1. **`PlainDisplay`** — first, to exercise the REPL end-to-end with no extra
   dependencies. Uses `print()` and `prompt_toolkit.prompt()`.
2. **`RichDisplay`** — after the REPL design is finalised. Layout, message
   history presentation, and other visual aspects to be agreed before
   implementation begins.

---

## Abstract Interface

`Display` is an ABC. The toggles (`verbose`, `markdown_enabled`) and their
mutation methods are implemented once in the base class and stored as `_verbose`
/ `_markdown_enabled`. All output and interaction methods are abstract.

### Mode flags

```python
@property
def verbose(self) -> bool: ...            # default False
def toggle_verbose(self) -> None: ...     # flip _verbose

@property
def markdown_enabled(self) -> bool: ...   # default True
def toggle_markdown(self) -> None: ...    # flip _markdown_enabled
```

`markdown_enabled` controls whether LLM text is rendered as Markdown or shown
raw. In `PlainDisplay` this flag has no effect (output is always raw).
In `RichDisplay` it switches between `rich.markdown.Markdown` rendering and
plain text — useful when the model emits content that Markdown would mangle
(e.g. code with backticks inside a fenced block, or unknown HTML-like tags).

### Streaming text output

The LLM response arrives as a stream of text deltas. The interface supports
incremental rendering so the user sees output as it is generated rather than
waiting for the full response.

```python
def begin_assistant_turn(self) -> None:
    """Called once before the first text delta arrives."""

def stream_text(self, delta: str) -> None:
    """Called for each text chunk as it arrives from the LLM."""

def stream_reasoning(self, delta: str) -> None:
    """Called for each reasoning/thinking chunk as it arrives.
    Default implementation is a no-op; subclasses override to show it.
    See LLM Backend section for how reasoning chunks are emitted."""

def end_assistant_turn(self) -> None:
    """Called once after the last chunk. Finalise/flush any buffered output."""
```

**Why three calls instead of one `show_assistant_message(text)`?**
A single-call design forces the REPL to buffer the entire response before
displaying anything. The three-call design lets `RichDisplay` render a live
updating panel; `PlainDisplay` simply `print(delta, end="", flush=True)` on
each chunk.

`stream_reasoning` is non-abstract with a no-op default so that existing
`PlainDisplay` (and future backends) do not need to implement it until they
are ready to surface reasoning content.

### Tool activity

```python
def show_tool_call(self, name: str, args: dict) -> None:
    """Notify the user that a tool is about to run."""

def show_tool_result(self, name: str, result: dict,
                     display_str: str | None = None) -> None:
    """Show the outcome of a tool call.

    display_str is an optional ANSI-capable string returned by
    tool.format_display(args, result).  When present, backends render it
    (with line wrapping and framing) in place of the default table/JSON
    dump.  ANSI colour codes in display_str are valid; PlainDisplay passes
    them through as-is, RichDisplay uses Text.from_ansi().
    """
```

In **summary mode** `show_tool_call` prints one compact line; `show_tool_result`
is silent unless `display_str` is provided, in which case a brief formatted
version is shown (the LLM incorporates the full result in its next message).
In **verbose mode** both show full detail.

### Usage update

```python
def update_usage(self, usage: dict, context_window: int) -> None:
    """Called by the REPL after each LLM turn with token usage data.

    usage: {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    context_window: total token budget from model config.

    Default: no-op.  RichDisplay uses this to update the bottom toolbar's
    context bar.
    """
```

### History viewer

```python
@abstractmethod
def show_history(self, messages: list[dict]) -> None:
    """Render the full conversation history in a scrollable/pageable view.

    messages is the list returned by session.get_messages().  Each entry is
    a standard OpenAI-format message dict (role + content).  Called by the
    REPL when the user runs /history.
    """
```

### Status and errors

```python
def show_status(self, message: str) -> None:
    """Informational, non-error messages (compaction notice, session saved, etc.)."""

def show_error(self, message: str) -> None:
    """User-visible error. Does not raise — caller decides whether to abort."""
```

### Permission prompt

```python
def show_permission_prompt(
    self,
    question: str,
    extra_options: list[str],
) -> tuple[str, str]:
    """
    Render an interactive permission prompt and return the user's decision.

    Universal choices (yes / no / always / custom) are always presented first.
    extra_options is a list of tool-specific option strings (e.g.
    'file:./src/foo.py', 'dir:./src/') appended below the universal set.

    Returns (choice, user_text) where:
      choice    — one of 'yes', 'no', 'always', 'custom', or a verbatim
                  string from extra_options
      user_text — the user's free-text message when choice == 'custom',
                  empty string otherwise
    """
```

### Session picker

```python
def show_session_list(self, sessions: list[SessionMeta]) -> SessionMeta | None:
    """
    Render a list of resumable sessions and let the user pick one.

    Returns the chosen SessionMeta, or None if the user declines / the list
    is empty. Each row shows at minimum: index, started_at, session_id,
    first_user_message (truncated), message_count.
    """
```

---

## Verbose vs Summary Mode

| Event            | Summary                         | Verbose                          |
|------------------|---------------------------------|----------------------------------|
| LLM text         | Full text                       | Full text                        |
| Tool call        | One-line (`▶ tool_name …`)      | Name + full args (pretty-printed)|
| Tool result      | Silent                          | Full result dict (pretty-printed)|
| Status           | Shown                           | Shown                            |
| Error            | Shown                           | Shown                            |
| Permission prompt| Shown                           | Shown                            |

---

## PlainDisplay

Simple backend using `print()` for output and `prompt_toolkit.prompt()` for
all interactive input.

- `begin_assistant_turn()` → no-op
- `stream_text(delta)` → `print(delta, end="", flush=True)`
- `stream_reasoning(delta)` → verbose only: `print(delta, end="", flush=True)` prefixed by `[thinking] ` on first call per turn; silent in summary mode
- `end_assistant_turn()` → `print()` (trailing newline)
- `show_tool_call(name, args)` → summary: `▶ name(key=val …)` one-liner; verbose: full JSON
- `show_tool_result(name, result, display_str=None)` → if `display_str`: print it; elif verbose: full JSON; else: silent
- `show_status(message)` → `print(f"# {message}")`
- `show_error(message)` → `print(f"✗ {message}", file=sys.stderr)`
- `update_usage(usage, context_window)` → no-op (no toolbar available)
- `show_history(messages)` → formats each message with `---` separators and role labels, pipes through `$PAGER` if set or `less -R` if available, otherwise prints directly
- `show_permission_prompt(question, extra_options)` → numbered menu via `print()` + `prompt_toolkit.prompt()`
- `show_session_list(sessions)` → numbered list via `print()` + `prompt_toolkit.prompt()`

`PlainDisplay` is also the easiest backend for unit-testing REPL logic without
mocking every Display method.

---

## RichDisplay

Built on [`rich`](https://github.com/Textualize/rich) for output and
`prompt_toolkit` for all interactive input.  Uses a **scrolling output** model
(no fixed TUI panels) so that it works correctly inside `screen`, `tmux`, and
any standard terminal emulator.  A future `TUIDisplay` implementation using
Textual may replace this if a fixed-panel layout becomes desirable.

---

### Colour palette

| Element | Rich style |
|---|---|
| User turn separator / border | `bold green` |
| Assistant turn separator / border | `bold cyan` |
| Reasoning block separator | `dim` |
| Reasoning text | `dim italic` |
| Tool call separator / border | `bold yellow` |
| Status message | `green` |
| Error message | `bold red` |
| Toolbar | prompt_toolkit default (inherits terminal) |

---

### Turn separators and left borders

Every conversation turn is framed by:

1. A full-width `Rule` with the speaker label, left-aligned:
   ```
   ── You ────────────────────────────────────────────────────────────────────
   ```
2. The message content indented and bordered on the left by a coloured vertical
   bar, using a `Panel` with a custom single-left-edge `Box` style (no top,
   right, or bottom borders):
   ```
   │ What does this function do?
   │
   ```

`Rule` is rendered at print time using `console.width`, so it automatically
fits the current terminal width.  If the terminal is resized during a
`Live`-streamed turn, Rich re-lays the live content to the new width on the
next refresh.

`console.print(Rule(title="You", style="bold green", align="left"))` followed
by `console.print(Panel(content, box=LEFT_BOX, border_style="bold green",
expand=True))`.  `LEFT_BOX` is a `rich.box.Box` constant with only the left
border characters set (all others are spaces).

---

### Streaming and end-of-turn rendering

**During streaming** a `Live(transient=True)` context is active.  `transient=True`
means the Live area is erased when it stops, leaving no artefacts.  The live
renderable shows:

- If reasoning content has arrived: a dim italic line showing the last ~200
  characters of the reasoning buffer, prefixed with `⟨thinking…⟩`.  This
  gives the user visible proof the model is working even when no response text
  has arrived yet.
- The accumulated raw response text below it.

Example live area (not yet formatted):
```
⟨thinking…⟩ …so if I consider the base case where n=0, the loop terminates…
Sure! The function implements a recursive Fibonacci with memoisation.
```

The turn separator `Rule` is printed by `begin_assistant_turn()` **before**
the `Live` context starts, so it appears immediately and remains on screen
throughout streaming.

**At `end_assistant_turn()`:**

1. Stop the `Live` context — the live area is erased.
2. Print the final formatted turn below the already-visible separator:
   - If `verbose` and reasoning content is non-empty: print a dim `Rule`
     titled `"Reasoning"`, the full reasoning text in `dim italic` style,
     then a dim `Rule` titled `"Response"`.
   - Print the response content inside a left-bordered `Panel`:
     - `markdown_enabled=True`: `Markdown(text)` (full Rich Markdown rendering)
     - `markdown_enabled=False`: `Text(text)` with word-wrap

This sequence means the user sees raw streaming text arrive, then it is
replaced by the properly formatted version when the turn completes — matching
the behaviour of other tools the user is familiar with.

**Sequence diagram:**

```
begin_assistant_turn()
    console.print(Rule("Assistant", style="bold cyan", align="left"))
    self._live = Live(transient=True, console=console, refresh_per_second=10)
    self._live.__enter__()

stream_text(delta) / stream_reasoning(delta)
    append to buffer(s)
    self._live.update(_build_live_renderable())   # redraws in place

end_assistant_turn()
    self._live.__exit__(...)        # erases live area
    self._live = None
    # reasoning block (verbose only)
    if verbose and reasoning_buffer:
        console.print(Rule("Reasoning", style="dim", align="left"))
        console.print(Panel(Text(reasoning_buffer, style="dim italic"),
                            box=LEFT_BOX, border_style="dim", expand=True))
        console.print(Rule("Response", style="dim cyan", align="left"))
    # response block
    content = Markdown(text_buffer) if markdown_enabled else Text(text_buffer)
    console.print(Panel(content, box=LEFT_BOX,
                        border_style="bold cyan", expand=True))
```

---

### Reasoning content

The LLMClient emits `{"type": "reasoning", "delta": str}` chunks, sourced from:

- **`reasoning_content` field** in OpenAI streaming deltas — used by `o1`,
  `o3`, and compatible models.  The LLMClient extracts this alongside the
  normal `content` field.
- **`<think>…</think>` tags** embedded in the text stream — used by DeepSeek
  R1, QwQ, and similar open-source reasoning models.  A stateful tag parser
  in the LLMClient strips the tags and re-emits their content as `reasoning`
  chunks; text outside the tags continues as normal `text` chunks.  The parser
  must handle tag boundaries that fall mid-chunk.

The scaffolding (new chunk type + stateful parser hook) is designed so that
additional reasoning sources (e.g. Anthropic thinking blocks, future models)
can be plugged in without changing the Display or REPL layers.

`RichDisplay.stream_reasoning()` appends to `_reasoning_buffer` and updates
the `Live` renderable to show the truncated preview.

`PlainDisplay.stream_reasoning()` in verbose mode prints reasoning deltas with
a `[thinking] ` label on the first call of the turn, then raw deltas thereafter.

---

### Tool call display

**`show_tool_call(name, args)`** prints:

```
── Tool: read_file ────────────────────────────────────────────────────────
│ path         ./src/main.py
│ start_line   10
│ end_line     30
```

Implementation: `Rule(title=f"Tool: {name}", style="bold yellow", align="left")`
followed by a `Table` (no headers, two columns: arg name and value) inside a
left-bordered `Panel`.  Long string values are truncated to ~80 characters with
`…` appended.

**`show_tool_result(name, result, display_str=None)`**

If `display_str` is provided (from `tool.format_display(args, result)`):

- Render it as `Text.from_ansi(display_str)` so ANSI colour codes are
  interpreted.  Existing newlines in the string are preserved; lines that
  exceed `console.width - 4` are wrapped at word boundaries.
- Wrap in a left-bordered `Panel` with `border_style="bold yellow"`.

If no `display_str`:
- Summary mode: print a one-line outcome indicator:
  `✓ read_file` (green) on success, `✗ read_file: <message>` (red) on error.
- Verbose mode: print the full result dict as a `Syntax`-highlighted JSON block.

**`tool.format_display(args, result) -> str | None`** (on the Tool base class):

```python
def format_display(self, *, args: dict, result: dict) -> str | None:
    """Return an ANSI-capable display string, or None for default rendering.

    The string may contain ANSI escape codes and newlines.  Long lines are
    wrapped by the display backend; existing newlines are preserved.  The
    method is called after execute() completes, so both args and result
    are available.  Return None (default) to use the standard arg table
    and result summary.
    """
    return None
```

This is a concrete method on `Tool` — not abstract.  Subclasses override it
when they can provide a richer display (e.g. `write_file` showing a coloured
unified diff, `find_files` showing a tree).  Both `RichDisplay` and
`PlainDisplay` use the returned string; a backend that does not support ANSI
can strip escape codes with `re.sub(r'\x1b\[[0-9;]*m', '', display_str)`.

The REPL is responsible for calling `format_display` and passing the result to
`show_tool_result`:

```python
display.show_tool_call(call["name"], call["arguments"])
result = tool_registry.execute(call["name"], call["arguments"])
display_str = None
tool = tool_registry.get(call["name"])
if tool is not None:
    try:
        display_str = tool.format_display(args=call["arguments"], result=result)
    except Exception:
        pass  # best effort; fall back to default
display.show_tool_result(call["name"], result, display_str=display_str)
```

---

### Status and errors

- `show_status(message)` → `console.print(f"  {message}", style="green")`
  Also saves `message` to `_last_status` for the bottom toolbar.
- `show_error(message)` → `console.print(f"  ✗ {message}", style="bold red")`
  Also saves `f"✗ {message}"` to `_last_status` for the bottom toolbar.

---

### Bottom toolbar

`prompt_toolkit`'s `PromptSession` accepts a `bottom_toolbar` callable and a
`refresh_interval` (in seconds).  Setting `refresh_interval=1` causes the
toolbar to be re-evaluated every second, making the elapsed-time clocks live.
The toolbar is visible only while the user is at the input prompt (it is not
shown during LLM streaming — that interval is where the timer information is
most useful to show before the next prompt appears).

**Toolbar content (left to right):**

```
 [ctx: ████████████░░░░░░░░ 61%]  ⏱ 01:23  ⚡ 00:04  ✓ Session compacted
```

| Segment | Source | Notes |
|---|---|---|
| `[ctx: ████░░ 61%]` | `update_usage()` | Filled blocks = prompt_tokens / context_window; updates after each turn |
| `⏱ MM:SS` | Internal timer | Time since `begin_assistant_turn()` was last called (i.e. since the user sent their last message); resets each turn |
| `⚡ MM:SS` | Internal timer | Time since the last `stream_text()` or `stream_reasoning()` call; helps detect network stalls; only shown during/after a streaming turn |
| Last status/error | `show_status` / `show_error` | Most recent message; truncated to fit available width |

The `⚡` timer is particularly useful for detecting stalled responses: if no
chunk has arrived for, say, 30 seconds, the user can see this and decide to
abort (Ctrl+C) and resend.

**Implementation note:** a persistent top bar was considered but requires ANSI
scrolling-region terminal escape sequences plus careful cursor management.
This is fragile, terminal-dependent, and incompatible with the scrolling output
model.  If a true persistent header becomes necessary, it should be implemented
as part of a dedicated `TUIDisplay` (future work).

**Timing is tracked internally by `RichDisplay`:**

- `begin_assistant_turn()` → records `_turn_start_time = datetime.now()`
- `stream_text()` / `stream_reasoning()` → records `_last_chunk_time = datetime.now()`
- `end_assistant_turn()` → clears `_last_chunk_time` (replaces with turn end time for post-turn display)
- `update_usage()` → updates `_prompt_tokens` and `_context_window`

No new method calls are needed from the REPL for timing.

---

### Permission prompt

A small `prompt_toolkit.application.Application` is constructed inline for each
permission request.  It shows a styled box with the question, the universal
options (y/n/a/c), and any tool-specific extra options.

Navigation:
- **Arrow keys (↑↓)**: move highlight between options
- **Hotkeys**: `y` (yes), `n` (no), `a` (always), `c` (custom); numeric digits
  for extra options (0, 1, 2…)
- **Enter**: confirm highlighted option
- **Esc / Ctrl+C**: treated as "no"

For the `custom` option, after selection a second prompt asks for the free-text
rejection message.

The application uses a `RadioList`-style layout built from prompt_toolkit
`FormattedText` and key bindings, styled to match the overall colour scheme.

---

### History viewer (`/history`)

`show_history(messages)` renders the session's current history to a Rich
`Console` wrapped in `console.pager()`, which pipes output through the system
pager (`$PAGER`, defaulting to `less -R`).  This works correctly in `screen`
and `tmux` sessions and does not depend on the terminal's scroll buffer size.
Because the history is read from `history_current.jsonl` (via
`session.get_messages()`), it reflects the state after any compaction and
persists across session resumes.

Rendering format inside the pager:

- Each message is preceded by a full-width separator Rule (same colour palette
  as live turns).
- Message content uses the same Markdown / left-border treatment as the main
  turn rendering.
- Tool messages (role `"tool"`) are rendered with the tool call/result style.
- Image content blocks are shown as `[image: image/png, 12.4 KB]` placeholders
  (the base64 data is not reprinted).

---

### Slash-command output (tables and lists)

`show_help`, `show_tool_list`, `show_tool_list_all`, `show_tool_info`,
`show_session_info` all use `rich.table.Table` for structured output and
`rich.text.Text` for styled labels.  No `Live` involvement — these are
synchronous print calls.

---

### Console setup

```python
self._console = Console(highlight=False)
```

`highlight=False` disables Rich's automatic syntax detection (which would
mis-highlight things like file paths and numbers in tool output).  Explicit
styles are applied per element instead.

---

## Factory

```python
def create_display(config: ConfigManager, verbose: bool = False) -> Display:
    """
    Read 'display_backend' from config (default 'plain') and return the
    appropriate Display implementation.

    Recognised values: 'plain', 'rich'.
    An unknown value logs a warning and falls back to 'plain'.
    """
```

Selectable via:
- Config key `display_backend: rich` in `~/.ai-cli/config.yaml` or `.ai-cli/config.yaml`
- CLI flag `--display plain` (passed as a cli_override, takes precedence)
