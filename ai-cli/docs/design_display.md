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

def end_assistant_turn(self) -> None:
    """Called once after the last chunk. Finalise/flush any buffered output."""
```

**Why three calls instead of one `show_assistant_message(text)`?**
A single-call design forces the REPL to buffer the entire response before
displaying anything. The three-call design lets `RichDisplay` render a live
updating panel; `PlainDisplay` simply `print(delta, end="", flush=True)` on
each chunk.

### Tool activity

```python
def show_tool_call(self, name: str, args: dict) -> None:
    """Notify the user that a tool is about to run."""

def show_tool_result(self, name: str, result: dict) -> None:
    """Show the outcome of a tool call."""
```

In **summary mode** `show_tool_call` prints one compact line; `show_tool_result`
is silent (the LLM incorporates the result in its next message).
In **verbose mode** both show full detail.

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
- `end_assistant_turn()` → `print()` (trailing newline)
- `show_tool_call(name, args)` → summary: `▶ name(key=val …)` one-liner; verbose: full JSON
- `show_tool_result(name, result)` → verbose only: full JSON; silent in summary mode
- `show_status(message)` → `print(f"# {message}")`
- `show_error(message)` → `print(f"✗ {message}", file=sys.stderr)`
- `show_permission_prompt(question, extra_options)` → numbered menu via `print()` + `prompt_toolkit.prompt()`
- `show_session_list(sessions)` → numbered list via `print()` + `prompt_toolkit.prompt()`

`PlainDisplay` is also the easiest backend for unit-testing REPL logic without
mocking every Display method.

---

## RichDisplay

To be designed in detail before implementation. Key points agreed so far:

- Built on the [`rich`](https://github.com/Textualize/rich) library
- Streaming text via `rich.live.Live` + `rich.markdown.Markdown` (or plain text
  when `markdown_enabled=False`)
- `prompt_toolkit` for all interactive input (permission prompts, session picker,
  and the REPL input line)
- Default colour scheme (no custom theme for now)
- Layout, message history presentation, and other visual design to be discussed
  before implementation

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
