# Architecture: ai-cli

## Implementation Status

Legend: ✅ implemented and tested · 🔲 planned

---

## Module Structure

```
ai-cli/
├── ai_cli/                       # Python package root
│   ├── __main__.py               # ✅ Entry point — --init, --workspace, --resume, --continue
│   ├── core/
│   │   ├── config_manager.py     # ✅ Layered YAML config loading
│   │   ├── workspace.py          # ✅ Workspace root resolution, file ops, ignore rules
│   │   ├── permission_manager.py # ✅ In-memory permission state
│   │   ├── tool_registry.py      # ✅ Three-tier tool discovery, loading, settings
│   │   ├── llm_client.py         # ✅ Abstract LLMClient + OpenAI-compatible implementation
│   │   ├── mcp_manager.py        # 🔲 MCP server connections, tool exposure
│   │   └── session_manager.py    # ✅ Session create/resume/compact/persist
│   ├── tools/
│   │   ├── base.py               # ✅ Tool abstract base class
│   │   ├── read_file.py          # ✅ Read a file or line range from the workspace
│   │   ├── write_file.py         # ✅ Write or partially replace a file in the workspace
│   │   ├── find_files.py         # ✅ Glob-pattern file search with ignore-rule enforcement
│   │   └── tool_manager.py       # ✅ Context-saving tool gatekeeper
│   ├── cli/
│   │   ├── repl.py               # ✅ Main REPL loop, slash commands, keyboard shortcuts, streaming abort
│   │   ├── completer.py          # ✅ Tab completion for slash commands, tool names, @path references
│   │   └── display.py            # ✅ Display ABC + PlainDisplay + RichDisplay
│   └── utils/
│       ├── ignore_filter.py      # ✅ .gitignore-style pattern matching
│       └── logging_utils.py      # 🔲 JSONL structured logging
└── tests/                        # ✅ mirrors ai_cli/ structure (865 tests)
```

---

## Dependency Flow

Dependencies are strictly one-way — no circular imports.

```
repl → session_manager → llm_client
repl → tool_registry → permission_manager
repl → display
session_manager → workspace
tool_registry → workspace
tool_registry → config_manager
workspace → config_manager
workspace → ignore_filter
mcp_manager → tool_registry
```

---

## Class Interfaces

### ConfigManager ✅

```python
class ConfigManager:
    def __init__(self, project_root: Path | None, cli_overrides: dict): ...

    def get(self, key: str, default=None) -> Any:
        """Layered lookup: cli_overrides > project config > global config > default."""

    def get_project(self, key: str, default=None) -> Any:
        """Project config layer only — used to detect untrusted project-level settings."""

    def get_backend(self) -> str:
        """Returns 'openai' or 'lmstudio'. Defaults to 'openai'."""

    def get_model_config(self) -> dict:
        """Returns merged model/backend config. Raises ConfigError if none found.
        Resolves api_key_env to the actual key from the environment."""
```

Raises `ConfigError` with a helpful message if required config (model/backend) is missing at all levels.

**Note**: the `tools` config section is a dict keyed by tool name (not a list). Access via
`config.get("tools", {})` for the merged view and `config.get_project("tools", {})` for
the project-only layer (used by `ToolRegistry._apply_config()` for security checks).

---

### Workspace ✅

```python
class Workspace:
    def __init__(self, root: Path, config_manager: ConfigManager): ...

    @staticmethod
    def find_root(start: Path) -> Path | None:
        """Walk up from start, return first .ai-cli/ parent (skip ~/.ai-cli/)."""

    @staticmethod
    def initialise(path: Path) -> None:
        """Create .ai-cli/ scaffold with template files. Called by --init."""

    def is_ignored(self, path: Path) -> bool:
        """Check path against global + project .ignore rules."""

    def resolve(self, relative: str) -> Path:
        """Resolve a relative path against workspace root. Raises WorkspaceError if it escapes."""

    def file_exists(self, relative: str) -> bool:
        """Returns False for ignored paths (no info leakage)."""

    def read_file(self, relative: str, start_line=None, end_line=None) -> str:
        """1-based inclusive line range. Raises WorkspaceError on any failure."""

    def write_file(self, relative: str, content: str,
                   start_line=None, end_line=None) -> str:
        """Full write (no line args): creates file + parent dirs.
        Partial write (both line args): file must exist.
          - Replacement: 1 ≤ start_line ≤ end_line ≤ total_lines
          - Append-at-EOF: start_line == end_line == total_lines + 1
        Returns a human-readable summary string."""
```

Owns the ignore filter internally — callers use `is_ignored()`. Used by `ToolRegistry` and the `@` file picker.

---

### Tool (base class) ✅

```python
class Tool(ABC):
    # Required class attributes — validated at registration time:
    NAME: str
    DESCRIPTION: str
    PERMISSION_REQUIRED: bool

    # Optional class attribute:
    DISABLED_BY_DEFAULT: bool  # default False — set True to start disabled

    def __init__(
        self,
        workspace: Workspace,
        permission_manager: PermissionManager,
        permission_required: bool,  # tool's own default, overridable via config
        name: str,
        description: str,
    ): ...

    # --- Must implement ---

    @abstractmethod
    def definition(self) -> dict:
        """Return OpenAI function-calling schema: {"type": "function", "function": {...}}"""

    @abstractmethod
    def execute(self, **kwargs) -> dict:
        """Run the tool. Use _ok()/_err() to build the return value."""

    # --- May override ---

    def extra_permission_options(self, **kwargs) -> list[str]:
        """
        Tool-specific permission options beyond the universal set.
        Returns opaque label strings — PermissionManager passes them through
        unchanged and the tool interprets them in on_permission_granted().
        By convention, file tools use 'file:./path/to/file.txt' or 'dir:./path/to/dir/'.
        Default: []
        """
        return []

    def on_permission_granted(self, choice: str, **kwargs) -> None:
        """
        Called by the registry when the user grants permission via a named
        extra option from extra_permission_options(). Not called for universal
        choices (yes/always) which return an empty choice string.
        Default: no-op.
        """

    def reset_session_state(self) -> None:
        """
        Clear all session-scoped in-memory state (e.g. per-path allow-lists).
        Called by ToolRegistry.reset_session_overrides() on session resume.
        Default: no-op.
        """

    def request_permission(self, action: str, **kwargs) -> tuple[bool, str]:
        """
        Check permission. If permission_required is False, returns (True, '') immediately.
        Otherwise checks PermissionManager (and any tool-level allow-lists in subclasses),
        then prompts the user. Returns (allowed, choice_or_reason).
        Not normally overridden — but file tools override it to check their own allow-lists.
        """

    # --- Result helpers ---

    @staticmethod
    def _ok(data: dict | None = None) -> dict:
        """Return {"status": "success", "data": data or {}} — None is normalised to {}."""

    @staticmethod
    def _err(error: str, message: str, code: int = 400, details: dict | None = None) -> dict:
        """Return {"status": "error", "error": error, "message": message, "code": code}"""
```

---

### PermissionManager ✅

```python
PERM_YES = "yes"
PERM_NO = "no"
PERM_ALWAYS = "always"
PERM_CUSTOM = "custom"

class PermissionManager:
    def __init__(self, prompt_fn: Callable[[str, list[str]], tuple[str, str]]): ...
    # prompt_fn(question, extra_options) -> (choice, user_text)

    def request(
        self,
        tool_name: str,
        question: str,
        extra_options: list[str] | None = None,
    ) -> tuple[bool, str]:
        """
        Check if tool has 'always' grant. If not, prompt the user.
        Returns (allowed, detail).
        For 'yes'/'always' (including an existing always-grant bypass), detail is "".
        If a tool-specific extra option is chosen, detail is that option string.
        For a 'custom' rejection, detail is the user's message.
        """

    def grant_always(self, tool_name: str) -> None:
        """Record in-memory always-allow for this tool."""

    def reset(self) -> None:
        """Clear all grants. Called on session resume."""
```

`prompt_fn` is provided by the REPL layer — keeps permission logic decoupled from the UI.

---

### ToolRegistry ✅

```python
class ToolRegistry:
    def __init__(
        self,
        workspace: Workspace,
        config_manager: ConfigManager,
        permission_manager: PermissionManager,
    ): ...

    def load(self) -> None:
        """
        Load tools in order: bundled (via importlib.import_module) →
        global (~/.ai-cli/tools/) → project (.ai-cli/tools/) (latter two via
        spec_from_file_location). Warn on name collisions. Apply per-tool
        settings from config via _apply_config().
        """

    def register(self, tool_cls: type[Tool], tier: str = "programmatic") -> None:
        """Programmatically register a Tool subclass without file discovery."""

    def get(self, name: str) -> Tool | None: ...

    def all_enabled(self) -> list[Tool]: ...

    def definitions(self) -> list[dict]:
        """Return OpenAI-format schemas for all currently enabled tools."""

    def execute(self, name: str, kwargs: dict, *, allow_transient: bool = False) -> dict:
        """
        Look up tool, request permission (skipped if not permission_required),
        call on_permission_granted if a named extra option was chosen, then execute.
        Returns a canonical result dict. allow_transient=True skips the enabled check,
        for use after enable_transient().
        """

    def enable(self, name: str) -> None:
        """Enable tool, clear any session override, persist to project config.yaml."""

    def disable(self, name: str) -> None:
        """Disable tool, clear any session override, persist to project config.yaml."""

    def enable_session(self, name: str) -> None:
        """Enable tool for this session only — no config write."""

    def disable_session(self, name: str) -> None:
        """Disable tool for this session only — no config write."""

    def reset_session_overrides(self) -> None:
        """
        Clear all session-level overrides and call reset_session_state() on
        every tool (guarded — one tool failing does not prevent others from
        being reset). Called on session resume.
        """

    def enable_transient(self, name: str) -> dict | None:
        """
        Return the named tool's OpenAI-format schema for one-call injection
        without changing its enabled state. Returns None if unknown.
        Used by tool_manager to inject a tool into a single API call only.
        """

    def set_permission_required(self, name: str, value: bool) -> None:
        """
        Toggle permission_required and persist to project config.yaml.
        When value=False, also writes user_confirmed=True so the lowering
        survives reloads without being blocked as an untrusted project config entry.
        """
```

**Config trust model**: `_apply_config()` applies the merged (global+project) tools dict, but
lowering `permission_required` from True to False is only allowed when:
- The setting comes from global config only (not present in project layer), OR
- The project layer entry has `user_confirmed: true` (written by `set_permission_required()`).

**Enabled state**: two layers in descending priority:
1. Session override (`_session_overrides`) — cleared on `reset_session_overrides()`
2. Persistent enabled state (`_enabled`) — initialised from `DISABLED_BY_DEFAULT` at registration time, then overridden by config via `_apply_config()`

At runtime `_is_enabled()` checks session overrides first, then falls back to `_enabled`. There is no separate third layer — the `DISABLED_BY_DEFAULT` attribute only affects the initial value of `_enabled` at load time.

`enable()`/`disable()` clear any session override for the tool before updating persistent state.

---

### Bundled Tools ✅

#### `read_file`
- `PERMISSION_REQUIRED = False`, `DISABLED_BY_DEFAULT = True` (enabled by `tool_manager` on demand)
- Parameters: `path` (required), `start_line` (optional, 1-based), `end_line` (optional, 1-based)
- Response data: `{content, path, start_line, end_line, lines_returned, total_lines}`
  - For an empty file: `start_line=0, end_line=0, lines_returned=0, total_lines=0`
- Overrides `request_permission()` to check a session-scoped allow-list before prompting
- `extra_permission_options()` generates `file:./…` and `dir:./…/` options for each path level up to workspace root
- `on_permission_granted()` adds the resolved path/dir to `_session_allowed_files` / `_session_allowed_dirs`
- `reset_session_state()` clears both allow-lists

#### `write_file`
- `PERMISSION_REQUIRED = True`, `DISABLED_BY_DEFAULT = True`
- Parameters: `path` (required), `content` (required), `start_line` + `end_line` (both optional, must be provided together)
- Full write (no line args): creates file and any missing parent directories
- Partial write (file must exist): two modes:
  - Replacement: `1 ≤ start_line ≤ end_line ≤ total_lines` — replaces those lines
  - Append-at-EOF: `start_line == end_line == total_lines + 1` — appends after the last line
- Response data: `{path, summary, lines_written}`
- Same session-scoped allow-list pattern as `read_file`

---

### ToolManager (bundled tool) ✅

A context-saving gatekeeper that prevents the LLM from being overwhelmed by tool schemas. At startup most
bundled tools are disabled; the LLM calls `tool_manager` to discover and transiently enable tools it needs.

Actions:
- **`list`** — returns each tool's name, one-line description, and `enabled` flag so the LLM can make informed requests without seeing full schemas.
- **`enable`** — accepts a `tool_names` array; calls `ToolRegistry.enable_transient()` for each name; the REPL injects those schemas into the immediately following API call only (no persistent state change).

This implements the **transient** enable mode (the weakest of three — see `project_plan.md` Key Features §3).

---

### LLMClient ✅ (OpenAI-compatible REST only)

**Known limitation**: when using a local server (e.g. LM Studio) that needs to
load the requested model on first request, `send()` may hang indefinitely until
the model finishes loading. A configurable request timeout on `OpenAIClient` is
🔲 planned.



```python
# Chunk variants yielded by LLMClient.send():
#   {"type": "text",      "delta": str}          — streamed text token
#   {"type": "reasoning", "delta": str}          — reasoning / thinking token
#                                                   (from reasoning_content field on OpenAI o1/o3,
#                                                    or from <think>…</think> tags when
#                                                    extract_think_tags: true in config)
#   {"type": "tool_call", "name": str,
#    "call_id": str,      "arguments": dict}      — complete tool invocation
#   {"type": "done",      "stop_reason": str,
#    "usage": {"prompt_tokens": int,              — always present; zeros if server
#              "completion_tokens": int,             omits usage data
#              "total_tokens": int}}              — stream finished

class LLMClient(ABC):
    @abstractmethod
    def send(
        self,
        messages: list[dict],
        tools: list[dict],
        stream: bool = True,
    ) -> Generator[dict, None, None]: ...
    # Yields the same Chunk types regardless of stream=True/False.
    # stream=True (default): text deltas arrive immediately, tool calls assembled from deltas.
    # stream=False: entire response awaited first, same Chunk sequence produced.
    # Returns Generator (not Iterator) so callers can call .close() to cancel mid-stream.

    @abstractmethod
    def get_model_metadata(self) -> dict:
        """Returns at minimum: {'context_window': int, 'max_response_tokens': int}"""

    @abstractmethod
    def count_tokens(self, messages: list[dict]) -> int: ...


class OpenAIClient(LLMClient):
    def __init__(self, config: dict): ...
    # config provides: base_url, api_key, model, context_window, max_response_tokens


class LMStudioClient(LLMClient):
    def __init__(self, config: dict): ...
    # Connects via WebSocket, fetches model metadata automatically


def create_llm_client(config_manager: ConfigManager) -> LLMClient:
    """Factory: reads backend from config, returns appropriate implementation."""
```

`send()` always yields streamed `Chunk` dicts. The REPL inspects the `type` field: `"text"` chunks are forwarded to `Display`; `"tool_call"` chunks are routed to `ToolRegistry.execute()`; `"done"` signals end of stream.

---

### SessionManager / Session ✅

```python
class SessionManager:
    def __init__(
        self,
        workspace: Workspace,
        llm_client: LLMClient,
        sessions_dir: Path,  # ~/.ai-cli/sessions/
    ): ...

    def new(self) -> Session: ...

    def list(self, workspace_path: Path) -> list[SessionMeta]:
        """Return sessions associated with this project, newest first."""

    def load(self, session_id: str) -> Session: ...

    def most_recent(self, workspace_path: Path) -> Session | None: ...


class Session:
    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        llm_client: LLMClient,
    ): ...

    def add_message(self, role: str, content: str) -> None:
        """Append to both history_full.jsonl and history_current.jsonl."""

    def get_messages(self) -> list[dict]:
        """Return messages from history_current.jsonl."""

    def compact(self, instructions: str = "") -> None:
        """Request summary from LLM, rewrite history_current.jsonl."""

    def clear(self) -> None:
        """Delete history_current.jsonl and reset message metadata. Preserves history_full.jsonl."""

    def get_meta(self) -> dict:
        """Return a copy of the session's metadata as a plain dict."""

    def set_name(self, name: str) -> None:
        """Write name to metadata.yaml."""

    def token_usage(self) -> tuple[int, int]:
        """Returns (used_tokens, context_window)."""

    def should_compact(self) -> bool:
        """True if (used_tokens + overhead) > context_window * 0.9."""
```

---

### SessionMeta ✅

```python
@dataclass
class SessionMeta:
    session_id: str
    workspace_path: Path
    started_at: datetime
    message_count: int
    name: str | None
    first_user_message: str    # truncated
    last_message_role: str
    last_message_preview: str  # truncated
```

Session folder layout:

```
~/.ai-cli/sessions/<session-id>/
├── metadata.yaml
├── history_full.jsonl
└── history_current.jsonl
```

---

### REPL ✅

```python
class REPL:
    def __init__(
        self,
        session: Session,
        tool_registry: ToolRegistry,
        llm_client: LLMClient,
        display: Display,
        workspace: Workspace,
    ): ...

    def run(self, *, _prompt_session: PromptSession | None = None) -> None:
        """
        Main loop: read input → route → render.
        _prompt_session is injectable for testing (avoids real terminal/filesystem).
        Uses PromptSession with FileHistory at ~/.ai-cli/history by default.
        Reads repl_behavior config for: complete_while_typing, enable_suspend,
        completion_max_results.  Injects key bindings and display toolbar kwargs.
        """

    def _handle_input(self, raw: str) -> None:
        """Route to _handle_slash_command or _send_to_llm."""

    def _handle_slash_command(self, command: str) -> None:
        """Dispatch /help, /exit, /clear, /verbose, /markdown, /tools,
        /compact, /session, /history, /rounds."""

    def _preprocess_at_references(self, text: str) -> str | list[dict]:
        """
        Replace @path and @!path tokens with file content wrapped in [file: path]...[/file].
        @path respects ignore rules; @!path bypasses them.
        On any error the token is left in place and an error is shown.
        Returns list[dict] (content blocks) for image attachments; str otherwise.
        """

    def _send_to_llm(self, user_input: str | list[dict]) -> None:
        """
        Append user message, create abort event + _AbortMonitor, then delegate
        to _send_rounds().  Monitor is always stopped in a finally block.
        """

    def _send_rounds(
        self, user_input: str | list[dict], abort: threading.Event
    ) -> None:
        """
        Inner loop that drives multi-round tool calls (capped at _MAX_TOOL_ROUNDS = 10).
        Checks abort.is_set() at the start of each round and after each tool call.
        Explicitly calls stream.close() in a finally block to release the HTTP connection.
        """

    def _check_compaction(self) -> None:
        """Called after each exchange. Auto-compact if should_compact()."""


class _AbortMonitor:
    """
    Background thread that watches stdin for a lone ESC or Ctrl+C and sets a
    threading.Event to signal the streaming loop to stop.

    Uses tty.setcbreak + select.select for raw single-keypress detection.
    ESC is disambiguated from arrow-key sequences with a 20 ms peek:
    if more bytes follow within the window they are drained and iteration
    continues; only a bare ESC (no following bytes) triggers abort.
    Only started when _HAS_TTY is True and sys.stdin.isatty() is True.
    """


def _make_key_bindings() -> KeyBindings:
    """
    Return prompt_toolkit KeyBindings injected into every PromptSession.
    Ctrl+L — clears the terminal screen.
    Ctrl+G — opens the current prompt buffer in $VISUAL / $EDITOR
             (shlex.split handles arguments; OSError prints to stderr).
    """


def _build_keyboard_shortcuts(*, enable_suspend: bool) -> list[tuple[str, str]]:
    """
    Return the keyboard-shortcut rows used by /help.
    Ctrl+Z row is only included when enable_suspend=True AND _HAS_TTY AND isatty().
    """
```

---

### Display ✅ (PlainDisplay + RichDisplay)

```python
class Display(ABC):
    """
    Mode flags (concrete on the ABC, not abstract):
      verbose: bool                — show full tool args/results
      markdown_enabled: bool       — render Markdown in LLM output
      toggle_verbose() -> None
      toggle_markdown() -> None
    """

    # Streaming — called once per LLM response
    @abstractmethod
    def begin_assistant_turn(self) -> None: ...
    @abstractmethod
    def stream_text(self, delta: str) -> None: ...
    @abstractmethod
    def end_assistant_turn(self) -> None: ...

    # Tool activity
    @abstractmethod
    def show_tool_call(self, name: str, args: dict) -> None:
        """Summary mode: one line. Verbose: full JSON args."""
    @abstractmethod
    def show_tool_result(
        self, name: str, result: dict, *, display_str: str | None = None
    ) -> None:
        """Summary mode: ✓/✗ one-liner. Verbose: Syntax JSON.
        display_str, if provided, is always shown regardless of verbose mode."""

    # Informational
    @abstractmethod
    def show_status(self, message: str) -> None: ...
    @abstractmethod
    def show_error(self, message: str) -> None:
        """Writes to sys.stderr."""

    # Slash-command output
    @abstractmethod
    def show_help(self, commands: list[tuple[str, str]]) -> None: ...
    @abstractmethod
    def show_tool_list(self, tools: list[Tool]) -> None: ...
    @abstractmethod
    def show_session_info(self, session: Session) -> None:
        """Calls session.get_meta() — not _read_meta()."""

    # Interactive prompts
    @abstractmethod
    def show_permission_prompt(
        self,
        question: str,
        extra_options: list[str],
    ) -> tuple[str, str]:
        """
        Render permission prompt, return (choice, user_text).
        Universal options: yes/no/always/custom (with message).
        EOFError / KeyboardInterrupt → ("no", "").
        """
    @abstractmethod
    def show_session_list(self, sessions: list[SessionMeta]) -> SessionMeta | None:
        """Render interactive resume picker, return chosen session or None."""


    # Non-abstract with default {} — RichDisplay overrides to add toolbar kwargs:
    def prompt_session_kwargs(self) -> dict: ...

    def stream_reasoning(self, delta: str) -> None: ...          # no-op default
    def update_usage(self, usage: dict, context_window: int) -> None: ...  # no-op default

    @abstractmethod
    def show_history(self, messages: list[dict]) -> None: ...


def create_display(config: ConfigManager, *, verbose: bool = False) -> Display:
    """
    Factory: reads display_backend (default 'rich') and display_markdown from config.
    'plain' → PlainDisplay, 'rich' → RichDisplay. Unknown backends warn and fall back to PlainDisplay.
    """


class RichDisplay(Display):
    """
    Rich-based display using Live(transient=True) during streaming.
    _LiveRenderable re-invokes its build function on every refresh tick so the
    spinner and toolbar animate even when no LLM chunks are arriving.
    _LeftBorderRenderable adds '│ ' to each rendered line without blank padding.
    Console(highlight=False) prevents unintended auto-highlighting.
    Bottom toolbar provided via prompt_session_kwargs() → {"bottom_toolbar": ..., "refresh_interval": 1}.
    """
```

---

### MCPManager 🔲

```python
class MCPManager:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        config_manager: ConfigManager,
    ): ...

    def connect_all(self) -> None:
        """
        Read mcp_servers.yaml from global + project .ai-cli/.
        Connect to each server via stdio or SSE.
        Register their tools into tool_registry.

        Security: project-level mcp_servers.yaml is untrusted. Any stdio
        command defined there must be explicitly approved by the user before
        execution. Global (~/.ai-cli/) entries are considered trusted.
        """

    def disconnect_all(self) -> None: ...


@dataclass
class MCPServerConfig:
    name: str
    transport: Literal["stdio", "sse"]
    command: list[str] | None  # for stdio
    url: str | None            # for sse
    enabled: bool = True
```

MCP tools are wrapped in a thin `MCPTool(Tool)` subclass that forwards `execute()` calls to the MCP server and uses the server-provided schema for `definition()`.

---

### REPLCompleter ✅

```python
DEFAULT_MAX_PATH_COMPLETIONS = 200

class REPLCompleter(Completer):
    """prompt_toolkit Completer for the ai-cli REPL."""

    def __init__(
        self,
        slash_commands: list[str],
        tool_registry: ToolRegistry | None = None,
        workspace: Workspace | None = None,
        max_path_completions: int = DEFAULT_MAX_PATH_COMPLETIONS,
    ): ...
    # max_path_completions must be >= 1; raises ValueError otherwise.

    def get_completions(self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """
        Dispatches to path completion when the cursor is after an @ token,
        otherwise to slash-command completion when the line starts with /.
        @-detection runs first so @/path does not trigger slash completion.
        """

    # Slash command routing:
    # /tools  → subcommands (list/info/enable/disable/allow/disallow) + tool names + --session flag
    # /session → 'name' subcommand
    # /rounds  → --session flag only (numeric arg is free-form)
    # All other commands: prefix-match against slash_commands list (case-insensitive)

    # @ path completion:
    # Regex _AT_PARTIAL_RE matches an @ token at end of text before cursor.
    # Group 1: '!' (bypass-ignore flag); Group 2: partial path.
    # Supports workspace-relative, '../', and absolute '/' paths.
    # Paths inside workspace root filtered via workspace.is_ignored() unless bypass=True.
    # Paths outside workspace root are never filtered.
    # Results capped at max_path_completions; directories get trailing '/'.
    # OSError during scan skips the entry rather than crashing.
```

Built on `prompt_toolkit`. The `@`-detection regex runs before slash-command detection so that `@/abs/path` is correctly treated as a file reference, not a slash command.

---

## Entry Point and Startup Sequence ✅

```python
# __main__.py
def main():
    args = parse_args()
    # args: --init, --workspace, --resume, --resume <id>, --continue

    # 1. Resolve workspace
    start = Path(args.workspace) if args.workspace else Path.cwd()

    if args.init:
        Workspace.initialise(start)  # create .ai-cli/ scaffold, exit
        return

    root = Workspace.find_root(start)
    if root is None:
        root = prompt_init_or_exit(start)

    # 2. Bootstrap core objects
    config = ConfigManager(root, cli_overrides={})
    workspace = Workspace(root, config)
    display = create_display(config)
    permission_manager = PermissionManager(prompt_fn=display.show_permission_prompt)
    tool_registry = ToolRegistry(workspace, config, permission_manager)
    llm_client = create_llm_client(config)
    session_manager = SessionManager(workspace, llm_client, SESSIONS_DIR)

    # 3. Load tools
    tool_registry.load()

    # 4. Resolve session (--resume / --resume <id> / --continue / new)
    session = _pick_session(args, session_manager, workspace, display)

    # 5. Start REPL
    repl = REPL(session, tool_registry, llm_client, display, workspace)
    repl.run()
```

`_pick_session()` handles `--resume` (interactive picker via `display.show_session_list()`),
`--resume <id>` (direct load), `--continue` (most recent or new), and bare start (new session).
On resume, `permission_manager.reset()` and `tool_registry.reset_session_overrides()` are called
to clear all session-scoped state.

MCP support (`mcp_manager.connect_all()` / `disconnect_all()`) is 🔲 planned and will be inserted at step 3.
