# Architecture: ai-cli

## Module Structure

```
ai-cli/
├── core/
│   ├── config_manager.py     # Layered YAML config loading
│   ├── workspace.py          # Workspace root resolution, file ops, ignore rules
│   ├── tool_registry.py      # Three-tier tool discovery, loading, settings
│   ├── permission_manager.py # In-memory permission state
│   ├── llm_client.py         # Abstract LLMClient + OpenAI/LMStudio implementations
│   ├── mcp_manager.py        # MCP server connections, tool exposure
│   └── session_manager.py    # Session create/resume/compact/persist
├── tools/                    # Bundled tools (read_file, write_file, bash, etc.)
├── cli/
│   ├── repl.py               # Main REPL loop, input handling, slash commands
│   ├── completer.py          # Tab completion + @ file picker
│   └── display.py            # Rich output, summary/verbose modes
└── utils/
    ├── ignore_filter.py      # .gitignore-style pattern matching
    └── logging_utils.py      # JSONL structured logging
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

### ConfigManager

```python
class ConfigManager:
    def __init__(self, project_root: Path | None, cli_overrides: dict): ...

    def get(self, key: str, default=None) -> Any:
        """Layered lookup: cli_overrides > project config > global config > default"""

    def get_backend(self) -> str:
        """Returns 'openai' or 'lmstudio'"""

    def get_model_config(self) -> dict:
        """Returns merged model/backend config. Raises ConfigError if none found."""

    def get_tools_config(self) -> list[dict]:
        """Returns merged tools list from global + project config."""
```

Raises `ConfigError` with a helpful message if required config (model/backend) is missing at all levels.

---

### Workspace

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
        """Resolve a relative path against workspace root."""

    def file_exists(self, relative: str) -> bool: ...

    def read_file(self, relative: str, start_line=None, end_line=None) -> str: ...

    def write_file(self, relative: str, content: str,
                   start_line=None, end_line=None) -> str: ...
```

Owns the ignore filter internally — callers use `is_ignored()`. Used by `ToolRegistry` and the `@` file picker.

---

### Tool (base class)

```python
class Tool:
    def __init__(
        self,
        workspace: Workspace,
        permission_manager: PermissionManager,
        permission_required: bool,  # tool's own default, overridable via config
        name: str,
        description: str,
    ): ...

    def definition(self) -> dict:
        """Return OpenAI function-calling schema. Must be implemented by subclass."""
        raise NotImplementedError

    def extra_permission_options(self, **kwargs) -> list[str]:
        """Tool-specific permission options beyond the universal set. Default: []"""
        return []

    def request_permission(self, **kwargs) -> tuple[bool, str]:
        """Check permission_manager, prompt if needed. Returns (allowed, reason)."""

    def execute(self, **kwargs) -> dict:
        """
        Run the tool and return the canonical tool result as a JSON-serializable
        dict (e.g., {"status": "success", "data": {...}}). Must be implemented
        by subclass.
        """
        raise NotImplementedError
```

---

### PermissionManager

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
        Returns (allowed, reason_or_suggestion).
        """

    def grant_always(self, tool_name: str) -> None:
        """Record in-memory always-allow for this tool."""

    def reset(self) -> None:
        """Clear all grants. Called on session resume."""
```

`prompt_fn` is provided by the REPL layer — keeps permission logic decoupled from the UI.

---

### ToolRegistry

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
        Load tools in order: bundled → global (~/.ai-cli/tools/) → project (.ai-cli/tools/).
        Warn on name collisions. Apply per-tool settings from config.
        """

    def get(self, name: str) -> Tool | None: ...

    def all_enabled(self) -> list[Tool]: ...

    def definitions(self) -> list[dict]:
        """Return OpenAI-format schemas for all enabled tools."""

    def execute(self, name: str, kwargs: dict) -> dict:
        """Look up tool, request permission, then execute. Returns the tool's canonical result dict."""

    def enable(self, name: str) -> None:
        """Enable tool and persist change to project config.yaml."""

    def disable(self, name: str) -> None:
        """Disable tool and persist change to project config.yaml."""

    def set_permission_required(self, name: str, value: bool) -> None:
        """Toggle permission_required and persist to project config.yaml."""
```

---

### LLMClient

```python
# Chunk variants yielded by LLMClient.send():
#   {"type": "text",      "delta": str}          — streamed text token
#   {"type": "tool_call", "name": str,
#    "call_id": str,      "arguments": dict}      — complete tool invocation
#   {"type": "done",      "stop_reason": str}     — stream finished

class LLMClient(ABC):
    @abstractmethod
    def send(
        self,
        messages: list[dict],
        tools: list[dict],
        stream: bool = True,
    ) -> Iterator[dict]: ...
    # Yields Chunk dicts (see types above). The REPL routes "tool_call" chunks
    # to ToolRegistry.execute() and streams "text" chunks to Display.

    @abstractmethod
    def get_model_metadata(self) -> dict:
        """Returns at minimum: {'context_window': int, 'max_tokens': int}"""

    @abstractmethod
    def count_tokens(self, messages: list[dict]) -> int: ...


class OpenAIClient(LLMClient):
    def __init__(self, config: dict): ...
    # config provides: base_url, api_key, model, context_window, max_tokens


class LMStudioClient(LLMClient):
    def __init__(self, config: dict): ...
    # Connects via WebSocket, fetches model metadata automatically


def create_llm_client(config_manager: ConfigManager) -> LLMClient:
    """Factory: reads backend from config, returns appropriate implementation."""
```

`send()` always yields streamed `Chunk` dicts. The REPL inspects the `type` field: `"text"` chunks are forwarded to `Display`; `"tool_call"` chunks are routed to `ToolRegistry.execute()`; `"done"` signals end of stream.

---

### SessionManager / Session

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
    def __init__(self, session_id: str, session_dir: Path): ...

    def add_message(self, role: str, content: str) -> None:
        """Append to both history_full.jsonl and history_current.jsonl."""

    def get_messages(self) -> list[dict]:
        """Return messages from history_current.jsonl."""

    def compact(self, instructions: str = "") -> None:
        """Request summary from LLM, rewrite history_current.jsonl."""

    def set_name(self, name: str) -> None:
        """Write name to metadata.yaml."""

    def token_usage(self) -> tuple[int, int]:
        """Returns (used_tokens, context_window)."""

    def should_compact(self) -> bool:
        """True if used_tokens > context_window * 0.9."""
```

---

### SessionMeta

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

### REPL

```python
class REPL:
    def __init__(
        self,
        session: Session,
        tool_registry: ToolRegistry,
        llm_client: LLMClient,
        display: Display,
        permission_manager: PermissionManager,
    ): ...

    def run(self) -> None:
        """Main loop: read input → route → render."""

    def _handle_input(self, raw: str) -> None:
        """Route to _handle_slash_command or _send_to_llm."""

    def _handle_slash_command(self, command: str) -> None:
        """Dispatch /help, /exit, /clear, /tools, /compact, /session."""

    def _send_to_llm(self, user_input: str) -> None:
        """
        Append user message, stream response, detect tool calls,
        execute via tool_registry, feed results back, continue streaming.
        """

    def _check_compaction(self) -> None:
        """Called after each exchange. Auto-compact if should_compact()."""
```

---

### Display

```python
class Display:
    def __init__(self, verbose: bool = False): ...

    def toggle_verbose(self) -> None: ...

    def show_assistant_message(self, text: str) -> None:
        """Always shown in full regardless of mode."""

    def show_tool_call(self, name: str, args: dict) -> None:
        """Summary mode: one line. Verbose: full args."""

    def show_tool_result(self, name: str, result: str) -> None:
        """Summary mode: hidden. Verbose: full output."""

    def show_status(self, message: str) -> None:
        """Informational messages (compaction notices, warnings, etc.)."""

    def show_error(self, message: str) -> None: ...

    def show_permission_prompt(
        self,
        question: str,
        universal_options: list[str],
        extra_options: list[str],
    ) -> tuple[str, str]:
        """Render permission prompt, return (choice, user_text)."""

    def show_session_list(self, sessions: list[SessionMeta]) -> SessionMeta | None:
        """Render interactive resume picker, return chosen session."""
```

---

### MCPManager

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

### Completer

```python
class Completer:
    def __init__(
        self,
        workspace: Workspace,
        tool_registry: ToolRegistry,
    ): ...

    def get_completions(self, document: Document) -> list[Completion]:
        """
        Called by prompt_toolkit on every keystroke.
        Routes to appropriate completer based on context.
        """

    def _complete_slash_command(self, text: str) -> list[Completion]:
        """Complete /help, /exit, /tools etc."""

    def _complete_file_path(self, text: str) -> list[Completion]:
        """Standard tab completion for file paths."""

    def _complete_at_picker(
        self,
        text: str,
        explicit: bool,  # True if triggered by @!
    ) -> list[Completion]:
        """
        Popup file picker. Filters via workspace.is_ignored()
        unless explicit=True.
        """
```

Built on `prompt_toolkit`. Detects `@` / `@!` by inspecting the current word in the `Document`.

---

## Entry Point and Startup Sequence

```python
# __main__.py
def main():
    args = parse_args()
    # args: --init, --workspace, --backend, --resume, --continue, --session-id

    # 1. Resolve workspace
    start = Path(args.workspace) if args.workspace else Path.cwd()

    if args.init:
        Workspace.initialise(start)  # create .ai-cli/ scaffold, exit
        return

    root = Workspace.find_root(start)
    if root is None:
        root = prompt_init_or_exit(start)

    # 2. Bootstrap core objects
    config = ConfigManager(root, cli_overrides_from(args))
    workspace = Workspace(root, config)
    permission_manager = PermissionManager(prompt_fn=...)
    tool_registry = ToolRegistry(workspace, config, permission_manager)
    llm_client = create_llm_client(config)
    session_manager = SessionManager(workspace, llm_client, SESSIONS_DIR)
    mcp_manager = MCPManager(tool_registry, config)

    # 3. Load tools and MCP servers
    tool_registry.load()
    mcp_manager.connect_all()

    # 4. Resolve session
    session = resolve_session(args, session_manager, workspace)

    # 5. Start REPL
    display = Display()
    repl = REPL(session, tool_registry, llm_client, display, permission_manager)
    repl.run()

    # 6. Cleanup
    mcp_manager.disconnect_all()
```

`resolve_session()` handles `--resume` / `--resume <id>` / `--continue` / new session logic, including the last-message role check and permission reset.
