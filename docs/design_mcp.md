# Design: MCP (Model Context Protocol) Support

## Purpose

`MCPManager` connects the CLI to external *MCP servers* — separate processes or
network services that expose additional tools via Anthropic's Model Context
Protocol.  From the LLM's perspective MCP tools are indistinguishable from
built-in tools: they appear in the same `tools` array in every API call and
return results through the same tool-call mechanism.  The CLI acts as the MCP
*client*, intercepting tool calls from the LLM, forwarding them to the
appropriate server, and returning the result.

The system is entirely additive.  When no MCP servers are configured, nothing
changes — no extra startup cost, no new registry entries.

---

## Backward Compatibility

MCP tools are registered into the existing `ToolRegistry` via
`register_instance()` alongside built-in tools.  The REPL, `Agent`, and
`LLMClient` layers are unaware that any given tool is backed by an MCP server.

```
Without MCP:                   With MCP:
ToolRegistry                   ToolRegistry
  └─ read_file (built-in)        └─ read_file         (built-in)
  └─ write_file (built-in)       └─ write_file        (built-in)
  └─ tool_manager                └─ tool_manager
                                 └─ context7__resolve-library-id  (MCP proxy)
                                 └─ context7__get-library-docs    (MCP proxy)
```

---

## Transports

### stdio

The CLI spawns a child process.  Communication is newline-delimited JSON-RPC 2.0
over the child's stdin/stdout.  The process lives for the duration of the CLI
session and is terminated on exit.

Config example:
```yaml
servers:
  filesystem:
    transport: stdio
    command: uvx
    args:
      - mcp-server-filesystem
      - /home/user/projects
```

### SSE (Server-Sent Events)

The CLI connects to a long-running HTTP server.  Requests are sent as HTTP POST;
responses arrive as a Server-Sent Events stream.  The server outlives the client.

Config example:
```yaml
servers:
  context7:
    transport: sse
    url: https://mcp.context7.com/mcp
    api_key_env: CONTEXT7_API_KEY
    api_key_header: CONTEXT7_API_KEY
    api_key_prefix: ""
```

`api_key_prefix` is prepended to the resolved key value before it is written
into `api_key_header`.  Set it to `"Bearer "` for servers that expect OAuth-style
tokens; leave it empty for servers that expect a bare key.

---

## Configuration

### File locations

| Layer   | Path                           | Precedence |
|---------|--------------------------------|------------|
| Global  | `~/.ai-cli/mcp.yaml`           | Lower      |
| Project | `<project>/.ai-cli/mcp.yaml`   | Higher     |

Project config is **field-level merged** on top of global config when the
same server name appears in both files.  Connection fields (`transport`,
`url`, `command`, `args`, auth fields) come from the global entry unless
the project entry overrides them.  State fields (`disabled`, `allowed`,
`tools` per-tool overrides) are merged the same way, with project winning
on any individual key collision.  The `tools` sub-dict is deep-merged so
per-tool overrides from both files are combined rather than replaced.

### Schema

```yaml
servers:
  <server-name>:
    transport: stdio | sse

    # stdio only:
    command: <executable>
    args:
      - <arg1>
      - ...

    # sse only:
    url: <endpoint-url>
    api_key_env: <ENV_VAR_NAME>       # name of env var holding the key
    api_key_header: <HEADER_NAME>     # HTTP header to send the key in
    api_key_prefix: <PREFIX>          # prepended to key value (e.g. "Bearer ")
```

All fields other than `transport` (and the transport-specific required fields)
are optional.

### Global example — Context7

`~/.ai-cli/mcp.yaml`:
```yaml
servers:
  context7:
    transport: sse
    url: https://mcp.context7.com/mcp
    api_key_env: CONTEXT7_API_KEY
    api_key_header: CONTEXT7_API_KEY
    api_key_prefix: ""
```

---

## Startup Lifecycle

1. `__main__.py` calls `MCPManager.connect_all()` after `ToolRegistry.load()`.
2. For each configured server, `MCPManager`:
   a. Resolves the API key from the environment (SSE only).  If `api_key_env`
      is set but the variable is absent → **skip** with a clear warning message;
      do not attempt to connect.
   b. Opens the transport (spawns subprocess / opens HTTP session).
   c. Performs the JSON-RPC handshake (`initialize` → `initialized`).
   d. Calls `tools/list` to discover the server's tool schemas.
   e. Creates an `MCPProxyTool` instance for each discovered tool and registers
      it via `ToolRegistry.register_instance()`.
3. Any connection failure (process won't start, handshake error, network
   timeout) is logged as a warning and that server is skipped.  The CLI
   continues to start normally.
4. Disabled servers (see [Runtime State](#runtime-state)) are still connected
   at startup so that `/mcp info` and tab completion work.  Their tools are
   registered but marked disabled so the LLM cannot use them.

---

## Tool Naming

MCP tool names are namespaced using the server name and a double-underscore
separator to avoid collisions with built-in tools and between servers:

```
<server-name>__<tool-name>
```

Example: server `context7`, tool `resolve-library-id` →
registered as `context7__resolve-library-id`.

**Constraints**:
- The full namespaced name must match `^[a-zA-Z0-9_][a-zA-Z0-9_-]{0,63}$`.
  This means the first character must be alphanumeric or underscore; a leading
  hyphen is not allowed.
- If the combined name exceeds 64 characters it is skipped at registration
  with a warning naming the server and tool.
- Double-underscore is chosen because single underscores are common in both
  server names and tool names, making `__` an unambiguous separator that stays
  within the allowed character set.

---

## `MCPProxyTool`

Each MCP tool is represented by an `MCPProxyTool` instance.  It implements the
`Tool` ABC so it is handled identically to built-in tools by `ToolRegistry`.

```python
class MCPProxyTool(Tool):
    NAME: str                  # set per-instance to "<server>__<tool>"
    DESCRIPTION: str           # set per-instance from MCP tool description
    PERMISSION_REQUIRED = False # MCP tools do not use interactive permission prompts
    DISABLED_BY_DEFAULT = False

    def __init__(
        self,
        server_name: str,
        mcp_tool_name: str,       # original name from tools/list
        namespaced_name: str,     # "<server>__<tool>"
        description: str,
        input_schema: dict,       # JSON Schema from tools/list
        transport: "_MCPTransport",
        workspace: Workspace,
        permission_manager: PermissionManager,
    ): ...

    def definition(self) -> dict:
        """Wrap the MCP server's inputSchema into an OpenAI function-calling dict."""

    def execute(self, **kwargs) -> dict:
        """Forward the call to transport.call_tool(); translate MCP result to canonical format."""
```

`PERMISSION_REQUIRED = False` because MCP servers are pre-vetted by the user
at configuration time, so MCP tools do not use interactive execution approval
prompts.  `/mcp enable`/`disable` and `/mcp allow`/`disallow` control whether
a tool is available to the LLM, matching the behaviour of the `/tools` command.
When used with a server name (e.g. `/mcp allow context7`) they apply to all
tools exposed by that server; when used with a tool name (e.g.
`/mcp allow context7 resolve-library-id`) they apply to that single tool.

---

## `MCPManager` Interface

```python
class MCPServerConfig:
    name: str
    transport: Literal["stdio", "sse"]

    # stdio
    command: str | None
    args: list[str]

    # sse
    url: str | None
    api_key_env: str | None
    api_key_header: str | None
    api_key_prefix: str

class ServerStatus:
    name: str
    connected: bool
    error: str | None           # None if connected
    tool_count: int
    tools: list[str]            # original (un-namespaced) tool names

class MCPManager:
    def __init__(
        self,
        global_config_path: Path,    # ~/.ai-cli/mcp.yaml
        project_config_path: Path | None,  # <project>/.ai-cli/mcp.yaml or None
        tool_registry: ToolRegistry,
        workspace: Workspace,
        permission_manager: PermissionManager,
    ): ...

    def connect_all(self) -> None:
        """
        Load mcp.yaml (global then project, project wins on collision).
        For each server: resolve key, open transport, handshake, discover tools,
        register MCPProxyTool instances. Skips servers that fail at any step.
        Logs a warning for each skipped server with the reason.
        """

    def status(self) -> list[ServerStatus]:
        """Return connection status and tool list for every configured server."""

    def get_server_tools(self, server_name: str) -> list[str]:
        """Return original tool names for a server (for tab completion)."""

    def enable_server(self, server_name: str, *, persist: bool = False) -> None:
    def disable_server(self, server_name: str, *, persist: bool = False) -> None:
    def allow_server(self, server_name: str, *, persist: bool = False) -> None:
    def disallow_server(self, server_name: str, *, persist: bool = False) -> None:

    def enable_tool(self, server_name: str, tool_name: str, *, persist: bool = False) -> None:
    def disable_tool(self, server_name: str, tool_name: str, *, persist: bool = False) -> None:
    def allow_tool(self, server_name: str, tool_name: str, *, persist: bool = False) -> None:
    def disallow_tool(self, server_name: str, tool_name: str, *, persist: bool = False) -> None:
```

`MCPError` is a module-level exception raised on unrecoverable transport
failures (connection dropped mid-session, process exited, etc.).

---

## Runtime State

Server- and tool-level enable/disable/allow/disallow state follows the same
two-layer model as `ToolRegistry`:

| Layer   | Scope                            | Written by       |
|---------|----------------------------------|------------------|
| Session | In-memory, reset on exit/resume  | `/mcp` commands  |
| Persist | Written to project `mcp.yaml`    | `--persist` flag |

**Default behaviour** (no `--persist`): all runtime mutations are in-memory
only for the current session.  This is the intended new behaviour that
`/tools` will adopt in a future refactor; `/mcp` implements it from the start.

Disabling a *server* disables all of that server's tools at once.  Disallowing
a server disallows all its tools and prevents transient enables.

---

## `/mcp` Slash Command

All subcommands support tab completion: command name → server name → tool name.

| Command | Action |
|---------|--------|
| `/mcp list` | List all configured servers with connection status and tool count |
| `/mcp info <server>` | Show all tools exposed by that server (name + description) |
| `/mcp enable <server>` | Enable all tools for that server (in-memory) |
| `/mcp enable <server> <tool>` | Enable one tool (in-memory) |
| `/mcp disable <server>` | Disable all tools for that server (in-memory) |
| `/mcp disable <server> <tool>` | Disable one tool (in-memory) |
| `/mcp allow <server>` | Allow all tools for that server (in-memory) |
| `/mcp allow <server> <tool>` | Allow one tool (in-memory) |
| `/mcp disallow <server>` | Disallow all tools for that server (in-memory) |
| `/mcp disallow <server> <tool>` | Disallow one tool (in-memory) |

Append `--persist` to any enable/disable/allow/disallow command to write the
change to the project `.ai-cli/mcp.yaml`.

### `/mcp list` output (example)

```
MCP Servers
  context7    connected   2 tools   (SSE)
  filesystem  ERROR: command not found
```

### `/mcp info context7` output (example)

```
context7  [SSE — https://mcp.context7.com/mcp]
  context7__resolve-library-id    Resolves a library name to a Context7 ID
  context7__get-library-docs      Fetches documentation for a Context7 library ID
```

---

## MCP JSON-RPC Protocol Sequence

```
client                              server
  │──── initialize ────────────────▶│
  │◀─── result (capabilities) ──────│
  │──── initialized (notification) ▶│
  │──── tools/list ─────────────────▶│
  │◀─── result [{name, description, inputSchema}, …] ──│
  │
  │   (LLM requests a tool call)
  │──── tools/call {name, arguments} ──────────────────▶│
  │◀─── result {content: [{type, text}]} ───────────────│
```

---

## MCP Result → Canonical Tool Response Translation

MCP `tools/call` results use a `content` array of typed blocks.  These are
translated to the CLI's canonical tool response format:

| MCP content block type | Canonical response |
|---|---|
| `{"type": "text", "text": "..."}` | `{"status": "success", "data": {"text": "..."}}` |
| Multiple text blocks | Text values joined with `\n\n` into a single `"text"` field |
| `{"type": "error", ...}` | `{"status": "error", "error": "mcp_error", "message": "...", "code": 500}` |
| Transport / JSON-RPC error | `{"status": "error", "error": "mcp_transport_error", "message": "...", "code": 503}` |

---

## Error Handling

| Failure point | Behaviour |
|---|---|
| `api_key_env` set but env var absent | Skip server at startup; log warning |
| Process won't start (stdio) | Skip server at startup; log warning |
| Handshake timeout / error | Skip server at startup; log warning |
| `tools/list` returns empty list | Connect succeeds; no tools registered; log info |
| Namespaced tool name > 64 chars | Skip that tool; log warning; other tools from same server register normally |
| Transport error during `tools/call` | Return `{"status": "error", "error": "mcp_transport_error", ...}` to LLM |
| Server process exits mid-session | Subsequent calls return transport error to LLM |

---

## Files Changed / Created

| File | Change |
|---|---|
| `ai_cli/core/mcp_manager.py` | **New** — `MCPManager`, `MCPProxyTool`, `MCPServerConfig`, `ServerStatus`, `MCPError` |
| `ai_cli/__main__.py` | Call `MCPManager.connect_all()` after `ToolRegistry.load()`; wire `/mcp` command |
| `ai_cli/cli/repl.py` | Add `/mcp` slash command handler and tab completion |
| `ai_cli/cli/completer.py` | Add completion for `/mcp` subcommands, server names, tool names |
| `~/.ai-cli/mcp.yaml` | **New** global config — Context7 server definition |
| `tests/test_mcp_manager.py` | **New** — unit tests for `MCPManager` and `MCPProxyTool` |
