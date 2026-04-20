"""
mcp_manager.py — MCP (Model Context Protocol) server connections and tool exposure.

Connects to configured MCP servers at startup, discovers their tools via
JSON-RPC ``tools/list``, and registers each tool as an :class:`MCPProxyTool`
in the ``ToolRegistry``.  Tool calls from the LLM are forwarded to the
appropriate server via ``tools/call`` and the result is translated to the
canonical tool response format.

Configuration
-------------
Global:   ``~/.ai-cli/mcp.yaml``
Project:  ``<project>/.ai-cli/mcp.yaml``

Project config is field-level merged on top of global config: connection
fields (transport, url, command, …) come from global unless overridden in
the project entry; state fields (disabled, allowed, tools overrides) are
read from whichever file defines them, with project winning on collision.

Schema::

    servers:
      <name>:
        transport: stdio | sse
        # stdio:
        command: <executable>
        args: [<arg>, ...]
        # sse:
        url: <endpoint>
        api_key_env: <ENV_VAR>        # env-var name
        api_key_header: <HEADER>      # HTTP header name
        api_key_prefix: <PREFIX>      # prepended to key value

Tool naming
-----------
MCP tools are registered as ``<server>__<tool>`` (double underscore).
The combined name must satisfy ``^[a-zA-Z0-9_][a-zA-Z0-9_-]{0,63}$``
(must not start with a hyphen); names that violate this are skipped with
a warning.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import yaml

from ai_cli.core.tool_registry import TOOL_NAME_RE
from ai_cli.tools.base import Tool, ToolArgument

if TYPE_CHECKING:
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.tool_registry import ToolRegistry
    from ai_cli.core.workspace import Workspace

logger = logging.getLogger(__name__)

# MCP JSON-RPC protocol version advertised in ``initialize``.
_MCP_PROTOCOL_VERSION = "2024-11-05"


def _client_info() -> dict[str, str]:
    """Return the ``clientInfo`` payload for the MCP ``initialize`` handshake.

    Resolves the version from the installed ``ai-cli`` package metadata so
    it stays in sync with ``pyproject.toml`` and makes server-side
    diagnostics accurate.  Falls back to ``"unknown"`` if the metadata
    cannot be read (e.g. running from a source tree without installation).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return {"name": "ai-cli", "version": version("ai-cli")}
        except PackageNotFoundError:
            return {"name": "ai-cli", "version": "unknown"}
    except Exception:  # pragma: no cover — defensive catch-all
        return {"name": "ai-cli", "version": "unknown"}


# Double-underscore namespace separator.
_NS_SEP = "__"

# Server names must be valid as the prefix of "<server>__<tool>" and must be
# usable as whitespace-delimited tokens in `/mcp` commands.  Same charset as
# TOOL_NAME_RE but without the 64-char total length cap (the combined
# namespaced name is validated separately at registration time).
_SERVER_NAME_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_-]*$")


# HTTP timeout for SSE transport (seconds).
_CONNECT_TIMEOUT = 15.0
_CALL_TIMEOUT = 60.0

# Timeout waiting for a response from a stdio server (seconds).
_STDIO_TIMEOUT = 60.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MCPError(Exception):
    """Raised on unrecoverable MCP transport or protocol errors."""


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class MCPServerConfig:
    name: str
    transport: Literal["stdio", "sse"]
    # stdio
    command: str | None = None
    args: list[str] = field(default_factory=list)
    # sse
    url: str | None = None
    api_key_env: str | None = None
    api_key_header: str | None = None
    api_key_prefix: str = ""
    # persisted state (loaded from config, applied after tool registration)
    disabled: bool = False
    allowed: bool = True
    tool_overrides: dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Status dataclass
# ---------------------------------------------------------------------------


@dataclass
class ServerStatus:
    name: str
    connected: bool
    error: str | None  # None when connected
    tool_count: int
    tools: list[str]  # original (un-namespaced) tool names


# ---------------------------------------------------------------------------
# Internal schema wrapper
# ---------------------------------------------------------------------------


class _MCPToolSchema:
    """ToolSchema-compatible wrapper for MCP tool schemas.

    Stores the raw OpenAI-format schema for ``schema()`` (called by
    ``definitions()``) and a list of :class:`ToolArgument` objects for
    ``_validate_args()`` compatibility.  The argument list is derived from
    the MCP ``inputSchema``; complex / nested types are mapped to their
    closest JSON Schema primitive so basic validation still works.
    """

    def __init__(self, openai_schema: dict, arguments: list[ToolArgument]) -> None:
        self._openai_schema = openai_schema
        self.arguments = arguments

    def schema(self) -> dict:
        return self._openai_schema


def _input_schema_to_arguments(input_schema: dict) -> list[ToolArgument]:
    """Convert an MCP ``inputSchema`` to a list of :class:`ToolArgument` objects."""
    raw_properties = input_schema.get("properties")
    properties: dict = raw_properties if isinstance(raw_properties, dict) else {}

    raw_required = input_schema.get("required")
    required: set[str] = (
        {item for item in raw_required if isinstance(item, str)}
        if isinstance(raw_required, list)
        else set()
    )
    args: list[ToolArgument] = []
    for param_name, prop in properties.items():
        # Non-string keys would later flow into ToolArgument.name and crash
        # downstream string operations (e.g. ", ".join(missing)) — skip them.
        if not isinstance(param_name, str):
            continue
        if not isinstance(prop, dict):
            continue
        raw_type = prop.get("type", "string")
        # Map to the JSON Schema primitives _check_type understands.
        arg_type: str = (
            raw_type
            if raw_type in ("string", "integer", "number", "boolean", "array", "object")
            else "string"
        )
        raw_desc = prop.get("description", "")
        description: str = raw_desc if isinstance(raw_desc, str) else ""
        args.append(
            ToolArgument(
                name=param_name,
                description=description,
                argument_type=arg_type,
                required=param_name in required,
            )
        )
    return args


def _build_openai_schema(
    namespaced_name: str, description: str, input_schema: dict
) -> dict:
    """Wrap an MCP ``inputSchema`` in the OpenAI function-calling envelope."""
    raw_properties = input_schema.get("properties")
    # Only keep string keys whose values are dicts — anything else would
    # violate the JSON Schema contract embedded in the OpenAI function
    # parameters and could break registration or tool calls at runtime.
    if isinstance(raw_properties, dict):
        properties: dict = {
            k: v
            for k, v in raw_properties.items()
            if isinstance(k, str) and isinstance(v, dict)
        }
    else:
        properties = {}

    parameters: dict = {
        "type": "object",
        "properties": properties,
    }

    raw_required = input_schema.get("required")
    if isinstance(raw_required, list):
        required = [
            item
            for item in raw_required
            if isinstance(item, str) and item in properties
        ]
        if required:
            parameters["required"] = required
    return {
        "type": "function",
        "function": {
            "name": namespaced_name,
            "description": description,
            "parameters": parameters,
        },
    }


# ---------------------------------------------------------------------------
# Transport layer
# ---------------------------------------------------------------------------


class _MCPTransport(ABC):
    """Abstract JSON-RPC transport for a single MCP server."""

    @abstractmethod
    def initialize(self) -> dict:
        """Perform the MCP handshake and return the server's capabilities dict."""

    @abstractmethod
    def list_tools(self) -> list[dict]:
        """Return the server's tool list (each entry has name, description, inputSchema)."""

    @abstractmethod
    def call_tool(self, mcp_name: str, arguments: dict) -> dict:
        """Call a tool by its original (un-namespaced) name and return a canonical result."""

    @abstractmethod
    def close(self) -> None:
        """Tear down the connection/process."""


class _SSETransport(_MCPTransport):
    """HTTP POST + SSE transport for remote MCP servers."""

    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self._url = url
        self._headers: dict[str, str] = dict(headers)
        self._lock = threading.Lock()
        self._next_id = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> dict:
        resp = self._post(
            "initialize",
            {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": _client_info(),
            },
        )
        result = self._extract_result(resp, "initialize")
        # Send the ``initialized`` notification (no response expected).
        with contextlib.suppress(MCPError):  # best-effort; some servers don't expect it
            self._post("initialized", {}, is_notification=True)
        return result

    def list_tools(self) -> list[dict]:
        resp = self._post("tools/list", {})
        result = self._extract_result(resp, "tools/list")
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise MCPError("tools/list result missing 'tools' list")
        return tools

    def call_tool(self, mcp_name: str, arguments: dict) -> dict:
        try:
            resp = self._post("tools/call", {"name": mcp_name, "arguments": arguments})
        except MCPError as exc:
            return Tool._err("mcp_transport_error", str(exc), 503)
        try:
            result = self._extract_result(resp, "tools/call")
        except MCPError as exc:
            return Tool._err("mcp_error", str(exc), 500)
        return _mcp_result_to_canonical(result)

    def close(self) -> None:
        pass  # stateless HTTP; nothing to tear down

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(
        self, method: str, params: dict, *, is_notification: bool = False
    ) -> dict | None:
        """Send a JSON-RPC POST and return the parsed response.

        The lock is held only for the short sections that touch shared state
        (``_next_id`` allocation, snapshotting ``_headers``, and later storing
        a captured ``Mcp-Session-Id``). The network round-trip itself runs
        without the lock so that concurrent callers (e.g. parallel agents) are
        not serialised by head-of-line blocking on a single in-flight request.
        """
        with self._lock:
            req_id: int | None = None
            if not is_notification:
                req_id = self._next_id
                self._next_id += 1
            # Snapshot headers under the lock so a concurrent update does not
            # mutate the dict while we iterate it below.
            header_snapshot = dict(self._headers)

        payload: dict = {"jsonrpc": "2.0", "method": method}
        if req_id is not None:
            payload["id"] = req_id
        if params:
            payload["params"] = params

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **header_snapshot,
        }
        req = urllib.request.Request(
            self._url, data=body, headers=headers, method="POST"
        )
        timeout = (
            _CONNECT_TIMEOUT
            if method in ("initialize", "tools/list")
            else _CALL_TIMEOUT
        )
        # Overall deadline for the whole request — used to bound SSE
        # streaming so a chatty server cannot keep us alive indefinitely
        # with non-matching keepalives.
        sse_deadline = time.monotonic() + timeout

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # Capture session ID from the first response. Only the store
                # itself needs the lock; the read does not.
                session_id = resp.getheader("Mcp-Session-Id")
                if session_id:
                    with self._lock:
                        if "Mcp-Session-Id" not in self._headers:
                            self._headers["Mcp-Session-Id"] = session_id

                if is_notification:
                    return None

                content_type = resp.getheader("Content-Type", "")

                if "text/event-stream" in content_type:
                    return self._stream_sse(resp, req_id, deadline=sse_deadline)

                raw = resp.read()
                return cast(dict, json.loads(raw))

        except urllib.error.HTTPError as exc:
            raise MCPError(f"HTTP {exc.code} from {self._url}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise MCPError(f"Request timed out for {self._url}") from exc
            raise MCPError(f"Connection failed to {self._url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise MCPError(f"Invalid JSON from {self._url}: {exc}") from exc
        except TimeoutError as exc:
            raise MCPError(f"Request timed out for {self._url}") from exc

    @staticmethod
    def _stream_sse(
        resp: Any, req_id: int | None, deadline: float | None = None
    ) -> dict:
        """Stream an SSE response line-by-line, returning the first matching JSON-RPC object.

        Unlike reading the entire body at once, this avoids blocking
        indefinitely on servers that keep the event-stream connection open.

        If *deadline* (a ``time.monotonic()`` value) is given, the stream is
        abandoned with ``MCPError`` once that moment passes.  Without a
        deadline, a chatty server emitting non-matching keepalives forever
        would never trigger the socket's read-inactivity timeout.
        """
        data_parts: list[str] = []
        for raw_line in resp:
            if deadline is not None and time.monotonic() >= deadline:
                raise MCPError(
                    "Timed out waiting for matching JSON-RPC response in SSE stream"
                )
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("data:"):
                data_parts.append(line[5:].lstrip())
            elif line == "" and data_parts:
                event_text = "\n".join(data_parts)
                data_parts = []
                try:
                    obj = json.loads(event_text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                # Return immediately if this matches our request ID or if
                # we have no ID to match (first valid object wins).
                if req_id is None or obj.get("id") == req_id:
                    return obj
        # End of stream — try any trailing data.  Apply the same req_id match
        # as the in-stream path so a dangling non-matching notification does
        # not get mistaken for the response.
        if data_parts:
            try:
                obj = json.loads("\n".join(data_parts))
                if isinstance(obj, dict) and (
                    req_id is None or obj.get("id") == req_id
                ):
                    return obj
            except json.JSONDecodeError:
                pass
        raise MCPError("No valid JSON-RPC data in SSE response")

    @staticmethod
    def _parse_sse(raw: bytes) -> dict:
        """Parse a complete SSE response body and return the first JSON-RPC object.

        Kept for cases where the full body is already available (e.g. tests).
        """
        text = raw.decode("utf-8", errors="replace")
        data_parts: list[str] = []
        for line in text.splitlines():
            if line.startswith("data:"):
                data_parts.append(line[5:].lstrip())
            elif line == "" and data_parts:
                event_text = "\n".join(data_parts)
                data_parts = []
                try:
                    return cast(dict, json.loads(event_text))
                except json.JSONDecodeError:
                    continue
        if data_parts:
            try:
                return cast(dict, json.loads("\n".join(data_parts)))
            except json.JSONDecodeError:
                pass
        raise MCPError("No valid JSON-RPC data in SSE response")

    @staticmethod
    def _extract_result(resp: dict | None, method: str) -> dict:
        """Raise MCPError if *resp* is a JSON-RPC error; otherwise return result."""
        if resp is None:
            raise MCPError(f"{method}: no response received")
        if not isinstance(resp, dict):
            raise MCPError(f"{method}: malformed response (expected JSON object)")
        if "error" in resp:
            err = resp["error"]
            if not isinstance(err, dict):
                raise MCPError(f"{method}: malformed error response")
            code = err.get("code", "?")
            msg = err.get("message", "Unknown error")
            raise MCPError(f"{method} error {code}: {msg}")
        result = resp.get("result")
        if result is None:
            raise MCPError(f"{method}: response has no 'result' field")
        if not isinstance(result, dict):
            raise MCPError(f"{method}: malformed result (expected object)")
        return result


class _StdioTransport(_MCPTransport):
    """stdio transport — the MCP server runs as a child subprocess."""

    def __init__(self, command: str, args: list[str]) -> None:
        try:
            self._proc = subprocess.Popen(
                [command, *args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # avoid stderr buffer deadlock
            )
        except OSError as exc:
            raise MCPError(
                f"Failed to start MCP server command {command!r}: {exc}"
            ) from exc
        self._lock = threading.Lock()
        self._next_id = 1
        # Background reader thread drains stdout into a queue so that
        # _request() can use a timeout instead of blocking on readline().
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="mcp-stdio-reader"
        )
        self._reader_thread.start()

    def _read_loop(self) -> None:
        """Read stdout lines into the queue until the process exits."""
        if self._proc.stdout is None:
            return
        with contextlib.suppress(Exception):
            for line in self._proc.stdout:
                self._queue.put(line)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> dict:
        resp = self._request(
            "initialize",
            {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": _client_info(),
            },
        )
        result = _SSETransport._extract_result(resp, "initialize")
        with contextlib.suppress(MCPError):  # best-effort; some servers don't expect it
            self._request("initialized", {}, is_notification=True)
        return result

    def list_tools(self) -> list[dict]:
        resp = self._request("tools/list", {})
        result = _SSETransport._extract_result(resp, "tools/list")
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise MCPError("tools/list result missing 'tools' list")
        return tools

    def call_tool(self, mcp_name: str, arguments: dict) -> dict:
        try:
            resp = self._request(
                "tools/call", {"name": mcp_name, "arguments": arguments}
            )
        except MCPError as exc:
            return Tool._err("mcp_transport_error", str(exc), 503)
        try:
            result = _SSETransport._extract_result(resp, "tools/call")
        except MCPError as exc:
            return Tool._err("mcp_error", str(exc), 500)
        return _mcp_result_to_canonical(result)

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            with contextlib.suppress(Exception):
                self._proc.kill()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(
        self, method: str, params: dict, *, is_notification: bool = False
    ) -> dict | None:
        with self._lock:
            req_id: int | None = None
            if not is_notification:
                req_id = self._next_id
                self._next_id += 1

            payload: dict = {"jsonrpc": "2.0", "method": method}
            if req_id is not None:
                payload["id"] = req_id
            if params:
                payload["params"] = params

            if self._proc.stdin is None:
                raise MCPError("Server process stdin is closed")
            try:
                self._proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise MCPError(
                    f"Failed to send request to server process: "
                    f"broken stdio pipe ({exc})"
                ) from exc

            if is_notification:
                return None

            # Read from the background-reader queue with a timeout.
            # Use an overall deadline so that a flood of non-matching
            # notifications/log messages cannot keep resetting the per-get
            # timeout and block indefinitely.
            deadline = time.monotonic() + _STDIO_TIMEOUT
            while True:
                # Fail fast if the process is already dead.
                exit_code = self._proc.poll()
                if exit_code is not None:
                    raise MCPError(
                        f"Server process exited before responding (exit code {exit_code})"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPError(
                        f"No response from server after {_STDIO_TIMEOUT:.0f}s"
                    )
                try:
                    raw_line = self._queue.get(timeout=min(remaining, 5.0))
                except queue.Empty as exc:
                    exit_code = self._proc.poll()
                    if exit_code is not None:
                        raise MCPError(
                            f"Server process exited before responding (exit code {exit_code})"
                        ) from exc
                    if time.monotonic() >= deadline:
                        raise MCPError(
                            f"No response from server after {_STDIO_TIMEOUT:.0f}s"
                        ) from exc
                    continue  # still within deadline; retry
                if not raw_line:
                    raise MCPError("Server process closed connection")
                try:
                    response = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise MCPError(f"Invalid JSON from server: {exc}") from exc
                if not isinstance(response, dict):
                    raise MCPError(
                        "Invalid JSON-RPC response from server: expected object"
                    )
                if response.get("id") == req_id:
                    return cast(dict, response)


# ---------------------------------------------------------------------------
# MCP result translation
# ---------------------------------------------------------------------------


def _mcp_result_to_canonical(result: dict) -> dict:
    """Translate a ``tools/call`` result to the canonical ``{status, data}`` format."""
    content = result.get("content") or []
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type == "text":
            text = block.get("text", "")
            if isinstance(text, str) and text:
                text_parts.append(text)
        elif block_type == "error":
            error_text = block.get("text", "Unknown MCP error")
            if not isinstance(error_text, str) or not error_text:
                error_text = "Unknown MCP error"
            return Tool._err("mcp_error", error_text, 500)
    combined = "\n\n".join(text_parts)
    return Tool._ok({"text": combined})


# ---------------------------------------------------------------------------
# MCPProxyTool
# ---------------------------------------------------------------------------


class MCPProxyTool(Tool):
    """Proxy tool that forwards LLM calls to an MCP server.

    One instance is created per MCP tool discovered during ``connect_all()``.
    Registered via :meth:`~ai_cli.core.tool_registry.ToolRegistry.register_instance`
    so non-standard constructor arguments are allowed.
    """

    # Class-level attrs required by ToolRegistry validation (register_instance
    # path does not call _validate_tool_class, but define them for clarity).
    PERMISSION_REQUIRED = False
    DISABLED_BY_DEFAULT = False

    def __init__(
        self,
        server_name: str,
        mcp_tool_name: str,
        namespaced_name: str,
        description: str,
        input_schema: dict,
        transport: _MCPTransport,
        workspace: Workspace,
        permission_manager: PermissionManager,
    ) -> None:
        super().__init__(
            workspace=workspace,
            permission_manager=permission_manager,
            permission_required=False,
            name=namespaced_name,
            description=description,
        )
        self._server_name = server_name
        self._mcp_tool_name = mcp_tool_name
        self._transport = transport

        arguments = _input_schema_to_arguments(input_schema)
        openai_schema = _build_openai_schema(namespaced_name, description, input_schema)
        self._schema = _MCPToolSchema(openai_schema, arguments)

    def definition(self) -> _MCPToolSchema:  # type: ignore[override]
        return self._schema

    def execute(self, **kwargs: Any) -> dict:
        return self._transport.call_tool(self._mcp_tool_name, kwargs)

    def execute_log(self, **kwargs: Any) -> str | None:
        return f"mcp {self._server_name}/{self._mcp_tool_name}"


# ---------------------------------------------------------------------------
# MCPManager
# ---------------------------------------------------------------------------


class MCPManager:
    """Connect to MCP servers, register their tools, and handle /mcp commands."""

    def __init__(
        self,
        global_config_path: Path,
        project_config_path: Path | None,
        tool_registry: ToolRegistry,
        workspace: Workspace,
        permission_manager: PermissionManager,
    ) -> None:
        self._global_config_path = global_config_path
        self._project_config_path = project_config_path
        self._tool_registry = tool_registry
        self._workspace = workspace
        self._permission_manager = permission_manager

        # Populated by connect_all()
        self._statuses: dict[str, ServerStatus] = {}
        self._transports: dict[str, _MCPTransport] = {}
        # server_name → list of original (un-namespaced) tool names
        self._server_tools: dict[str, list[str]] = {}
        # Raw merged config dict; used to seed project entries when persisting
        # state for servers defined only in the global config.
        self._raw_merged: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def connect_all(self) -> None:
        """Load config, connect to all servers, and register tools."""
        configs = self._load_configs()
        for cfg in configs:
            self._connect_server(cfg)

    def _load_configs(self) -> list[MCPServerConfig]:
        """Return merged server configs.

        Global and project entries are field-level merged: connection fields
        (transport, url, command, …) come from global unless the project entry
        overrides them; state fields (disabled, allowed, tools) are merged the
        same way.  Config files are parsed as YAML.
        """
        global_servers: dict[str, dict] = {}
        project_servers: dict[str, dict] = {}

        for path, target in (
            (self._global_config_path, global_servers),
            (self._project_config_path, project_servers),
        ):
            if path is None or not path.is_file():
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                logger.warning("Could not read MCP config %s: %s", path, exc)
                continue
            if not isinstance(data, dict):
                logger.warning(
                    "MCP config %s: expected a YAML mapping at root, got %s; ignoring.",
                    path,
                    type(data).__name__,
                )
                continue
            raw_servers = data.get("servers")
            if raw_servers is None:
                continue
            if not isinstance(raw_servers, dict):
                logger.warning(
                    "MCP config %s: 'servers' must be an object; ignoring.", path
                )
                continue
            servers = raw_servers
            for sname, sval in servers.items():
                if not isinstance(sval, dict):
                    logger.warning(
                        "MCP config %s: server %r must be an object; skipping.",
                        path,
                        sname,
                    )
                    continue
                target[sname] = sval

        # Field-level merge: start from global, overlay project fields on top.
        merged: dict[str, dict] = {}
        for name, global_entry in global_servers.items():
            merged[name] = dict(global_entry)
        for name, project_entry in project_servers.items():
            if name in merged:
                # Deep-merge the per-tool 'tools' sub-dict field-by-field so
                # that global and project per-tool keys are combined rather than
                # the project entry wholesale replacing the global one.
                global_tools: dict[str, dict] = {}
                raw_global_tools = merged[name].get("tools")
                if isinstance(raw_global_tools, dict):
                    global_tools = {
                        k: dict(v)
                        for k, v in raw_global_tools.items()
                        if isinstance(v, dict)
                    }
                raw_project_tools = project_entry.get("tools")
                project_tools_invalid = "tools" in project_entry and not isinstance(
                    raw_project_tools, dict
                )
                if project_tools_invalid:
                    logger.warning(
                        "MCP server %r: project 'tools' must be an object; ignoring.",
                        name,
                    )
                if isinstance(raw_project_tools, dict):
                    for tool_name, tool_entry in raw_project_tools.items():
                        if isinstance(tool_entry, dict):
                            global_tools[tool_name] = {
                                **global_tools.get(tool_name, {}),
                                **tool_entry,
                            }
                merged[name].update(project_entry)
                if global_tools:
                    merged[name]["tools"] = global_tools
                elif project_tools_invalid:
                    # Don't let an invalid project 'tools' value leak into the
                    # merged dict via update(); drop it here so downstream
                    # validation doesn't see the bogus value.
                    merged[name].pop("tools", None)
            else:
                merged[name] = dict(project_entry)

        self._raw_merged = merged

        configs: list[MCPServerConfig] = []
        for name, raw in merged.items():
            if not isinstance(name, str) or not _SERVER_NAME_RE.match(name):
                logger.warning(
                    "MCP server %r: name must match %s; skipping.",
                    name,
                    _SERVER_NAME_RE.pattern,
                )
                continue
            if not isinstance(raw, dict):
                logger.warning(
                    "MCP server %r: config must be a mapping; skipping.", name
                )
                continue
            transport = raw.get("transport", "")
            if transport not in ("stdio", "sse"):
                logger.warning(
                    "MCP server %r: unknown transport %r; skipping.", name, transport
                )
                continue

            # Validate field types.
            raw_args = raw.get("args")
            if raw_args is None:
                args: list[str] = []
            elif not isinstance(raw_args, list) or not all(
                isinstance(a, str) for a in raw_args
            ):
                logger.warning(
                    "MCP server %r: 'args' must be a list of strings; skipping.", name
                )
                continue
            else:
                args = raw_args
            command = raw.get("command")
            if command is not None and not isinstance(command, str):
                logger.warning(
                    "MCP server %r: 'command' must be a string; skipping.", name
                )
                continue
            url = raw.get("url")
            if url is not None and not isinstance(url, str):
                logger.warning("MCP server %r: 'url' must be a string; skipping.", name)
                continue

            # Validate auth string fields.
            api_key_env = raw.get("api_key_env")
            if api_key_env is not None and not isinstance(api_key_env, str):
                logger.warning(
                    "MCP server %r: 'api_key_env' must be a string; skipping.", name
                )
                continue
            api_key_header = raw.get("api_key_header")
            if api_key_header is not None and not isinstance(api_key_header, str):
                logger.warning(
                    "MCP server %r: 'api_key_header' must be a string; skipping.", name
                )
                continue
            api_key_prefix = raw.get("api_key_prefix", "")
            if not isinstance(api_key_prefix, str):
                logger.warning(
                    "MCP server %r: 'api_key_prefix' must be a string; skipping.", name
                )
                continue

            # Validate boolean state flags — require actual booleans.
            raw_disabled = raw.get("disabled", False)
            if not isinstance(raw_disabled, bool):
                logger.warning(
                    "MCP server %r: 'disabled' must be true/false; ignoring.", name
                )
                raw_disabled = False
            raw_allowed = raw.get("allowed", True)
            if not isinstance(raw_allowed, bool):
                logger.warning(
                    "MCP server %r: 'allowed' must be true/false; ignoring.", name
                )
                raw_allowed = True

            # Parse per-tool overrides — validate tools is a mapping.
            raw_tools_cfg = raw.get("tools")
            if raw_tools_cfg is None:
                tool_overrides: dict[str, dict] = {}
            elif not isinstance(raw_tools_cfg, dict):
                logger.warning(
                    "MCP server %r: 'tools' must be an object/mapping; ignoring.", name
                )
                tool_overrides = {}
            else:
                tool_overrides = {}
                for k, v in raw_tools_cfg.items():
                    if not isinstance(v, dict):
                        continue
                    cleaned = dict(v)
                    # Validate boolean flags — mirror server-level validation.
                    for flag, default in (("disabled", False), ("allowed", True)):
                        if flag in cleaned and not isinstance(cleaned[flag], bool):
                            logger.warning(
                                "MCP server %r tool %r: '%s' must be true/false; ignoring.",
                                name,
                                k,
                                flag,
                            )
                            cleaned[flag] = default
                    tool_overrides[k] = cleaned

            cfg = MCPServerConfig(
                name=name,
                transport=transport,
                command=command,
                args=args,
                url=url,
                api_key_env=api_key_env,
                api_key_header=api_key_header,
                api_key_prefix=api_key_prefix,
                disabled=raw_disabled,
                allowed=raw_allowed,
                tool_overrides=tool_overrides,
            )
            configs.append(cfg)
        return configs

    def _connect_server(self, cfg: MCPServerConfig) -> None:
        """Connect to one server and register its tools.  Errors warn and skip."""
        name = cfg.name
        try:
            transport = self._build_transport(cfg)
        except MCPError as exc:
            logger.warning("MCP server %r: skipping — %s", name, exc)
            self._statuses[name] = ServerStatus(
                name=name, connected=False, error=str(exc), tool_count=0, tools=[]
            )
            return

        try:
            transport.initialize()
            raw_tools = transport.list_tools()
        except MCPError as exc:
            logger.warning("MCP server %r: skipping — %s", name, exc)
            self._statuses[name] = ServerStatus(
                name=name, connected=False, error=str(exc), tool_count=0, tools=[]
            )
            transport.close()
            return

        self._transports[name] = transport
        tool_names: list[str] = []

        for tool_def in raw_tools:
            if not isinstance(tool_def, dict):
                continue

            raw_mcp_name = tool_def.get("name", "")
            if not isinstance(raw_mcp_name, str) or not raw_mcp_name:
                logger.warning(
                    "MCP server %r: skipping tool with invalid name %r.",
                    name,
                    raw_mcp_name,
                )
                continue
            mcp_name: str = raw_mcp_name

            raw_description = tool_def.get("description", "")
            if isinstance(raw_description, str):
                description: str = raw_description
            else:
                logger.warning(
                    "MCP server %r: tool %r has non-string description; "
                    "using empty string.",
                    name,
                    mcp_name,
                )
                description = ""

            raw_input_schema = tool_def.get("inputSchema")
            if raw_input_schema is None:
                input_schema: dict = {}
            elif isinstance(raw_input_schema, dict):
                input_schema = raw_input_schema
            else:
                logger.warning(
                    "MCP server %r: tool %r has invalid inputSchema; skipping.",
                    name,
                    mcp_name,
                )
                continue

            ns_name = f"{name}{_NS_SEP}{mcp_name}"
            if not TOOL_NAME_RE.match(ns_name):
                logger.warning(
                    "MCP server %r: tool %r → namespaced name %r violates %s; skipping.",
                    name,
                    mcp_name,
                    ns_name,
                    TOOL_NAME_RE.pattern,
                )
                continue

            proxy = MCPProxyTool(
                server_name=name,
                mcp_tool_name=mcp_name,
                namespaced_name=ns_name,
                description=description,
                input_schema=input_schema,
                transport=transport,
                workspace=self._workspace,
                permission_manager=self._permission_manager,
            )
            self._tool_registry.register_instance(proxy, tier="mcp")
            if self._tool_registry.get(ns_name) is None:
                logger.warning(
                    "MCP server %r: tool %r was not registered; skipping from "
                    "server tool list.",
                    name,
                    mcp_name,
                )
                continue
            tool_names.append(mcp_name)

        self._server_tools[name] = tool_names

        # Apply persisted state loaded from mcp.yaml via session overrides so
        # we don't rewrite the loaded mcp.yaml state here.
        if cfg.disabled:
            for ns_name in self._namespaced_tools(name):
                self._tool_registry.disable_session(ns_name)
        if not cfg.allowed:
            for ns_name in self._namespaced_tools(name):
                self._tool_registry.disallow_session(ns_name)
        for tool_name, tool_cfg in cfg.tool_overrides.items():
            if tool_name not in tool_names:
                continue  # tool was skipped (e.g. name too long)
            ns_name = self._ns(name, tool_name)
            if tool_cfg.get("disabled", False):
                self._tool_registry.disable_session(ns_name)
            if not tool_cfg.get("allowed", True):
                self._tool_registry.disallow_session(ns_name)

        self._statuses[name] = ServerStatus(
            name=name,
            connected=True,
            error=None,
            tool_count=len(tool_names),
            tools=list(tool_names),
        )
        logger.info(
            "MCP server %r: connected, %d tool(s) registered.", name, len(tool_names)
        )

    def _build_transport(self, cfg: MCPServerConfig) -> _MCPTransport:
        """Instantiate the appropriate transport or raise :class:`MCPError`."""
        if cfg.transport == "sse":
            return self._build_sse_transport(cfg)
        return self._build_stdio_transport(cfg)

    def _build_sse_transport(self, cfg: MCPServerConfig) -> _SSETransport:
        if not cfg.url:
            raise MCPError("SSE transport requires 'url'")
        headers: dict[str, str] = {}
        if cfg.api_key_env:
            key_value = os.environ.get(cfg.api_key_env)
            if key_value is None:
                raise MCPError(
                    f"env var {cfg.api_key_env!r} is not set; "
                    "configure it or remove 'api_key_env' from mcp.yaml"
                )
            header_name = cfg.api_key_header or "Authorization"
            headers[header_name] = f"{cfg.api_key_prefix}{key_value}"
        return _SSETransport(cfg.url, headers)

    def _build_stdio_transport(self, cfg: MCPServerConfig) -> _StdioTransport:
        if not cfg.command:
            raise MCPError("stdio transport requires 'command'")
        return _StdioTransport(cfg.command, cfg.args)

    # ------------------------------------------------------------------
    # Status / info
    # ------------------------------------------------------------------

    def status(self) -> list[ServerStatus]:
        """Return connection status for all configured servers."""
        return list(self._statuses.values())

    def get_server_tools(self, server_name: str) -> list[str]:
        """Return original (un-namespaced) tool names for *server_name*."""
        return list(self._server_tools.get(server_name, []))

    def server_names(self) -> list[str]:
        """Return all configured server names."""
        return list(self._statuses.keys())

    # ------------------------------------------------------------------
    # Enable / disable / allow / disallow — server level
    # ------------------------------------------------------------------

    def enable_server(self, server_name: str, *, persist: bool = False) -> None:
        for ns_name in self._namespaced_tools(server_name):
            self._tool_registry.enable_session(ns_name)
        if persist:
            self._persist_server_setting(server_name, "disabled", False)

    def disable_server(self, server_name: str, *, persist: bool = False) -> None:
        for ns_name in self._namespaced_tools(server_name):
            self._tool_registry.disable_session(ns_name)
        if persist:
            self._persist_server_setting(server_name, "disabled", True)

    def allow_server(self, server_name: str, *, persist: bool = False) -> None:
        for ns_name in self._namespaced_tools(server_name):
            self._tool_registry.allow_session(ns_name)
        if persist:
            self._persist_server_setting(server_name, "allowed", True)

    def disallow_server(self, server_name: str, *, persist: bool = False) -> None:
        for ns_name in self._namespaced_tools(server_name):
            self._tool_registry.disallow_session(ns_name)
        if persist:
            self._persist_server_setting(server_name, "allowed", False)

    # ------------------------------------------------------------------
    # Enable / disable / allow / disallow — tool level
    # ------------------------------------------------------------------

    def enable_tool(
        self, server_name: str, tool_name: str, *, persist: bool = False
    ) -> None:
        ns = self._ns(server_name, tool_name)
        self._tool_registry.enable_session(ns)
        if persist:
            self._persist_tool_setting(server_name, tool_name, "disabled", False)

    def disable_tool(
        self, server_name: str, tool_name: str, *, persist: bool = False
    ) -> None:
        ns = self._ns(server_name, tool_name)
        self._tool_registry.disable_session(ns)
        if persist:
            self._persist_tool_setting(server_name, tool_name, "disabled", True)

    def allow_tool(
        self, server_name: str, tool_name: str, *, persist: bool = False
    ) -> None:
        ns = self._ns(server_name, tool_name)
        self._tool_registry.allow_session(ns)
        if persist:
            self._persist_tool_setting(server_name, tool_name, "allowed", True)

    def disallow_tool(
        self, server_name: str, tool_name: str, *, persist: bool = False
    ) -> None:
        ns = self._ns(server_name, tool_name)
        self._tool_registry.disallow_session(ns)
        if persist:
            self._persist_tool_setting(server_name, tool_name, "allowed", False)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close_all(self) -> None:
        """Close all transport connections (called on REPL exit)."""
        for name, transport in self._transports.items():
            try:
                transport.close()
            except Exception as exc:
                logger.debug("Error closing MCP transport for %r: %s", name, exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ns(server_name: str, tool_name: str) -> str:
        return f"{server_name}{_NS_SEP}{tool_name}"

    def _namespaced_tools(self, server_name: str) -> list[str]:
        return [
            self._ns(server_name, t) for t in self._server_tools.get(server_name, [])
        ]

    def _project_mcp_path(self) -> Path | None:
        return self._project_config_path

    def _load_project_mcp_yaml(self, *, raise_on_error: bool = False) -> dict:
        """Load the project mcp.yaml.

        When *raise_on_error* is ``True`` (used by the persist path) an
        :class:`MCPError` is raised if the file exists but cannot be parsed,
        so that a corrupt file is never silently overwritten with empty content.
        When ``False`` (default), parse errors fall back to an empty dict.
        """
        path = self._project_mcp_path()
        if path is None or not path.is_file():
            return {"servers": {}}
        try:
            parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                if raise_on_error:
                    raise MCPError(
                        f"Project mcp.yaml ({path}) must contain a YAML mapping, "
                        f"got {type(parsed).__name__}. "
                        "Fix or delete the file before using --persist."
                    )
                logger.warning(
                    "Project mcp.yaml %s: expected mapping, got %s; ignoring.",
                    path,
                    type(parsed).__name__,
                )
                return {"servers": {}}
            return parsed
        except (OSError, yaml.YAMLError) as exc:
            if raise_on_error:
                raise MCPError(
                    f"Cannot parse project mcp.yaml ({path}): {exc}. "
                    "Fix or delete the file before using --persist."
                ) from exc
            logger.warning("Could not parse project mcp.yaml %s: %s", path, exc)
            return {"servers": {}}

    def _save_project_mcp_yaml(self, data: dict) -> None:
        path = self._project_mcp_path()
        if path is None:
            logger.warning("No project mcp.yaml path configured; cannot persist.")
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise MCPError(
                f"Cannot save project mcp.yaml ({path}): {exc}. "
                "Check that the workspace is writable and that you have "
                "permission to create or modify this file before using --persist."
            ) from exc

    def _persist_server_setting(
        self, server_name: str, key: str, value: object
    ) -> None:
        data = self._load_project_mcp_yaml(raise_on_error=True)
        servers = data.setdefault("servers", {})
        if not isinstance(servers, dict):
            raise MCPError(
                "'servers' in project mcp.yaml must be an object/mapping. "
                "Fix the file before using --persist."
            )
        # If the server has no project entry yet, seed connection fields from
        # the merged config so a state-only project entry doesn't shadow and
        # break the global connection definition on the next run.
        if server_name not in servers:
            seed = {
                k: v
                for k, v in self._raw_merged.get(server_name, {}).items()
                if k
                in (
                    "transport",
                    "command",
                    "args",
                    "url",
                    "api_key_env",
                    "api_key_header",
                    "api_key_prefix",
                )
            }
            servers[server_name] = seed
        entry = servers.get(server_name)
        if not isinstance(entry, dict):
            raise MCPError(
                f"Server entry '{server_name}' in project mcp.yaml must be an "
                "object/mapping. Fix the file before using --persist."
            )
        entry[key] = value
        self._save_project_mcp_yaml(data)

    def _persist_tool_setting(
        self, server_name: str, tool_name: str, key: str, value: object
    ) -> None:
        data = self._load_project_mcp_yaml(raise_on_error=True)
        servers = data.setdefault("servers", {})
        if not isinstance(servers, dict):
            raise MCPError(
                "'servers' in project mcp.yaml must be an object/mapping. "
                "Fix the file before using --persist."
            )
        if server_name not in servers:
            seed = {
                k: v
                for k, v in self._raw_merged.get(server_name, {}).items()
                if k
                in (
                    "transport",
                    "command",
                    "args",
                    "url",
                    "api_key_env",
                    "api_key_header",
                    "api_key_prefix",
                )
            }
            servers[server_name] = seed
        entry = servers[server_name]
        if not isinstance(entry, dict):
            raise MCPError(
                f"Server entry '{server_name}' in project mcp.yaml must be an "
                "object/mapping. Fix the file before using --persist."
            )
        tools_cfg = entry.setdefault("tools", {})
        if not isinstance(tools_cfg, dict):
            raise MCPError(
                f"'tools' for server '{server_name}' in project mcp.yaml must be "
                "an object/mapping. Fix the file before using --persist."
            )
        tool_entry = tools_cfg.get(tool_name)
        if tool_entry is None:
            tool_entry = {}
            tools_cfg[tool_name] = tool_entry
        elif not isinstance(tool_entry, dict):
            raise MCPError(
                f"Tool entry '{tool_name}' for server '{server_name}' in "
                "project mcp.yaml must be an object/mapping. Fix the file "
                "before using --persist."
            )
        tool_entry[key] = value
        self._save_project_mcp_yaml(data)
