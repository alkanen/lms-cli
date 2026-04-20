"""
Unit tests for MCPManager, MCPProxyTool, and transport helpers.

Transport calls (HTTP, subprocess) are always mocked — no real network or
process activity occurs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ai_cli.core.mcp_manager import (
    MCPError,
    MCPManager,
    MCPProxyTool,
    _build_openai_schema,
    _client_info,
    _input_schema_to_arguments,
    _mcp_result_to_canonical,
    _MCPToolSchema,
    _SSETransport,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace():
    ws = MagicMock()
    ws.ai_cli_dir = Path("/fake/.ai-cli")
    return ws


@pytest.fixture()
def permission_manager():
    return MagicMock()


@pytest.fixture()
def tool_registry():
    tr = MagicMock()
    # Track tools registered via register_instance so that get() returns
    # non-None for them (needed by _connect_server's post-registration check).
    _registered: dict[str, object] = {}
    tr.register_instance.side_effect = lambda proxy, **kw: _registered.__setitem__(
        proxy.name, proxy
    )
    tr.get.side_effect = lambda name: _registered.get(name)
    return tr


@pytest.fixture()
def global_mcp_path(tmp_path):
    return tmp_path / "global" / "mcp.yaml"


@pytest.fixture()
def project_mcp_path(tmp_path):
    return tmp_path / "project" / ".ai-cli" / "mcp.yaml"


def _make_manager(
    global_path,
    project_path,
    tool_registry,
    workspace,
    permission_manager,
):
    return MCPManager(
        global_config_path=global_path,
        project_config_path=project_path,
        tool_registry=tool_registry,
        workspace=workspace,
        permission_manager=permission_manager,
    )


# ---------------------------------------------------------------------------
# _client_info
# ---------------------------------------------------------------------------


def test_client_info_reports_installed_package_version():
    """clientInfo must advertise the real installed ai-cli version, not a placeholder."""
    info = _client_info()
    assert info["name"] == "ai-cli"
    # Version must be a non-empty string (either the real package version or
    # the documented "unknown" fallback — never the hard-coded "1.0").
    assert isinstance(info["version"], str)
    assert info["version"]
    assert info["version"] != "1.0"


def test_client_info_matches_pyproject_version():
    """The reported version should match the installed package metadata."""
    from importlib.metadata import version

    assert _client_info()["version"] == version("ai-cli")


# ---------------------------------------------------------------------------
# _input_schema_to_arguments
# ---------------------------------------------------------------------------


def test_input_schema_to_arguments_basic():
    schema = {
        "type": "object",
        "properties": {
            "libraryName": {"type": "string", "description": "The library."},
            "tokens": {"type": "number"},
        },
        "required": ["libraryName"],
    }
    args = _input_schema_to_arguments(schema)
    assert len(args) == 2
    by_name = {a.name: a for a in args}
    assert by_name["libraryName"].required is True
    assert by_name["libraryName"].argument_type == "string"
    assert by_name["libraryName"].description == "The library."
    assert by_name["tokens"].required is False
    assert by_name["tokens"].argument_type == "number"


def test_input_schema_to_arguments_unknown_type_defaults_to_string():
    schema = {
        "properties": {"x": {"type": "null"}},
        "required": [],
    }
    args = _input_schema_to_arguments(schema)
    assert args[0].argument_type == "string"


def test_input_schema_to_arguments_empty():
    args = _input_schema_to_arguments({})
    assert args == []


def test_input_schema_to_arguments_array_and_object():
    schema = {
        "properties": {
            "tags": {"type": "array"},
            "meta": {"type": "object"},
        },
    }
    args = _input_schema_to_arguments(schema)
    by_name = {a.name: a for a in args}
    assert by_name["tags"].argument_type == "array"
    assert by_name["meta"].argument_type == "object"


def test_input_schema_to_arguments_non_dict_properties():
    """Non-dict 'properties' should be treated as empty."""
    args = _input_schema_to_arguments({"properties": ["bad"]})
    assert args == []


def test_input_schema_to_arguments_non_string_property_keys_skipped():
    """Non-string property keys must not flow into ToolArgument.name."""
    schema = {
        "properties": {
            "good": {"type": "string"},
            42: {"type": "integer"},  # non-string key
            None: {"type": "boolean"},  # non-string key
        },
        "required": ["good"],
    }
    args = _input_schema_to_arguments(schema)
    names = [a.name for a in args]
    assert names == ["good"]
    assert all(isinstance(a.name, str) for a in args)


def test_input_schema_to_arguments_non_list_required():
    """Non-list 'required' should be treated as empty set."""
    schema = {
        "properties": {"foo": {"type": "string"}},
        "required": "foo",  # string, not list
    }
    args = _input_schema_to_arguments(schema)
    assert len(args) == 1
    assert args[0].required is False


def test_input_schema_to_arguments_required_with_non_string_entries():
    """Non-string entries in 'required' should be filtered out."""
    schema = {
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a", 42],
    }
    args = _input_schema_to_arguments(schema)
    by_name = {a.name: a for a in args}
    assert by_name["a"].required is True
    assert by_name["b"].required is False


# ---------------------------------------------------------------------------
# _build_openai_schema
# ---------------------------------------------------------------------------


def test_build_openai_schema_structure():
    input_schema = {
        "properties": {"q": {"type": "string", "description": "Query"}},
        "required": ["q"],
    }
    result = _build_openai_schema("myserver__mytool", "Does something.", input_schema)
    assert result["type"] == "function"
    fn = result["function"]
    assert fn["name"] == "myserver__mytool"
    assert fn["description"] == "Does something."
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "q" in params["properties"]
    assert params["required"] == ["q"]


def test_build_openai_schema_no_required():
    input_schema = {"properties": {"x": {"type": "integer"}}}
    result = _build_openai_schema("s__t", "desc", input_schema)
    assert "required" not in result["function"]["parameters"]


def test_build_openai_schema_non_dict_properties():
    """Non-dict properties should be coerced to empty dict."""
    result = _build_openai_schema("s__t", "desc", {"properties": "bad"})
    assert result["function"]["parameters"]["properties"] == {}


def test_build_openai_schema_filters_malformed_property_entries():
    """Property entries that are not dicts, or keys that are not strings, are dropped."""
    result = _build_openai_schema(
        "s__t",
        "desc",
        {
            "properties": {
                "good": {"type": "string"},
                "bad_value": "not-a-dict",
                "also_bad": 42,
                7: {"type": "integer"},  # non-string key
            },
            "required": ["good", "bad_value"],
        },
    )
    props = result["function"]["parameters"]["properties"]
    assert props == {"good": {"type": "string"}}
    # Required list must also drop entries that were sanitized away.
    assert result["function"]["parameters"]["required"] == ["good"]


def test_build_openai_schema_non_list_required():
    """Non-list required should be omitted."""
    result = _build_openai_schema(
        "s__t", "desc", {"properties": {"a": {"type": "string"}}, "required": "a"}
    )
    assert "required" not in result["function"]["parameters"]


def test_build_openai_schema_required_filters_non_strings():
    """Non-string entries in required should be filtered out."""
    result = _build_openai_schema(
        "s__t",
        "desc",
        {"properties": {"a": {"type": "string"}}, "required": ["a", 42]},
    )
    assert result["function"]["parameters"]["required"] == ["a"]


def test_build_openai_schema_required_filters_unknown_keys():
    """Required entries not present in properties should be filtered out."""
    result = _build_openai_schema(
        "s__t",
        "desc",
        {"properties": {"a": {"type": "string"}}, "required": ["a", "missing"]},
    )
    assert result["function"]["parameters"]["required"] == ["a"]


def test_build_openai_schema_required_all_unknown_omitted():
    """If all required entries are unknown, 'required' key should be omitted."""
    result = _build_openai_schema(
        "s__t",
        "desc",
        {"properties": {"a": {"type": "string"}}, "required": ["x", "y"]},
    )
    assert "required" not in result["function"]["parameters"]


# ---------------------------------------------------------------------------
# _SSETransport._stream_sse
# ---------------------------------------------------------------------------


def test_stream_sse_returns_matching_event():
    """_stream_sse returns the first event matching the request ID."""
    lines = [
        b'data: {"jsonrpc":"2.0","method":"log","params":{}}\n',
        b"\n",
        b'data: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n',
        b"\n",
    ]
    result = _SSETransport._stream_sse(iter(lines), req_id=1)
    assert result["id"] == 1
    assert result["result"] == {"tools": []}


def test_stream_sse_skips_non_matching_ids():
    """_stream_sse ignores events with non-matching IDs."""
    lines = [
        b'data: {"jsonrpc":"2.0","id":99,"result":{}}\n',
        b"\n",
        b'data: {"jsonrpc":"2.0","id":2,"result":{"ok":true}}\n',
        b"\n",
    ]
    result = _SSETransport._stream_sse(iter(lines), req_id=2)
    assert result["id"] == 2


def test_stream_sse_raises_on_no_match():
    """_stream_sse raises MCPError if no matching event found."""
    lines = [
        b'data: {"jsonrpc":"2.0","id":99,"result":{}}\n',
        b"\n",
    ]
    with pytest.raises(MCPError, match="No valid JSON-RPC"):
        _SSETransport._stream_sse(iter(lines), req_id=1)


def test_stream_sse_trailing_data_rejects_wrong_req_id():
    """Trailing unterminated data with a non-matching id must not be returned."""
    # Last event has no trailing blank line — it would hit the end-of-stream
    # fallback.  Its id is 99 but the caller asked for 1, so the fallback must
    # reject it and raise MCPError instead of returning a bogus response.
    lines = [
        b'data: {"jsonrpc":"2.0","id":99,"result":{"wrong":true}}\n',
    ]
    with pytest.raises(MCPError, match="No valid JSON-RPC"):
        _SSETransport._stream_sse(iter(lines), req_id=1)


def test_stream_sse_trailing_data_accepts_matching_req_id():
    """Trailing unterminated data with a matching id is returned."""
    lines = [
        b'data: {"jsonrpc":"2.0","id":7,"result":{"ok":true}}\n',
    ]
    result = _SSETransport._stream_sse(iter(lines), req_id=7)
    assert result["id"] == 7


def test_stream_sse_deadline_enforced_against_chatty_server():
    """Non-matching events with a past deadline must raise MCPError."""
    import time

    def chatty_lines():
        # Infinite stream of non-matching keepalives.  Without a deadline,
        # this would loop forever.
        while True:
            yield b'data: {"jsonrpc":"2.0","method":"keepalive"}\n'
            yield b"\n"

    # Deadline already in the past — the very first check must trip.
    past_deadline = time.monotonic() - 1.0
    with pytest.raises(MCPError, match="Timed out"):
        _SSETransport._stream_sse(chatty_lines(), req_id=1, deadline=past_deadline)


# ---------------------------------------------------------------------------
# _mcp_result_to_canonical
# ---------------------------------------------------------------------------


def test_mcp_result_to_canonical_text():
    result = {"content": [{"type": "text", "text": "hello"}]}
    canon = _mcp_result_to_canonical(result)
    assert canon["status"] == "success"
    assert canon["data"]["text"] == "hello"


def test_mcp_result_to_canonical_multiple_text_blocks():
    result = {
        "content": [
            {"type": "text", "text": "part 1"},
            {"type": "text", "text": "part 2"},
        ]
    }
    canon = _mcp_result_to_canonical(result)
    assert "part 1" in canon["data"]["text"]
    assert "part 2" in canon["data"]["text"]


def test_mcp_result_to_canonical_error_block():
    result = {"content": [{"type": "error", "text": "server failed"}]}
    canon = _mcp_result_to_canonical(result)
    assert canon["status"] == "error"
    assert canon["error"] == "mcp_error"
    assert "server failed" in canon["message"]


def test_mcp_result_to_canonical_empty():
    canon = _mcp_result_to_canonical({})
    assert canon["status"] == "success"
    assert canon["data"]["text"] == ""


def test_mcp_result_to_canonical_non_string_text_ignored():
    """Non-string text values in content blocks should be ignored."""
    result = {"content": [{"type": "text", "text": 42}]}
    canon = _mcp_result_to_canonical(result)
    assert canon["status"] == "success"
    assert canon["data"]["text"] == ""


def test_mcp_result_to_canonical_non_string_error_text():
    """Non-string error text should fall back to default message."""
    result = {"content": [{"type": "error", "text": None}]}
    canon = _mcp_result_to_canonical(result)
    assert canon["status"] == "error"
    assert "Unknown MCP error" in canon["message"]


# ---------------------------------------------------------------------------
# _SSETransport — parse_sse
# ---------------------------------------------------------------------------


def test_parse_sse_single_event():
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
    raw = f"data: {json.dumps(payload)}\n\n".encode()
    result = _SSETransport._parse_sse(raw)
    assert result == payload


def test_parse_sse_with_event_prefix():
    payload = {"jsonrpc": "2.0", "id": 1, "result": {}}
    raw = f"event: message\ndata: {json.dumps(payload)}\n\n".encode()
    result = _SSETransport._parse_sse(raw)
    assert result == payload


def test_parse_sse_trailing_data_no_blank_line():
    """Accumulated data with no trailing blank line is still returned."""
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"x": 1}}
    raw = f"data: {json.dumps(payload)}".encode()
    result = _SSETransport._parse_sse(raw)
    assert result == payload


def test_parse_sse_no_data_raises():
    with pytest.raises(MCPError, match="No valid JSON-RPC"):
        _SSETransport._parse_sse(b"event: ping\n\n")


# ---------------------------------------------------------------------------
# _SSETransport — extract_result
# ---------------------------------------------------------------------------


def test_extract_result_success():
    resp = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
    assert _SSETransport._extract_result(resp, "tools/list") == {"tools": []}


def test_extract_result_error_raises():
    resp = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "fail"}}
    with pytest.raises(MCPError, match="fail"):
        _SSETransport._extract_result(resp, "tools/list")


def test_extract_result_none_raises():
    with pytest.raises(MCPError):
        _SSETransport._extract_result(None, "tools/list")


def test_extract_result_missing_result_raises():
    with pytest.raises(MCPError):
        _SSETransport._extract_result({"jsonrpc": "2.0", "id": 1}, "tools/list")


def test_extract_result_non_dict_response_raises():
    """Non-dict response should raise MCPError."""
    with pytest.raises(MCPError, match="malformed response"):
        _SSETransport._extract_result([1, 2, 3], "tools/list")


def test_extract_result_non_dict_error_raises():
    """Non-dict error field should raise MCPError."""
    with pytest.raises(MCPError, match="malformed error"):
        _SSETransport._extract_result(
            {"jsonrpc": "2.0", "id": 1, "error": "bad"}, "tools/list"
        )


def test_extract_result_non_dict_result_raises():
    """Non-dict result field should raise MCPError."""
    with pytest.raises(MCPError, match="malformed result"):
        _SSETransport._extract_result(
            {"jsonrpc": "2.0", "id": 1, "result": "string"}, "tools/list"
        )


# ---------------------------------------------------------------------------
# _input_schema_to_arguments — non-string description
# ---------------------------------------------------------------------------


def test_input_schema_to_arguments_non_string_description():
    """Non-string description should be coerced to empty string."""
    schema = {
        "properties": {"a": {"type": "string", "description": 42}},
    }
    args = _input_schema_to_arguments(schema)
    assert len(args) == 1
    assert args[0].description == ""


# ---------------------------------------------------------------------------
# MCPProxyTool
# ---------------------------------------------------------------------------


def _make_proxy_tool(workspace, permission_manager, transport=None):
    if transport is None:
        transport = MagicMock()
    input_schema = {
        "type": "object",
        "properties": {"libraryName": {"type": "string", "description": "name"}},
        "required": ["libraryName"],
    }
    return MCPProxyTool(
        server_name="context7",
        mcp_tool_name="resolve-library-id",
        namespaced_name="context7__resolve-library-id",
        description="Resolve a library.",
        input_schema=input_schema,
        transport=transport,
        workspace=workspace,
        permission_manager=permission_manager,
    )


def test_proxy_tool_name(workspace, permission_manager):
    tool = _make_proxy_tool(workspace, permission_manager)
    assert tool.name == "context7__resolve-library-id"


def test_proxy_tool_description(workspace, permission_manager):
    tool = _make_proxy_tool(workspace, permission_manager)
    assert tool.description == "Resolve a library."


def test_proxy_tool_definition_returns_mcp_schema(workspace, permission_manager):
    tool = _make_proxy_tool(workspace, permission_manager)
    schema = tool.definition()
    assert isinstance(schema, _MCPToolSchema)
    d = schema.schema()
    assert d["function"]["name"] == "context7__resolve-library-id"
    assert "libraryName" in d["function"]["parameters"]["properties"]


def test_proxy_tool_definition_arguments_for_validation(workspace, permission_manager):
    tool = _make_proxy_tool(workspace, permission_manager)
    args = tool.definition().arguments
    assert len(args) == 1
    assert args[0].name == "libraryName"
    assert args[0].required is True


def test_proxy_tool_execute_delegates_to_transport(workspace, permission_manager):
    transport = MagicMock()
    transport.call_tool.return_value = {"status": "success", "data": {"text": "result"}}
    tool = _make_proxy_tool(workspace, permission_manager, transport=transport)
    result = tool.execute(libraryName="react")
    transport.call_tool.assert_called_once_with(
        "resolve-library-id", {"libraryName": "react"}
    )
    assert result["status"] == "success"


def test_proxy_tool_permission_not_required(workspace, permission_manager):
    tool = _make_proxy_tool(workspace, permission_manager)
    assert tool.permission_required is False


# ---------------------------------------------------------------------------
# MCPManager — config loading
# ---------------------------------------------------------------------------


def test_load_configs_global_only(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "context7": {
                        "transport": "sse",
                        "url": "https://mcp.context7.com/mcp",
                    }
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    configs = mgr._load_configs()
    assert len(configs) == 1
    assert configs[0].name == "context7"
    assert configs[0].transport == "sse"


def test_load_configs_project_overrides_global(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "context7": {
                        "transport": "sse",
                        "url": "https://global.example.com/mcp",
                    },
                    "only-global": {
                        "transport": "sse",
                        "url": "https://global.example.com",
                    },
                }
            }
        )
    )
    project_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "context7": {
                        "transport": "sse",
                        "url": "https://project.example.com/mcp",
                    },
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    configs = mgr._load_configs()
    by_name = {c.name: c for c in configs}
    assert by_name["context7"].url == "https://project.example.com/mcp"
    assert "only-global" in by_name


def test_load_configs_missing_files_returns_empty(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    configs = mgr._load_configs()
    assert configs == []


def test_load_configs_unknown_transport_skipped(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"bad": {"transport": "websocket", "url": "ws://example.com"}}}
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    assert mgr._load_configs() == []


def test_load_configs_invalid_yaml_logged(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text("not: valid: yaml: {{{{")
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    assert mgr._load_configs() == []


def test_load_configs_non_object_root_ignored(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    """A YAML list at the root of mcp.yaml should be safely ignored."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text("[]")
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    assert mgr._load_configs() == []


def test_load_configs_non_dict_servers_ignored(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    """'servers' as a non-dict value should be safely ignored."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(yaml.dump({"servers": "oops"}))
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    assert mgr._load_configs() == []


def test_load_configs_falsy_non_dict_servers_warns(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    """Falsy non-dict 'servers' (e.g. []) should warn, not silently coerce to {}."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(yaml.dump({"servers": []}))
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    assert mgr._load_configs() == []


def test_load_configs_no_servers_key(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    """Config with no 'servers' key should produce no configs."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(yaml.dump({"other": "stuff"}))
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    assert mgr._load_configs() == []


def test_load_configs_non_dict_server_entry_skipped(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    """A non-dict server entry should be skipped; valid siblings still load."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "bad": "not-a-dict",
                    "good": {"transport": "sse", "url": "https://example.com/mcp"},
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    configs = mgr._load_configs()
    assert len(configs) == 1
    assert configs[0].name == "good"


def test_load_configs_invalid_server_name_skipped(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    """Server names with whitespace or disallowed chars are skipped."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "has space": {"transport": "sse", "url": "https://a.com"},
                    "-leading-dash": {"transport": "sse", "url": "https://b.com"},
                    "good_name": {"transport": "sse", "url": "https://c.com"},
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    configs = mgr._load_configs()
    assert len(configs) == 1
    assert configs[0].name == "good_name"


# ---------------------------------------------------------------------------
# MCPManager — connect_all
# ---------------------------------------------------------------------------


def _fake_transport(tools):
    """Return a mock transport that returns *tools* from list_tools()."""
    t = MagicMock()
    t.initialize.return_value = {}
    t.list_tools.return_value = tools
    return t


def test_connect_all_registers_tools(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "context7": {
                        "transport": "sse",
                        "url": "https://mcp.context7.com/mcp",
                    }
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    mock_transport = _fake_transport(
        [
            {
                "name": "resolve-library-id",
                "description": "Resolve a library.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"libraryName": {"type": "string"}},
                    "required": ["libraryName"],
                },
            }
        ]
    )
    with patch.object(mgr, "_build_transport", return_value=mock_transport):
        mgr.connect_all()

    assert tool_registry.register_instance.call_count == 1
    registered = tool_registry.register_instance.call_args[0][0]
    assert isinstance(registered, MCPProxyTool)
    assert registered.name == "context7__resolve-library-id"


def test_connect_all_missing_api_key_env_skips_server(
    global_mcp_path, tool_registry, workspace, permission_manager, monkeypatch
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "context7": {
                        "transport": "sse",
                        "url": "https://mcp.context7.com/mcp",
                        "api_key_env": "CONTEXT7_API_KEY",
                    }
                }
            }
        )
    )
    monkeypatch.delenv("CONTEXT7_API_KEY", raising=False)
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    mgr.connect_all()

    tool_registry.register_instance.assert_not_called()
    statuses = mgr.status()
    assert len(statuses) == 1
    assert statuses[0].connected is False
    assert "CONTEXT7_API_KEY" in statuses[0].error


def test_connect_all_transport_error_skips_server(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "myserver": {"transport": "sse", "url": "https://example.com/mcp"}
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    with patch.object(mgr, "_build_transport", side_effect=MCPError("refused")):
        mgr.connect_all()

    tool_registry.register_instance.assert_not_called()
    assert mgr.status()[0].connected is False


def test_connect_all_initialize_error_skips_server(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"s": {"transport": "sse", "url": "https://example.com"}}}
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    bad_transport = MagicMock()
    bad_transport.initialize.side_effect = MCPError("timeout")
    with patch.object(mgr, "_build_transport", return_value=bad_transport):
        mgr.connect_all()

    assert mgr.status()[0].connected is False
    bad_transport.close.assert_called_once()


def test_connect_all_tool_name_too_long_skipped(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    long_name = "a" * 60  # context7__<60 chars> = 70 chars > 64
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "context7": {
                        "transport": "sse",
                        "url": "https://mcp.context7.com/mcp",
                    }
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": long_name, "description": "too long", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    tool_registry.register_instance.assert_not_called()


def test_connect_all_api_key_sent_in_header(
    global_mcp_path, tool_registry, workspace, permission_manager, monkeypatch
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "context7": {
                        "transport": "sse",
                        "url": "https://mcp.context7.com/mcp",
                        "api_key_env": "CONTEXT7_API_KEY",
                        "api_key_header": "CONTEXT7_API_KEY",
                        "api_key_prefix": "",
                    }
                }
            }
        )
    )
    monkeypatch.setenv("CONTEXT7_API_KEY", "my-secret-key")
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )

    captured: dict = {}

    def fake_build(cfg):
        t = _fake_transport([])
        captured["transport"] = t
        # Verify the correct header was going to be set.
        transport = mgr._build_sse_transport(cfg)
        captured["headers"] = transport._headers
        return _fake_transport([])

    with patch.object(mgr, "_build_transport", side_effect=fake_build):
        mgr.connect_all()

    assert captured["headers"].get("CONTEXT7_API_KEY") == "my-secret-key"


def test_connect_all_bearer_prefix(
    global_mcp_path, tool_registry, workspace, permission_manager, monkeypatch
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "myapi": {
                        "transport": "sse",
                        "url": "https://api.example.com/mcp",
                        "api_key_env": "MY_API_KEY",
                        "api_key_header": "Authorization",
                        "api_key_prefix": "Bearer ",
                    }
                }
            }
        )
    )
    monkeypatch.setenv("MY_API_KEY", "tok123")
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    cfg = mgr._load_configs()[0]
    transport = mgr._build_sse_transport(cfg)
    assert transport._headers["Authorization"] == "Bearer tok123"


# ---------------------------------------------------------------------------
# MCPManager — status / get_server_tools
# ---------------------------------------------------------------------------


def test_status_after_connect(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "my-tool", "description": "desc", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    statuses = mgr.status()
    assert len(statuses) == 1
    s = statuses[0]
    assert s.name == "srv"
    assert s.connected is True
    assert s.tool_count == 1
    assert s.tools == ["my-tool"]


def test_get_server_tools(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [
            {"name": "tool-a", "description": "", "inputSchema": {}},
            {"name": "tool-b", "description": "", "inputSchema": {}},
        ]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    assert set(mgr.get_server_tools("srv")) == {"tool-a", "tool-b"}
    assert mgr.get_server_tools("nonexistent") == []


# ---------------------------------------------------------------------------
# MCPManager — enable / disable / allow / disallow (session)
# ---------------------------------------------------------------------------


def _connected_manager(global_mcp_path, tool_registry, workspace, permission_manager):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [
            {"name": "tool-a", "description": "", "inputSchema": {}},
            {"name": "tool-b", "description": "", "inputSchema": {}},
        ]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()
    return mgr


def test_disable_server_session(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    mgr = _connected_manager(
        global_mcp_path, tool_registry, workspace, permission_manager
    )
    mgr.disable_server("srv")
    calls = [str(c) for c in tool_registry.disable_session.call_args_list]
    assert any("tool-a" in c for c in calls)
    assert any("tool-b" in c for c in calls)
    tool_registry.disable.assert_not_called()


def test_enable_server_session(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    mgr = _connected_manager(
        global_mcp_path, tool_registry, workspace, permission_manager
    )
    mgr.enable_server("srv")
    tool_registry.enable_session.assert_called()
    tool_registry.enable.assert_not_called()


def test_disallow_tool_session(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    mgr = _connected_manager(
        global_mcp_path, tool_registry, workspace, permission_manager
    )
    mgr.disallow_tool("srv", "tool-a")
    tool_registry.disallow_session.assert_called_once_with("srv__tool-a")
    tool_registry.disallow.assert_not_called()


def test_allow_tool_session(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    mgr = _connected_manager(
        global_mcp_path, tool_registry, workspace, permission_manager
    )
    mgr.allow_tool("srv", "tool-a")
    tool_registry.allow_session.assert_called_once_with("srv__tool-a")


# ---------------------------------------------------------------------------
# MCPManager — persist path
# ---------------------------------------------------------------------------


def test_disable_server_persist_writes_to_project_mcp(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "tool-a", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    mgr.disable_server("srv", persist=True)

    written = yaml.safe_load(project_mcp_path.read_text())
    assert written["servers"]["srv"]["disabled"] is True
    # Runtime effect via session override only — must not touch mcp.yaml.
    tool_registry.disable_session.assert_called()
    tool_registry.disable.assert_not_called()


def test_disallow_tool_persist_writes_to_project_mcp(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "tool-a", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    mgr.disallow_tool("srv", "tool-a", persist=True)

    written = yaml.safe_load(project_mcp_path.read_text())
    assert written["servers"]["srv"]["tools"]["tool-a"]["allowed"] is False


# ---------------------------------------------------------------------------
# MCPManager — close_all
# ---------------------------------------------------------------------------


def test_close_all_calls_transport_close(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport([])
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    mgr.close_all()
    transport.close.assert_called_once()


def test_close_all_exception_is_silenced(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport([])
    transport.close.side_effect = RuntimeError("bang")
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    mgr.close_all()  # should not raise


# ---------------------------------------------------------------------------
# YAML comment support
# ---------------------------------------------------------------------------


def test_load_configs_accepts_yaml_comments(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        "# Top-level comment\n"
        "servers:\n"
        "  # Inline comment before server\n"
        "  srv:\n"
        "    transport: sse\n"
        "    url: https://example.com/mcp\n"
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    configs = mgr._load_configs()
    assert len(configs) == 1
    assert configs[0].name == "srv"


# ---------------------------------------------------------------------------
# Field-level merging and persisted state
# ---------------------------------------------------------------------------


def test_load_configs_field_level_merge(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """Project state keys merge with global connection fields; neither wins entirely."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "srv": {
                        "transport": "sse",
                        "url": "https://global.example.com/mcp",
                    }
                }
            }
        )
    )
    # Project only has a state field; no connection fields.
    project_mcp_path.write_text(yaml.dump({"servers": {"srv": {"disabled": True}}}))
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    configs = mgr._load_configs()
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.url == "https://global.example.com/mcp"  # from global
    assert cfg.disabled is True  # from project


def test_persisted_state_applied_on_startup(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    """disabled=true in config should call tool_registry.disable() at startup."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "srv": {
                        "transport": "sse",
                        "url": "https://example.com/mcp",
                        "disabled": True,
                    }
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "my-tool", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    # Session override used — must not touch mcp.yaml via persistent mutator.
    tool_registry.disable_session.assert_called()
    tool_registry.disable.assert_not_called()
    disabled_names = [c.args[0] for c in tool_registry.disable_session.call_args_list]
    assert "srv__my-tool" in disabled_names


def test_persisted_tool_state_applied_on_startup(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    """Per-tool allowed=false in config should call tool_registry.disallow()."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "srv": {
                        "transport": "sse",
                        "url": "https://example.com/mcp",
                        "tools": {"my-tool": {"allowed": False}},
                    }
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "my-tool", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    # Session override used — must not touch mcp.yaml via persistent mutator.
    tool_registry.disallow_session.assert_called()
    tool_registry.disallow.assert_not_called()
    disallowed_names = [
        c.args[0] for c in tool_registry.disallow_session.call_args_list
    ]
    assert "srv__my-tool" in disallowed_names


def test_persist_global_only_server_seeds_connection_fields(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """Persisting state for a global-only server includes connection fields in project file."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "srv": {
                        "transport": "sse",
                        "url": "https://example.com/mcp",
                    }
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "tool-a", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    mgr.disable_server("srv", persist=True)

    written = yaml.safe_load(project_mcp_path.read_text())
    srv_entry = written["servers"]["srv"]
    # Connection fields must be present so the project entry is self-contained.
    assert srv_entry["transport"] == "sse"
    assert srv_entry["url"] == "https://example.com/mcp"
    assert srv_entry["disabled"] is True


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_load_configs_invalid_args_type_skipped(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "bad": {
                        "transport": "stdio",
                        "command": "uvx",
                        "args": "not-a-list",
                    }
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    assert mgr._load_configs() == []


def test_load_configs_invalid_command_type_skipped(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "bad": {
                        "transport": "stdio",
                        "command": 42,
                    }
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    assert mgr._load_configs() == []


# ---------------------------------------------------------------------------
# Per-tool deep merge
# ---------------------------------------------------------------------------


def test_tools_deep_merge_combines_keys(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """Same tool in global and project should have fields from both (project wins)."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "srv": {
                        "transport": "sse",
                        "url": "https://example.com/mcp",
                        "tools": {"my-tool": {"allowed": False}},
                    }
                }
            }
        )
    )
    project_mcp_path.write_text(
        yaml.dump({"servers": {"srv": {"tools": {"my-tool": {"disabled": True}}}}})
    )
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    configs = mgr._load_configs()
    assert len(configs) == 1
    tool_cfg = configs[0].tool_overrides.get("my-tool", {})
    assert tool_cfg.get("allowed") is False
    assert tool_cfg.get("disabled") is True


def test_invalid_project_tools_value_warns_and_is_dropped(
    global_mcp_path,
    project_mcp_path,
    tool_registry,
    workspace,
    permission_manager,
    caplog,
):
    """A non-mapping project 'tools' (e.g. []) should warn and not leak into
    the merged config — regardless of whether global has valid tools."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    # Global has a valid tools dict; project tries to replace it with a list.
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "srv": {
                        "transport": "sse",
                        "url": "https://example.com/mcp",
                        "tools": {"my-tool": {"allowed": False}},
                    }
                }
            }
        )
    )
    project_mcp_path.write_text(yaml.dump({"servers": {"srv": {"tools": []}}}))
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    with caplog.at_level("WARNING"):
        configs = mgr._load_configs()
    assert any(
        "project 'tools' must be an object" in rec.message for rec in caplog.records
    )
    assert len(configs) == 1
    # Global tools survive; the invalid project value did not wipe them.
    assert configs[0].tool_overrides.get("my-tool", {}).get("allowed") is False


def test_invalid_project_tools_without_global_tools_is_dropped(
    global_mcp_path,
    project_mcp_path,
    tool_registry,
    workspace,
    permission_manager,
    caplog,
):
    """A non-mapping project 'tools' should warn and be dropped even when
    global has no tools dict of its own (nothing should leak through
    update())."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "srv": {
                        "transport": "sse",
                        "url": "https://example.com/mcp",
                    }
                }
            }
        )
    )
    project_mcp_path.write_text(
        yaml.dump({"servers": {"srv": {"tools": "not-a-dict"}}})
    )
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    with caplog.at_level("WARNING"):
        configs = mgr._load_configs()
    assert any(
        "project 'tools' must be an object" in rec.message for rec in caplog.records
    )
    assert len(configs) == 1
    # Invalid project 'tools' string did not leak into the merged config.
    assert configs[0].tool_overrides == {}


# ---------------------------------------------------------------------------
# Boolean flag validation
# ---------------------------------------------------------------------------


def test_non_bool_disabled_flag_ignored(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "srv": {
                        "transport": "sse",
                        "url": "https://example.com/mcp",
                        "disabled": "false",  # string, not bool
                    }
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    configs = mgr._load_configs()
    assert len(configs) == 1
    assert configs[0].disabled is False  # invalid type ignored; default used


# ---------------------------------------------------------------------------
# Auth field type validation
# ---------------------------------------------------------------------------


def test_non_string_api_key_env_skips_server(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "srv": {
                        "transport": "sse",
                        "url": "https://example.com/mcp",
                        "api_key_env": 123,
                    }
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    assert mgr._load_configs() == []


# ---------------------------------------------------------------------------
# Malformed tool definitions in _connect_server
# ---------------------------------------------------------------------------


def test_non_string_tool_name_skipped(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [
            {"name": 42, "description": "", "inputSchema": {}},  # bad name
            {"name": "good-tool", "description": "", "inputSchema": {}},
        ]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    assert tool_registry.register_instance.call_count == 1
    registered = tool_registry.register_instance.call_args[0][0]
    assert registered.name == "srv__good-tool"


def test_non_dict_input_schema_skipped(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "bad-tool", "description": "", "inputSchema": "not-a-dict"}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    tool_registry.register_instance.assert_not_called()


# ---------------------------------------------------------------------------
# Persist safety — corrupt project mcp.yaml
# ---------------------------------------------------------------------------


def test_persist_raises_on_corrupt_project_mcp(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """--persist must not silently overwrite a corrupt project mcp.yaml."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    project_mcp_path.write_text("not: valid: yaml: !!!")
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "tool-a", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    with pytest.raises(MCPError, match="Cannot parse"):
        mgr.disable_server("srv", persist=True)

    assert "not: valid: yaml" in project_mcp_path.read_text()


def test_persist_raises_mcperror_on_oserror_writing_project_mcp(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """--persist must raise MCPError (not bare OSError) when the file cannot be written."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    project_mcp_path.write_text(yaml.dump({"servers": {}}))
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "tool-a", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    # Simulate a permission error on write_text.
    with (
        patch.object(
            Path, "write_text", side_effect=PermissionError("read-only filesystem")
        ),
        pytest.raises(MCPError, match="Cannot save project mcp.yaml"),
    ):
        mgr.disable_server("srv", persist=True)


# ---------------------------------------------------------------------------
# Per-tool boolean flag validation
# ---------------------------------------------------------------------------


def test_non_bool_tool_disabled_flag_ignored(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    """String 'true' in a per-tool disabled field must be treated as default (False)."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "srv": {
                        "transport": "sse",
                        "url": "https://example.com/mcp",
                        "tools": {
                            "my-tool": {"disabled": "true"},
                        },
                    }
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    configs = mgr._load_configs()
    assert len(configs) == 1
    assert configs[0].tool_overrides["my-tool"]["disabled"] is False


def test_non_bool_tool_allowed_flag_ignored(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    """String 'false' in a per-tool allowed field must be treated as default (True)."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "srv": {
                        "transport": "sse",
                        "url": "https://example.com/mcp",
                        "tools": {
                            "my-tool": {"allowed": "false"},
                        },
                    }
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    configs = mgr._load_configs()
    assert len(configs) == 1
    assert configs[0].tool_overrides["my-tool"]["allowed"] is True


# ---------------------------------------------------------------------------
# _load_project_mcp_yaml root type validation
# ---------------------------------------------------------------------------


def test_load_project_mcp_yaml_non_object_fallback(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """Non-mapping root (e.g. a YAML list) should fall back to empty servers."""
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.write_text("[]")
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    result = mgr._load_project_mcp_yaml()
    assert result == {"servers": {}}


def test_load_project_mcp_yaml_non_object_raises(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """Non-object root must raise MCPError when raise_on_error=True."""
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.write_text("[]")
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    with pytest.raises(MCPError, match="must contain a YAML mapping"):
        mgr._load_project_mcp_yaml(raise_on_error=True)


# ---------------------------------------------------------------------------
# _persist_*_setting — servers key validation
# ---------------------------------------------------------------------------


def test_persist_server_setting_raises_on_non_dict_servers(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """Persist must raise MCPError when 'servers' is not a dict."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    project_mcp_path.write_text(yaml.dump({"servers": "not-a-dict"}))
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "tool-a", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    with pytest.raises(MCPError, match="must be an object/mapping"):
        mgr.disable_server("srv", persist=True)


def test_persist_tool_setting_raises_on_non_dict_servers(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """Persist tool setting must raise MCPError when 'servers' is not a dict."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    project_mcp_path.write_text(yaml.dump({"servers": "not-a-dict"}))
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "tool-a", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    with pytest.raises(MCPError, match="must be an object/mapping"):
        mgr.disable_tool("srv", "tool-a", persist=True)


# ---------------------------------------------------------------------------
# _persist_tool_setting uses raise_on_error=True
# ---------------------------------------------------------------------------


def test_persist_tool_setting_raises_on_corrupt_file(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """_persist_tool_setting must raise MCPError on corrupt mcp.yaml, not overwrite it."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    project_mcp_path.write_text("corrupt: yaml: !!!")
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "tool-a", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    with pytest.raises(MCPError, match="Cannot parse"):
        mgr.disable_tool("srv", "tool-a", persist=True)

    # File should not have been overwritten
    assert "corrupt: yaml" in project_mcp_path.read_text()


# ---------------------------------------------------------------------------
# _persist_*_setting — server entry / tools key validation
# ---------------------------------------------------------------------------


def test_persist_server_setting_raises_on_non_dict_server_entry(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """Persist must raise MCPError when a server entry is not a dict."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    # Start with a valid project file so connect_all succeeds.
    project_mcp_path.write_text(yaml.dump({"servers": {}}))
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "tool-a", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    # Now corrupt the project file before persisting.
    project_mcp_path.write_text(yaml.dump({"servers": {"srv": "not-a-dict"}}))

    with pytest.raises(MCPError, match="Server entry.*must be an object"):
        mgr.disable_server("srv", persist=True)


def test_persist_server_setting_raises_on_null_server_entry(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """Persist must raise MCPError when a server entry is YAML null."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    project_mcp_path.write_text(yaml.dump({"servers": {}}))
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "tool-a", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    project_mcp_path.write_text(yaml.dump({"servers": {"srv": None}}))

    with pytest.raises(MCPError, match="Server entry.*must be an object"):
        mgr.disable_server("srv", persist=True)


def test_persist_tool_setting_raises_on_non_dict_server_entry(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """Persist tool must raise MCPError when a server entry is not a dict."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    project_mcp_path.write_text(yaml.dump({"servers": {}}))
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "tool-a", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    project_mcp_path.write_text(yaml.dump({"servers": {"srv": "not-a-dict"}}))

    with pytest.raises(MCPError, match="Server entry.*must be an object"):
        mgr.disable_tool("srv", "tool-a", persist=True)


def test_persist_tool_setting_raises_on_non_dict_tools_key(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """Persist tool must raise MCPError when 'tools' in a server entry is not a dict."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    project_mcp_path.write_text(yaml.dump({"servers": {}}))
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "tool-a", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    project_mcp_path.write_text(
        yaml.dump({"servers": {"srv": {"transport": "sse", "tools": []}}})
    )

    with pytest.raises(MCPError, match="'tools' for server.*must be an object"):
        mgr.disable_tool("srv", "tool-a", persist=True)


# ---------------------------------------------------------------------------
# _StdioTransport._request — overall deadline
# ---------------------------------------------------------------------------


def test_stdio_request_deadline_not_reset_by_notifications():
    """Non-matching notifications must not extend the deadline indefinitely."""
    import queue
    import time

    from ai_cli.core.mcp_manager import _StdioTransport

    transport = object.__new__(_StdioTransport)
    transport._lock = __import__("threading").Lock()
    transport._next_id = 1
    transport._queue = queue.Queue()

    # Mock the subprocess
    proc = MagicMock()
    proc.poll.return_value = None
    proc.stdin = MagicMock()
    transport._proc = proc

    # Pre-fill queue with non-matching JSON-RPC notifications (no "id" field)
    # — enough to outlast a short timeout.
    for _ in range(200):
        transport._queue.put(
            json.dumps({"jsonrpc": "2.0", "method": "log", "params": {}}).encode()
            + b"\n"
        )

    import ai_cli.core.mcp_manager as _mod

    original_timeout = _mod._STDIO_TIMEOUT
    _mod._STDIO_TIMEOUT = 0.5  # very short for test speed
    try:
        start = time.monotonic()
        with pytest.raises(MCPError, match="No response from server"):
            transport._request("test/method", {"foo": "bar"})
        elapsed = time.monotonic() - start
        # Must complete within a reasonable margin of the timeout, not
        # 200 × per-get-timeout seconds.
        assert elapsed < 3.0, f"Took {elapsed:.1f}s — deadline was not enforced"
    finally:
        _mod._STDIO_TIMEOUT = original_timeout


def test_stdio_request_raises_on_non_dict_response():
    """A valid-JSON but non-object response must raise MCPError, not AttributeError."""
    import queue

    from ai_cli.core.mcp_manager import _StdioTransport

    transport = object.__new__(_StdioTransport)
    transport._lock = __import__("threading").Lock()
    transport._next_id = 1
    transport._queue = queue.Queue()

    proc = MagicMock()
    proc.poll.return_value = None
    proc.stdin = MagicMock()
    transport._proc = proc

    # Server emits a valid JSON array (not an object) on stdout.
    transport._queue.put(b"[1, 2, 3]\n")

    with pytest.raises(MCPError, match="expected object"):
        transport._request("test/method", {})


def test_stdio_transport_init_raises_mcperror_on_missing_command():
    """Popen(FileNotFoundError) must surface as MCPError, not a bare OSError."""
    from ai_cli.core.mcp_manager import _StdioTransport

    with pytest.raises(MCPError, match="Failed to start MCP server command"):
        _StdioTransport("/nonexistent/definitely-not-a-real-binary", [])


def test_connect_all_skips_stdio_server_with_missing_command(
    global_mcp_path, tool_registry, workspace, permission_manager
):
    """An unstartable stdio server should be marked as failed, not crash startup."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {
                "servers": {
                    "bad": {
                        "transport": "stdio",
                        "command": "/nonexistent/definitely-not-a-real-binary",
                    },
                }
            }
        )
    )
    mgr = _make_manager(
        global_mcp_path, None, tool_registry, workspace, permission_manager
    )
    # Should not raise.
    mgr.connect_all()
    statuses = {s.name: s for s in mgr.status()}
    assert "bad" in statuses
    assert statuses["bad"].connected is False
    assert "Failed to start" in (statuses["bad"].error or "")


# ---------------------------------------------------------------------------
# _persist_tool_setting — non-dict tool entry validation
# ---------------------------------------------------------------------------


def test_persist_tool_setting_raises_on_non_dict_tool_entry(
    global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
):
    """Persist must raise MCPError when an existing tool entry is not a dict."""
    global_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    project_mcp_path.parent.mkdir(parents=True, exist_ok=True)
    global_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "url": "https://example.com/mcp"}}}
        )
    )
    project_mcp_path.write_text(yaml.dump({"servers": {}}))
    mgr = _make_manager(
        global_mcp_path, project_mcp_path, tool_registry, workspace, permission_manager
    )
    transport = _fake_transport(
        [{"name": "tool_a", "description": "", "inputSchema": {}}]
    )
    with patch.object(mgr, "_build_transport", return_value=transport):
        mgr.connect_all()

    # Corrupt the tool entry after connect.
    project_mcp_path.write_text(
        yaml.dump(
            {"servers": {"srv": {"transport": "sse", "tools": {"tool_a": "bad"}}}}
        )
    )

    with pytest.raises(MCPError, match="Tool entry.*must be an object"):
        mgr.disable_tool("srv", "tool_a", persist=True)
