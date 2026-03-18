"""Tests for ai_cli.tools.tool_manager.ToolManagerTool."""

from typing import Any
from unittest.mock import MagicMock

from ai_cli.tools.tool_manager import ToolManagerTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(registry: Any = None) -> ToolManagerTool:
    """Return a ToolManagerTool with a mock workspace and permission manager."""
    tool = ToolManagerTool(
        workspace=MagicMock(),
        permission_manager=MagicMock(),
        permission_required=False,
        name=ToolManagerTool.NAME,
        description=ToolManagerTool.DESCRIPTION,
    )
    if registry is not None:
        tool.set_registry(registry)
    return tool


def _make_registry(
    tools: list[dict] | None = None,
    transient_map: dict[str, dict] | None = None,
) -> MagicMock:
    """
    Return a mock ToolRegistry.

    Parameters
    ----------
    tools:
        Value returned by ``list_all()``.
    transient_map:
        Mapping of tool-name → schema returned by ``enable_transient(name)``.
        Unknown names should be absent from the map (``enable_transient`` returns
        ``None`` for them).
    """
    registry = MagicMock()
    registry.list_all.return_value = tools or []
    transient_map = transient_map or {}
    registry.enable_transient.side_effect = lambda name: transient_map.get(name)
    return registry


# ---------------------------------------------------------------------------
# Class attributes
# ---------------------------------------------------------------------------


class TestClassAttributes:
    def test_name(self):
        assert ToolManagerTool.NAME == "tool_manager"

    def test_permission_required_false(self):
        assert ToolManagerTool.PERMISSION_REQUIRED is False

    def test_not_disabled_by_default(self):
        assert ToolManagerTool.DISABLED_BY_DEFAULT is False


# ---------------------------------------------------------------------------
# set_registry
# ---------------------------------------------------------------------------


class TestSetRegistry:
    def test_registry_is_none_before_injection(self):
        tool = _make_tool()
        assert tool._registry is None

    def test_registry_set_after_injection(self):
        registry = _make_registry()
        tool = _make_tool(registry)
        assert tool._registry is registry


# ---------------------------------------------------------------------------
# list action
# ---------------------------------------------------------------------------


class TestListAction:
    def test_returns_all_tools(self):
        tools = [
            {"name": "read_file", "description": "Read a file.", "enabled": True},
            {"name": "write_file", "description": "Write a file.", "enabled": False},
        ]
        tool = _make_tool(_make_registry(tools=tools))
        result = tool.execute(action="list")
        assert result["status"] == "success"
        assert result["data"]["tools"] == tools

    def test_empty_registry(self):
        tool = _make_tool(_make_registry(tools=[]))
        result = tool.execute(action="list")
        assert result["status"] == "success"
        assert result["data"]["tools"] == []

    def test_error_when_registry_not_set(self):
        tool = _make_tool()
        result = tool.execute(action="list")
        assert result["status"] == "error"
        assert result["error"] == "internal_error"


# ---------------------------------------------------------------------------
# enable action
# ---------------------------------------------------------------------------


class TestEnableAction:
    def test_enable_known_tools_returns_schemas(self):
        schema_a = {"type": "function", "function": {"name": "tool_a"}}
        schema_b = {"type": "function", "function": {"name": "tool_b"}}
        registry = _make_registry(
            transient_map={"tool_a": schema_a, "tool_b": schema_b}
        )
        tool = _make_tool(registry)
        result = tool.execute(action="enable", tool_names=["tool_a", "tool_b"])
        assert result["status"] == "success"
        assert result["data"]["enabled"] == ["tool_a", "tool_b"]
        assert result["data"]["transient_schemas"] == [schema_a, schema_b]
        assert "unknown" not in result["data"]

    def test_enable_unknown_tool_recorded_in_unknown(self):
        registry = _make_registry(transient_map={})
        tool = _make_tool(registry)
        result = tool.execute(action="enable", tool_names=["ghost"])
        assert result["status"] == "success"
        assert result["data"]["enabled"] == []
        assert result["data"]["unknown"] == ["ghost"]
        assert result["data"]["transient_schemas"] == []

    def test_enable_mixed_known_and_unknown(self):
        schema = {"type": "function", "function": {"name": "read_file"}}
        registry = _make_registry(transient_map={"read_file": schema})
        tool = _make_tool(registry)
        result = tool.execute(action="enable", tool_names=["read_file", "ghost"])
        assert result["status"] == "success"
        assert result["data"]["enabled"] == ["read_file"]
        assert result["data"]["unknown"] == ["ghost"]
        assert result["data"]["transient_schemas"] == [schema]

    def test_enable_empty_tool_names_returns_error(self):
        tool = _make_tool(_make_registry())
        result = tool.execute(action="enable", tool_names=[])
        assert result["status"] == "error"
        assert result["error"] == "invalid_input"

    def test_enable_without_tool_names_returns_error(self):
        tool = _make_tool(_make_registry())
        result = tool.execute(action="enable")
        assert result["status"] == "error"
        assert result["error"] == "invalid_input"

    def test_enable_calls_enable_transient_for_each_name(self):
        schema = {"type": "function", "function": {"name": "find_files"}}
        registry = _make_registry(transient_map={"find_files": schema})
        tool = _make_tool(registry)
        tool.execute(action="enable", tool_names=["find_files"])
        registry.enable_transient.assert_called_once_with("find_files")

    def test_error_when_registry_not_set(self):
        tool = _make_tool()
        result = tool.execute(action="enable", tool_names=["read_file"])
        assert result["status"] == "error"
        assert result["error"] == "internal_error"


# ---------------------------------------------------------------------------
# Unknown action
# ---------------------------------------------------------------------------


class TestUnknownAction:
    def test_unknown_action_returns_error(self):
        tool = _make_tool(_make_registry())
        result = tool.execute(action="dance")
        assert result["status"] == "error"
        assert result["error"] == "invalid_input"
        assert "dance" in result["message"]


# ---------------------------------------------------------------------------
# definition()
# ---------------------------------------------------------------------------


class TestDefinition:
    def test_schema_structure(self):
        tool = _make_tool()
        defn = tool.definition()
        assert defn["type"] == "function"
        fn = defn["function"]
        assert fn["name"] == "tool_manager"
        params = fn["parameters"]
        assert "action" in params["properties"]
        assert params["properties"]["action"]["enum"] == ["list", "enable"]
        assert "tool_names" in params["properties"]
        assert params["required"] == ["action"]
