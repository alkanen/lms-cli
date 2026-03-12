"""Tests for ai_cli.core.tool_registry.ToolRegistry."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import yaml

from ai_cli.core.tool_registry import ToolRegistry
from ai_cli.tools.base import Tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EchoTool(Tool):
    NAME = "echo"
    DESCRIPTION = "Echoes input."
    PERMISSION_REQUIRED = False

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {},
            },
        }

    def execute(self, **kwargs: Any) -> dict:
        return self._ok({"echo": kwargs.get("message", "")})


class _PermTool(Tool):
    NAME = "perm_tool"
    DESCRIPTION = "Requires permission."
    PERMISSION_REQUIRED = True

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {},
            },
        }

    def execute(self, **kwargs: Any) -> dict:
        return self._ok()


class _DisabledByDefaultTool(Tool):
    NAME = "disabled_tool"
    DESCRIPTION = "Disabled by default."
    PERMISSION_REQUIRED = False
    DISABLED_BY_DEFAULT = True

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {},
            },
        }

    def execute(self, **kwargs: Any) -> dict:
        return self._ok()


def make_registry(
    tmp_path: Path,
    tool_classes: list[type[Tool]] | None = None,
    config_tools: dict | None = None,
    project_config_tools: dict | None = None,
) -> ToolRegistry:
    """Build a ToolRegistry with a fake workspace and pre-registered tools.

    *config_tools* simulates the merged (global + project) tools config.
    *project_config_tools* simulates the project-layer-only tools config
    (used by ``_apply_config`` to detect untrusted lowerings); defaults to
    an empty dict (all settings treated as coming from global config).
    """
    workspace = MagicMock()
    workspace.root = tmp_path
    (tmp_path / ".ai-cli").mkdir(exist_ok=True)

    config = MagicMock()
    config.get.return_value = config_tools or {}
    config.get_project.return_value = project_config_tools or {}

    pm = MagicMock()
    pm.request.return_value = (True, "")

    registry = ToolRegistry(workspace, config, pm)

    # Bypass file discovery and register tool classes directly.
    for cls in tool_classes or [_EchoTool]:
        registry._register(cls, tier="bundled")

    registry._apply_config()
    return registry


# ---------------------------------------------------------------------------
# get / all_enabled / definitions
# ---------------------------------------------------------------------------


class TestQueries:
    def test_get_known_tool(self, tmp_path):
        reg = make_registry(tmp_path)
        assert reg.get("echo") is not None

    def test_get_unknown_returns_none(self, tmp_path):
        reg = make_registry(tmp_path)
        assert reg.get("nonexistent") is None

    def test_all_enabled_returns_enabled_tools(self, tmp_path):
        reg = make_registry(tmp_path, tool_classes=[_EchoTool, _DisabledByDefaultTool])
        names = [t.name for t in reg.all_enabled()]
        assert "echo" in names
        assert "disabled_tool" not in names

    def test_definitions_returns_schemas(self, tmp_path):
        reg = make_registry(tmp_path)
        defs = reg.definitions()
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "echo"


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_executes_known_enabled_tool(self, tmp_path):
        reg = make_registry(tmp_path)
        result = reg.execute("echo", {"message": "hi"})
        assert result["status"] == "success"
        assert result["data"]["echo"] == "hi"

    def test_unknown_tool_returns_error(self, tmp_path):
        reg = make_registry(tmp_path)
        result = reg.execute("no_such_tool", {})
        assert result["status"] == "error"
        assert result["error"] == "unknown_tool"

    def test_disabled_tool_returns_error(self, tmp_path):
        reg = make_registry(tmp_path, tool_classes=[_DisabledByDefaultTool])
        result = reg.execute("disabled_tool", {})
        assert result["status"] == "error"
        assert result["error"] == "tool_disabled"

    def test_tool_exception_returns_error(self, tmp_path):
        class BrokenTool(_EchoTool):
            NAME = "broken"

            def execute(self, **kwargs: Any) -> dict:
                raise RuntimeError("something went wrong")

        reg = make_registry(tmp_path, tool_classes=[BrokenTool])
        result = reg.execute("broken", {})
        assert result["status"] == "error"
        assert result["error"] == "tool_execution_error"
        assert result["code"] == 500
        assert "logs" in result["message"]  # generic message, no exception details

    def test_kwargs_forwarded_to_request_permission(self, tmp_path):
        class FolderTool(_EchoTool):
            NAME = "folder_tool"
            PERMISSION_REQUIRED = True  # must be True so permission prompt fires

            def extra_permission_options(self, **kwargs: Any) -> list[str]:
                return ["always_in_this_folder"] if kwargs.get("path") else []

        workspace = MagicMock()
        workspace.root = tmp_path
        (tmp_path / ".ai-cli").mkdir(exist_ok=True)
        config = MagicMock()
        config.get.return_value = {}
        config.get_project.return_value = {}
        captured: list[Any] = []
        pm = MagicMock()
        pm.request.side_effect = lambda **kw: captured.append(kw) or (True, "")
        reg = ToolRegistry(workspace, config, pm)
        reg._register(FolderTool, tier="bundled")
        reg._apply_config()
        reg.execute("folder_tool", {"path": "/tmp/foo"})
        assert any("always_in_this_folder" in str(c) for c in captured)

    def test_permission_denied_returns_error(self, tmp_path):
        workspace = MagicMock()
        workspace.root = tmp_path
        (tmp_path / ".ai-cli").mkdir(exist_ok=True)
        config = MagicMock()
        config.get.return_value = {}
        config.get_project.return_value = {}
        pm = MagicMock()
        pm.request.return_value = (False, "Permission denied.")
        reg = ToolRegistry(workspace, config, pm)
        reg._register(_PermTool, tier="bundled")
        reg._apply_config()
        result = reg.execute("perm_tool", {})
        assert result["status"] == "error"
        assert result["error"] == "permission_denied"

    def test_on_permission_granted_called_with_choice(self, tmp_path):
        """When the user picks an extra option, on_permission_granted receives it."""
        granted_calls: list[tuple[str, dict]] = []

        class HookTool(_EchoTool):
            NAME = "hook_tool"
            PERMISSION_REQUIRED = True

            def extra_permission_options(self, **kwargs: Any) -> list[str]:
                return ["always_in_this_folder"]

            def on_permission_granted(self, choice: str, **kwargs: Any) -> None:
                granted_calls.append((choice, dict(kwargs)))

        workspace = MagicMock()
        workspace.root = tmp_path
        (tmp_path / ".ai-cli").mkdir(exist_ok=True)
        config = MagicMock()
        config.get.return_value = {}
        config.get_project.return_value = {}
        pm = MagicMock()
        pm.request.return_value = (True, "always_in_this_folder")
        reg = ToolRegistry(workspace, config, pm)
        reg._register(HookTool, tier="bundled")
        reg._apply_config()
        reg.execute("hook_tool", {"message": "hi"})
        assert granted_calls == [("always_in_this_folder", {"message": "hi"})]

    def test_on_permission_granted_not_called_when_choice_empty(self, tmp_path):
        """on_permission_granted is not called when the permission choice is empty."""
        granted_calls: list = []

        class HookTool(_EchoTool):
            NAME = "hook_tool"
            PERMISSION_REQUIRED = True

            def on_permission_granted(self, choice: str, **kwargs: Any) -> None:
                granted_calls.append(choice)

        workspace = MagicMock()
        workspace.root = tmp_path
        (tmp_path / ".ai-cli").mkdir(exist_ok=True)
        config = MagicMock()
        config.get.return_value = {}
        config.get_project.return_value = {}
        pm = MagicMock()
        pm.request.return_value = (True, "")
        reg = ToolRegistry(workspace, config, pm)
        reg._register(HookTool, tier="bundled")
        reg._apply_config()
        reg.execute("hook_tool", {"message": "hi"})
        assert granted_calls == []


# ---------------------------------------------------------------------------
# Persistent enable / disable
# ---------------------------------------------------------------------------


class TestPersistentEnable:
    def test_enable_enables_tool(self, tmp_path):
        reg = make_registry(tmp_path, tool_classes=[_DisabledByDefaultTool])
        assert not reg._is_enabled("disabled_tool")
        reg.enable("disabled_tool")
        assert reg._is_enabled("disabled_tool")

    def test_disable_disables_tool(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disable("echo")
        assert not reg._is_enabled("echo")

    def test_enable_writes_config(self, tmp_path):
        reg = make_registry(tmp_path, tool_classes=[_DisabledByDefaultTool])
        reg.enable("disabled_tool")
        cfg = yaml.safe_load((tmp_path / ".ai-cli" / "config.yaml").read_text())
        assert cfg["tools"]["disabled_tool"]["disabled"] is False

    def test_disable_writes_config(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disable("echo")
        cfg = yaml.safe_load((tmp_path / ".ai-cli" / "config.yaml").read_text())
        assert cfg["tools"]["echo"]["disabled"] is True

    def test_enable_unknown_tool_is_noop(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.enable("nonexistent")  # should not raise

    def test_enable_clears_session_override(self, tmp_path):
        # disable_session then enable (persistent) — tool must become enabled.
        reg = make_registry(tmp_path)
        reg.disable_session("echo")
        assert not reg._is_enabled("echo")
        reg.enable("echo")
        assert reg._is_enabled("echo")

    def test_disable_clears_session_override(self, tmp_path):
        # enable_session on a disabled tool then disable (persistent) — tool must become disabled.
        reg = make_registry(tmp_path, tool_classes=[_DisabledByDefaultTool])
        reg.enable_session("disabled_tool")
        assert reg._is_enabled("disabled_tool")
        reg.disable("disabled_tool")
        assert not reg._is_enabled("disabled_tool")


# ---------------------------------------------------------------------------
# Session enable / disable
# ---------------------------------------------------------------------------


class TestSessionEnable:
    def test_session_enable_overrides_persistent_disabled(self, tmp_path):
        reg = make_registry(tmp_path, tool_classes=[_DisabledByDefaultTool])
        assert not reg._is_enabled("disabled_tool")
        reg.enable_session("disabled_tool")
        assert reg._is_enabled("disabled_tool")

    def test_session_disable_overrides_persistent_enabled(self, tmp_path):
        reg = make_registry(tmp_path)
        assert reg._is_enabled("echo")
        reg.disable_session("echo")
        assert not reg._is_enabled("echo")

    def test_reset_clears_session_overrides(self, tmp_path):
        reg = make_registry(tmp_path, tool_classes=[_DisabledByDefaultTool])
        reg.enable_session("disabled_tool")
        reg.reset_session_overrides()
        assert not reg._is_enabled("disabled_tool")

    def test_reset_continues_if_one_tool_raises(self, tmp_path):
        # A bug in one tool's reset hook must not prevent others from being reset.
        class BrokenResetTool(_EchoTool):
            NAME = "broken_reset"

            def reset_session_state(self) -> None:
                raise RuntimeError("oops")

        reg = make_registry(tmp_path, tool_classes=[_EchoTool, BrokenResetTool])
        reset_called: list[str] = []
        original = reg.get("echo").reset_session_state

        def tracked_reset() -> None:
            reset_called.append("echo")
            original()

        reg.get("echo").reset_session_state = tracked_reset  # type: ignore[method-assign]
        reg.reset_session_overrides()  # must not raise
        assert "echo" in reset_called

    def test_session_does_not_write_config(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disable_session("echo")
        config_path = tmp_path / ".ai-cli" / "config.yaml"
        assert not config_path.exists()


# ---------------------------------------------------------------------------
# Transient enable
# ---------------------------------------------------------------------------


class TestTransientEnable:
    def test_returns_schema_for_known_tool(self, tmp_path):
        reg = make_registry(tmp_path)
        schema = reg.enable_transient("echo")
        assert schema is not None
        assert schema["function"]["name"] == "echo"

    def test_returns_none_for_unknown_tool(self, tmp_path):
        reg = make_registry(tmp_path)
        assert reg.enable_transient("nonexistent") is None

    def test_does_not_change_enabled_state(self, tmp_path):
        reg = make_registry(tmp_path, tool_classes=[_DisabledByDefaultTool])
        assert not reg._is_enabled("disabled_tool")
        reg.enable_transient("disabled_tool")
        assert not reg._is_enabled("disabled_tool")

    def test_execute_allow_transient_bypasses_disabled_check(self, tmp_path):
        reg = make_registry(tmp_path, tool_classes=[_DisabledByDefaultTool])
        assert not reg._is_enabled("disabled_tool")
        result = reg.execute("disabled_tool", {}, allow_transient=True)
        assert result["status"] == "success"

    def test_execute_without_allow_transient_still_blocked(self, tmp_path):
        reg = make_registry(tmp_path, tool_classes=[_DisabledByDefaultTool])
        result = reg.execute("disabled_tool", {})
        assert result["error"] == "tool_disabled"


# ---------------------------------------------------------------------------
# Config application
# ---------------------------------------------------------------------------


class TestConfigApplication:
    def test_config_overrides_permission_required(self, tmp_path):
        reg = make_registry(
            tmp_path,
            tool_classes=[_EchoTool],
            config_tools={"echo": {"permission_required": True}},
        )
        assert reg.get("echo").permission_required is True

    def test_config_can_disable_tool(self, tmp_path):
        reg = make_registry(
            tmp_path,
            tool_classes=[_EchoTool],
            config_tools={"echo": {"disabled": True}},
        )
        assert not reg._is_enabled("echo")

    def test_config_unknown_tool_ignored(self, tmp_path):
        reg = make_registry(
            tmp_path,
            tool_classes=[_EchoTool],
            config_tools={"nonexistent": {"disabled": True}},
        )
        assert reg._is_enabled("echo")

    def test_non_bool_config_value_ignored(self, tmp_path):
        # "false" as a string is truthy — must not be applied as a boolean.
        reg = make_registry(
            tmp_path,
            tool_classes=[_EchoTool],
            config_tools={"echo": {"permission_required": "false"}},
        )
        # Default should be preserved (False), not coerced to True.
        assert reg.get("echo").permission_required is False

    def test_project_config_cannot_lower_permission_required(self, tmp_path):
        # Project config is untrusted — it must not silently disable permissions.
        reg = make_registry(
            tmp_path,
            tool_classes=[_PermTool],
            config_tools={"perm_tool": {"permission_required": False}},
            project_config_tools={"perm_tool": {"permission_required": False}},
        )
        assert reg.get("perm_tool").permission_required is True

    def test_global_config_can_lower_permission_required(self, tmp_path):
        # Global (~/.ai-cli/config.yaml) is trusted — lowering is allowed without marker.
        reg = make_registry(
            tmp_path,
            tool_classes=[_PermTool],
            config_tools={"perm_tool": {"permission_required": False}},
            # project_config_tools left empty → lowering came from global only
        )
        assert reg.get("perm_tool").permission_required is False


# ---------------------------------------------------------------------------
# set_permission_required
# ---------------------------------------------------------------------------


class TestSetPermissionRequired:
    def test_updates_tool(self, tmp_path):
        reg = make_registry(tmp_path)
        assert reg.get("echo").permission_required is False
        reg.set_permission_required("echo", True)
        assert reg.get("echo").permission_required is True

    def test_persists_to_config(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.set_permission_required("echo", True)
        cfg = yaml.safe_load((tmp_path / ".ai-cli" / "config.yaml").read_text())
        assert cfg["tools"]["echo"]["permission_required"] is True

    def test_lowering_permission_requires_user_confirmed_marker(self, tmp_path):
        # set_permission_required(False) writes user_confirmed=True so the
        # setting survives a config reload without being blocked as untrusted.
        reg = make_registry(tmp_path, tool_classes=[_PermTool])
        assert reg.get("perm_tool").permission_required is True
        reg.set_permission_required("perm_tool", False)
        cfg = yaml.safe_load((tmp_path / ".ai-cli" / "config.yaml").read_text())
        assert cfg["tools"]["perm_tool"]["permission_required"] is False
        assert cfg["tools"]["perm_tool"]["user_confirmed"] is True
        # Simulated reload — the marker in the project layer allows the lowering.
        reg2 = make_registry(
            tmp_path,
            tool_classes=[_PermTool],
            config_tools=cfg["tools"],
            project_config_tools=cfg["tools"],
        )
        assert reg2.get("perm_tool").permission_required is False

    def test_untrusted_config_cannot_lower_permission(self, tmp_path):
        # Project config without user_confirmed must not lower permission_required.
        reg = make_registry(
            tmp_path,
            tool_classes=[_PermTool],
            config_tools={"perm_tool": {"permission_required": False}},
            project_config_tools={"perm_tool": {"permission_required": False}},
        )
        assert reg.get("perm_tool").permission_required is True
