"""Tests for ai_cli.core.tool_registry.ToolRegistry."""

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import yaml

from ai_cli.core.tool_registry import ToolRegistry
from ai_cli.tools.base import Tool, ToolSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EchoTool(Tool):
    NAME = "echo"
    DESCRIPTION = "Echoes input."
    PERMISSION_REQUIRED = False

    def definition(self) -> ToolSchema:
        from ai_cli.tools.base import ToolArgument

        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    name="message",
                    description="Message to echo.",
                    argument_type="string",
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        return self._ok({"echo": kwargs.get("message", "")})


class _PermTool(Tool):
    NAME = "perm_tool"
    DESCRIPTION = "Requires permission."
    PERMISSION_REQUIRED = True

    def definition(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description)

    def execute(self, **kwargs: Any) -> dict:
        return self._ok()


class _DisabledByDefaultTool(Tool):
    NAME = "disabled_tool"
    DESCRIPTION = "Disabled by default."
    PERMISSION_REQUIRED = False
    DISABLED_BY_DEFAULT = True

    def definition(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description)

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
# definition() validation
# ---------------------------------------------------------------------------


class TestDefinitionValidation:
    def _make_bad_tool(self, bad_defn: object) -> type:
        class _BadSchema(ToolSchema):
            def __init__(self) -> None:
                pass  # skip normal __init__

            def schema(self) -> dict:  # type: ignore[override]
                return bad_defn  # type: ignore[return-value]

        class _BadDefnTool(_EchoTool):
            NAME = "bad_tool"

            def definition(self) -> ToolSchema:
                return _BadSchema()

        return _BadDefnTool

    def test_valid_definition_is_registered(self, tmp_path):
        reg = make_registry(tmp_path)
        assert reg.get("echo") is not None

    def test_missing_type_field_rejected(self, tmp_path, caplog):
        import logging

        cls = self._make_bad_tool(
            {
                "function": {
                    "name": "bad_tool",
                    "description": "x",
                    "parameters": {"type": "object", "properties": {}},
                }
            }
        )
        with caplog.at_level(logging.WARNING, logger="ai_cli.core.tool_registry"):
            reg = make_registry(tmp_path, tool_classes=[cls])
        assert reg.get("bad_tool") is None
        assert any("invalid definition" in r.message for r in caplog.records)

    def test_missing_function_name_rejected(self, tmp_path, caplog):
        import logging

        cls = self._make_bad_tool(
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "x",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
        with caplog.at_level(logging.WARNING, logger="ai_cli.core.tool_registry"):
            reg = make_registry(tmp_path, tool_classes=[cls])
        assert reg.get("bad_tool") is None

    def test_required_param_not_in_properties_rejected(self, tmp_path, caplog):
        import logging

        cls = self._make_bad_tool(
            {
                "type": "function",
                "function": {
                    "name": "bad_tool",
                    "description": "x",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": ["missing"],
                    },
                },
            }
        )
        with caplog.at_level(logging.WARNING, logger="ai_cli.core.tool_registry"):
            reg = make_registry(tmp_path, tool_classes=[cls])
        assert reg.get("bad_tool") is None
        assert any("missing" in r.message for r in caplog.records)

    def test_schema_name_mismatch_rejected(self, tmp_path, caplog):
        import logging

        cls = self._make_bad_tool(
            {
                "type": "function",
                "function": {
                    "name": "wrong_name",
                    "description": "x",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        )
        with caplog.at_level(logging.WARNING, logger="ai_cli.core.tool_registry"):
            reg = make_registry(tmp_path, tool_classes=[cls])
        assert reg.get("bad_tool") is None
        assert any("wrong_name" in r.message for r in caplog.records)

    def test_non_dict_definition_rejected(self, tmp_path, caplog):
        import logging

        cls = self._make_bad_tool("not a dict")  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING, logger="ai_cli.core.tool_registry"):
            reg = make_registry(tmp_path, tool_classes=[cls])
        assert reg.get("bad_tool") is None

    def test_raising_definition_warns_and_tool_not_registered(self, tmp_path, caplog):
        import logging

        class _RaisingDefnTool(_EchoTool):
            NAME = "raising_defn"

            def definition(self) -> ToolSchema:
                raise RuntimeError("schema generation failed")

        with caplog.at_level(logging.WARNING, logger="ai_cli.core.tool_registry"):
            reg = make_registry(tmp_path, tool_classes=[_RaisingDefnTool])

        assert reg.get("raising_defn") is None
        assert any("definition()" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class _ArgTool(Tool):
    """Tool with a mix of required and optional typed arguments for validation tests."""

    NAME = "arg_tool"
    DESCRIPTION = "Tool with typed arguments."
    PERMISSION_REQUIRED = False

    def definition(self) -> ToolSchema:
        from ai_cli.tools.base import ToolArgument

        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    name="count",
                    description="An integer count.",
                    argument_type="integer",
                    required=True,
                ),
                ToolArgument(
                    name="label",
                    description="An optional string label.",
                    argument_type="string",
                ),
                ToolArgument(
                    name="active",
                    description="An optional boolean flag.",
                    argument_type="boolean",
                ),
                ToolArgument(
                    name="ratio",
                    description="An optional float.",
                    argument_type="number",
                ),
                ToolArgument(
                    name="score",
                    description="An integer score from 0 to 100.",
                    argument_type="integer",
                    minimum=0,
                    maximum=100,
                ),
                ToolArgument(
                    name="temperature",
                    description="A float from 0.0 to 1.0.",
                    argument_type="number",
                    minimum=0.0,
                    maximum=1.0,
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        return self._ok(kwargs)


class TestArgumentValidation:
    def _reg(self, tmp_path: Any) -> Any:
        return make_registry(tmp_path, tool_classes=[_ArgTool])

    def test_valid_args_pass_through(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 3})
        assert result["status"] == "success"
        assert result["data"]["count"] == 3

    def test_missing_required_arg_returns_error(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {})
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert "count" in result["message"]

    def test_optional_arg_can_be_omitted(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 1})
        assert result["status"] == "success"
        assert "label" not in result["data"]

    def test_unknown_arg_is_stripped(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 1, "surprise": "x"})
        assert result["status"] == "success"
        assert "surprise" not in result["data"]

    def test_unknown_arg_logs_warning(self, tmp_path, caplog):
        import logging

        reg = self._reg(tmp_path)
        with caplog.at_level(logging.WARNING, logger="ai_cli.core.tool_registry"):
            reg.execute("arg_tool", {"count": 1, "surprise": "x"})
        assert any("surprise" in r.message for r in caplog.records)

    def test_wrong_type_returns_error(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": "not-an-int"})
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert "count" in result["message"]
        assert "integer" in result["message"]

    def test_bool_rejected_for_integer_arg(self, tmp_path):
        # bool is a subclass of int in Python but not a valid JSON integer.
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": True})
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_string_rejected_for_boolean_arg(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 1, "active": "true"})
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert "active" in result["message"]

    def test_string_rejected_for_number_arg(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 1, "ratio": "3.14"})
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert "ratio" in result["message"]

    def test_correct_types_pass_through(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute(
            "arg_tool", {"count": 5, "label": "hi", "active": True, "ratio": 2.5}
        )
        assert result["status"] == "success"
        assert result["data"] == {
            "count": 5,
            "label": "hi",
            "active": True,
            "ratio": 2.5,
        }

    def test_int_accepted_for_number_arg(self, tmp_path):
        # JSON "number" covers both int and float.
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 1, "ratio": 3})
        assert result["status"] == "success"
        assert result["data"]["ratio"] == 3


# ---------------------------------------------------------------------------
# Bounds validation
# ---------------------------------------------------------------------------


class TestBoundsValidation:
    def _reg(self, tmp_path: Any) -> Any:
        return make_registry(tmp_path, tool_classes=[_ArgTool])

    def test_value_within_bounds_passes(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 1, "score": 50})
        assert result["status"] == "success"
        assert result["data"]["score"] == 50

    def test_value_at_minimum_passes(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 1, "score": 0})
        assert result["status"] == "success"

    def test_value_at_maximum_passes(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 1, "score": 100})
        assert result["status"] == "success"

    def test_value_below_minimum_returns_error(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 1, "score": -1})
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert "score" in result["message"]
        assert "minimum" in result["message"]

    def test_value_above_maximum_returns_error(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 1, "score": 101})
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert "score" in result["message"]
        assert "maximum" in result["message"]

    def test_float_within_bounds_passes(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 1, "temperature": 0.5})
        assert result["status"] == "success"

    def test_float_below_minimum_returns_error(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 1, "temperature": -0.1})
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert "temperature" in result["message"]

    def test_float_above_maximum_returns_error(self, tmp_path):
        reg = self._reg(tmp_path)
        result = reg.execute("arg_tool", {"count": 1, "temperature": 1.1})
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert "temperature" in result["message"]

    def test_minimum_only_enforced(self, tmp_path):
        reg = self._reg(tmp_path)
        # ratio has no bounds — any number passes
        result = reg.execute("arg_tool", {"count": 1, "ratio": -999.9})
        assert result["status"] == "success"

    def test_bounds_in_schema_output(self, tmp_path):
        reg = self._reg(tmp_path)
        tool = reg.get("arg_tool")
        schema = tool.definition().schema()
        props = schema["function"]["parameters"]["properties"]
        assert props["score"]["minimum"] == 0
        assert props["score"]["maximum"] == 100
        assert props["temperature"]["minimum"] == 0.0
        assert props["temperature"]["maximum"] == 1.0

    def test_unbounded_arg_has_no_bounds_in_schema(self, tmp_path):
        reg = self._reg(tmp_path)
        tool = reg.get("arg_tool")
        schema = tool.definition().schema()
        props = schema["function"]["parameters"]["properties"]
        assert "minimum" not in props["ratio"]
        assert "maximum" not in props["ratio"]

    def test_inverted_bounds_skipped_with_warning(self, tmp_path, caplog):
        import logging

        from ai_cli.tools.base import ToolArgument

        # Bypass __init__ validation by mutating after construction.
        arg = ToolArgument(
            name="x", description="x", argument_type="integer", minimum=0, maximum=10
        )
        arg.minimum = 99
        arg.maximum = 1  # now inverted

        from ai_cli.core.tool_registry import _check_bounds

        with caplog.at_level(logging.WARNING, logger="ai_cli.core.tool_registry"):
            result = _check_bounds(5, arg)
        assert result is None  # skipped, not an error
        assert any("inverted" in r.message for r in caplog.records)

    def test_non_numeric_bound_skipped_with_warning(self, tmp_path, caplog):
        import logging

        from ai_cli.core.tool_registry import _check_bounds
        from ai_cli.tools.base import ToolArgument

        arg = ToolArgument(
            name="x", description="x", argument_type="integer", minimum=0, maximum=10
        )
        arg.minimum = "bad"  # type: ignore[assignment]

        with caplog.at_level(logging.WARNING, logger="ai_cli.core.tool_registry"):
            result = _check_bounds(5, arg)
        assert result is None
        assert any("non-numeric" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# ToolArgument bounds validation
# ---------------------------------------------------------------------------


class TestToolArgumentBoundsValidation:
    def test_valid_integer_bounds_accepted(self):
        from ai_cli.tools.base import ToolArgument

        arg = ToolArgument(
            name="n", description="n", argument_type="integer", minimum=0, maximum=100
        )
        assert arg.minimum == 0
        assert arg.maximum == 100

    def test_valid_number_bounds_accepted(self):
        from ai_cli.tools.base import ToolArgument

        arg = ToolArgument(
            name="r", description="r", argument_type="number", minimum=0.0, maximum=1.0
        )
        assert arg.minimum == 0.0
        assert arg.maximum == 1.0

    def test_minimum_only_accepted(self):
        from ai_cli.tools.base import ToolArgument

        arg = ToolArgument(
            name="n", description="n", argument_type="integer", minimum=1
        )
        assert arg.minimum == 1
        assert arg.maximum is None

    def test_maximum_only_accepted(self):
        from ai_cli.tools.base import ToolArgument

        arg = ToolArgument(
            name="n", description="n", argument_type="integer", maximum=10
        )
        assert arg.minimum is None
        assert arg.maximum == 10

    def test_bounds_on_string_type_raises(self):
        import pytest

        from ai_cli.tools.base import ToolArgument

        with pytest.raises(ValueError, match="integer.*number"):
            ToolArgument(name="s", description="s", argument_type="string", minimum=0)

    def test_bounds_on_boolean_type_raises(self):
        import pytest

        from ai_cli.tools.base import ToolArgument

        with pytest.raises(ValueError, match="integer.*number"):
            ToolArgument(name="b", description="b", argument_type="boolean", maximum=1)

    def test_non_numeric_minimum_raises(self):
        import pytest

        from ai_cli.tools.base import ToolArgument

        with pytest.raises(ValueError, match="numeric"):
            ToolArgument(
                name="n", description="n", argument_type="integer", minimum="zero"
            )  # type: ignore[arg-type]

    def test_bool_minimum_raises(self):
        import pytest

        from ai_cli.tools.base import ToolArgument

        with pytest.raises(ValueError, match="numeric"):
            ToolArgument(
                name="n", description="n", argument_type="integer", minimum=True
            )  # type: ignore[arg-type]

    def test_inverted_bounds_raises(self):
        import pytest

        from ai_cli.tools.base import ToolArgument

        with pytest.raises(ValueError, match="minimum.*maximum|<="):
            ToolArgument(
                name="n",
                description="n",
                argument_type="integer",
                minimum=10,
                maximum=5,
            )

    def test_equal_bounds_accepted(self):
        from ai_cli.tools.base import ToolArgument

        arg = ToolArgument(
            name="n", description="n", argument_type="integer", minimum=5, maximum=5
        )
        assert arg.minimum == 5
        assert arg.maximum == 5

    def test_invalid_bounds_cause_tool_to_be_skipped(self, tmp_path, caplog):
        import logging

        class _BadBoundsTool(_EchoTool):
            NAME = "bad_bounds"

            def definition(self) -> ToolSchema:
                # Will raise ValueError in ToolArgument.__init__
                from ai_cli.tools.base import ToolArgument

                return ToolSchema(
                    name=self.name,
                    description=self.description,
                    arguments=[
                        ToolArgument(
                            name="x",
                            description="x",
                            argument_type="integer",
                            minimum=100,
                            maximum=1,
                        ),
                    ],
                )

        with caplog.at_level(logging.WARNING, logger="ai_cli.core.tool_registry"):
            reg = make_registry(tmp_path, tool_classes=[_BadBoundsTool])
        assert reg.get("bad_bounds") is None
        assert any("definition()" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_executes_known_enabled_tool(self, tmp_path):
        reg = make_registry(tmp_path)
        result = reg.execute("echo", {"message": "hi"})
        assert result["status"] == "success"
        assert result["data"]["echo"] == "hi"

    def test_non_dict_kwargs_returns_error(self, tmp_path):
        reg = make_registry(tmp_path)
        for bad in (None, [], "args", 42):
            result = reg.execute("echo", bad)  # type: ignore[arg-type]
            assert result["status"] == "error", f"expected error for {bad!r}"
            assert result["error"] == "invalid_arguments"

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

            def definition(self) -> ToolSchema:
                from ai_cli.tools.base import ToolArgument

                return ToolSchema(
                    name=self.name,
                    description=self.description,
                    arguments=[
                        ToolArgument(
                            name="path",
                            description="Path argument.",
                            argument_type="string",
                            required=True,
                        ),
                    ],
                )

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
# Allow / disallow
# ---------------------------------------------------------------------------


class TestAllowed:
    def test_disallowed_tool_not_in_all_enabled(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disallow("echo")
        assert not any(t.name == "echo" for t in reg.all_enabled())

    def test_disallowed_tool_not_in_list_all(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disallow("echo")
        names = [t["name"] for t in reg.list_all()]
        assert "echo" not in names

    def test_disallowed_tool_execute_returns_error(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disallow("echo")
        result = reg.execute("echo", {})
        assert result["status"] == "error"
        assert result["error"] == "tool_disallowed"

    def test_disallow_persists_to_config(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disallow("echo")
        cfg = yaml.safe_load((tmp_path / ".ai-cli" / "config.yaml").read_text())
        assert cfg["tools"]["echo"]["allowed"] is False

    def test_allow_re_enables_disallowed_tool(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disallow("echo")
        assert not reg._is_allowed("echo")
        reg.allow("echo")
        assert reg._is_allowed("echo")

    def test_allow_writes_to_project_config(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disallow("echo")
        reg.allow("echo")
        cfg = yaml.safe_load((tmp_path / ".ai-cli" / "config.yaml").read_text())
        assert cfg["tools"]["echo"]["allowed"] is True

    def test_disallow_session_hides_tool(self, tmp_path):
        reg = make_registry(tmp_path)
        assert reg._is_allowed("echo")
        reg.disallow_session("echo")
        assert not reg._is_allowed("echo")

    def test_allow_session_overrides_persistent_disallow(self, tmp_path):
        reg = make_registry(tmp_path, config_tools={"echo": {"allowed": False}})
        assert not reg._is_allowed("echo")
        reg.allow_session("echo")
        assert reg._is_allowed("echo")

    def test_session_disallow_does_not_write_config(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disallow_session("echo")
        config_path = tmp_path / ".ai-cli" / "config.yaml"
        assert not config_path.exists()

    def test_transient_enable_blocked_for_disallowed(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disallow("echo")
        assert reg.enable_transient("echo") is None

    def test_reset_clears_session_allowed_overrides(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disallow_session("echo")
        assert not reg._is_allowed("echo")
        reg.reset_session_overrides()
        assert reg._is_allowed("echo")

    def test_all_tools_info_includes_disallowed(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disallow("echo")
        info_list = reg.all_tools_info()
        echo_info = next(i for i in info_list if i["name"] == "echo")
        assert echo_info["allowed"] is False

    def test_tool_info_returns_none_for_unknown(self, tmp_path):
        reg = make_registry(tmp_path)
        assert reg.tool_info("ghost") is None

    def test_tool_info_returns_dict_for_known(self, tmp_path):
        reg = make_registry(tmp_path)
        info = reg.tool_info("echo")
        assert info is not None
        assert info["name"] == "echo"
        assert "parameters" in info

    def test_allow_unknown_tool_is_noop(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.allow("nonexistent")  # should not raise

    def test_disallow_unknown_tool_is_noop(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.disallow("nonexistent")  # should not raise


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

    def test_project_config_can_lower_permission_required(self, tmp_path):
        # Project config overrides global — it can lower permission_required.
        reg = make_registry(
            tmp_path,
            tool_classes=[_PermTool],
            config_tools={"perm_tool": {"permission_required": False}},
            project_config_tools={"perm_tool": {"permission_required": False}},
        )
        assert reg.get("perm_tool").permission_required is False

    def test_global_config_can_lower_permission_required(self, tmp_path):
        # Global config can lower permission_required when no project override exists.
        reg = make_registry(
            tmp_path,
            tool_classes=[_PermTool],
            config_tools={"perm_tool": {"permission_required": False}},
        )
        assert reg.get("perm_tool").permission_required is False

    def test_lowering_permission_required_logs_warning(self, tmp_path, caplog):
        # Config is allowed to lower permission_required, but a warning must be logged
        # so the user knows their prompts are being skipped.
        import logging

        with caplog.at_level(logging.WARNING, logger="ai_cli.core.tool_registry"):
            make_registry(
                tmp_path,
                tool_classes=[_PermTool],
                config_tools={"perm_tool": {"permission_required": False}},
            )
        assert any(
            "perm_tool" in r.message and "permission_required" in r.message
            for r in caplog.records
        )

    def test_config_can_re_allow_tool(self, tmp_path):
        # A tool disallowed by config can be re-allowed by a higher-precedence config entry.
        reg_disallowed = make_registry(
            tmp_path,
            tool_classes=[_EchoTool],
            config_tools={"echo": {"allowed": False}},
        )
        assert not reg_disallowed._is_allowed("echo")

        reg_reallowed = make_registry(
            tmp_path,
            tool_classes=[_EchoTool],
            config_tools={"echo": {"allowed": True}},
        )
        assert reg_reallowed._is_allowed("echo")

    def test_project_config_can_re_allow_tool(self, tmp_path):
        # Project config overrides global — project allowed: true re-allows a globally
        # disallowed tool (project config is the higher-precedence layer).
        reg_global_disallowed = make_registry(
            tmp_path,
            tool_classes=[_EchoTool],
            config_tools={"echo": {"allowed": False}},
        )
        assert not reg_global_disallowed._is_allowed("echo")

        reg_project_reallowed = make_registry(
            tmp_path,
            tool_classes=[_EchoTool],
            # merged config reflects project overriding global disallow
            config_tools={"echo": {"allowed": True}},
            project_config_tools={"echo": {"allowed": True}},
        )
        assert reg_project_reallowed._is_allowed("echo")


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

    def test_lowering_permission_persists_and_survives_reload(self, tmp_path):
        # set_permission_required(False) writes to project config and survives reload.
        reg = make_registry(tmp_path, tool_classes=[_PermTool])
        reg.set_permission_required("perm_tool", False)
        cfg = yaml.safe_load((tmp_path / ".ai-cli" / "config.yaml").read_text())
        assert cfg["tools"]["perm_tool"]["permission_required"] is False
        # Reload — project config overrides the tool default, lowering is honoured.
        reg2 = make_registry(
            tmp_path,
            tool_classes=[_PermTool],
            config_tools=cfg["tools"],
            project_config_tools=cfg["tools"],
        )
        assert reg2.get("perm_tool").permission_required is False


# ---------------------------------------------------------------------------
# set_registry hook safety
# ---------------------------------------------------------------------------


class TestSetRegistryHook:
    def test_set_registry_called_on_tool_that_defines_it(self, tmp_path):
        calls = []

        class _RegistryAwareTool(_EchoTool):
            NAME = "reg_aware"

            def set_registry(self, registry):
                calls.append(registry)

        reg = make_registry(tmp_path, tool_classes=[_RegistryAwareTool])
        assert len(calls) == 1
        assert calls[0] is reg

    def test_non_callable_set_registry_warns_and_does_not_crash(self, tmp_path, caplog):
        import logging

        class _BadAttrTool(_EchoTool):
            NAME = "bad_attr"
            set_registry = "not_callable"  # type: ignore[assignment]

        with caplog.at_level(logging.WARNING, logger="ai_cli.core.tool_registry"):
            reg = make_registry(tmp_path, tool_classes=[_BadAttrTool])

        assert reg.get("bad_attr") is not None
        assert any("non-callable" in r.message for r in caplog.records)

    def test_raising_set_registry_warns_and_does_not_crash(self, tmp_path, caplog):
        import logging

        class _RaisingTool(_EchoTool):
            NAME = "raising_tool"

            def set_registry(self, registry):
                raise RuntimeError("boom")

        with caplog.at_level(logging.WARNING, logger="ai_cli.core.tool_registry"):
            reg = make_registry(tmp_path, tool_classes=[_RaisingTool])

        assert reg.get("raising_tool") is not None
        assert any("set_registry" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# REGISTER_VIA_INSTANCE — file loader skip
# ---------------------------------------------------------------------------


class TestRegisterViaInstance:
    """Tools with REGISTER_VIA_INSTANCE=True must be silently skipped by
    both the bundled module loader and the directory file loader."""

    def _make_bare_registry(self, tmp_path: Path) -> ToolRegistry:
        """Return a ToolRegistry with no tools pre-registered."""
        workspace = MagicMock()
        workspace.root = tmp_path
        (tmp_path / ".ai-cli").mkdir(exist_ok=True)
        config = MagicMock()
        config.get.return_value = {}
        config.get_project.return_value = {}
        pm = MagicMock()
        return ToolRegistry(workspace, config, pm)

    def test_load_bundled_skips_register_via_instance(self, tmp_path):
        """_load_bundled must not attempt to instantiate a class marked
        REGISTER_VIA_INSTANCE=True, even though it passes _validate_tool_class.

        Calls _load_bundled() directly: writes a .py file into a temp directory,
        pre-populates sys.modules under the expected 'ai_cli.tools.<stem>' key so
        importlib.import_module returns the fake module, then verifies the tool is
        not registered and its incompatible __init__ is never called.
        """
        import sys
        import types

        tool_src = textwrap.dedent("""\
            from typing import Any
            from ai_cli.tools.base import Tool, ToolSchema

            class NonStandardBundledTool(Tool):
                NAME = "non_standard_bundled"
                DESCRIPTION = "Requires extra constructor args."
                PERMISSION_REQUIRED = False
                REGISTER_VIA_INSTANCE = True

                def __init__(self, workspace, permission_manager, extra_arg):
                    raise AssertionError("should never be instantiated by the loader")

                def definition(self):
                    return ToolSchema(name=self.name, description=self.description)

                def execute(self, **kwargs: Any) -> dict:
                    return self._ok()
        """)
        tool_file = tmp_path / "non_standard_bundled.py"
        tool_file.write_text(tool_src)

        # Build the fake module and inject it so importlib.import_module finds it
        # under the name _load_bundled will request.
        module_name = "ai_cli.tools.non_standard_bundled"
        fake_module = types.ModuleType(module_name)
        exec(compile(tool_src, str(tool_file), "exec"), fake_module.__dict__)  # noqa: S102
        sys.modules[module_name] = fake_module

        try:
            reg = self._make_bare_registry(tmp_path)
            reg._load_bundled(tmp_path)  # exercises the real production path
        finally:
            sys.modules.pop(module_name, None)

        assert reg.get("non_standard_bundled") is None

    def test_load_from_file_skips_register_via_instance(self, tmp_path):
        """_load_from_file must not attempt to instantiate a class marked
        REGISTER_VIA_INSTANCE=True discovered in a user tool file."""
        tool_src = textwrap.dedent("""\
            from typing import Any
            from ai_cli.tools.base import Tool, ToolSchema

            class _NonStandardFileTool(Tool):
                NAME = "non_standard_file"
                DESCRIPTION = "Non-standard constructor tool."
                PERMISSION_REQUIRED = False
                REGISTER_VIA_INSTANCE = True

                def __init__(self, workspace, permission_manager, extra_arg):
                    raise AssertionError("should never be instantiated by the loader")

                def definition(self):
                    return ToolSchema(name=self.name, description=self.description)

                def execute(self, **kwargs: Any) -> dict:
                    return self._ok()
        """)
        tool_file = tmp_path / "non_standard_file_tool.py"
        tool_file.write_text(tool_src)

        reg = self._make_bare_registry(tmp_path)
        # Must not raise; tool must not be registered.
        reg._load_from_file(tool_file, tier="user")

        assert reg.get("non_standard_file") is None
