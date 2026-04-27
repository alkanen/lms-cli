"""Tests for ai_cli.tools.base.Tool."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

# ---------------------------------------------------------------------------
# Minimal concrete tool for testing
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    """A minimal Tool subclass that echoes its input."""

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    name="message",
                    description="Text to echo.",
                    argument_type="string",
                    required=True,
                )
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        return self._ok({"echo": kwargs.get("message", "")})


def make_tool(
    permission_required: bool = False,
) -> tuple[EchoTool, MagicMock, MagicMock]:
    workspace = MagicMock()
    pm = MagicMock()
    pm.request.return_value = (True, "")
    tool = EchoTool(
        workspace=workspace,
        permission_manager=pm,
        permission_required=permission_required,
        name="echo",
        description="Echoes a message.",
    )
    return tool, workspace, pm


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_attributes_set(self):
        tool, _, _ = make_tool()
        assert tool.name == "echo"
        assert tool.description == "Echoes a message."
        assert tool.permission_required is False

    def test_definition_returns_schema(self):
        tool, _, _ = make_tool()
        d = tool.definition().schema()
        assert d["type"] == "function"
        assert d["function"]["name"] == "echo"

    def test_execute_returns_ok(self):
        tool, _, _ = make_tool()
        result = tool.execute(message="hello")
        assert result["status"] == "success"
        assert result["data"]["echo"] == "hello"


# ---------------------------------------------------------------------------
# NotImplementedError for abstract methods
# ---------------------------------------------------------------------------


class TestAbstractMethods:
    def test_tool_cannot_be_instantiated_directly(self):
        workspace = MagicMock()
        pm = MagicMock()
        with pytest.raises(TypeError):
            Tool(workspace, pm, False, "base", "Base tool")  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# extra_permission_options default
# ---------------------------------------------------------------------------


class TestExtraPermissionOptions:
    def test_default_returns_empty_list(self):
        tool, _, _ = make_tool()
        assert tool.extra_permission_options() == []


# ---------------------------------------------------------------------------
# request_permission
# ---------------------------------------------------------------------------


class TestRequestPermission:
    def test_no_permission_required_skips_prompt(self):
        tool, _, pm = make_tool(permission_required=False)
        allowed, reason = tool.request_permission("Read file")
        assert allowed is True
        assert reason == ""
        pm.request.assert_not_called()

    def test_permission_required_calls_manager(self):
        tool, _, pm = make_tool(permission_required=True)
        pm.request.return_value = (True, "")
        allowed, reason = tool.request_permission("Read file")
        assert allowed is True
        pm.request.assert_called_once()

    def test_denial_propagated(self):
        tool, _, pm = make_tool(permission_required=True)
        pm.request.return_value = (False, "Permission denied.")
        allowed, reason = tool.request_permission("Read file")
        assert allowed is False
        assert reason == "Permission denied."

    def test_question_includes_tool_name(self):
        tool, _, pm = make_tool(permission_required=True)
        pm.request.return_value = (True, "")
        tool.request_permission("Read /etc/hosts")
        call_kwargs = pm.request.call_args
        assert "echo" in call_kwargs.kwargs["tool_name"]
        assert "Read /etc/hosts" in call_kwargs.kwargs["question"]

    def test_extra_options_forwarded(self):
        class FolderTool(EchoTool):
            def extra_permission_options(self, **kwargs: Any) -> list[str]:
                return ["always_in_this_folder"]

        workspace, pm_mock = MagicMock(), MagicMock()
        pm_mock.request.return_value = (True, "")
        tool = FolderTool(
            workspace, pm_mock, True, "folder_echo", "Echo with folder option."
        )
        tool.request_permission("Read file")
        pm_mock.request.assert_called_once()
        assert pm_mock.request.call_args.kwargs["extra_options"] == [
            "always_in_this_folder"
        ]


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


class TestFormatDisplay:
    def test_returns_none_by_default(self):
        tool, _, _ = make_tool()
        result = tool.format_display(
            args={"message": "hi"}, result={"status": "success"}
        )
        assert result is None

    def test_can_be_overridden_to_return_string(self):
        class FancyTool(EchoTool):
            def format_display(self, *, args, result):
                return f"echo: {args.get('message', '')}"

        workspace, pm = MagicMock(), MagicMock()
        pm.request.return_value = (True, "")
        tool = FancyTool(workspace, pm, False, "fancy_echo", "Fancy echo.")
        out = tool.format_display(args={"message": "hi"}, result={})
        assert out == "echo: hi"


class TestResultHelpers:
    def test_ok_empty_data(self):
        result = Tool._ok()
        assert result == {"status": "success", "data": {}}

    def test_ok_with_data(self):
        result = Tool._ok({"key": "value"})
        assert result["data"] == {"key": "value"}

    def test_err_minimal(self):
        result = Tool._err("not_found", "File not found.", 404)
        assert result["status"] == "error"
        assert result["error"] == "not_found"
        assert result["message"] == "File not found."
        assert result["code"] == 404
        assert "details" not in result

    def test_err_with_details(self):
        result = Tool._err("bad_input", "Invalid range.", 400, {"field": "start_line"})
        assert result["details"] == {"field": "start_line"}

    def test_err_with_empty_dict_details_included(self):
        result = Tool._err("bad_input", "Invalid range.", 400, {})
        assert "details" in result
        assert result["details"] == {}


class TestTypedErrorWrappers:
    """Each wrapper must encode the correct error code and HTTP status."""

    def _assert_error(
        self,
        result: dict,
        *,
        error: str,
        code: int,
        message: str = "msg",
    ) -> None:
        assert result["status"] == "error"
        assert result["error"] == error
        assert result["code"] == code
        assert result["message"] == message
        assert "details" not in result

    def test_err_invalid_arguments(self):
        self._assert_error(
            Tool._err_invalid_arguments("msg"),
            error="invalid_arguments",
            code=400,
        )

    def test_err_invalid_range(self):
        self._assert_error(
            Tool._err_invalid_range("msg"),
            error="invalid_range",
            code=400,
        )

    def test_err_read_error(self):
        self._assert_error(
            Tool._err_read_error("msg"),
            error="read_error",
            code=400,
        )

    def test_err_write_error(self):
        self._assert_error(
            Tool._err_write_error("msg"),
            error="write_error",
            code=400,
        )

    def test_err_execution_error(self):
        self._assert_error(
            Tool._err_execution_error("msg"),
            error="execution_error",
            code=400,
        )

    def test_err_timeout(self):
        self._assert_error(
            Tool._err_timeout("msg"),
            error="timeout",
            code=408,
        )

    def test_err_internal_error(self):
        self._assert_error(
            Tool._err_internal_error("msg"),
            error="internal_error",
            code=500,
        )

    def test_details_forwarded_when_provided(self):
        result = Tool._err_invalid_arguments("bad", {"field": "path"})
        assert result["details"] == {"field": "path"}

    def test_details_absent_when_not_provided(self):
        result = Tool._err_timeout("timed out")
        assert "details" not in result
