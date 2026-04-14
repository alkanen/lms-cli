"""Tests for CallAgentTool, CallAgentsParallelTool and the end-to-end agent dispatch path."""

from unittest.mock import MagicMock, patch

from ai_cli.core.agent import AgentResult, AgentSpec
from ai_cli.core.agent_registry import AgentRegistry
from ai_cli.tools.base import Tool, ToolSchema
from ai_cli.tools.call_agent import CallAgentsParallelTool, CallAgentTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    name: str,
    *,
    tools: list[str] | None = None,
    persistence: str = "ephemeral",
    system_message: str = "You are a helper. Do what is asked.",
) -> AgentSpec:
    return AgentSpec(
        name=name,
        system_message=system_message,
        tools=tools or ["read_file"],
        model="llama3:8b",
        persistence=persistence,
    )


def _make_tool(
    specs: dict[str, AgentSpec] | None = None,
    *,
    tool_info_allowed: bool = False,
) -> tuple[CallAgentTool, AgentRegistry]:
    """Return a CallAgentTool backed by mocked dependencies.

    *tool_info_allowed* controls whether ``global_tool_registry.is_allowed()``
    returns ``True`` (simulating available tools) or ``False``
    (simulating unknown/disallowed tools).
    """
    if specs is None:
        specs = {"explore": _make_spec("explore", tools=["read_file"])}
    registry = AgentRegistry(specs)
    workspace = MagicMock()
    permission_manager = MagicMock()
    config = MagicMock()
    coordinator_llm = MagicMock()
    global_tool_registry = MagicMock()
    global_tool_registry.is_allowed.return_value = tool_info_allowed
    tool = CallAgentTool(
        workspace,
        permission_manager,
        registry,
        config,
        coordinator_llm,
        global_tool_registry,
    )
    return tool, registry


# ---------------------------------------------------------------------------
# Description building
# ---------------------------------------------------------------------------


class TestBuildDescription:
    def test_includes_base_text(self):
        tool, _ = _make_tool()
        desc = tool._build_description()
        assert "Delegate" in desc
        assert "sub-agent" in desc

    def test_includes_agent_names(self):
        specs = {
            "explore": _make_spec("explore", tools=["read_file", "find_files"]),
            "coder": _make_spec("coder", tools=["write_file"]),
        }
        tool, _ = _make_tool(specs, tool_info_allowed=True)
        desc = tool._build_description()
        assert "explore" in desc
        assert "coder" in desc

    def test_includes_tool_names(self):
        specs = {
            "explore": _make_spec("explore", tools=["read_file", "find_files"]),
        }
        tool, _ = _make_tool(specs, tool_info_allowed=True)
        desc = tool._build_description()
        assert "read_file" in desc
        assert "find_files" in desc

    def test_excludes_disallowed_tool_names(self):
        specs = {
            "explore": _make_spec("explore", tools=["read_file", "find_files"]),
        }

        # read_file allowed, find_files disallowed
        tool, _ = _make_tool(specs)
        tool._global_tool_registry.is_allowed.side_effect = lambda name: (
            name == "read_file"
        )
        desc = tool._build_description()
        assert "read_file" in desc
        assert "find_files" not in desc

    def test_description_rebuilt_on_each_call(self):
        """description() reflects current allowed state, not a stale snapshot."""
        specs = {"explore": _make_spec("explore", tools=["read_file"])}
        tool, _ = _make_tool(specs, tool_info_allowed=True)
        desc1 = tool._build_description()
        assert "read_file" in desc1
        # Now simulate the tool being disallowed at runtime.
        tool._global_tool_registry.is_allowed.return_value = False
        desc2 = tool._build_description()
        assert "read_file" not in desc2

    def test_self_reference_in_spec_tools_does_not_recurse(self):
        """call_agent listed in spec.tools must not cause infinite recursion."""
        specs = {"explore": _make_spec("explore", tools=["call_agent", "read_file"])}
        tool, _ = _make_tool(specs, tool_info_allowed=True)
        # Should not raise RecursionError; call_agent is silently excluded.
        desc = tool._build_description()
        assert "call_agent" not in desc
        assert "read_file" in desc

    def test_system_message_excerpt_in_description(self):
        specs = {
            "helper": _make_spec("helper", system_message="Analyse code carefully.")
        }
        tool, _ = _make_tool(specs)
        assert "Analyse code carefully" in tool._build_description()

    def test_long_system_message_truncated(self):
        long_msg = "A" * 200 + " extra text."
        specs = {"big": _make_spec("big", system_message=long_msg)}
        tool, _ = _make_tool(specs)
        assert len(tool._build_description()) < 500


# ---------------------------------------------------------------------------
# definition() / ToolSchema
# ---------------------------------------------------------------------------


class TestDefinition:
    def test_returns_tool_schema(self):
        tool, _ = _make_tool()
        from ai_cli.tools.base import ToolSchema

        assert isinstance(tool.definition(), ToolSchema)

    def test_schema_has_correct_name(self):
        tool, _ = _make_tool()
        schema = tool.definition().schema()
        assert schema["function"]["name"] == "call_agent"

    def test_agent_type_enum_matches_registry(self):
        specs = {
            "explore": _make_spec("explore"),
            "coder": _make_spec("coder"),
        }
        tool, _ = _make_tool(specs)
        schema = tool.definition().schema()
        props = schema["function"]["parameters"]["properties"]
        assert "agent_type" in props
        assert set(props["agent_type"]["enum"]) == {"explore", "coder"}

    def test_prompt_argument_present(self):
        tool, _ = _make_tool()
        schema = tool.definition().schema()
        props = schema["function"]["parameters"]["properties"]
        assert "prompt" in props

    def test_both_args_required(self):
        tool, _ = _make_tool()
        schema = tool.definition().schema()
        required = schema["function"]["parameters"]["required"]
        assert "agent_type" in required
        assert "prompt" in required

    def test_enum_sorted(self):
        specs = {
            "z_agent": _make_spec("z_agent"),
            "a_agent": _make_spec("a_agent"),
        }
        tool, _ = _make_tool(specs)
        schema = tool.definition().schema()
        enum_vals = schema["function"]["parameters"]["properties"]["agent_type"]["enum"]
        assert enum_vals == sorted(enum_vals)


# ---------------------------------------------------------------------------
# execute() — error cases
# ---------------------------------------------------------------------------


class TestExecuteErrors:
    def test_unknown_agent_type_returns_error(self):
        tool, _ = _make_tool()
        result = tool.execute(agent_type="nonexistent", prompt="hello")
        assert result["status"] == "error"
        assert "nonexistent" in result.get("message", "") or "nonexistent" in str(
            result
        )

    def test_none_agent_type_returns_error(self):
        tool, _ = _make_tool()
        result = tool.execute(agent_type=None, prompt="hello")
        assert result["status"] == "error"

    def test_non_string_prompt_returns_error(self):
        tool, _ = _make_tool()
        result = tool.execute(agent_type="explore", prompt=42)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# execute() — success cases
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    def test_returns_success_with_agent_result_fields(self):
        tool, registry = _make_tool()
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            text="Found 3 files.", status="ok", partial=False, error_message=""
        )
        with patch.object(registry, "get_or_create", return_value=mock_agent):
            result = tool.execute(agent_type="explore", prompt="find python files")

        assert result["status"] == "success"
        data = result["data"]
        assert data["result"] == "Found 3 files."
        assert data["agent_status"] == "ok"
        assert data["partial"] is False
        assert data["error_message"] == ""

    def test_passes_prompt_to_agent_run(self):
        tool, registry = _make_tool()
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(text="", status="ok")
        with patch.object(registry, "get_or_create", return_value=mock_agent):
            tool.execute(agent_type="explore", prompt="my task prompt")

        mock_agent.run.assert_called_once_with("my task prompt")

    def test_agent_error_status_still_returns_success(self):
        """Tool call succeeds even when agent's run() ends in error status."""
        tool, registry = _make_tool()
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            text="", status="error", partial=True, error_message="LLM failed"
        )
        with patch.object(registry, "get_or_create", return_value=mock_agent):
            result = tool.execute(agent_type="explore", prompt="do something")

        assert result["status"] == "success"
        assert result["data"]["agent_status"] == "error"
        assert result["data"]["error_message"] == "LLM failed"
        assert result["data"]["partial"] is True

    def test_agent_tool_limit_status_propagated(self):
        tool, registry = _make_tool()
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            text="partial result", status="tool_limit", partial=True
        )
        with patch.object(registry, "get_or_create", return_value=mock_agent):
            result = tool.execute(agent_type="explore", prompt="search everything")

        assert result["data"]["agent_status"] == "tool_limit"
        assert result["data"]["result"] == "partial result"

    def test_get_or_create_called_with_correct_args(self):
        tool, registry = _make_tool()
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(text="", status="ok")
        with patch.object(
            registry, "get_or_create", return_value=mock_agent
        ) as mock_get:
            tool.execute(agent_type="explore", prompt="test")

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs.args[0] == "explore"
        assert "workspace" in call_kwargs.kwargs
        assert "config" in call_kwargs.kwargs
        assert "coordinator_llm" in call_kwargs.kwargs
        assert "global_tool_registry" in call_kwargs.kwargs
        assert "coordinator_display" not in call_kwargs.kwargs

    def test_agent_run_exception_returns_error(self):
        """Unexpected exception from agent.run() is caught and returned as a tool error."""
        tool, registry = _make_tool()
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("something went very wrong")
        with patch.object(registry, "get_or_create", return_value=mock_agent):
            result = tool.execute(agent_type="explore", prompt="crash")

        assert result["status"] == "error"
        assert result["error"] == "agent_run_error"
        assert "explore" in result["message"]
        assert "something went very wrong" in result["message"]


# ---------------------------------------------------------------------------
# execute() — session-persistent vs ephemeral agents
# ---------------------------------------------------------------------------


class TestAgentPersistence:
    def test_ephemeral_agent_get_or_create_called_each_time(self):
        specs = {"explore": _make_spec("explore", persistence="ephemeral")}
        tool, registry = _make_tool(specs)
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(text="ok", status="ok")
        with patch.object(
            registry, "get_or_create", return_value=mock_agent
        ) as mock_get:
            tool.execute(agent_type="explore", prompt="first")
            tool.execute(agent_type="explore", prompt="second")

        assert mock_get.call_count == 2

    def test_session_agent_cached_in_registry(self):
        """For session-persistent agents, AgentRegistry caches the instance."""
        specs = {"helper": _make_spec("helper", persistence="session")}
        tool, registry = _make_tool(specs)
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(text="ok", status="ok")
        with patch.object(
            registry, "get_or_create", return_value=mock_agent
        ) as mock_get:
            tool.execute(agent_type="helper", prompt="first")
            tool.execute(agent_type="helper", prompt="second")

        # get_or_create is called twice — caching is internal to the registry
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Integration: real registry + mocked LLM
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_end_to_end_with_real_registry(self):
        """Build a real registry, mock only the innards, verify the call path."""
        specs = {"explore": _make_spec("explore", tools=["read_file"])}
        registry = AgentRegistry(specs)
        workspace = MagicMock()
        permission_manager = MagicMock()
        config = MagicMock()
        coordinator_llm = MagicMock()
        coordinator_llm.get_model_metadata.return_value = {
            "model": "test",
            "context_window": 4096,
            "max_response_tokens": 512,
        }
        global_tool_registry = MagicMock()
        global_tool_registry.get.return_value = None  # no real tools needed

        tool = CallAgentTool(
            workspace,
            permission_manager,
            registry,
            config,
            coordinator_llm,
            global_tool_registry,
        )

        # Mock Agent.run at class level so the real _build_agent path is exercised
        # up until the actual LLM call.
        from ai_cli.core.agent import Agent

        with patch.object(
            Agent,
            "run",
            return_value=AgentResult(text="result text", status="ok"),
        ):
            result = tool.execute(agent_type="explore", prompt="search for py files")

        assert result["status"] == "success"
        assert result["data"]["result"] == "result text"
        assert result["data"]["agent_status"] == "ok"


# ---------------------------------------------------------------------------
# build_agent_tool_registry — call_agent self-reference guard
# ---------------------------------------------------------------------------


class TestBuildAgentToolRegistry:
    def test_call_agent_in_spec_tools_is_skipped_silently(self):
        """call_agent listed in spec.tools must not produce warnings or errors.

        build_agent_tool_registry() cannot register CallAgentTool via the
        generic register() path (non-standard constructor), so it must skip
        'call_agent' explicitly instead of attempting registration and emitting
        a noisy warning on every sub-agent build.
        """
        from unittest.mock import patch

        from ai_cli.cli.display import SubAgentDisplay
        from ai_cli.core.agent import AgentSpec, build_agent_tool_registry

        spec = AgentSpec(
            name="loopy",
            system_message="I try to call myself.",
            tools=["call_agent", "read_file"],
            model="llama3:8b",
        )

        workspace = MagicMock()
        config = MagicMock()
        config.get.return_value = None
        display = SubAgentDisplay()
        global_tool_registry = MagicMock()
        # Simulate call_agent present but read_file absent in global registry.
        global_tool_registry.get.return_value = None

        with patch("ai_cli.core.agent.logger") as mock_logger:
            build_agent_tool_registry(
                spec, workspace, config, display, global_tool_registry
            )

        # No warning should mention call_agent — it is silently skipped.
        for call in mock_logger.warning.call_args_list:
            assert "call_agent" not in str(call)

    def test_instance_only_tools_are_cloned_per_registry(self):
        """Instance-only tools must not be shared across scoped registries.

        Permission overrides are applied by mutating ``permission_required``.
        If a shared object is reused, overrides leak across agents/coordinator.
        """
        from ai_cli.cli.display import SubAgentDisplay
        from ai_cli.core.agent import AgentSpec, build_agent_tool_registry

        class InstanceOnlyTool(Tool):
            NAME = "instance_only"
            DESCRIPTION = "Instance-only test tool"
            PERMISSION_REQUIRED = False
            REGISTER_VIA_INSTANCE = True

            def __init__(self, task_manager, workspace, permission_manager) -> None:
                super().__init__(
                    workspace,
                    permission_manager,
                    self.PERMISSION_REQUIRED,
                    self.NAME,
                    self.DESCRIPTION,
                )
                self._tm = task_manager

            def definition(self) -> ToolSchema:
                return ToolSchema(name=self.name, description=self.description)

            def execute(self, **kwargs: object) -> dict:
                return self._ok({"ok": True})

        spec = AgentSpec(
            name="loopy",
            system_message="I use one instance-only tool.",
            tools=["instance_only"],
            model="llama3:8b",
            tool_permission_overrides={"instance_only": True},
        )

        workspace = MagicMock()
        config = MagicMock()
        config.get.return_value = None
        display = SubAgentDisplay()

        task_manager = MagicMock()
        global_instance = InstanceOnlyTool(task_manager, MagicMock(), MagicMock())
        global_instance.permission_required = False

        global_tool_registry = MagicMock()
        global_tool_registry.get.return_value = global_instance

        registry = build_agent_tool_registry(
            spec, workspace, config, display, global_tool_registry
        )

        scoped_instance = registry.get("instance_only")
        assert scoped_instance is not None
        assert scoped_instance is not global_instance
        assert getattr(scoped_instance, "_tm", None) is task_manager
        assert global_instance.permission_required is False
        assert scoped_instance.permission_required is True


# ---------------------------------------------------------------------------
# CallAgentsParallelTool
# ---------------------------------------------------------------------------


def _make_parallel_tool(
    specs: dict[str, AgentSpec] | None = None,
) -> tuple[CallAgentsParallelTool, AgentRegistry]:
    if specs is None:
        specs = {
            "explore": _make_spec("explore"),
            "coder": _make_spec("coder"),
        }
    registry = AgentRegistry(specs)
    workspace = MagicMock()
    permission_manager = MagicMock()
    config = MagicMock()
    coordinator_llm = MagicMock()
    global_tool_registry = MagicMock()
    global_tool_registry.is_allowed.return_value = True
    tool = CallAgentsParallelTool(
        workspace,
        permission_manager,
        registry,
        config,
        coordinator_llm,
        global_tool_registry,
    )
    return tool, registry


class TestCallAgentsParallelTool:
    def test_two_calls_returned_in_input_order(self):
        """Two parallel calls return results in input order regardless of completion order."""
        tool, registry = _make_parallel_tool()

        mock_explore = MagicMock()
        mock_explore.run.return_value = AgentResult(
            text="explore result", status="ok", partial=False, error_message=""
        )
        mock_coder = MagicMock()
        mock_coder.run.return_value = AgentResult(
            text="coder result", status="ok", partial=False, error_message=""
        )

        def _get_or_create(name, **kwargs):
            return mock_explore if name == "explore" else mock_coder

        with patch.object(registry, "get_or_create", side_effect=_get_or_create):
            result = tool.execute(
                calls=[
                    {"agent_type": "explore", "prompt": "find files"},
                    {"agent_type": "coder", "prompt": "write code"},
                ]
            )

        assert result["status"] == "success"
        results = result["data"]["results"]
        assert len(results) == 2
        assert results[0]["agent_type"] == "explore"
        assert results[0]["result"] == "explore result"
        assert results[1]["agent_type"] == "coder"
        assert results[1]["result"] == "coder result"

    def test_calls_not_a_list_returns_error(self):
        tool, _ = _make_parallel_tool()
        result = tool.execute(calls="not a list")
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_exceeds_default_max_parallel_calls_returns_error(self):
        tool, _ = _make_parallel_tool()
        # 11 calls > default limit of 10
        calls = [{"agent_type": "explore", "prompt": f"task {i}"} for i in range(11)]
        result = tool.execute(calls=calls)
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert "10" in result["message"]

    def test_configurable_max_parallel_calls_respected(self):
        tool, registry = _make_parallel_tool()
        tool._config.get.return_value = {"max_parallel_calls": 3}
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(text="ok", status="ok")
        with patch.object(registry, "get_or_create", return_value=mock_agent):
            # 3 calls ≤ limit of 3 — should succeed
            calls = [{"agent_type": "explore", "prompt": f"task {i}"} for i in range(3)]
            result = tool.execute(calls=calls)
        assert result["status"] == "success"

    def test_exceeds_configurable_max_parallel_calls_returns_error(self):
        tool, _ = _make_parallel_tool()
        tool._config.get.return_value = {"max_parallel_calls": 2}
        calls = [{"agent_type": "explore", "prompt": f"task {i}"} for i in range(3)]
        result = tool.execute(calls=calls)
        assert result["status"] == "error"
        assert "2" in result["message"]

    def test_calls_none_returns_error(self):
        tool, _ = _make_parallel_tool()
        result = tool.execute(calls=None)
        assert result["status"] == "error"

    def test_item_with_unknown_agent_type_returns_error(self):
        tool, _ = _make_parallel_tool()
        result = tool.execute(
            calls=[{"agent_type": "nonexistent", "prompt": "do something"}]
        )
        assert result["status"] == "error"
        assert result["error"] == "invalid_agent_type"

    def test_item_with_non_string_prompt_returns_error(self):
        tool, _ = _make_parallel_tool()
        result = tool.execute(calls=[{"agent_type": "explore", "prompt": 42}])
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_item_not_a_dict_returns_error(self):
        tool, _ = _make_parallel_tool()
        result = tool.execute(calls=["not a dict"])
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_definition_schema_has_calls_array(self):
        tool, _ = _make_parallel_tool()
        schema = tool.definition().schema()
        props = schema["function"]["parameters"]["properties"]
        assert "calls" in props
        assert props["calls"]["type"] == "array"
        assert "items" in props["calls"]

    def test_definition_name_is_correct(self):
        tool, _ = _make_parallel_tool()
        schema = tool.definition().schema()
        assert schema["function"]["name"] == "call_agents_parallel"

    def test_duplicate_session_agent_returns_error(self):
        """Duplicate session-persistent agent in one parallel call is rejected."""
        specs = {
            "helper": _make_spec("helper", persistence="session"),
        }
        tool, _ = _make_parallel_tool(specs)
        result = tool.execute(
            calls=[
                {"agent_type": "helper", "prompt": "first"},
                {"agent_type": "helper", "prompt": "second"},
            ]
        )
        assert result["status"] == "error"
        assert "helper" in result["message"]

    def test_duplicate_ephemeral_agent_is_allowed(self):
        """Duplicate ephemeral agents in one parallel call are permitted (each gets a fresh instance)."""
        specs = {"explore": _make_spec("explore", persistence="ephemeral")}
        tool, registry = _make_parallel_tool(specs)
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(text="ok", status="ok")
        with patch.object(registry, "get_or_create", return_value=mock_agent):
            result = tool.execute(
                calls=[
                    {"agent_type": "explore", "prompt": "task one"},
                    {"agent_type": "explore", "prompt": "task two"},
                ]
            )
        assert result["status"] == "success"
        assert len(result["data"]["results"]) == 2
