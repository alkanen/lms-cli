"""
CallAgentTool — delegate a focused task to a specialised sub-agent.

The tool is registered in the coordinator's ``ToolRegistry`` only when at
least one agent spec is present in the project config.  The LLM selects an
agent type by name; ``AgentRegistry.get_or_create()`` builds or retrieves the
corresponding ``Agent``; ``Agent.run()`` drives the sub-agent's send/tool loop;
the result is returned in the canonical tool-response format so the coordinator
can incorporate it into the conversation.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import TYPE_CHECKING

from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

if TYPE_CHECKING:
    from ai_cli.core.agent_registry import AgentRegistry
    from ai_cli.core.config_manager import ConfigManager
    from ai_cli.core.llm_client import LLMClient
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.tool_registry import ToolRegistry
    from ai_cli.core.workspace import Workspace

logger = logging.getLogger(__name__)

# Maximum characters of an agent's system_message shown as its description
# in the dynamic tool description.
_DESC_MAX_CHARS = 80


class CallAgentTool(Tool):
    """Delegate a focused task to a specialised sub-agent.

    Unlike bundled tools, ``CallAgentTool`` requires non-standard constructor
    arguments and must be registered via
    :meth:`~ai_cli.core.tool_registry.ToolRegistry.register_instance` rather
    than the three-tier file loader.
    """

    NAME = "call_agent"
    DESCRIPTION = "Delegate a focused task to a specialised sub-agent."
    PERMISSION_REQUIRED = False

    def __init__(
        self,
        workspace: Workspace,
        permission_manager: PermissionManager,
        agent_registry: AgentRegistry,
        config: ConfigManager,
        coordinator_llm: LLMClient,
        global_tool_registry: ToolRegistry,
    ) -> None:
        super().__init__(
            workspace,
            permission_manager,
            self.PERMISSION_REQUIRED,
            self.NAME,
            self.DESCRIPTION,
        )
        self._agent_registry = agent_registry
        self._config = config
        self._coordinator_llm = coordinator_llm
        self._global_tool_registry = global_tool_registry

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.NAME,
            description=self._build_description(),
            arguments=[
                ToolArgument(
                    "agent_type",
                    "Name of the agent type to delegate the task to.",
                    "string",
                    required=True,
                    enum=sorted(self._agent_registry.specs),
                ),
                ToolArgument(
                    "prompt",
                    "The task or question for the agent.",
                    "string",
                    required=True,
                ),
            ],
        )

    def execute(self, **kwargs: object) -> dict:
        agent_type = kwargs.get("agent_type")
        prompt = kwargs.get("prompt")

        if (
            not isinstance(agent_type, str)
            or agent_type not in self._agent_registry.specs
        ):
            return self._err(
                "invalid_agent_type",
                f"Unknown agent type {agent_type!r}. "
                f"Available: {sorted(self._agent_registry.specs)}.",
                400,
            )

        if not isinstance(prompt, str):
            return self._err("invalid_arguments", "'prompt' must be a string.", 400)

        try:
            agent = self._agent_registry.get_or_create(
                agent_type,
                workspace=self._workspace,
                config=self._config,
                coordinator_llm=self._coordinator_llm,
                global_tool_registry=self._global_tool_registry,
            )
        except KeyError as exc:
            return self._err("invalid_agent_type", str(exc), 400)
        except Exception as exc:
            logger.exception("Failed to build agent %r: %s", agent_type, exc)
            return self._err(
                "agent_build_error",
                f"Failed to initialize agent {agent_type!r}: {exc}",
                500,
            )

        try:
            t0 = time.monotonic()
            result = agent.run(prompt)
            elapsed = time.monotonic() - t0
            logger.info(
                "Agent dispatch: agent=%r prompt_len=%d status=%r elapsed=%.3fs",
                agent_type,
                len(prompt),
                result.status,
                elapsed,
            )
        except Exception as exc:
            logger.exception("Unexpected error running agent %r: %s", agent_type, exc)
            return self._err(
                "agent_run_error",
                f"Agent {agent_type!r} raised an unexpected error: {exc}",
                500,
            )

        return self._ok(
            {
                "result": result.text,
                "agent_status": result.status,
                "partial": result.partial,
                "error_message": result.error_message,
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_description(self) -> str:
        """Build a dynamic description listing all configured agent types.

        Only tools that are actually available in the global registry are
        shown — tools listed in spec but not registered are silently omitted
        so the coordinator is not misled about an agent's real capabilities.
        """
        lines = [self.DESCRIPTION, "", "Available agent types:"]
        for name, spec in sorted(self._agent_registry.specs.items()):
            # Use the first sentence of system_message as a brief description.
            first_sentence = spec.system_message.split(".")[0].strip()
            if len(first_sentence) > _DESC_MAX_CHARS:
                first_sentence = first_sentence[: _DESC_MAX_CHARS - 3] + "..."
            available_tools = [
                t
                for t in spec.tools
                # Skip self to prevent recursive definition() → _build_description()
                # → tool_info(self.NAME) → definition() loop.
                if t != self.NAME and self._global_tool_registry.is_allowed(t)
            ]
            tools_str = ", ".join(available_tools) if available_tools else "(none)"
            lines.append(f"  {name:<14}{first_sentence}")
            lines.append(f"  {'':>14}Tools: {tools_str}")
        return "\n".join(lines)


class CallAgentsParallelTool(Tool):
    """Run multiple sub-agent tasks in parallel and return all results.

    Like :class:`CallAgentTool`, this tool requires non-standard constructor
    arguments and must be registered via
    :meth:`~ai_cli.core.tool_registry.ToolRegistry.register_instance`.
    """

    NAME = "call_agents_parallel"
    DESCRIPTION = "Run multiple sub-agent tasks in parallel and return all results."
    PERMISSION_REQUIRED = False

    def __init__(
        self,
        workspace: Workspace,
        permission_manager: PermissionManager,
        agent_registry: AgentRegistry,
        config: ConfigManager,
        coordinator_llm: LLMClient,
        global_tool_registry: ToolRegistry,
    ) -> None:
        super().__init__(
            workspace,
            permission_manager,
            self.PERMISSION_REQUIRED,
            self.NAME,
            self.DESCRIPTION,
        )
        self._agent_registry = agent_registry
        self._config = config
        self._coordinator_llm = coordinator_llm
        self._global_tool_registry = global_tool_registry

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.NAME,
            description=self.DESCRIPTION,
            arguments=[
                ToolArgument(
                    "calls",
                    "List of agent calls to run in parallel. Each item must have "
                    "'agent_type' (string) and 'prompt' (string).",
                    "array",
                    required=True,
                    items={
                        "type": "object",
                        "properties": {
                            "agent_type": {
                                "type": "string",
                                "description": "Name of the agent type to delegate to.",
                                "enum": sorted(self._agent_registry.specs),
                            },
                            "prompt": {
                                "type": "string",
                                "description": "The task or question for the agent.",
                            },
                        },
                        "required": ["agent_type", "prompt"],
                    },
                ),
            ],
        )

    # Default maximum number of parallel calls allowed per invocation.
    # Overridable via agent_settings.max_parallel_calls in config.
    _DEFAULT_MAX_PARALLEL_CALLS = 10

    def execute(self, **kwargs: object) -> dict:
        calls = kwargs.get("calls")
        if not isinstance(calls, list):
            return self._err(
                "invalid_arguments", "'calls' must be a list of objects.", 400
            )

        # Enforce a configurable maximum batch size to prevent runaway resource use.
        agent_settings = self._config.get("agent_settings") or {}
        max_calls: int = self._DEFAULT_MAX_PARALLEL_CALLS
        if isinstance(agent_settings, dict):
            cfg_max = agent_settings.get("max_parallel_calls")
            if (
                isinstance(cfg_max, int)
                and not isinstance(cfg_max, bool)
                and cfg_max > 0
            ):
                max_calls = cfg_max
        if len(calls) > max_calls:
            return self._err(
                "invalid_arguments",
                f"'calls' contains {len(calls)} items but the maximum allowed is "
                f"{max_calls} (agent_settings.max_parallel_calls).",
                400,
            )

        # Snapshot specs once — AgentRegistry.specs returns a copy each time.
        specs = self._agent_registry.specs

        # Validate all items before dispatching any.
        seen_session_agents: set[str] = set()
        for i, item in enumerate(calls):
            if not isinstance(item, dict):
                return self._err(
                    "invalid_arguments",
                    f"calls[{i}] must be an object with 'agent_type' and 'prompt'.",
                    400,
                )
            agent_type = item.get("agent_type")
            prompt = item.get("prompt")
            if not isinstance(agent_type, str) or agent_type not in specs:
                return self._err(
                    "invalid_agent_type",
                    f"calls[{i}]: unknown agent type {agent_type!r}. "
                    f"Available: {sorted(specs)}.",
                    400,
                )
            if not isinstance(prompt, str):
                return self._err(
                    "invalid_arguments",
                    f"calls[{i}]: 'prompt' must be a string.",
                    400,
                )
            # Session-persistent agents share state and cannot safely be run
            # concurrently — reject duplicate session agent types up front.
            if specs[agent_type].persistence == "session":
                if agent_type in seen_session_agents:
                    return self._err(
                        "invalid_arguments",
                        f"calls[{i}]: agent type {agent_type!r} appears more than once. "
                        f"Session-persistent agents cannot be run concurrently.",
                        400,
                    )
                seen_session_agents.add(agent_type)

        def _run_one(index: int, agent_type: str, prompt: str) -> tuple[int, dict]:
            try:
                agent = self._agent_registry.get_or_create(
                    agent_type,
                    workspace=self._workspace,
                    config=self._config,
                    coordinator_llm=self._coordinator_llm,
                    global_tool_registry=self._global_tool_registry,
                )
            except Exception as exc:
                logger.exception("Failed to build agent %r: %s", agent_type, exc)
                return index, {
                    "agent_type": agent_type,
                    "agent_status": "error",
                    "result": "",
                    "partial": True,
                    "error_message": f"Failed to initialize agent {agent_type!r}: {exc}",
                }

            try:
                t0 = time.monotonic()
                result = agent.run(prompt)
                elapsed = time.monotonic() - t0
                logger.info(
                    "Agent dispatch: agent=%r prompt_len=%d status=%r elapsed=%.3fs",
                    agent_type,
                    len(prompt),
                    result.status,
                    elapsed,
                )
            except Exception as exc:
                logger.exception(
                    "Unexpected error running agent %r: %s", agent_type, exc
                )
                return index, {
                    "agent_type": agent_type,
                    "agent_status": "error",
                    "result": "",
                    "partial": True,
                    "error_message": f"Agent {agent_type!r} raised an unexpected error: {exc}",
                }

            return index, {
                "agent_type": agent_type,
                "result": result.text,
                "agent_status": result.status,
                "partial": result.partial,
                "error_message": result.error_message,
            }

        indexed_results: list[tuple[int, dict]] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_run_one, i, item["agent_type"], item["prompt"])
                for i, item in enumerate(calls)
            ]
            for future in concurrent.futures.as_completed(futures):
                indexed_results.append(future.result())

        # Restore input order.
        indexed_results.sort(key=lambda x: x[0])
        results = [r for _, r in indexed_results]
        return self._ok({"results": results})
