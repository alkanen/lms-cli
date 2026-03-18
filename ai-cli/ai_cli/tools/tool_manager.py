"""
tool_manager — context-saving tool gatekeeper.

Exposes two actions:

  list    — return all available tools with name, description, and current
             enabled/disabled status so the LLM can make informed enable
             requests without receiving every tool's full schema upfront.

  enable  — request one or more tools for the immediately following LLM
             response only.  The REPL pops the returned ``transient_schemas``
             from the tool result (so they never enter conversation history)
             and injects them into the next API call's ``tools`` list.  They
             disappear automatically afterwards; no state change is made.

Workflow::

    # Step 1 — discover what's available
    tool_manager(action="list")

    # Step 2 — activate specific tools for the next response
    tool_manager(action="enable", tool_names=["find_files", "write_file"])

    # Step 3 — use the enabled tools; they vanish after this response
    find_files(pattern="**/*.py")

``tool_manager`` is enabled by default and requires no permission.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

if TYPE_CHECKING:
    from ai_cli.core.tool_registry import ToolRegistry


class ToolManagerTool(Tool):
    NAME = "tool_manager"
    DESCRIPTION = (
        "Manage available tools. "
        "Use action='list' to see all tools and their enabled/disabled status. "
        "Use action='enable' with a tool_names list to activate tools for the "
        "next response only — they disappear automatically afterwards."
    )
    PERMISSION_REQUIRED = False
    DISABLED_BY_DEFAULT = False

    def __init__(
        self,
        workspace: Any,
        permission_manager: Any,
        permission_required: bool,
        name: str,
        description: str,
    ) -> None:
        super().__init__(
            workspace, permission_manager, permission_required, name, description
        )
        self._registry: ToolRegistry | None = None

    def set_registry(self, registry: ToolRegistry) -> None:
        """Inject the registry after construction.  Called by ToolRegistry._register()."""
        self._registry = registry

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def definition(self) -> dict:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    name="action",
                    description=(
                        "'list' — show all available tools and their status. "
                        "'enable' — activate tools for the next response "
                        "(requires tool_names)."
                    ),
                    argument_type="string",
                    required=True,
                    enum=["list", "enable"],
                ),
                ToolArgument(
                    name="tool_names",
                    description=(
                        "Names of tools to activate. "
                        "Required when action='enable'; ignored otherwise. "
                        'Example: ["find_files", "write_file"].'
                    ),
                    argument_type="array",
                    items={"type": "string"},
                ),
            ],
        ).schema()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(  # type: ignore[override]
        self,
        *,
        action: str,
        tool_names: list[str] | None = None,
    ) -> dict:
        if action == "list":
            return self._do_list()
        if action == "enable":
            return self._do_enable(tool_names or [])
        return self._err(
            "invalid_input",
            f"Unknown action {action!r}.  Valid actions: 'list', 'enable'.",
            400,
        )

    def _do_list(self) -> dict:
        if self._registry is None:
            return self._err("internal_error", "Registry not available.", 500)
        return self._ok({"tools": self._registry.list_all()})

    def _do_enable(self, tool_names: list[str]) -> dict:
        if not tool_names:
            return self._err(
                "invalid_input",
                "'tool_names' must be a non-empty list when action='enable'.",
                400,
            )
        if self._registry is None:
            return self._err("internal_error", "Registry not available.", 500)

        enabled: list[str] = []
        unknown: list[str] = []
        transient_schemas: list[dict] = []

        for name in tool_names:
            schema = self._registry.enable_transient(name)
            if schema is None:
                unknown.append(name)
            else:
                enabled.append(name)
                transient_schemas.append(schema)

        # ``transient_schemas`` is consumed and removed by the REPL before the
        # result is added to conversation history, so the LLM never sees the
        # raw schema JSON.  It only sees the tidy ``enabled``/``unknown`` summary.
        data: dict = {"enabled": enabled, "transient_schemas": transient_schemas}
        if unknown:
            data["unknown"] = unknown
        return self._ok(data)
