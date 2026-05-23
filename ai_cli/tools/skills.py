"""Model-facing skills tool.

Exposes validated skills loaded by SkillRegistry to the coordinator model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_cli.core.skill_registry import SkillRegistry
from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

if TYPE_CHECKING:
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.workspace import Workspace


class SkillsTool(Tool):
    """Return skill instructions for an exact canonical skill name."""

    NAME = "skills"
    DESCRIPTION = "Get skill instructions by canonical skill name."
    PERMISSION_REQUIRED = False
    REGISTER_VIA_INSTANCE = True

    def __init__(
        self,
        skill_registry: SkillRegistry,
        workspace: Workspace,
        permission_manager: PermissionManager,
    ) -> None:
        super().__init__(
            workspace,
            permission_manager,
            self.PERMISSION_REQUIRED,
            self.NAME,
            self.DESCRIPTION,
        )
        self._skills = skill_registry

    def definition(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            arguments=[
                ToolArgument(
                    name="name",
                    description="Canonical skill name (exact match).",
                    argument_type="string",
                    required=True,
                )
            ],
        )

    def execute(self, **kwargs: Any) -> dict:
        requested_name = kwargs.get("name")
        if not isinstance(requested_name, str):
            return self._err_invalid_arguments("'name' must be a string.")
        if not requested_name.strip():
            return self._err_invalid_arguments("'name' must be a non-empty string.")

        skill = self._skills.get(requested_name)
        if skill is None:
            return self._ok(
                {
                    "found": False,
                    "requested_name": requested_name,
                    "available_skills": sorted(self._skills.skills.keys()),
                }
            )

        return self._ok(
            {
                "name": skill.name,
                "description": skill.description,
                "instructions": skill.instructions,
                "base_dir": str(skill.base_dir),
            }
        )
