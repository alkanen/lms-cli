"""Tests for ai_cli.tools.skills.SkillsTool."""

from pathlib import Path
from unittest.mock import MagicMock

from ai_cli.core.skill_registry import SkillRegistry, SkillSpec
from ai_cli.tools.skills import SkillsTool


def _registry_with_skills(tmp_path: Path) -> SkillRegistry:
    alpha_dir = tmp_path / "skills" / "alpha"
    beta_dir = tmp_path / "skills" / "beta"
    alpha_dir.mkdir(parents=True)
    beta_dir.mkdir(parents=True)

    return SkillRegistry(
        {
            "alpha": SkillSpec(
                name="alpha",
                description="Alpha skill",
                instructions="Do alpha tasks.",
                base_dir=alpha_dir,
                scope="project",
            ),
            "beta": SkillSpec(
                name="beta",
                description="Beta skill",
                instructions="Do beta tasks.",
                base_dir=beta_dir,
                scope="project",
            ),
        }
    )


class TestSkillsToolSchema:
    def test_schema(self, tmp_path: Path):
        tool = SkillsTool(_registry_with_skills(tmp_path), MagicMock(), MagicMock())
        schema = tool.definition().schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "skills"
        assert schema["function"]["parameters"]["required"] == ["name"]
        assert (
            schema["function"]["parameters"]["properties"]["name"]["type"] == "string"
        )


class TestSkillsToolExecute:
    def test_found_payload(self, tmp_path: Path):
        reg = _registry_with_skills(tmp_path)
        tool = SkillsTool(reg, MagicMock(), MagicMock())

        result = tool.execute(name="alpha")

        assert result["status"] == "success"
        data = result["data"]
        assert data["name"] == "alpha"
        assert data["description"] == "Alpha skill"
        assert data["instructions"] == "Do alpha tasks."
        assert data["base_dir"] == str(reg.get("alpha").base_dir)
        assert "found" not in data

    def test_not_found_payload(self, tmp_path: Path):
        tool = SkillsTool(_registry_with_skills(tmp_path), MagicMock(), MagicMock())

        result = tool.execute(name="missing")

        assert result["status"] == "success"
        assert result["data"] == {
            "found": False,
            "requested_name": "missing",
            "available_skills": ["alpha", "beta"],
        }

    def test_name_is_exact_match(self, tmp_path: Path):
        tool = SkillsTool(_registry_with_skills(tmp_path), MagicMock(), MagicMock())

        result = tool.execute(name="Alpha")

        assert result["status"] == "success"
        assert result["data"]["found"] is False
        assert result["data"]["requested_name"] == "Alpha"

    def test_missing_name_returns_invalid_arguments(self, tmp_path: Path):
        tool = SkillsTool(_registry_with_skills(tmp_path), MagicMock(), MagicMock())

        result = tool.execute()

        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert result["code"] == 400
        assert "must be a string" in result["message"]

    def test_blank_name_returns_invalid_arguments(self, tmp_path: Path):
        tool = SkillsTool(_registry_with_skills(tmp_path), MagicMock(), MagicMock())

        result = tool.execute(name="   ")

        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"
        assert result["code"] == 400
        assert "non-empty" in result["message"]
