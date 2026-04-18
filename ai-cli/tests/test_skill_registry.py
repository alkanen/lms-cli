"""Tests for ai_cli.core.skill_registry."""

from pathlib import Path

import pytest

from ai_cli.core.skill_registry import (
    MAX_DESCRIPTION_CHARS,
    MAX_SKILL_FILE_BYTES,
    MAX_SKILL_NAME_CHARS,
    SKILL_FILENAME,
    SkillRegistry,
)


def _write_skill(
    skill_dir: Path,
    *,
    name: str = "example",
    description: str = "desc",
    instructions: str = "Do X",
) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    body = f"---\nname: {name}\ndescription: {description}\n---\n{instructions}\n"
    (skill_dir / SKILL_FILENAME).write_text(body, encoding="utf-8")


@pytest.fixture()
def global_dir(tmp_path: Path) -> Path:
    d = tmp_path / "global"
    d.mkdir()
    return d


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    p = tmp_path / "project"
    (p / ".ai-cli").mkdir(parents=True)
    return p


class TestSkillRegistryValidation:
    def test_missing_skill_md_skipped_with_warning(self, project_root: Path, caplog):
        (project_root / ".ai-cli" / "skills" / "missing").mkdir(parents=True)
        with caplog.at_level("WARNING"):
            reg = SkillRegistry.load(project_root, global_dir=project_root / "unused")
        assert reg.skills == {}
        assert any("missing SKILL.md" in w for w in reg.warnings)
        assert "missing SKILL.md" in caplog.text

    def test_malformed_frontmatter_skipped_with_warning(
        self, project_root: Path, caplog
    ):
        sdir = project_root / ".ai-cli" / "skills" / "bad"
        sdir.mkdir(parents=True)
        (sdir / SKILL_FILENAME).write_text("name: no-fm\n", encoding="utf-8")
        with caplog.at_level("WARNING"):
            reg = SkillRegistry.load(project_root, global_dir=project_root / "unused")
        assert reg.skills == {}
        assert any("malformed YAML frontmatter" in w for w in reg.warnings)
        assert "malformed YAML frontmatter" in caplog.text

    def test_missing_required_name_skipped_with_warning(
        self, project_root: Path, caplog
    ):
        sdir = project_root / ".ai-cli" / "skills" / "bad"
        sdir.mkdir(parents=True)
        (sdir / SKILL_FILENAME).write_text(
            "---\ndescription: nope\n---\nbody\n", encoding="utf-8"
        )
        with caplog.at_level("WARNING"):
            reg = SkillRegistry.load(project_root, global_dir=project_root / "unused")
        assert reg.skills == {}
        assert any("required frontmatter field 'name'" in w for w in reg.warnings)
        assert "required frontmatter field 'name'" in caplog.text

    def test_missing_required_description_skipped_with_warning(
        self, project_root: Path, caplog
    ):
        sdir = project_root / ".ai-cli" / "skills" / "bad"
        sdir.mkdir(parents=True)
        (sdir / SKILL_FILENAME).write_text(
            "---\nname: ok\n---\nbody\n", encoding="utf-8"
        )
        with caplog.at_level("WARNING"):
            reg = SkillRegistry.load(project_root, global_dir=project_root / "unused")
        assert reg.skills == {}
        assert any(
            "required frontmatter field 'description'" in w for w in reg.warnings
        )
        assert "required frontmatter field 'description'" in caplog.text

    def test_description_too_long_skipped_with_warning(
        self, project_root: Path, caplog
    ):
        too_long = "x" * (MAX_DESCRIPTION_CHARS + 1)
        _write_skill(
            project_root / ".ai-cli" / "skills" / "bad",
            name="bad",
            description=too_long,
        )
        with caplog.at_level("WARNING"):
            reg = SkillRegistry.load(project_root, global_dir=project_root / "unused")
        assert reg.skills == {}
        assert any(f"exceeds {MAX_DESCRIPTION_CHARS}" in w for w in reg.warnings)
        assert f"exceeds {MAX_DESCRIPTION_CHARS}" in caplog.text

    def test_skill_md_size_cap_skipped_with_warning(self, project_root: Path, caplog):
        sdir = project_root / ".ai-cli" / "skills" / "oversize"
        sdir.mkdir(parents=True)
        big = "-" * (MAX_SKILL_FILE_BYTES + 1)
        (sdir / SKILL_FILENAME).write_text(big, encoding="utf-8")
        with caplog.at_level("WARNING"):
            reg = SkillRegistry.load(project_root, global_dir=project_root / "unused")
        assert reg.skills == {}
        assert any(f"exceeds {MAX_SKILL_FILE_BYTES}" in w for w in reg.warnings)
        assert f"exceeds {MAX_SKILL_FILE_BYTES}" in caplog.text

    def test_folder_name_mismatch_warns_but_loads(self, project_root: Path, caplog):
        _write_skill(
            project_root / ".ai-cli" / "skills" / "folder-name",
            name="canonical-name",
            description="ok",
        )
        with caplog.at_level("WARNING"):
            reg = SkillRegistry.load(project_root, global_dir=project_root / "unused")
        assert "canonical-name" in reg.skills
        assert any("folder name differs" in w for w in reg.warnings)
        assert "folder name differs" in caplog.text

    def test_invalid_skill_name_chars_skipped_with_warning(
        self, project_root: Path, caplog
    ):
        _write_skill(
            project_root / ".ai-cli" / "skills" / "bad-name",
            name="Bad_Name",
            description="ok",
        )
        with caplog.at_level("WARNING"):
            reg = SkillRegistry.load(project_root, global_dir=project_root / "unused")
        assert reg.skills == {}
        assert any("lowercase letters, numbers, and hyphens" in w for w in reg.warnings)

    def test_skill_name_surrounding_whitespace_is_normalized(self, project_root: Path):
        _write_skill(
            project_root / ".ai-cli" / "skills" / "alpha",
            name="  alpha  ",
            description="ok",
        )
        reg = SkillRegistry.load(project_root, global_dir=project_root / "unused")
        assert reg.get("alpha") is not None

    def test_skill_name_too_long_skipped_with_warning(self, project_root: Path, caplog):
        too_long_name = "a" * (MAX_SKILL_NAME_CHARS + 1)
        _write_skill(
            project_root / ".ai-cli" / "skills" / "bad-name",
            name=too_long_name,
            description="ok",
        )
        with caplog.at_level("WARNING"):
            reg = SkillRegistry.load(project_root, global_dir=project_root / "unused")
        assert reg.skills == {}
        assert any("at most 64 characters" in w for w in reg.warnings)


class TestSkillRegistryPrecedence:
    def test_project_overrides_global_on_canonical_name_collision(
        self,
        global_dir: Path,
        project_root: Path,
        caplog,
    ):
        _write_skill(
            global_dir / "skills" / "shared_global",
            name="shared",
            description="global description",
            instructions="global instructions",
        )
        _write_skill(
            project_root / ".ai-cli" / "skills" / "shared_project",
            name="shared",
            description="project description",
            instructions="project instructions",
        )

        with caplog.at_level("WARNING"):
            reg = SkillRegistry.load(project_root, global_dir=global_dir)

        skill = reg.get("shared")
        assert skill is not None
        assert skill.scope == "project"
        assert skill.description == "project description"
        assert "project scope overrides global definition" in caplog.text

    def test_canonical_name_mapping_is_deterministic(self, project_root: Path):
        _write_skill(
            project_root / ".ai-cli" / "skills" / "z_last",
            name="zeta",
            description="z",
        )
        _write_skill(
            project_root / ".ai-cli" / "skills" / "a_first",
            name="alpha",
            description="a",
        )

        reg1 = SkillRegistry.load(project_root, global_dir=project_root / "unused")
        reg2 = SkillRegistry.load(project_root, global_dir=project_root / "unused")

        keys1 = list(reg1.skills.keys())
        keys2 = list(reg2.skills.keys())
        assert keys1 == keys2
        assert set(keys1) == {"alpha", "zeta"}


class TestSkillRegistryLen:
    def test_len_returns_skill_count_without_needing_skills_copy(
        self, project_root: Path
    ):
        _write_skill(
            project_root / ".ai-cli" / "skills" / "alpha",
            name="alpha",
            description="a",
        )
        _write_skill(
            project_root / ".ai-cli" / "skills" / "beta",
            name="beta",
            description="b",
        )

        reg = SkillRegistry.load(project_root, global_dir=project_root / "unused")

        assert len(reg) == 2

    def test_names_returns_live_view_without_copying(self, project_root: Path):
        _write_skill(
            project_root / ".ai-cli" / "skills" / "alpha",
            name="alpha",
            description="a",
        )

        reg = SkillRegistry.load(project_root, global_dir=project_root / "unused")

        assert list(reg.names()) == ["alpha"]

    def test_items_returns_live_view_without_copying(self, project_root: Path):
        _write_skill(
            project_root / ".ai-cli" / "skills" / "alpha",
            name="alpha",
            description="a",
        )

        reg = SkillRegistry.load(project_root, global_dir=project_root / "unused")

        assert [(name, spec.name) for name, spec in reg.items()] == [("alpha", "alpha")]
