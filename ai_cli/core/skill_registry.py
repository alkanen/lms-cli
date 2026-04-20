"""Skill registry discovery and validation.

PR1 scope:
- Discover skills from project/global scopes.
- Parse SKILL.md YAML frontmatter + instructions body.
- Validate required fields and hard caps.
- Apply project-over-global precedence by canonical skill name.
- Collect user-visible warning messages and skipped metadata.
"""

from __future__ import annotations

import logging
import re
from collections.abc import ItemsView, KeysView
from dataclasses import dataclass
from pathlib import Path

import yaml

from ai_cli.core.workspace import _DOT_AI_CLI, get_global_dir

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"
MAX_SKILL_FILE_BYTES = 40 * 1024
MAX_DESCRIPTION_CHARS = 1024
MAX_SKILL_NAME_CHARS = 64
SKILL_NAME_RE = re.compile(rf"^[a-z0-9][a-z0-9-]{{0,{MAX_SKILL_NAME_CHARS - 1}}}$")


@dataclass(frozen=True)
class SkillSpec:
    """Canonical skill payload loaded from SKILL.md."""

    name: str
    description: str
    instructions: str
    base_dir: Path
    scope: str  # "project" or "global"


@dataclass(frozen=True)
class SkippedSkill:
    """Diagnostic record for a skipped skill directory."""

    path: Path
    reason: str


class SkillRegistry:
    """In-memory map of canonical skill name -> SkillSpec."""

    def __init__(
        self,
        skills: dict[str, SkillSpec],
        *,
        warnings: list[str] | None = None,
        skipped: list[SkippedSkill] | None = None,
    ) -> None:
        self._skills = dict(skills)
        self._warnings = list(warnings or [])
        self._skipped = list(skipped or [])

    @property
    def skills(self) -> dict[str, SkillSpec]:
        """Return a copy of canonical skill mapping."""
        return dict(self._skills)

    @property
    def warnings(self) -> list[str]:
        """Return warning messages suitable for user-visible output."""
        return list(self._warnings)

    @property
    def skipped(self) -> list[SkippedSkill]:
        """Return skipped skill metadata for diagnostics."""
        return list(self._skipped)

    @property
    def has_skills(self) -> bool:
        """True when at least one valid skill is loaded."""
        return bool(self._skills)

    def get(self, name: str) -> SkillSpec | None:
        """Return the skill for canonical *name*, or None."""
        return self._skills.get(name)

    def __len__(self) -> int:
        """Return the number of loaded skills without copying the mapping."""
        return len(self._skills)

    def names(self) -> KeysView[str]:
        """Return a dynamic view of canonical skill names without copying."""
        return self._skills.keys()

    def items(self) -> ItemsView[str, SkillSpec]:
        """Return a dynamic view of skill entries without copying."""
        return self._skills.items()

    @classmethod
    def load(
        cls,
        project_root: Path | None,
        *,
        global_dir: Path | None = None,
    ) -> SkillRegistry:
        """Load skills from global + project scopes.

        Load order is global then project so project collisions override global.
        """
        resolved_global = global_dir if global_dir is not None else get_global_dir()
        global_skills_dir = resolved_global / "skills"
        project_skills_dir = (
            project_root / _DOT_AI_CLI / "skills" if project_root is not None else None
        )

        loader = _SkillLoader()
        loader.load_scope("global", global_skills_dir)
        if project_skills_dir is not None:
            loader.load_scope("project", project_skills_dir)
        return cls(loader.skills, warnings=loader.warnings, skipped=loader.skipped)


class _SkillLoader:
    """Stateful loader used by SkillRegistry.load()."""

    def __init__(self) -> None:
        self.skills: dict[str, SkillSpec] = {}
        self.warnings: list[str] = []
        self.skipped: list[SkippedSkill] = []

    def _warn(self, message: str) -> None:
        logger.warning(message)
        self.warnings.append(message)

    def _skip(self, path: Path, reason: str) -> None:
        self.skipped.append(SkippedSkill(path=path, reason=reason))
        self._warn(
            f"Skill '{path.name}' skipped: {reason}. "
            "Fix the skill directory contents and run '/skills reload' (or restart ai-cli)."
        )

    def load_scope(self, scope: str, skills_dir: Path | None) -> None:
        if skills_dir is None or not skills_dir.is_dir():
            return
        for entry in sorted(skills_dir.iterdir(), key=lambda p: p.name):
            if not entry.is_dir():
                continue
            self._load_skill_dir(scope, entry)

    def _load_skill_dir(self, scope: str, skill_dir: Path) -> None:
        skill_file = skill_dir / SKILL_FILENAME
        if not skill_file.is_file():
            self._skip(skill_dir, f"missing {SKILL_FILENAME}")
            return

        try:
            size_bytes = skill_file.stat().st_size
        except OSError as exc:
            self._skip(skill_dir, f"cannot stat {SKILL_FILENAME}: {exc}")
            return
        if size_bytes > MAX_SKILL_FILE_BYTES:
            self._skip(
                skill_dir,
                f"{SKILL_FILENAME} exceeds {MAX_SKILL_FILE_BYTES} bytes",
            )
            return

        try:
            raw = skill_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self._skip(skill_dir, f"cannot read {SKILL_FILENAME}: {exc}")
            return

        parsed = _parse_skill_markdown(raw)
        if isinstance(parsed, str):
            self._skip(skill_dir, parsed)
            return
        frontmatter, instructions = parsed

        name = frontmatter.get("name")
        description = frontmatter.get("description")

        if not isinstance(name, str) or not name.strip():
            self._skip(skill_dir, "missing required frontmatter field 'name'")
            return
        name = name.strip()
        if not SKILL_NAME_RE.match(name):
            self._skip(
                skill_dir,
                f"frontmatter 'name' must contain only lowercase letters, numbers, and hyphens, and be at most {MAX_SKILL_NAME_CHARS} characters",
            )
            return
        if not isinstance(description, str) or not description.strip():
            self._skip(skill_dir, "missing required frontmatter field 'description'")
            return
        if len(description) > MAX_DESCRIPTION_CHARS:
            self._skip(
                skill_dir,
                f"frontmatter 'description' exceeds {MAX_DESCRIPTION_CHARS} characters",
            )
            return

        canonical_name = name
        cleaned_description = description.strip()

        if canonical_name != skill_dir.name:
            self._warn(
                f"Skill '{skill_dir.name}': folder name differs from frontmatter name '{canonical_name}'. "
                "Rename the folder or update frontmatter field 'name' to keep references predictable."
            )

        spec = SkillSpec(
            name=canonical_name,
            description=cleaned_description,
            instructions=instructions,
            base_dir=skill_dir.resolve(),
            scope=scope,
        )

        existing = self.skills.get(canonical_name)
        if existing is None:
            self.skills[canonical_name] = spec
            return

        if scope == "project" and existing.scope == "global":
            self._warn(
                f"Skill '{canonical_name}': project scope overrides global definition. "
                "Remove one definition if this override is unintended."
            )
            self.skills[canonical_name] = spec
            return

        self._skip(
            skill_dir,
            f"duplicate canonical name '{canonical_name}' in {scope} scope",
        )


def _parse_skill_markdown(raw: str) -> tuple[dict, str] | str:
    """Parse SKILL.md content.

    Returns:
    - ``(frontmatter_dict, instructions_text)`` on success.
    - error string on failure.
    """
    if not raw.startswith("---"):
        return "malformed YAML frontmatter: missing opening '---'"

    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return "malformed YAML frontmatter: first line must be '---'"

    closing_idx = -1
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing_idx = idx
            break
    if closing_idx == -1:
        return "malformed YAML frontmatter: missing closing '---'"

    frontmatter_text = "".join(lines[1:closing_idx])
    instructions = "".join(lines[closing_idx + 1 :]).strip()
    try:
        parsed = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        return f"malformed YAML frontmatter: {exc}"
    if not isinstance(parsed, dict):
        return "malformed YAML frontmatter: expected a mapping"

    return parsed, instructions
