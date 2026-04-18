"""Tests for ai_cli.cli.completer.REPLCompleter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from prompt_toolkit.document import Document

from ai_cli.cli.completer import (
    DEFAULT_MAX_PATH_COMPLETIONS,
    REPLCompleter,
    _tokenize_command,
)
from ai_cli.core.skill_registry import SkillRegistry, SkillSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CMDS = [
    "clear",
    "compact",
    "exit",
    "help",
    "history",
    "index",
    "markdown",
    "rounds",
    "session",
    "skills",
    "tools",
    "verbose",
]


def _completer(
    cmds: list[str] | None = None,
    tool_names: list[str] | None = None,
    workspace=None,
    task_manager=None,
    skill_registry=None,
    skill_aliases=None,
) -> REPLCompleter:
    registry = None
    if tool_names is not None:
        registry = MagicMock()
        registry.all_tools_info.return_value = [{"name": n} for n in tool_names]
    return REPLCompleter(
        slash_commands=cmds if cmds is not None else CMDS,
        tool_registry=registry,
        workspace=workspace,
        task_manager=task_manager,
        skill_registry_getter=(lambda: skill_registry),
        skill_aliases_getter=(lambda: skill_aliases or {}),
    )


def _skill_registry(names: list[str]) -> SkillRegistry:
    return SkillRegistry(
        {
            name: SkillSpec(
                name=name,
                description=f"{name} desc",
                instructions=f"{name} instructions",
                base_dir=Path(f"/tmp/{name}"),
                scope="project",
            )
            for name in names
        }
    )


def _completions(completer: REPLCompleter, text: str) -> list[str]:
    """Return the completion *text* strings for *text* as typed so far."""
    doc = Document(text, cursor_position=len(text))
    event = MagicMock()
    return [c.text for c in completer.get_completions(doc, event)]


def _make_workspace(root: Path, is_ignored: bool = False) -> MagicMock:
    ws = MagicMock()
    ws.root = root
    ws.is_ignored.return_value = is_ignored
    return ws


def _make_task_detail(
    name: str,
    task_id: str,
    parent_id: str | None = None,
    subtask_names: list[str] | None = None,
) -> dict:
    subtasks = [
        {"id": f"{task_id}_{i}", "name": child_name, "status": "not_started"}
        for i, child_name in enumerate(subtask_names or [], start=1)
    ]
    return {
        "id": task_id,
        "name": name,
        "parent_id": parent_id,
        "status": "not_started",
        "priority": "medium",
        "description": "",
        "subtasks": subtasks,
    }


# ---------------------------------------------------------------------------
# Slash command — top-level names
# ---------------------------------------------------------------------------


class TestSlashTopLevel:
    def test_bare_slash_returns_all_commands(self):
        result = _completions(_completer(), "/")
        assert set(result) == {f"/{c}" for c in CMDS}

    def test_prefix_filters_commands(self):
        result = _completions(_completer(), "/h")
        assert set(result) == {"/help", "/history"}

    def test_exact_match_returned(self):
        result = _completions(_completer(), "/help")
        assert "/help" in result

    def test_no_match_returns_empty(self):
        result = _completions(_completer(), "/zzz")
        assert result == []

    def test_normal_text_returns_nothing(self):
        result = _completions(_completer(), "hello world")
        assert result == []

    def test_empty_input_returns_nothing(self):
        result = _completions(_completer(), "")
        assert result == []

    def test_trailing_space_after_unknown_cmd_returns_nothing(self):
        result = _completions(_completer(), "/zzz ")
        assert result == []

    def test_case_insensitive_prefix(self):
        result = _completions(_completer(), "/H")
        assert set(result) == {"/help", "/history"}

    def test_case_insensitive_mixed(self):
        result = _completions(_completer(), "/Ex")
        assert "/exit" in result

    def test_skill_aliases_are_included_in_top_level_completion(self):
        result = _completions(_completer(skill_aliases={"planner": "planner"}), "/pl")
        assert result == ["/planner"]

    def test_top_level_completion_reuses_cached_alias_union(self):
        alias_getter = MagicMock(return_value={"planner": "planner"})
        c = REPLCompleter(
            slash_commands=CMDS,
            skill_aliases_getter=alias_getter,
        )

        first = _completions(c, "/pl")
        assert first == ["/planner"]
        assert alias_getter.call_count == 1

        alias_getter.side_effect = RuntimeError("should not be called again")
        second = _completions(c, "/pl")
        assert second == ["/planner"]
        assert alias_getter.call_count == 2


# ---------------------------------------------------------------------------
# /tools subcommands
# ---------------------------------------------------------------------------


class TestToolsSubcommands:
    def test_tools_space_offers_subcommands(self):
        result = _completions(_completer(), "/tools ")
        assert set(result) == {"list", "info", "enable", "disable", "allow", "disallow"}

    def test_tools_partial_subcommand(self):
        result = _completions(_completer(), "/tools e")
        assert set(result) == {"enable"}

    def test_tools_partial_subcommand_uppercase(self):
        result = _completions(_completer(), "/tools E")
        assert set(result) == {"enable"}

    def test_tools_partial_d(self):
        result = _completions(_completer(), "/tools d")
        assert set(result) == {"disable", "disallow"}

    def test_tools_list_no_further_completions(self):
        result = _completions(_completer(tool_names=["read_file"]), "/tools list ")
        assert result == []

    def test_tools_info_offers_tool_names(self):
        c = _completer(tool_names=["read_file", "write_file"])
        result = _completions(c, "/tools info ")
        assert set(result) == {"read_file", "write_file"}

    def test_tools_info_prefix_filters_names(self):
        c = _completer(tool_names=["read_file", "write_file"])
        result = _completions(c, "/tools info r")
        assert result == ["read_file"]

    def test_tools_enable_offers_session_flag_and_names(self):
        c = _completer(tool_names=["bash"])
        result = _completions(c, "/tools enable ")
        assert "--session" in result
        assert "bash" in result

    def test_tools_disable_offers_session_flag_and_names(self):
        c = _completer(tool_names=["bash"])
        result = _completions(c, "/tools disable ")
        assert "--session" in result
        assert "bash" in result

    def test_tools_allow_offers_session_flag_and_names(self):
        c = _completer(tool_names=["bash"])
        result = _completions(c, "/tools allow ")
        assert "--session" in result
        assert "bash" in result

    def test_tools_disallow_offers_session_flag_and_names(self):
        c = _completer(tool_names=["bash"])
        result = _completions(c, "/tools disallow ")
        assert "--session" in result
        assert "bash" in result

    def test_tools_enable_after_session_flag_offers_names_only(self):
        c = _completer(tool_names=["bash", "read_file"])
        result = _completions(c, "/tools enable --session ")
        assert set(result) == {"bash", "read_file"}
        assert "--session" not in result

    def test_tools_enable_no_registry_no_crash(self):
        c = _completer(tool_names=None)
        result = _completions(c, "/tools enable ")
        assert "--session" in result


# ---------------------------------------------------------------------------
# /session subcommands
# ---------------------------------------------------------------------------


class TestSessionSubcommands:
    def test_session_space_offers_name(self):
        result = _completions(_completer(), "/session ")
        assert result == ["name"]

    def test_session_n_prefix(self):
        result = _completions(_completer(), "/session n")
        assert result == ["name"]

    def test_session_uppercase_prefix(self):
        result = _completions(_completer(), "/session N")
        assert result == ["name"]

    def test_session_x_no_match(self):
        result = _completions(_completer(), "/session x")
        assert result == []

    def test_session_name_no_further_completions(self):
        result = _completions(_completer(), "/session name ")
        assert result == []


# ---------------------------------------------------------------------------
# /skills
# ---------------------------------------------------------------------------


class TestSkillsSubcommands:
    def test_skills_space_offers_subcommands(self):
        result = _completions(_completer(), "/skills ")
        assert set(result) == {"info", "list", "reload"}

    def test_skills_partial_subcommand(self):
        result = _completions(_completer(), "/skills l")
        assert result == ["list"]

    def test_skills_info_offers_skill_names(self):
        result = _completions(
            _completer(skill_registry=_skill_registry(["grill-me", "planner"])),
            "/skills info ",
        )
        assert set(result) == {"grill-me", "planner"}

    def test_skills_info_prefix_filters_skill_names(self):
        result = _completions(
            _completer(skill_registry=_skill_registry(["grill-me", "planner"])),
            "/skills info g",
        )
        assert result == ["grill-me"]

    def test_skills_info_quoted_prefix_filters_skill_names(self):
        result = _completions(
            _completer(skill_registry=_skill_registry(["my-skill", "planner"])),
            '/skills info "my',
        )
        assert result == ["my-skill"]

    def test_skills_info_escaped_prefix_filters_skill_names(self):
        result = _completions(
            _completer(skill_registry=_skill_registry(["my-skill", "planner"])),
            "/skills info my\\-s",
        )
        assert result == ["my-skill"]

    def test_skills_reload_has_no_further_completions(self):
        result = _completions(_completer(), "/skills reload ")
        assert result == []

    def test_skills_info_reuses_cached_names_for_same_registry(self):
        registry = _skill_registry(["grill-me", "planner"])
        completer = _completer(skill_registry=registry)

        first = _completions(completer, "/skills info ")

        original_names = registry.names
        registry.names = MagicMock(side_effect=RuntimeError("should not be called"))
        try:
            second = _completions(completer, "/skills info ")
        finally:
            registry.names = original_names

        assert set(first) == {"grill-me", "planner"}
        assert second == first


# ---------------------------------------------------------------------------
# /tasks
# ---------------------------------------------------------------------------


class TestTasksSubcommands:
    def test_tasks_space_offers_all_subcommands(self):
        result = _completions(_completer(), "/tasks ")
        assert set(result) == {
            "add",
            "close",
            "delete",
            "edit",
            "info",
            "list",
            "note",
            "open",
            "tree",
        }

    def test_tasks_partial_subcommand(self):
        result = _completions(_completer(), "/tasks l")
        assert result == ["list"]

    def test_tasks_partial_matches_close(self):
        result = _completions(_completer(), "/tasks c")
        assert set(result) == {"close"}

    def test_tasks_partial_d(self):
        result = _completions(_completer(), "/tasks d")
        assert result == ["delete"]

    def test_tasks_uppercase_prefix(self):
        result = _completions(_completer(), "/tasks T")
        assert result == ["tree"]

    def test_tasks_no_match(self):
        result = _completions(_completer(), "/tasks z")
        assert result == []

    def test_tasks_note_offers_obsolete_verb(self):
        result = _completions(_completer(), "/tasks note ")
        assert result == ["obsolete"]

    def test_tasks_note_partial_obsolete(self):
        result = _completions(_completer(), "/tasks note o")
        assert result == ["obsolete"]

    def test_tasks_note_obsolete_path_completion(self):
        task_manager = MagicMock()
        task_manager.get_all_task_details_map.return_value = {
            "task_root": _make_task_detail("Root", "task_root")
        }
        result = _completions(
            _completer(task_manager=task_manager), "/tasks note obsolete R"
        )
        assert result == ["Root"]

    def test_tasks_note_obsolete_suggests_reason_flag_after_index(self):
        result = _completions(_completer(), "/tasks note obsolete Root 0 ")
        assert result == ["--reason"]

    def test_tasks_list_without_task_manager_has_no_path_completions(self):
        result = _completions(_completer(), "/tasks list ")
        assert result == []

    def test_tasks_info_offers_top_level_task_paths(self):
        task_manager = MagicMock()
        task_manager.get_all_task_details_map.return_value = {
            "task_root": _make_task_detail(
                "Root", "task_root", subtask_names=["Child"]
            ),
            "task_other": _make_task_detail("Other", "task_other"),
        }
        result = _completions(_completer(task_manager=task_manager), "/tasks info ")
        assert result == ["Other", "Root"]

    def test_tasks_list_offers_optional_path_completion(self):
        task_manager = MagicMock()
        task_manager.get_all_task_details_map.return_value = {
            "task_root": _make_task_detail("Root", "task_root")
        }
        result = _completions(_completer(task_manager=task_manager), "/tasks list ")
        assert result == ["Root"]

    def test_tasks_info_nested_completion_after_dot(self):
        task_manager = MagicMock()
        task_manager.get_all_task_details_map.return_value = {
            "task_root": _make_task_detail(
                "Root", "task_root", subtask_names=["Child"]
            ),
            "task_child": _make_task_detail(
                "Child", "task_child", parent_id="task_root"
            ),
            "task_other": _make_task_detail("Other", "task_other"),
        }
        result = _completions(
            _completer(task_manager=task_manager), "/tasks info Root."
        )
        assert result == ["Root.Child"]

    def test_tasks_completion_display_shows_subtask_count(self):
        task_manager = MagicMock()
        task_manager.get_all_task_details_map.return_value = {
            "task_root": _make_task_detail(
                "Root", "task_root", subtask_names=["ChildA", "ChildB"]
            )
        }
        completer = _completer(task_manager=task_manager)
        doc = Document("/tasks info R", cursor_position=len("/tasks info R"))
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        assert len(completions) == 1
        assert completions[0].text == "Root"
        assert completions[0].display_text == "Root (2 subtasks)"


# ---------------------------------------------------------------------------
# /rounds
# ---------------------------------------------------------------------------


class TestRoundsCompletion:
    def test_rounds_space_offers_session_flag(self):
        result = _completions(_completer(), "/rounds ")
        assert "--session" in result

    def test_rounds_partial_dashes(self):
        result = _completions(_completer(), "/rounds --")
        assert "--session" in result

    def test_rounds_no_numeric_completion(self):
        result = _completions(_completer(), "/rounds 1")
        assert result == []

    def test_rounds_after_session_flag_no_completions(self):
        result = _completions(_completer(), "/rounds --session ")
        assert result == []


# ---------------------------------------------------------------------------
# @ file-path completion
# ---------------------------------------------------------------------------


class TestAtPathCompletion:
    def test_bare_at_lists_workspace_root(self, tmp_path):
        (tmp_path / "foo.py").touch()
        (tmp_path / "bar.py").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@")
        assert "foo.py" in result
        assert "bar.py" in result

    def test_at_prefix_filters_files(self, tmp_path):
        (tmp_path / "alpha.txt").touch()
        (tmp_path / "beta.txt").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@a")
        assert result == ["alpha.txt"]

    def test_at_directory_gets_trailing_slash(self, tmp_path):
        (tmp_path / "src").mkdir()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@")
        assert "src/" in result

    def test_at_symlinked_directory_gets_trailing_slash(self, tmp_path):
        real = tmp_path / "real_dir"
        real.mkdir()
        link = tmp_path / "link_dir"
        link.symlink_to(real)
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@")
        assert "link_dir/" in result

    def test_at_path_in_subdir(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@src/")
        assert "src/main.py" in result

    def test_ignored_files_excluded(self, tmp_path):
        (tmp_path / "secret.env").touch()
        (tmp_path / "ok.py").touch()
        ws = _make_workspace(tmp_path)
        ws.is_ignored.side_effect = lambda p, **_: p.name == "secret.env"
        c = _completer(workspace=ws)
        result = _completions(c, "@")
        assert "secret.env" not in result
        assert "ok.py" in result

    def test_no_workspace_returns_no_at_completions(self):
        c = _completer(workspace=None)
        result = _completions(c, "@foo")
        assert result == []

    def test_at_bang_prefix_also_completes(self, tmp_path):
        (tmp_path / "diagram.png").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@!")
        assert "diagram.png" in result

    def test_bypass_flag_skips_ignore_filter(self, tmp_path):
        """@! completions include files that would otherwise be ignored."""
        (tmp_path / "ignored.py").touch()
        ws = _make_workspace(tmp_path)
        ws.is_ignored.side_effect = lambda p, **_: p.name == "ignored.py"
        c = _completer(workspace=ws)
        assert "ignored.py" not in _completions(c, "@i")
        assert "ignored.py" in _completions(c, "@!i")

    def test_start_position_replaces_partial(self, tmp_path):
        (tmp_path / "alpha.py").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        doc = Document("@al", cursor_position=3)
        event = MagicMock()
        completions = list(c.get_completions(doc, event))
        assert len(completions) == 1
        assert completions[0].text == "alpha.py"
        assert completions[0].start_position == -len("al")

    def test_completions_capped_at_max(self, tmp_path):
        for i in range(DEFAULT_MAX_PATH_COMPLETIONS + 10):
            (tmp_path / f"file{i:04d}.txt").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@")
        assert len(result) == DEFAULT_MAX_PATH_COMPLETIONS

    def test_custom_max_path_completions(self, tmp_path):
        """max_path_completions constructor arg overrides the default cap."""
        for i in range(20):
            (tmp_path / f"file{i:02d}.txt").touch()
        c = REPLCompleter(
            slash_commands=[],
            workspace=_make_workspace(tmp_path),
            max_path_completions=5,
        )
        result = _completions(c, "@")
        assert len(result) == 5

    def test_max_path_completions_zero_raises(self):
        """max_path_completions=0 raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="max_path_completions"):
            REPLCompleter(slash_commands=[], max_path_completions=0)

    def test_max_path_completions_negative_raises(self):
        """max_path_completions < 0 raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="max_path_completions"):
            REPLCompleter(slash_commands=[], max_path_completions=-1)

    def test_nonexistent_subdir_returns_nothing(self, tmp_path):
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@nonexistent/")
        assert result == []

    def test_at_in_middle_of_text_completes(self, tmp_path):
        (tmp_path / "main.py").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "look at @ma")
        assert "main.py" in result

    def test_dotdot_path_completes_parent_dir(self, tmp_path):
        """@../ completes files in the parent directory."""
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        (tmp_path / "sibling.txt").touch()
        c = _completer(workspace=_make_workspace(workspace_root))
        result = _completions(c, "@../")
        assert "../sibling.txt" in result

    def test_dotdot_outside_workspace_skips_ignore_check(self, tmp_path):
        """Files outside the workspace root are never filtered by ignore rules."""
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        (tmp_path / "outside.py").touch()
        ws = _make_workspace(workspace_root)
        ws.is_ignored.return_value = True  # would filter if checked
        c = _completer(workspace=ws)
        result = _completions(c, "@../")
        assert "../outside.py" in result
        ws.is_ignored.assert_not_called()

    def test_absolute_path_completes(self, tmp_path):
        """@/absolute/path completes from the filesystem root."""
        (tmp_path / "file.txt").touch()
        # Use a workspace rooted elsewhere so tmp_path is 'outside'.
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        c = _completer(workspace=_make_workspace(workspace_root))
        result = _completions(c, f"@{tmp_path}/")
        assert f"{tmp_path}/file.txt" in result

    def test_resolve_os_error_returns_nothing(self, tmp_path):
        """A dir_part that raises OSError on resolve() returns no completions."""
        from unittest.mock import patch

        c = _completer(workspace=_make_workspace(tmp_path))
        with patch("pathlib.Path.resolve", side_effect=OSError("symlink loop")):
            result = _completions(c, "@bad/")
        assert result == []

    def test_is_dir_os_error_skips_entry(self, tmp_path):
        """An entry whose is_dir() raises OSError is skipped; others still complete."""
        from unittest.mock import patch

        (tmp_path / "ok.py").touch()
        (tmp_path / "broken").touch()
        c = _completer(workspace=_make_workspace(tmp_path))

        import os as _os

        entry_type = type(next(iter(_os.scandir(tmp_path))))
        original_is_dir = entry_type.is_dir

        def _patched_is_dir(self, follow_symlinks=True):
            if self.name == "broken":
                raise OSError("stat failed")
            return original_is_dir(self, follow_symlinks=follow_symlinks)

        with patch.object(entry_type, "is_dir", _patched_is_dir):
            result = _completions(c, "@")
        assert "ok.py" in result
        assert "broken" not in result

    def test_space_in_name_is_backslash_escaped(self, tmp_path):
        """Completions for filenames with spaces use backslash escaping."""
        (tmp_path / "my file.txt").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "@")
        assert r"my\ file.txt" in result

    def test_space_in_name_start_position_covers_raw_partial(self, tmp_path):
        """start_position covers the raw (escaped) partial, not the decoded one."""
        (tmp_path / "my file.txt").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        # User has typed "my\ " (4 raw chars) which decodes to "my " (3 chars).
        raw_partial = r"my\ "
        doc = Document("@" + raw_partial, cursor_position=1 + len(raw_partial))
        event = MagicMock()
        completions = list(c.get_completions(doc, event))
        assert len(completions) == 1
        assert completions[0].text == r"my\ file.txt"
        assert completions[0].start_position == -len(raw_partial)

    def test_escaped_space_in_subdir_is_traversable(self, tmp_path):
        """@dir\\ with\\ spaces/ completes into the directory contents."""
        spaced = tmp_path / "my dir"
        spaced.mkdir()
        (spaced / "inside.txt").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, r"@my\ dir/")
        assert r"my\ dir/inside.txt" in result


# ---------------------------------------------------------------------------
# /index completions
# ---------------------------------------------------------------------------


class TestIndexCompletion:
    def test_index_space_offers_flags_and_path(self, tmp_path):
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "/index ")
        assert "--full" in result
        assert "--file" in result
        assert "--label" in result
        assert "--remove" in result

    def test_index_partial_flag_filters(self, tmp_path):
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "/index --f")
        assert "--file" in result
        assert "--full" in result
        assert "--label" not in result
        assert "--remove" not in result

    def test_index_file_flag_offers_path(self, tmp_path):
        (tmp_path / "book.txt").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "/index --file ")
        assert "book.txt" in result

    def test_index_file_flag_partial_path(self, tmp_path):
        (tmp_path / "alpha.txt").touch()
        (tmp_path / "beta.txt").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "/index --file a")
        assert "alpha.txt" in result
        assert "beta.txt" not in result

    def test_index_label_suppresses_completions(self, tmp_path):
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "/index --label ")
        assert result == []

    def test_index_space_in_path_backslash_escaped(self, tmp_path):
        """Completing a file with a space in its name inserts escaped text."""
        (tmp_path / "my book.txt").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "/index --file ")
        assert r"my\ book.txt" in result

    def test_index_file_with_escaped_partial(self, tmp_path):
        """Typing a backslash-escaped partial matches and replaces correctly."""
        (tmp_path / "my book.txt").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        raw_partial = r"my\ b"
        doc = Document(
            "/index --file " + raw_partial, cursor_position=14 + len(raw_partial)
        )
        event = MagicMock()
        completions = list(c.get_completions(doc, event))
        assert len(completions) == 1
        assert completions[0].text == r"my\ book.txt"
        assert completions[0].start_position == -len(raw_partial)

    def test_index_positional_path(self, tmp_path):
        """Without --file, the positional argument also gets path completion."""
        (tmp_path / "src").mkdir()
        c = _completer(workspace=_make_workspace(tmp_path))
        result = _completions(c, "/index ")
        assert "src/" in result

    def test_index_file_single_quoted_partial(self, tmp_path):
        """Single-quoted partial: partial_raw excludes the opening quote."""
        (tmp_path / "my file.txt").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        raw_partial = "'my fil"
        doc = Document(
            "/index --file " + raw_partial,
            cursor_position=14 + len(raw_partial),
        )
        completions = list(c.get_completions(doc, MagicMock()))
        assert len(completions) == 1
        # The replacement must NOT re-insert the opening quote.
        assert completions[0].start_position == -(len(raw_partial) - 1)

    def test_index_file_double_quoted_partial(self, tmp_path):
        """Double-quoted partial: partial_raw excludes the opening quote."""
        (tmp_path / "my file.txt").touch()
        c = _completer(workspace=_make_workspace(tmp_path))
        raw_partial = '"my fil'
        doc = Document(
            "/index --file " + raw_partial,
            cursor_position=14 + len(raw_partial),
        )
        completions = list(c.get_completions(doc, MagicMock()))
        assert len(completions) == 1
        assert completions[0].start_position == -(len(raw_partial) - 1)


# ---------------------------------------------------------------------------
# _tokenize_command — quote raw_start behaviour
# ---------------------------------------------------------------------------


class TestTokenizeCommandQuotes:
    def test_single_quoted_partial_raw_excludes_quote(self):
        """`partial_raw` for a single-quoted token starts after the opening quote."""
        completed, partial = _tokenize_command("/cmd 'some/path")
        assert completed == ["/cmd"]
        assert partial == "some/path"

    def test_double_quoted_partial_raw_excludes_quote(self):
        completed, partial = _tokenize_command('/cmd "some/path')
        assert completed == ["/cmd"]
        assert partial == "some/path"

    def test_mid_token_quote_does_not_shift_raw_start(self):
        """A quote that opens mid-token does not alter raw_start."""
        completed, partial = _tokenize_command("/cmd abc'def")
        assert completed == ["/cmd"]
        # raw_start was set when whitespace was consumed, not at the quote.
        assert partial == "abc'def"

    def test_completed_quoted_token(self):
        """A fully-closed quoted token is decoded correctly."""
        completed, partial = _tokenize_command("/cmd 'hello world' ")
        assert completed == ["/cmd", "hello world"]
        assert partial == ""
