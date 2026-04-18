"""Tests for ai_cli.cli.repl and Session.clear()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_cli.cli.completer import REPLCompleter
from ai_cli.cli.repl import (
    _DEFAULT_MAX_TOOL_ROUNDS,
    _SLASH_COMMANDS,
    REPL,
    _build_keyboard_shortcuts,
    _levenshtein_distance,
    skill_aliases_for_registry,
)
from ai_cli.core.llm_client import LLMError
from ai_cli.core.session_manager import Session, SessionError
from ai_cli.core.skill_registry import SkillRegistry, SkillSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(chunks=None):
    llm = MagicMock()
    if chunks is None:
        chunks = [
            {"type": "text", "delta": "Hello!"},
            {"type": "done", "stop_reason": "stop", "usage": {}},
        ]
    llm.send.return_value = iter(chunks)
    return llm


def _make_repl(
    session=None,
    tool_registry=None,
    llm=None,
    display=None,
    workspace=None,
    task_manager=None,
    skill_registry=None,
) -> REPL:
    return REPL(
        session=session or MagicMock(),
        tool_registry=tool_registry or MagicMock(),
        llm_client=llm or _make_llm(),
        display=display or MagicMock(),
        workspace=workspace or MagicMock(),
        task_manager=task_manager,
        skill_registry=skill_registry,
    )


def _make_skill_registry(tmp_path: Path, names: list[str]) -> SkillRegistry:
    specs: dict[str, SkillSpec] = {}
    for name in names:
        skill_dir = tmp_path / "skills" / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        specs[name] = SkillSpec(
            name=name,
            description=f"{name} description",
            instructions=f"{name} instructions",
            base_dir=skill_dir,
            scope="project",
        )
    return SkillRegistry(specs)


def _make_prompt_session(*inputs):
    """Return a mock PromptSession that yields *inputs* then raises EOFError."""
    pt = MagicMock()
    pt.prompt.side_effect = [*inputs, EOFError()]
    return pt


# ---------------------------------------------------------------------------
# Session.clear()
# ---------------------------------------------------------------------------


class TestSessionClear:
    def _make_session(self, tmp_path: Path) -> Session:
        session_dir = tmp_path / "s1"
        session_dir.mkdir()
        return Session("s1", session_dir, MagicMock())

    def test_clear_removes_current_history(self, tmp_path):
        s = self._make_session(tmp_path)
        s._write_meta({"message_count": 0})
        s.add_message("user", "Hello")
        assert s._current_path.exists()
        s.clear()
        assert not s._current_path.exists()

    def test_clear_preserves_full_history(self, tmp_path):
        s = self._make_session(tmp_path)
        s._write_meta({"message_count": 0})
        s.add_message("user", "Hello")
        assert s._full_path.exists()
        s.clear()
        assert s._full_path.exists()

    def test_clear_resets_message_count(self, tmp_path):
        s = self._make_session(tmp_path)
        s._write_meta({"message_count": 0})
        s.add_message("user", "Hello")
        s.add_message("assistant", "Hi")
        s.clear()
        assert s._read_meta()["message_count"] == 0

    def test_clear_resets_last_message_fields(self, tmp_path):
        s = self._make_session(tmp_path)
        s._write_meta({"message_count": 0})
        s.add_message("user", "Hello")
        s.clear()
        meta = s._read_meta()
        assert meta["last_message_role"] == ""
        assert meta["last_message_preview"] == ""

    def test_clear_is_idempotent(self, tmp_path):
        s = self._make_session(tmp_path)
        s._write_meta({"message_count": 0})
        s.clear()
        s.clear()  # should not raise

    def test_clear_raises_session_error_on_oserror(self, tmp_path):
        s = self._make_session(tmp_path)
        s._write_meta({"message_count": 0})
        s.add_message("user", "Hello")
        # Replace current_path with a directory so unlink raises
        s._current_path.unlink()
        s._current_path.mkdir()
        with pytest.raises(SessionError, match="Could not clear"):
            s.clear()


# ---------------------------------------------------------------------------
# REPL.run() — basic loop behaviour
# ---------------------------------------------------------------------------


class TestREPLRun:
    def test_constructor_preserves_provided_empty_skill_registry(self):
        empty_registry = SkillRegistry({}, warnings=["warn"], skipped=[])
        repl = _make_repl(skill_registry=empty_registry)
        assert repl._skill_registry is empty_registry

    def test_eof_exits_loop(self):
        repl = _make_repl()
        pt = _make_prompt_session()  # immediately EOFError
        repl.run(_prompt_session=pt)  # should return normally

    def test_keyboard_interrupt_re_prompts(self):
        repl = _make_repl()
        pt = _make_prompt_session(KeyboardInterrupt())
        repl.run(_prompt_session=pt)
        assert pt.prompt.call_count == 2  # once for interrupt, once for EOF

    def test_empty_input_is_ignored(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        pt = _make_prompt_session("   ")
        repl.run(_prompt_session=pt)
        display.begin_assistant_turn.assert_not_called()

    def test_slash_command_dispatched(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        pt = _make_prompt_session("/help")
        repl.run(_prompt_session=pt)
        display.show_help.assert_called_once_with(
            _SLASH_COMMANDS
            + [("", "")]
            + _build_keyboard_shortcuts(enable_suspend=True)
        )

    def test_plain_input_sent_to_llm(self):
        llm = _make_llm()
        session = MagicMock()
        session.should_compact.return_value = False
        repl = _make_repl(session=session, llm=llm)
        pt = _make_prompt_session("Hello there")
        repl.run(_prompt_session=pt)
        session.add_message.assert_any_call("user", "Hello there")

    def test_default_session_created_with_completer(self, tmp_path):
        """When no _prompt_session is injected, PromptSession gets a REPLCompleter."""
        repl = _make_repl()
        mock_pt = MagicMock()
        mock_pt.prompt.side_effect = EOFError()
        with (
            patch("ai_cli.cli.repl.PromptSession", return_value=mock_pt) as mock_cls,
            patch("ai_cli.cli.repl.get_global_dir", return_value=tmp_path),
        ):
            repl.run()
        _, kwargs = mock_cls.call_args
        assert isinstance(kwargs.get("completer"), REPLCompleter)
        assert kwargs.get("complete_while_typing") is False

    def test_complete_while_typing_enabled_via_config(self, tmp_path):
        """repl_behavior.complete_while_typing: true is passed to PromptSession."""
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: (
            {"complete_while_typing": True} if key == "repl_behavior" else default
        )
        repl = _make_repl()
        repl._config = config
        mock_pt = MagicMock()
        mock_pt.prompt.side_effect = EOFError()
        with (
            patch("ai_cli.cli.repl.PromptSession", return_value=mock_pt) as mock_cls,
            patch("ai_cli.cli.repl.get_global_dir", return_value=tmp_path),
        ):
            repl.run()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("complete_while_typing") is True

    def test_complete_while_typing_bad_repl_cfg_falls_back_to_false(self, tmp_path):
        """A non-dict repl_behavior value falls back to complete_while_typing=False."""
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: (
            "bad" if key == "repl_behavior" else default
        )
        repl = _make_repl()
        repl._config = config
        mock_pt = MagicMock()
        mock_pt.prompt.side_effect = EOFError()
        with (
            patch("ai_cli.cli.repl.PromptSession", return_value=mock_pt) as mock_cls,
            patch("ai_cli.cli.repl.get_global_dir", return_value=tmp_path),
        ):
            repl.run()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("complete_while_typing") is False

    def test_enable_suspend_default_true(self, tmp_path):
        """enable_suspend defaults to True when not in config."""
        repl = _make_repl()
        mock_pt = MagicMock()
        mock_pt.prompt.side_effect = EOFError()
        with (
            patch("ai_cli.cli.repl.PromptSession", return_value=mock_pt) as mock_cls,
            patch("ai_cli.cli.repl.get_global_dir", return_value=tmp_path),
        ):
            repl.run()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("enable_suspend") is True

    def test_enable_suspend_disabled_via_config(self, tmp_path):
        """repl_behavior.enable_suspend: false is passed to PromptSession."""
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: (
            {"enable_suspend": False} if key == "repl_behavior" else default
        )
        repl = _make_repl()
        repl._config = config
        mock_pt = MagicMock()
        mock_pt.prompt.side_effect = EOFError()
        with (
            patch("ai_cli.cli.repl.PromptSession", return_value=mock_pt) as mock_cls,
            patch("ai_cli.cli.repl.get_global_dir", return_value=tmp_path),
        ):
            repl.run()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("enable_suspend") is False

    def test_key_bindings_injected(self, tmp_path):
        """A KeyBindings object is always passed to PromptSession."""
        from prompt_toolkit.key_binding import KeyBindings

        repl = _make_repl()
        mock_pt = MagicMock()
        mock_pt.prompt.side_effect = EOFError()
        with (
            patch("ai_cli.cli.repl.PromptSession", return_value=mock_pt) as mock_cls,
            patch("ai_cli.cli.repl.get_global_dir", return_value=tmp_path),
        ):
            repl.run()
        _, kwargs = mock_cls.call_args
        assert isinstance(kwargs.get("key_bindings"), KeyBindings)

    def test_completion_max_results_passed_to_completer(self, tmp_path):
        """repl_behavior.completion_max_results is forwarded to REPLCompleter."""
        from ai_cli.cli.completer import REPLCompleter

        config = MagicMock()
        config.get.side_effect = lambda key, default=None: (
            {"completion_max_results": 42} if key == "repl_behavior" else default
        )
        repl = _make_repl()
        repl._config = config
        mock_pt = MagicMock()
        mock_pt.prompt.side_effect = EOFError()
        with (
            patch("ai_cli.cli.repl.PromptSession", return_value=mock_pt) as mock_cls,
            patch("ai_cli.cli.repl.get_global_dir", return_value=tmp_path),
        ):
            repl.run()
        _, kwargs = mock_cls.call_args
        completer = kwargs.get("completer")
        assert isinstance(completer, REPLCompleter)
        assert completer._max_path_completions == 42

    def test_completion_max_results_bad_value_uses_default(self, tmp_path):
        """An invalid completion_max_results falls back to the default."""
        from ai_cli.cli.completer import DEFAULT_MAX_PATH_COMPLETIONS, REPLCompleter

        config = MagicMock()
        config.get.side_effect = lambda key, default=None: (
            {"completion_max_results": "bad"} if key == "repl_behavior" else default
        )
        repl = _make_repl()
        repl._config = config
        mock_pt = MagicMock()
        mock_pt.prompt.side_effect = EOFError()
        with (
            patch("ai_cli.cli.repl.PromptSession", return_value=mock_pt) as mock_cls,
            patch("ai_cli.cli.repl.get_global_dir", return_value=tmp_path),
        ):
            repl.run()
        _, kwargs = mock_cls.call_args
        completer = kwargs.get("completer")
        assert isinstance(completer, REPLCompleter)
        assert completer._max_path_completions == DEFAULT_MAX_PATH_COMPLETIONS

    def test_keyboard_interrupt_shows_hint(self):
        """Ctrl+C at the prompt shows a hint rather than silently re-prompting."""
        display = MagicMock()
        repl = _make_repl(display=display)
        pt = _make_prompt_session(KeyboardInterrupt())
        repl.run(_prompt_session=pt)
        display.show_status.assert_called_once()
        msg = display.show_status.call_args[0][0]
        assert "/exit" in msg or "Ctrl+D" in msg

    def test_abort_during_streaming_shows_aborted(self):
        """If abort is signalled mid-stream, 'Aborted.' is shown."""
        import threading

        abort_event = threading.Event()

        # Chunk iterator that sets the abort event mid-stream.
        def _chunks():
            yield {"type": "text", "delta": "Hello"}
            abort_event.set()
            yield {"type": "text", "delta": " world"}
            yield {"type": "done", "stop_reason": "stop", "usage": {}}

        llm = MagicMock()
        llm.send.return_value = _chunks()
        session = MagicMock()
        session.get_messages.return_value = []
        session.should_compact.return_value = False
        display = MagicMock()
        repl = _make_repl(session=session, llm=llm, display=display)

        # Patch _AbortMonitor so the abort_event we control is used.
        with patch("ai_cli.cli.repl._AbortMonitor") as MockMonitor:
            instance = MockMonitor.return_value
            instance.start.side_effect = lambda: abort_event.clear()  # no-op start

            # Inject abort_event by overriding Agent.run to use our event.
            original_run = repl._main_agent.run

            def _patched_run(prompt, *, abort=None):
                return original_run(prompt, abort=abort_event)

            repl._main_agent.run = _patched_run
            repl._send_to_llm("Hello")

        # "Aborted." should have been shown.
        status_calls = [c[0][0] for c in display.show_status.call_args_list]
        assert any("Aborted" in s for s in status_calls)


# ---------------------------------------------------------------------------
# REPL._handle_slash_command()
# ---------------------------------------------------------------------------


class TestREPLSlashCommands:
    def test_help(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        with (
            patch("ai_cli.cli.repl._HAS_TTY", True),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = True
            repl._handle_slash_command("help")
            expected = (
                _SLASH_COMMANDS
                + [("", "")]
                + _build_keyboard_shortcuts(enable_suspend=True)
            )
        display.show_help.assert_called_once_with(expected)

    def test_help_omits_ctrl_z_when_suspend_disabled(self):
        """Ctrl+Z is not listed even when platform supports it, if disabled in config."""
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: (
            {"enable_suspend": False} if key == "repl_behavior" else default
        )
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._config = config
        # Patch both TTY guards to True so the only reason Ctrl+Z is absent is the
        # config flag — not an accident of the test environment.
        with (
            patch("ai_cli.cli.repl._HAS_TTY", True),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = True
            repl._handle_slash_command("help")
        displayed = display.show_help.call_args[0][0]
        assert not any("Ctrl+Z" in cmd for cmd, _ in displayed)

    def test_exit_raises_system_exit(self):
        repl = _make_repl()
        with pytest.raises(SystemExit):
            repl._handle_slash_command("exit")

    def test_clear_calls_session_clear(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("clear")
        session.clear.assert_called_once()
        display.show_status.assert_called_once()

    def test_clear_shows_error_on_session_error(self):
        session = MagicMock()
        session.clear.side_effect = SessionError("disk full")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("clear")
        display.show_error.assert_called_once()
        display.show_status.assert_not_called()

    def test_verbose_toggles_and_reports(self):
        display = MagicMock()
        display.verbose = False
        repl = _make_repl(display=display)
        repl._handle_slash_command("verbose")
        display.toggle_verbose.assert_called_once()
        display.show_status.assert_called_once()

    def test_verbose_status_reflects_new_state(self):
        display = MagicMock()
        display.verbose = False  # known starting state

        def _toggle():
            display.verbose = not display.verbose

        display.toggle_verbose.side_effect = _toggle
        repl = _make_repl(display=display)
        repl._handle_slash_command("verbose")
        assert display.verbose is True
        msg = display.show_status.call_args[0][0]
        assert "on" in msg

    def test_markdown_toggles_and_reports(self):
        display = MagicMock()
        display.markdown_enabled = True
        repl = _make_repl(display=display)
        repl._handle_slash_command("markdown")
        display.toggle_markdown.assert_called_once()
        display.show_status.assert_called_once()

    def test_compact_calls_session_compact(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("compact")
        session.compact.assert_called_once()

    def test_compact_shows_error_on_failure(self):
        session = MagicMock()
        session.compact.side_effect = SessionError("LLM down")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("compact")
        display.show_error.assert_called_once()

    def test_tools_calls_show_tool_list(self):
        tool_registry = MagicMock()
        tool_registry.all_enabled.return_value = []
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools")
        display.show_tool_list.assert_called_once_with([])

    def test_skills_calls_show_skills_simple(self, tmp_path: Path):
        display = MagicMock()
        repl = _make_repl(
            display=display,
            skill_registry=_make_skill_registry(tmp_path, ["planner", "grill-me"]),
        )
        repl._handle_slash_command("skills")
        rows = display.show_skills_simple.call_args[0][0]
        assert [row["name"] for row in rows] == ["grill-me", "planner"]

    def test_skill_alias_invokes_llm_with_injected_message(self, tmp_path: Path):
        repl = _make_repl(skill_registry=_make_skill_registry(tmp_path, ["planner"]))
        repl._send_to_llm = MagicMock()

        repl._handle_slash_command("planner draft a plan")

        repl._send_to_llm.assert_called_once()
        injected = repl._send_to_llm.call_args.args[0]
        assert "Call the skills tool with name='planner'" in injected
        assert "draft a plan" in injected

    def test_skill_alias_without_trailing_text_still_invokes_skill(
        self, tmp_path: Path
    ):
        repl = _make_repl(skill_registry=_make_skill_registry(tmp_path, ["planner"]))
        repl._send_to_llm = MagicMock()

        repl._handle_slash_command("planner")

        injected = repl._send_to_llm.call_args.args[0]
        assert "Call the skills tool with name='planner'" in injected
        assert "User request:" not in injected

    def test_skill_alias_preserves_multimodal_user_request(self, tmp_path: Path):
        repl = _make_repl(skill_registry=_make_skill_registry(tmp_path, ["planner"]))
        repl._send_to_llm = MagicMock()
        repl._preprocess_at_references = MagicMock(
            return_value=[
                {"type": "text", "text": "describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ]
        )

        repl._handle_slash_command("planner describe this @image.png")

        payload = repl._send_to_llm.call_args.args[0]
        assert isinstance(payload, list)
        assert payload[0]["type"] == "text"
        assert "Call the skills tool with name='planner'" in payload[0]["text"]
        assert "User request:" in payload[0]["text"]
        assert payload[1]["type"] == "text"
        assert payload[2]["type"] == "image_url"

    def test_skill_alias_collision_is_skipped_with_warning(self, tmp_path: Path):
        registry = _make_skill_registry(tmp_path, ["help", "planner"])
        aliases, warnings = skill_aliases_for_registry(registry)
        assert aliases == {"planner": "planner"}
        assert any(
            "conflicts with an existing command" in warning for warning in warnings
        )

    def test_skills_list_calls_show_skills_list(self, tmp_path: Path):
        display = MagicMock()
        repl = _make_repl(
            display=display,
            skill_registry=_make_skill_registry(tmp_path, ["planner"]),
        )
        repl._handle_slash_command("skills list")
        display.show_skills_list.assert_called_once()

    def test_skills_info_calls_show_skill_info(self, tmp_path: Path):
        display = MagicMock()
        repl = _make_repl(
            display=display,
            skill_registry=_make_skill_registry(tmp_path, ["planner"]),
        )
        repl._handle_slash_command("skills info planner")
        info = display.show_skill_info.call_args[0][0]
        assert info["name"] == "planner"
        assert "description" in info
        assert "instructions" in info

    def test_skills_info_usage_error(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._handle_slash_command("skills info")
        display.show_error.assert_called_once()
        assert "Usage: /skills info <name>" in display.show_error.call_args[0][0]

    def test_skills_parse_error_shows_error(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._handle_slash_command('skills info "broken')
        display.show_error.assert_called_once()
        assert "Could not parse /skills arguments" in display.show_error.call_args[0][0]

    def test_skills_zero_behavior(self):
        display = MagicMock()
        repl = _make_repl(display=display, skill_registry=SkillRegistry({}))
        repl._handle_slash_command("skills")
        display.show_skills_simple.assert_called_once_with([])

    def test_skills_reload_refreshes_registry_and_warns(self, tmp_path: Path):
        display = MagicMock()
        workspace = MagicMock()
        workspace.root = tmp_path
        tool_registry = MagicMock()
        tool_registry.permission_manager = MagicMock()
        read_file_tool = MagicMock()
        tool_registry.get.return_value = read_file_tool
        reloaded = SkillRegistry(
            _make_skill_registry(tmp_path, ["planner"]).skills,
            warnings=["Skill 'bad' skipped: missing SKILL.md"],
        )
        repl = _make_repl(
            display=display,
            workspace=workspace,
            tool_registry=tool_registry,
            skill_registry=SkillRegistry({}),
        )
        with (
            patch("ai_cli.cli.repl.SkillRegistry.load", return_value=reloaded),
            patch("ai_cli.cli.repl.get_global_dir", return_value=tmp_path / "global"),
        ):
            repl._handle_slash_command("skills reload")
        read_file_tool.set_skill_registry.assert_called_once_with(reloaded)
        tool_registry.unregister.assert_called_once_with("skills")
        tool_registry.register_instance.assert_called_once()
        statuses = [c[0][0] for c in display.show_status.call_args_list]
        assert any("missing SKILL.md" in msg for msg in statuses)
        assert any("1 skill(s) loaded" in msg for msg in statuses)

    def test_skills_reload_with_no_skills_clears_read_file_skill_registry(
        self, tmp_path: Path
    ):
        display = MagicMock()
        workspace = MagicMock()
        workspace.root = tmp_path
        tool_registry = MagicMock()
        tool_registry.permission_manager = MagicMock()
        read_file_tool = MagicMock()
        tool_registry.get.return_value = read_file_tool
        repl = _make_repl(
            display=display,
            workspace=workspace,
            tool_registry=tool_registry,
            skill_registry=_make_skill_registry(tmp_path, ["planner"]),
        )

        with (
            patch("ai_cli.cli.repl.SkillRegistry.load", return_value=SkillRegistry({})),
            patch("ai_cli.cli.repl.get_global_dir", return_value=tmp_path / "global"),
        ):
            repl._handle_slash_command("skills reload")

        read_file_tool.set_skill_registry.assert_called_once_with(None)
        tool_registry.unregister.assert_called_once_with("skills")
        tool_registry.register_instance.assert_not_called()
        display.show_error.assert_not_called()
        statuses = [c[0][0] for c in display.show_status.call_args_list]
        assert any("0 skill(s) loaded" in msg for msg in statuses)

    def test_skills_reload_load_failure_shows_error_and_keeps_state(
        self, tmp_path: Path
    ):
        display = MagicMock()
        workspace = MagicMock()
        workspace.root = tmp_path
        tool_registry = MagicMock()
        tool_registry.permission_manager = MagicMock()
        read_file_tool = MagicMock()
        old_skill_tool = MagicMock()

        def _get_tool(name: str):
            if name == "read_file":
                return read_file_tool
            if name == "skills":
                return old_skill_tool
            return None

        tool_registry.get.side_effect = _get_tool

        old_registry = _make_skill_registry(tmp_path, ["planner"])
        repl = _make_repl(
            display=display,
            workspace=workspace,
            tool_registry=tool_registry,
            skill_registry=old_registry,
        )

        with (
            patch(
                "ai_cli.cli.repl.SkillRegistry.load", side_effect=RuntimeError("boom")
            ),
            patch("ai_cli.cli.repl.get_global_dir", return_value=tmp_path / "global"),
        ):
            repl._handle_slash_command("skills reload")

        assert repl._skill_registry is old_registry
        display.show_error.assert_called_once()
        assert "Failed to reload skills" in display.show_error.call_args[0][0]
        tool_registry.unregister.assert_not_called()
        tool_registry.register_instance.assert_not_called()
        read_file_tool.set_skill_registry.assert_not_called()

    def test_skills_reload_apply_failure_rolls_back(self, tmp_path: Path):
        display = MagicMock()
        workspace = MagicMock()
        workspace.root = tmp_path
        tool_registry = MagicMock()
        tool_registry.permission_manager = MagicMock()
        read_file_tool = MagicMock()
        old_skill_tool = MagicMock()

        def _get_tool(name: str):
            if name == "read_file":
                return read_file_tool
            if name == "skills":
                return old_skill_tool
            return None

        tool_registry.get.side_effect = _get_tool
        tool_registry.register_instance.side_effect = RuntimeError("register failed")

        old_registry = _make_skill_registry(tmp_path, ["planner"])
        reloaded = _make_skill_registry(tmp_path, ["grill-me"])
        repl = _make_repl(
            display=display,
            workspace=workspace,
            tool_registry=tool_registry,
            skill_registry=old_registry,
        )

        with (
            patch("ai_cli.cli.repl.SkillRegistry.load", return_value=reloaded),
            patch("ai_cli.cli.repl.get_global_dir", return_value=tmp_path / "global"),
        ):
            repl._handle_slash_command("skills reload")

        assert repl._skill_registry is old_registry
        display.show_error.assert_called_once()
        assert "Failed to reload skills" in display.show_error.call_args[0][0]
        read_file_tool.set_skill_registry.assert_any_call(reloaded)
        read_file_tool.set_skill_registry.assert_any_call(old_registry)
        assert tool_registry.unregister.call_count >= 1
        tool_registry.register_instance.assert_any_call(old_skill_tool)
        display.show_status.assert_not_called()

    def test_skills_reload_apply_failure_rolls_back_to_none_for_empty_registry(
        self, tmp_path: Path
    ):
        display = MagicMock()
        workspace = MagicMock()
        workspace.root = tmp_path
        tool_registry = MagicMock()
        tool_registry.permission_manager = MagicMock()
        read_file_tool = MagicMock()

        def _get_tool(name: str):
            if name == "read_file":
                return read_file_tool
            return None

        tool_registry.get.side_effect = _get_tool
        tool_registry.register_instance.side_effect = RuntimeError("register failed")

        reloaded = _make_skill_registry(tmp_path, ["grill-me"])
        repl = _make_repl(
            display=display,
            workspace=workspace,
            tool_registry=tool_registry,
            skill_registry=SkillRegistry({}),
        )

        with (
            patch("ai_cli.cli.repl.SkillRegistry.load", return_value=reloaded),
            patch("ai_cli.cli.repl.get_global_dir", return_value=tmp_path / "global"),
        ):
            repl._handle_slash_command("skills reload")

        assert repl._skill_registry.has_skills is False
        read_file_tool.set_skill_registry.assert_any_call(reloaded)
        read_file_tool.set_skill_registry.assert_any_call(None)
        display.show_error.assert_called_once()

    def test_session_calls_show_session_info(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session")
        display.show_session_info.assert_called_once_with(session)

    def test_unknown_command_shows_error(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._handle_slash_command("foobar")
        display.show_error.assert_called_once()
        msg = display.show_error.call_args[0][0]
        assert "foobar" in msg

    def test_unknown_command_suggests_builtin_command(self):
        display = MagicMock()
        repl = _make_repl(display=display)

        repl._handle_slash_command("hep")

        msg = display.show_error.call_args[0][0]
        assert "Did you mean /help?" in msg

    def test_unknown_command_suggests_skill_alias(self, tmp_path: Path):
        display = MagicMock()
        repl = _make_repl(
            display=display,
            skill_registry=_make_skill_registry(tmp_path, ["planner"]),
        )

        repl._handle_slash_command("plnner")

        msg = display.show_error.call_args[0][0]
        assert "Did you mean /planner?" in msg

    def test_unknown_command_omits_suggestions_when_over_threshold(self):
        display = MagicMock()
        repl = _make_repl(display=display)

        repl._handle_slash_command("wildlydifferent")

        msg = display.show_error.call_args[0][0]
        assert "Did you mean" not in msg


class TestREPLHelpers:
    def test_levenshtein_distance(self):
        assert _levenshtein_distance("help", "hep") == 1
        assert _levenshtein_distance("planner", "plnner") == 1

    def test_suggestion_threshold_uses_twenty_percent_floor_of_two(self):
        repl = _make_repl()
        assert repl._suggest_slash_commands("hep") == ["help"]
        assert repl._suggest_slash_commands("wildlydifferent") == []

    def test_empty_command_shows_error(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._handle_slash_command("")
        display.show_error.assert_called_once()

    def test_command_is_case_insensitive(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._handle_slash_command("HELP")
        display.show_help.assert_called_once()


# ---------------------------------------------------------------------------
# REPL._send_to_llm()
# ---------------------------------------------------------------------------


class TestREPLSendToLLM:
    def test_basic_text_response(self):
        session = MagicMock()
        session.should_compact.return_value = False
        display = MagicMock()
        llm = _make_llm(
            [
                {"type": "text", "delta": "Hi!"},
                {"type": "done", "stop_reason": "stop", "usage": {}},
            ]
        )
        repl = _make_repl(session=session, llm=llm, display=display)
        repl._send_to_llm("Hello")
        display.begin_assistant_turn.assert_called_once()
        display.stream_text.assert_called_once_with("Hi!")
        display.end_assistant_turn.assert_called_once()
        session.add_message.assert_any_call("user", "Hello")
        session.add_message.assert_any_call("assistant", "Hi!")

    def test_llm_error_shows_error_and_returns(self):
        session = MagicMock()
        display = MagicMock()
        llm = MagicMock()
        llm.send.side_effect = LLMError("timeout")
        repl = _make_repl(session=session, llm=llm, display=display)
        repl._send_to_llm("Hello")
        display.show_error.assert_called_once()
        assert "timeout" in display.show_error.call_args[0][0]
        # assistant message must NOT be saved
        for c in session.add_message.call_args_list:
            assert c[0][0] != "assistant"

    def test_session_error_on_user_message_shows_error(self):
        session = MagicMock()
        session.add_message.side_effect = SessionError("disk full")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._send_to_llm("Hello")
        display.show_error.assert_called_once()

    def test_tool_call_executed_and_result_saved(self):
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {"status": "success", "data": {}}

        # First LLM call returns a tool_call; second returns text only
        llm = MagicMock()
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": "read_file",
                        "call_id": "1",
                        "arguments": {"path": "foo.py"},
                    },
                    {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                ]
            ),
            iter(
                [
                    {"type": "text", "delta": "Done."},
                    {"type": "done", "stop_reason": "stop", "usage": {}},
                ]
            ),
        ]

        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )
        repl._send_to_llm("Read foo.py")

        tool_registry.execute.assert_called_once_with(
            "read_file", {"path": "foo.py"}, allow_transient=False
        )
        display.show_tool_call.assert_called_once_with("read_file", {"path": "foo.py"})
        display.show_tool_result.assert_called_once()
        # assistant tool-call message and tool result saved via add_raw_message
        raw_calls = session.add_raw_message.call_args_list
        assert any(
            c[0][0]["role"] == "assistant" and "tool_calls" in c[0][0]
            for c in raw_calls
        )
        assert any(
            c[0][0]["role"] == "tool" and "tool_call_id" in c[0][0] for c in raw_calls
        )

    def test_show_tool_call_before_execute(self):
        """show_tool_call must be called before execute()."""
        call_order = []
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()
        display.show_tool_call.side_effect = lambda *a, **kw: call_order.append("show")
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.side_effect = lambda *a, **kw: (
            call_order.append("exec") or {"status": "success", "data": {}}
        )

        llm = MagicMock()
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": "read_file",
                        "call_id": "1",
                        "arguments": {},
                    },
                    {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                ]
            ),
            iter(
                [
                    {"type": "text", "delta": "Done."},
                    {"type": "done", "stop_reason": "stop", "usage": {}},
                ]
            ),
        ]

        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )
        repl._send_to_llm("go")
        assert call_order == ["show", "exec"]

    def test_tool_call_depth_limit(self):
        session = MagicMock()
        session.get_messages.return_value = []
        session.should_compact.return_value = False
        display = MagicMock()
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {"status": "success", "data": {}}

        # LLM always returns a tool_call — never stops
        llm = MagicMock()
        llm.send.return_value = iter(
            [
                {
                    "type": "tool_call",
                    "name": "read_file",
                    "call_id": "1",
                    "arguments": {},
                },
                {"type": "done", "stop_reason": "tool_calls", "usage": {}},
            ]
        )

        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )

        with patch.object(
            llm,
            "send",
            side_effect=[
                iter(
                    [
                        {
                            "type": "tool_call",
                            "name": "read_file",
                            "call_id": str(i),
                            "arguments": {},
                        },
                        {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                    ]
                )
                for i in range(_DEFAULT_MAX_TOOL_ROUNDS)
            ],
        ):
            repl._send_to_llm("loop forever")

        display.show_error.assert_called_once()
        assert str(_DEFAULT_MAX_TOOL_ROUNDS) in display.show_error.call_args[0][0]

    def test_empty_text_response_not_saved(self):
        session = MagicMock()
        session.should_compact.return_value = False
        display = MagicMock()
        llm = _make_llm([{"type": "done", "stop_reason": "stop", "usage": {}}])
        repl = _make_repl(session=session, llm=llm, display=display)
        repl._send_to_llm("Hello")
        saved_roles = [c[0][0] for c in session.add_message.call_args_list]
        assert "assistant" not in saved_roles

    def _tool_call_round(self, tool_name, result_data):
        """Helper: LLM returns one tool_call then stops; tool returns result_data."""
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {
            "status": "success",
            "data": result_data,
        }
        llm = MagicMock()
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": tool_name,
                        "call_id": "1",
                        "arguments": {},
                    },
                    {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                ]
            ),
            iter([{"type": "done", "stop_reason": "stop", "usage": {}}]),
        ]
        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )
        return repl, tool_registry

    def test_transient_schemas_only_accepted_from_tool_manager(self):
        # A non-tool_manager tool returning transient_schemas must be ignored.
        schema = {"type": "function", "function": {"name": "read_file"}}
        repl, tool_registry = self._tool_call_round(
            "malicious_tool", {"transient_schemas": [schema]}
        )
        tool_registry.get.return_value = MagicMock()  # simulate registered tool
        repl._send_to_llm("exploit")
        assert repl._main_agent._pending_transients == {}

    def test_transient_schemas_rejected_for_unregistered_names(self):
        # Even from tool_manager, schemas for unknown tools must not be accepted.
        schema = {"type": "function", "function": {"name": "ghost_tool"}}
        repl, tool_registry = self._tool_call_round(
            "tool_manager", {"transient_schemas": [schema]}
        )
        tool_registry.get.return_value = None  # "ghost_tool" not in registry
        repl._send_to_llm("enable ghost")
        assert repl._main_agent._pending_transients == {}

    def test_transient_schemas_accepted_from_tool_manager_for_known_tool(self):
        # tool_manager returning a schema for a registered tool name is accepted
        # and injected into the *next* LLM call's tools list.
        schema = {"type": "function", "function": {"name": "read_file"}}
        repl, tool_registry = self._tool_call_round(
            "tool_manager", {"transient_schemas": [schema]}
        )
        tool_registry.get.return_value = MagicMock()  # "read_file" is registered
        repl._send_to_llm("enable read_file")
        # The second llm.send call (round 2) receives the transient schema.
        llm = repl._llm
        second_tools = llm.send.call_args_list[1][1]["tools"]
        assert any(t["function"]["name"] == "read_file" for t in second_tools)

    def test_disallowed_tool_shows_user_hint_and_sends_unknown_tool_to_llm(self):
        # When the registry returns tool_disallowed, the REPL must:
        # 1. Show a user-facing hint via show_error (with correct wording).
        # 2. Replace the result with unknown_tool before it reaches the LLM.
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {
            "status": "error",
            "error": "tool_disallowed",
            "message": "Tool 'secret' is not available.",
            "code": 403,
        }

        llm = MagicMock()
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": "secret",
                        "call_id": "1",
                        "arguments": {},
                    },
                    {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                ]
            ),
            iter([{"type": "done", "stop_reason": "stop", "usage": {}}]),
        ]

        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )
        repl._send_to_llm("use secret")

        # User sees the hint with the correct wording.
        display.show_error.assert_called_once()
        hint = display.show_error.call_args[0][0]
        assert "secret" in hint
        assert "allow" in hint
        assert "list of available tools" in hint

        # The message saved for the LLM must contain unknown_tool, not tool_disallowed.
        tool_result_msgs = [
            c[0][0]
            for c in session.add_raw_message.call_args_list
            if c[0][0].get("role") == "tool"
        ]
        assert len(tool_result_msgs) == 1
        import json as _json

        content = _json.loads(tool_result_msgs[0]["content"])
        assert content.get("error") == "unknown_tool"
        assert content.get("error") != "tool_disallowed"

    def test_tools_list_deduplicates_by_name(self):
        # If a transient schema and an already-enabled definition share a name,
        # only one schema should be sent to the LLM (transient takes precedence).
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()

        enabled_schema = {
            "type": "function",
            "function": {"name": "read_file", "description": "enabled version"},
        }
        transient_schema = {
            "type": "function",
            "function": {"name": "read_file", "description": "transient version"},
        }

        tool_registry = MagicMock()
        tool_registry.definitions.return_value = [enabled_schema]
        tool_registry.execute.return_value = {"status": "success", "data": {}}
        tool_registry.get.return_value = MagicMock()

        llm = MagicMock()
        llm.send.return_value = iter(
            [{"type": "done", "stop_reason": "stop", "usage": {}}]
        )

        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )
        repl._main_agent._pending_transients = {"read_file": transient_schema}
        repl._send_to_llm("go")

        sent_tools = llm.send.call_args[1]["tools"]
        read_file_schemas = [
            t for t in sent_tools if t["function"]["name"] == "read_file"
        ]
        assert len(read_file_schemas) == 1
        assert read_file_schemas[0]["function"]["description"] == "transient version"

    def test_pause_resume_bracket_permission_prompt_and_fn_restored(self):
        """pause()/resume() bracket the permission prompt; prompt_fn restored after."""
        from ai_cli.core.permission_manager import PermissionManager

        call_order: list[str] = []
        original_prompt_fn = MagicMock(return_value=("yes", ""))
        pm = PermissionManager(prompt_fn=original_prompt_fn)

        tool_registry = MagicMock()
        tool_registry.permission_manager = pm
        tool_registry.definitions.return_value = []

        # Simulate a tool that internally calls pm.prompt_fn (as a real permission
        # check would).  At execution time pm.prompt_fn is the wrapped version
        # installed by _send_to_llm, so pause/resume are recorded through it.
        def _execute_with_permission(name, args, **kwargs):
            pm.prompt_fn("Allow?", [])
            return {"status": "success", "data": {}}

        tool_registry.execute.side_effect = _execute_with_permission

        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()

        llm = MagicMock()
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": "write_file",
                        "call_id": "1",
                        "arguments": {},
                    },
                    {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                ]
            ),
            iter([{"type": "done", "stop_reason": "stop", "usage": {}}]),
        ]

        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )

        with patch("ai_cli.cli.repl._AbortMonitor") as MockMonitor:
            monitor_instance = MockMonitor.return_value
            monitor_instance.pause.side_effect = lambda: call_order.append("pause")
            monitor_instance.resume.side_effect = lambda: call_order.append("resume")
            repl._send_to_llm("do something")

        # pause → (original prompt_fn) → resume must appear in that order.
        assert "pause" in call_order, "monitor.pause() was never called"
        assert "resume" in call_order, "monitor.resume() was never called"
        assert call_order.index("pause") < call_order.index("resume")
        # The original prompt_fn must be invoked (via the wrapper).
        original_prompt_fn.assert_called_once_with("Allow?", [])
        # prompt_fn must be restored to the original after _send_to_llm returns.
        assert pm.prompt_fn is original_prompt_fn

    def test_abort_injects_stub_tool_results_for_unexecuted_calls(self):
        """On abort after assistant message is saved, stub role:tool msgs are injected."""
        import json as _json
        import threading

        abort_event = threading.Event()

        # add_raw_message sets abort after the assistant message is persisted so
        # that the abort fires at the start of the tool-execution loop.
        raw_messages: list[dict] = []

        def _add_raw(msg):
            raw_messages.append(msg)
            if msg.get("role") == "assistant":
                abort_event.set()

        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        session.add_raw_message.side_effect = _add_raw
        display = MagicMock()

        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []

        # Two tool calls in the same response.
        llm = MagicMock()
        llm.send.return_value = iter(
            [
                {
                    "type": "tool_call",
                    "name": "write_file",
                    "call_id": "call-A",
                    "arguments": {},
                },
                {
                    "type": "tool_call",
                    "name": "write_file",
                    "call_id": "call-B",
                    "arguments": {},
                },
                {"type": "done", "stop_reason": "tool_calls", "usage": {}},
            ]
        )

        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )

        with patch("ai_cli.cli.repl._AbortMonitor") as MockMonitor:
            MockMonitor.return_value.start.side_effect = lambda: None

            # Replace abort in Agent.run with our controlled event.
            original_run = repl._main_agent.run

            def _patched_run(prompt, *, abort=None):
                return original_run(prompt, abort=abort_event)

            repl._main_agent.run = _patched_run
            repl._send_to_llm("write two files")

        # add_raw_message must have been called with stub results for BOTH call_ids.
        raw_tool_msgs = [m for m in raw_messages if m.get("role") == "tool"]
        tool_call_ids = {m["tool_call_id"] for m in raw_tool_msgs}
        assert "call-A" in tool_call_ids
        assert "call-B" in tool_call_ids

        # Each stub must carry the abort error payload.
        for msg in raw_tool_msgs:
            content = _json.loads(msg["content"])
            assert content["status"] == "error"
            assert content["error"] == "aborted"

        # User must be shown "Aborted."
        status_msgs = [c[0][0] for c in display.show_status.call_args_list]
        assert any("Aborted" in s for s in status_msgs)


# ---------------------------------------------------------------------------
# REPL._preprocess_at_references()
# ---------------------------------------------------------------------------


class TestREPLAtReferences:
    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self._root = tmp_path

    def _make_workspace(self, is_ignored: bool = False, root=None):
        ws = MagicMock()
        ws.root = root if root is not None else self._root
        ws.is_ignored.return_value = is_ignored
        return ws

    def test_no_references_unchanged(self):
        repl = _make_repl()
        assert repl._preprocess_at_references("Hello world") == "Hello world"

    def test_at_reference_replaced_with_content(self):
        (self._root / "src").mkdir()
        (self._root / "src" / "foo.py").write_text("line1\nline2\n")
        repl = _make_repl(workspace=self._make_workspace())
        result = repl._preprocess_at_references("Check @src/foo.py please")
        assert "[file: src/foo.py]" in result
        assert "line1" in result
        assert "[/file]" in result

    def test_at_reference_missing_file_aborts(self):
        display = MagicMock()
        repl = _make_repl(workspace=self._make_workspace(), display=display)
        result = repl._preprocess_at_references("Check @missing.py please")
        assert result is None
        display.show_error.assert_called_once()
        # Error message should not leak the resolved absolute path.
        msg = display.show_error.call_args[0][0]
        assert str(self._root) not in msg

    def test_at_bang_bypasses_ignore(self):
        (self._root / "secret.key").write_text("secret\n")
        repl = _make_repl(workspace=self._make_workspace(is_ignored=True))
        result = repl._preprocess_at_references("See @!secret.key")
        assert "secret" in result

    def test_dotdot_path_reads_from_parent(self):
        """@../file reads a file outside the workspace root."""
        workspace_root = self._root / "workspace"
        workspace_root.mkdir()
        (self._root / "escape.py").write_text("outside_content\n")
        repl = _make_repl(workspace=self._make_workspace(root=workspace_root))
        result = repl._preprocess_at_references("@../escape.py")
        assert "outside_content" in result

    def test_os_error_aborts(self):
        """A missing file (OSError) returns None and shows an error."""
        display = MagicMock()
        repl = _make_repl(workspace=self._make_workspace(), display=display)
        result = repl._preprocess_at_references("@nonexistent.py")
        assert result is None
        display.show_error.assert_called_once()
        # Error message should use strerror, not the full exception (no resolved path).
        msg = display.show_error.call_args[0][0]
        assert str(self._root) not in msg

    def test_resolve_os_error_aborts(self):
        """resolve() failure (e.g. symlink loop) returns None and shows an error."""
        display = MagicMock()
        repl = _make_repl(workspace=self._make_workspace(), display=display)
        with patch("pathlib.Path.resolve", side_effect=OSError("symlink loop")):
            result = repl._preprocess_at_references("@loop.py")
        assert result is None
        display.show_error.assert_called_once()

    def test_multiple_references_all_replaced(self):
        (self._root / "a.py").write_text("content_a\n")
        (self._root / "b.py").write_text("content_b\n")
        repl = _make_repl(workspace=self._make_workspace())
        result = repl._preprocess_at_references("@a.py and @b.py")
        assert "content_a" in result
        assert "content_b" in result


class TestREPLAtReferencesImages:
    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self._root = tmp_path

    @staticmethod
    def _valid_png() -> bytes:
        import io as _io

        from PIL import Image as _PILImage

        buf = _io.BytesIO()
        _PILImage.new("RGB", (1, 1)).save(buf, format="PNG")
        return buf.getvalue()

    def _make_image_workspace(self, img_bytes: bytes | None = None):
        if img_bytes is None:
            img_bytes = self._valid_png()
        workspace = MagicMock()
        workspace.root = self._root
        workspace.is_ignored.return_value = False
        for name in [
            "diagram.png",
            "shot.png",
            "photo.jpg",
            "pic.webp",
            "only.gif",
            "img.png",
            "img.jpg",
            "img.jpeg",
            "img.gif",
            "img.webp",
            "secret.png",
        ]:
            (self._root / name).write_bytes(img_bytes)
        return workspace

    def test_image_returns_content_block_list(self):
        repl = _make_repl(workspace=self._make_image_workspace())
        result = repl._preprocess_at_references("Look at @diagram.png")
        assert isinstance(result, list)

    def test_image_block_has_correct_structure(self):
        repl = _make_repl(workspace=self._make_image_workspace())
        result = repl._preprocess_at_references("@shot.png")
        assert isinstance(result, list)
        img_block = next(b for b in result if b.get("type") == "image_url")
        assert img_block["image_url"]["url"].startswith("data:image/png;base64,")
        assert img_block["image_url"]["detail"] == "auto"

    def test_image_text_preserved_as_text_block(self):
        repl = _make_repl(workspace=self._make_image_workspace())
        result = repl._preprocess_at_references("Describe this @photo.jpg please")
        assert isinstance(result, list)
        all_text = " ".join(b["text"] for b in result if b.get("type") == "text")
        assert "Describe this" in all_text
        assert "please" in all_text

    def test_image_token_removed_from_text_block(self):
        repl = _make_repl(workspace=self._make_image_workspace())
        result = repl._preprocess_at_references("See @pic.webp")
        assert isinstance(result, list)
        # Token itself should not appear in any text block
        for block in result:
            if block.get("type") == "text":
                assert "@pic.webp" not in block["text"]

    def test_image_only_no_text_block(self):
        repl = _make_repl(workspace=self._make_image_workspace())
        result = repl._preprocess_at_references("@only.gif")
        assert isinstance(result, list)
        assert not any(b.get("type") == "text" for b in result)
        assert any(b.get("type") == "image_url" for b in result)

    def test_supported_extensions(self):
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            repl = _make_repl(workspace=self._make_image_workspace())
            result = repl._preprocess_at_references(f"@img{ext}")
            assert isinstance(result, list), f"Expected list for {ext}"

    def test_image_mime_type_jpeg(self):
        repl = _make_repl(workspace=self._make_image_workspace())
        result = repl._preprocess_at_references("@photo.jpg")
        assert isinstance(result, list)
        img_block = next(b for b in result if b.get("type") == "image_url")
        assert "image/jpeg" in img_block["image_url"]["url"]

    def test_image_bypasses_ignore_rules(self):
        workspace = self._make_image_workspace()
        repl = _make_repl(workspace=workspace)
        repl._preprocess_at_references("@secret.png")
        workspace.file_exists.assert_not_called()

    def test_image_bang_also_works(self):
        workspace = self._make_image_workspace()
        repl = _make_repl(workspace=workspace)
        result = repl._preprocess_at_references("@!diagram.png")
        assert isinstance(result, list)
        assert any(b.get("type") == "image_url" for b in result)

    def test_image_read_error_aborts(self):
        # No file written — read_bytes() raises FileNotFoundError (subclass of OSError)
        workspace = MagicMock()
        workspace.root = self._root
        workspace.is_ignored.return_value = False
        display = MagicMock()
        repl = _make_repl(workspace=workspace, display=display)
        result = repl._preprocess_at_references("@bad.png")
        assert result is None
        display.show_error.assert_called_once()

    def test_mixed_text_and_image(self):
        (self._root / "src.py").write_text("def foo(): pass\n")
        (self._root / "shot.png").write_bytes(self._valid_png())
        workspace = MagicMock()
        workspace.root = self._root
        workspace.is_ignored.return_value = False
        repl = _make_repl(workspace=workspace)
        result = repl._preprocess_at_references("code @src.py image @shot.png")
        assert isinstance(result, list)
        assert any(b.get("type") == "text" for b in result)
        assert any(b.get("type") == "image_url" for b in result)

    def test_interleaved_ordering_preserved(self):
        """text @img text @img should produce text→image→text→image blocks."""
        png = self._valid_png()
        (self._root / "a.png").write_bytes(png)
        (self._root / "b.png").write_bytes(png)
        workspace = MagicMock()
        workspace.root = self._root
        workspace.is_ignored.return_value = False
        repl = _make_repl(workspace=workspace)
        result = repl._preprocess_at_references("before @a.png middle @b.png after")
        assert isinstance(result, list)
        types = [b.get("type") for b in result]
        assert types.count("image_url") == 2
        # text before first image, image, text between, image, text after
        first_img = types.index("image_url")
        second_img = types.index("image_url", first_img + 1)
        assert any(t == "text" for t in types[:first_img])
        assert any(t == "text" for t in types[first_img + 1 : second_img])
        assert any(t == "text" for t in types[second_img + 1 :])

    def test_image_size_limit_exceeded(self):
        """Images exceeding max_pixels_per_image are rejected."""
        import io as _io

        from PIL import Image as _PILImage

        buf = _io.BytesIO()
        _PILImage.new("RGB", (100, 100)).save(buf, format="PNG")
        (self._root / "large.png").write_bytes(buf.getvalue())

        workspace = MagicMock()
        workspace.root = self._root
        workspace.is_ignored.return_value = False
        display = MagicMock()

        config = MagicMock()
        # Set limit below 100×100 = 10000 pixels
        config.get.side_effect = lambda key, default=None: (
            50 * 50 if key == "max_pixels_per_image" else default
        )

        repl = REPL(
            session=MagicMock(),
            tool_registry=MagicMock(),
            llm_client=_make_llm(),
            display=display,
            workspace=workspace,
            config=config,
        )
        result = repl._preprocess_at_references("@large.png")
        assert result is None
        display.show_error.assert_called_once()

    def test_image_within_size_limit_accepted(self):
        """Images within max_pixels_per_image are accepted."""
        import io as _io

        from PIL import Image as _PILImage

        buf = _io.BytesIO()
        _PILImage.new("RGB", (10, 10)).save(buf, format="PNG")
        (self._root / "small.png").write_bytes(buf.getvalue())

        workspace = MagicMock()
        workspace.root = self._root
        workspace.is_ignored.return_value = False

        repl = _make_repl(workspace=workspace)
        result = repl._preprocess_at_references("@small.png")
        assert isinstance(result, list)
        assert any(b.get("type") == "image_url" for b in result)

    def test_image_corrupt_aborts(self):
        """Pillow decode failure aborts the send."""
        (self._root / "corrupt.png").write_bytes(b"not an image")
        workspace = MagicMock()
        workspace.root = self._root
        workspace.is_ignored.return_value = False
        display = MagicMock()
        repl = _make_repl(workspace=workspace, display=display)
        result = repl._preprocess_at_references("@corrupt.png")
        assert result is None
        display.show_error.assert_called_once()

    def test_max_pixels_zero_falls_back_to_default(self):
        """A zero max_pixels_per_image in config falls back to the default."""
        import io as _io

        from PIL import Image as _PILImage

        buf = _io.BytesIO()
        _PILImage.new("RGB", (10, 10)).save(buf, format="PNG")
        (self._root / "small.png").write_bytes(buf.getvalue())

        workspace = MagicMock()
        workspace.root = self._root
        workspace.is_ignored.return_value = False

        config = MagicMock()
        config.get.side_effect = lambda key, default=None: (
            0 if key == "max_pixels_per_image" else default
        )
        repl = REPL(
            session=MagicMock(),
            tool_registry=MagicMock(),
            llm_client=_make_llm(),
            display=MagicMock(),
            workspace=workspace,
            config=config,
        )
        # A 10×10 image (100 px) is well within the default limit (2,073,600),
        # so it should be accepted even though max_pixels=0 was configured.
        result = repl._preprocess_at_references("@small.png")
        assert isinstance(result, list)
        assert any(b.get("type") == "image_url" for b in result)

    def test_max_pixels_invalid_string_falls_back_to_default(self):
        """A non-numeric max_pixels_per_image falls back to the default."""
        import io as _io

        from PIL import Image as _PILImage

        buf = _io.BytesIO()
        _PILImage.new("RGB", (10, 10)).save(buf, format="PNG")
        (self._root / "small.png").write_bytes(buf.getvalue())

        workspace = MagicMock()
        workspace.root = self._root
        workspace.is_ignored.return_value = False

        config = MagicMock()
        config.get.side_effect = lambda key, default=None: (
            "bad" if key == "max_pixels_per_image" else default
        )
        repl = REPL(
            session=MagicMock(),
            tool_registry=MagicMock(),
            llm_client=_make_llm(),
            display=MagicMock(),
            workspace=workspace,
            config=config,
        )
        result = repl._preprocess_at_references("@small.png")
        assert isinstance(result, list)
        assert any(b.get("type") == "image_url" for b in result)

    def test_max_pixels_bool_falls_back_to_default(self):
        """A boolean max_pixels_per_image (e.g. YAML true/false) falls back to default."""
        workspace = self._make_image_workspace()
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: (
            True if key == "max_pixels_per_image" else default
        )
        repl = REPL(
            session=MagicMock(),
            tool_registry=MagicMock(),
            llm_client=_make_llm(),
            display=MagicMock(),
            workspace=workspace,
            config=config,
        )
        # True as int would be 1 pixel, rejecting almost everything.
        # With the bool guard the default limit applies, so a 1×1 image passes.
        result = repl._preprocess_at_references("@img.png")
        assert isinstance(result, list)
        assert any(b.get("type") == "image_url" for b in result)

    def test_send_to_llm_uses_add_raw_message_for_image(self):
        workspace = self._make_image_workspace()
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        llm = _make_llm()
        repl = _make_repl(workspace=workspace, session=session, llm=llm)
        repl._handle_input("Look @diagram.png")
        session.add_raw_message.assert_called_once()
        call_arg = session.add_raw_message.call_args[0][0]
        assert call_arg["role"] == "user"
        assert isinstance(call_arg["content"], list)

    def test_handle_input_aborts_send_on_image_error(self):
        """_handle_input does not call _send_to_llm when an @ reference fails."""
        # No file written — image read will fail.
        workspace = MagicMock()
        workspace.root = self._root
        workspace.is_ignored.return_value = False
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(workspace=workspace, session=session, display=display)
        repl._handle_input("See @missing.png")
        session.add_message.assert_not_called()
        session.add_raw_message.assert_not_called()
        display.show_error.assert_called_once()


# ---------------------------------------------------------------------------
# REPL._check_compaction()
# ---------------------------------------------------------------------------


class TestREPLCompaction:
    def test_no_compaction_when_not_needed(self):
        session = MagicMock()
        session.should_compact.return_value = False
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._check_compaction()
        session.compact.assert_not_called()
        display.show_status.assert_not_called()

    def test_compaction_triggered_when_needed(self):
        session = MagicMock()
        session.should_compact.return_value = True
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._check_compaction()
        session.compact.assert_called_once()
        assert display.show_status.call_count == 2  # before and after

    def test_compaction_error_shown(self):
        session = MagicMock()
        session.should_compact.return_value = True
        session.compact.side_effect = SessionError("LLM unavailable")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._check_compaction()
        display.show_error.assert_called_once()


# ---------------------------------------------------------------------------
# /compact with instructions
# ---------------------------------------------------------------------------


class TestCompactSubcommand:
    def test_compact_no_instructions_passes_empty_string(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("compact")
        session.compact.assert_called_once_with(instructions="")

    def test_compact_with_instructions_passes_them(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("compact Summarise the key decisions only")
        session.compact.assert_called_once_with(
            instructions="Summarise the key decisions only"
        )

    def test_compact_error_shows_error(self):
        session = MagicMock()
        session.compact.side_effect = SessionError("LLM down")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("compact focus on bugs")
        display.show_error.assert_called_once()


# ---------------------------------------------------------------------------
# /tools subcommands
# ---------------------------------------------------------------------------


class TestToolsSubcommand:
    def _reg(self, all_enabled=None, all_tools_info=None, tool_info_val=None):
        tr = MagicMock()
        tr.all_enabled.return_value = all_enabled if all_enabled is not None else []
        tr.all_tools_info.return_value = (
            all_tools_info if all_tools_info is not None else []
        )
        tr.tool_info.return_value = tool_info_val
        return tr

    def test_tools_no_subcommand_shows_enabled_list(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools")
        display.show_tool_list.assert_called_once_with([])

    def test_tools_list_shows_all_tools(self):
        info_list = [{"name": "echo", "enabled": True, "allowed": True}]
        tool_registry = self._reg(all_tools_info=info_list)
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools list")
        display.show_tool_list_all.assert_called_once_with(info_list)

    def test_tools_info_known_tool(self):
        info = {"name": "echo", "description": "Echoes."}
        tool_registry = self._reg(tool_info_val=info)
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools info echo")
        tool_registry.tool_info.assert_called_once_with("echo")
        display.show_tool_info.assert_called_once_with(info)

    def test_tools_info_unknown_tool_shows_error(self):
        tool_registry = self._reg(tool_info_val=None)
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools info ghost")
        display.show_error.assert_called_once()

    def test_tools_info_missing_name_shows_error(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools info")
        display.show_error.assert_called_once()

    def test_tools_enable_calls_enable(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools enable read_file")
        tool_registry.enable.assert_called_once_with("read_file")
        display.show_status.assert_called_once()

    def test_tools_disable_calls_disable(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools disable write_file")
        tool_registry.disable.assert_called_once_with("write_file")

    def test_tools_enable_session_calls_enable_session(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools enable --session read_file")
        tool_registry.enable_session.assert_called_once_with("read_file")
        tool_registry.enable.assert_not_called()

    def test_tools_disable_session_calls_disable_session(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools disable --session write_file")
        tool_registry.disable_session.assert_called_once_with("write_file")

    def test_tools_allow_calls_allow(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools allow bash")
        tool_registry.allow.assert_called_once_with("bash")

    def test_tools_disallow_calls_disallow(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools disallow bash")
        tool_registry.disallow.assert_called_once_with("bash")

    def test_tools_allow_session(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools allow --session bash")
        tool_registry.allow_session.assert_called_once_with("bash")
        tool_registry.allow.assert_not_called()

    def test_tools_disallow_session(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools disallow --session bash")
        tool_registry.disallow_session.assert_called_once_with("bash")
        tool_registry.disallow.assert_not_called()

    def test_tools_enable_missing_name_shows_error(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools enable")
        display.show_error.assert_called_once()

    def test_tools_enable_unknown_tool_shows_error(self):
        tool_registry = self._reg()
        tool_registry.get.return_value = None  # unknown tool
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools enable ghost_tool")
        display.show_error.assert_called_once()
        tool_registry.enable.assert_not_called()

    def test_tools_unknown_subcommand_shows_error(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools frobnicate")
        display.show_error.assert_called_once()
        assert "frobnicate" in display.show_error.call_args[0][0]

    def test_status_message_uses_correct_past_tense(self):
        for sub, expected_past in [
            ("enable", "enabled"),
            ("disable", "disabled"),
            ("allow", "allowed"),
            ("disallow", "disallowed"),
        ]:
            tool_registry = self._reg()
            display = MagicMock()
            repl = _make_repl(tool_registry=tool_registry, display=display)
            repl._handle_slash_command(f"tools {sub} read_file")
            msg = display.show_status.call_args[0][0]
            assert expected_past in msg, (
                f"{sub!r} → expected {expected_past!r} in {msg!r}"
            )

    def test_status_message_mentions_scope_persistent(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools enable read_file")
        msg = display.show_status.call_args[0][0]
        assert "persistently" in msg

    def test_status_message_mentions_scope_session(self):
        tool_registry = self._reg()
        display = MagicMock()
        repl = _make_repl(tool_registry=tool_registry, display=display)
        repl._handle_slash_command("tools enable --session read_file")
        msg = display.show_status.call_args[0][0]
        assert "session" in msg


# ---------------------------------------------------------------------------
# /session subcommands
# ---------------------------------------------------------------------------


class TestSessionSubcommand:
    def test_session_no_subcommand_shows_info(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session")
        display.show_session_info.assert_called_once_with(session)

    def test_session_name_calls_set_name(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session name my-chat")
        session.set_name.assert_called_once_with("my-chat")
        display.show_status.assert_called_once()

    def test_session_name_with_spaces(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session name bug fix session")
        session.set_name.assert_called_once_with("bug fix session")

    def test_session_name_missing_name_shows_error(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session name")
        display.show_error.assert_called_once()
        session.set_name.assert_not_called()

    def test_session_name_error_shows_error(self):
        session = MagicMock()
        session.set_name.side_effect = SessionError("disk full")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session name my-name")
        display.show_error.assert_called_once()

    def test_session_unknown_subcommand_shows_error(self):
        session = MagicMock()
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("session frobnicate")
        display.show_error.assert_called_once()
        assert "frobnicate" in display.show_error.call_args[0][0]


# ---------------------------------------------------------------------------
# /history command
# ---------------------------------------------------------------------------


class TestHistoryCommand:
    def test_history_calls_show_history(self):
        session = MagicMock()
        messages = [{"role": "user", "content": "hi"}]
        session.get_messages.return_value = messages
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("history")
        display.show_history.assert_called_once_with(messages)

    def test_history_session_error_shows_error(self):
        session = MagicMock()
        session.get_messages.side_effect = SessionError("disk full")
        display = MagicMock()
        repl = _make_repl(session=session, display=display)
        repl._handle_slash_command("history")
        display.show_error.assert_called_once()
        display.show_history.assert_not_called()


# ---------------------------------------------------------------------------
# REPL reasoning chunk routing and update_usage
# ---------------------------------------------------------------------------


class TestREPLReasoningAndUsage:
    def test_reasoning_chunk_routed_to_stream_reasoning(self):
        session = MagicMock()
        session.should_compact.return_value = False
        display = MagicMock()
        llm = _make_llm(
            [
                {"type": "reasoning", "delta": "thinking..."},
                {"type": "text", "delta": "answer"},
                {"type": "done", "stop_reason": "stop", "usage": {}},
            ]
        )
        repl = _make_repl(session=session, llm=llm, display=display)
        repl._send_to_llm("hello")
        display.stream_reasoning.assert_called_once_with("thinking...")

    def test_text_chunk_not_routed_to_stream_reasoning(self):
        session = MagicMock()
        session.should_compact.return_value = False
        display = MagicMock()
        llm = _make_llm(
            [
                {"type": "text", "delta": "answer"},
                {"type": "done", "stop_reason": "stop", "usage": {}},
            ]
        )
        repl = _make_repl(session=session, llm=llm, display=display)
        repl._send_to_llm("hello")
        display.stream_reasoning.assert_not_called()

    def test_update_usage_called_on_done_chunk(self):
        session = MagicMock()
        session.should_compact.return_value = False
        display = MagicMock()
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        llm = _make_llm(
            [
                {"type": "text", "delta": "Hi"},
                {"type": "done", "stop_reason": "stop", "usage": usage},
            ]
        )
        repl = _make_repl(session=session, llm=llm, display=display)
        repl._send_to_llm("hello")
        display.update_usage.assert_called_once()
        call_usage = display.update_usage.call_args[0][0]
        assert call_usage == usage


# ---------------------------------------------------------------------------
# REPL format_display plumbing
# ---------------------------------------------------------------------------


class TestREPLFormatDisplay:
    def _make_tool_call_round(self, display_str):
        """LLM calls read_file once, tool returns success; tool.format_display returns display_str."""
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()
        tool_obj = MagicMock()
        tool_obj.format_display.return_value = display_str
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {"status": "success", "data": {}}
        tool_registry.get.return_value = tool_obj

        llm = MagicMock()
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": "read_file",
                        "call_id": "1",
                        "arguments": {"path": "foo.py"},
                    },
                    {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                ]
            ),
            iter([{"type": "done", "stop_reason": "stop", "usage": {}}]),
        ]
        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )
        return repl, display, tool_obj

    def test_format_display_called_with_args_and_result(self):
        repl, display, tool_obj = self._make_tool_call_round("formatted")
        repl._send_to_llm("read foo.py")
        tool_obj.format_display.assert_called_once_with(
            args={"path": "foo.py"}, result={"status": "success", "data": {}}
        )

    def test_format_display_result_passed_to_show_tool_result(self):
        repl, display, _ = self._make_tool_call_round("my custom string")
        repl._send_to_llm("read foo.py")
        display.show_tool_result.assert_called_once_with(
            "read_file", {"status": "success", "data": {}}, "my custom string"
        )

    def test_format_display_none_passed_when_returns_none(self):
        repl, display, _ = self._make_tool_call_round(None)
        repl._send_to_llm("read foo.py")
        display.show_tool_result.assert_called_once_with(
            "read_file", {"status": "success", "data": {}}, None
        )

    def test_format_display_exception_does_not_propagate(self):
        session = MagicMock()
        session.should_compact.return_value = False
        session.get_messages.return_value = []
        display = MagicMock()
        tool_obj = MagicMock()
        tool_obj.format_display.side_effect = RuntimeError("crash")
        tool_registry = MagicMock()
        tool_registry.definitions.return_value = []
        tool_registry.execute.return_value = {"status": "success", "data": {}}
        tool_registry.get.return_value = tool_obj

        llm = MagicMock()
        llm.send.side_effect = [
            iter(
                [
                    {
                        "type": "tool_call",
                        "name": "read_file",
                        "call_id": "1",
                        "arguments": {},
                    },
                    {"type": "done", "stop_reason": "tool_calls", "usage": {}},
                ]
            ),
            iter([{"type": "done", "stop_reason": "stop", "usage": {}}]),
        ]
        repl = _make_repl(
            session=session, tool_registry=tool_registry, llm=llm, display=display
        )
        repl._send_to_llm("go")  # should not raise
        display.show_tool_result.assert_called_once_with(
            "read_file", {"status": "success", "data": {}}, None
        )


# ---------------------------------------------------------------------------
# /rounds command
# ---------------------------------------------------------------------------


class TestRoundsCommand:
    def test_rounds_session_updates_attribute(self):
        repl = _make_repl()
        repl._handle_input("/rounds --session 5")
        assert repl._max_tool_rounds == 5

    def test_rounds_session_no_persist(self, tmp_path):
        workspace = MagicMock()
        workspace.root = tmp_path
        repl = _make_repl(workspace=workspace)
        repl._handle_input("/rounds --session 3")
        config_path = tmp_path / ".ai-cli" / "config.yaml"
        assert not config_path.exists()

    def test_rounds_persistent_writes_config(self, tmp_path):
        import yaml as _yaml

        dot = tmp_path / ".ai-cli"
        dot.mkdir()
        workspace = MagicMock()
        workspace.root = tmp_path
        repl = _make_repl(workspace=workspace)
        repl._handle_input("/rounds 7")
        config_path = dot / "config.yaml"
        assert config_path.exists()
        data = _yaml.safe_load(config_path.read_text())
        assert data["max_tool_rounds"] == 7

    def test_rounds_invalid_value(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._handle_input("/rounds abc")
        display.show_error.assert_called_once()
        assert repl._max_tool_rounds == _DEFAULT_MAX_TOOL_ROUNDS

    def test_rounds_zero_rejected(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._handle_input("/rounds 0")
        display.show_error.assert_called_once()
        assert repl._max_tool_rounds == _DEFAULT_MAX_TOOL_ROUNDS

    def test_rounds_missing_value(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._handle_input("/rounds")
        display.show_error.assert_called_once()

    def test_rounds_config_initial_value(self):
        config = MagicMock()
        config.get.return_value = 25
        repl = REPL(
            session=MagicMock(),
            tool_registry=MagicMock(),
            llm_client=_make_llm(),
            display=MagicMock(),
            workspace=MagicMock(),
            config=config,
        )
        assert repl._max_tool_rounds == 25

    def test_rounds_in_slash_commands(self):
        cmds = [cmd for cmd, _ in _SLASH_COMMANDS]
        assert any("rounds" in cmd for cmd in cmds)

    def test_rounds_config_invalid_string_falls_back_to_default(self):
        config = MagicMock()
        config.get.return_value = "not-a-number"
        repl = REPL(
            session=MagicMock(),
            tool_registry=MagicMock(),
            llm_client=_make_llm(),
            display=MagicMock(),
            workspace=MagicMock(),
            config=config,
        )
        assert repl._max_tool_rounds == _DEFAULT_MAX_TOOL_ROUNDS

    def test_rounds_config_zero_falls_back_to_default(self):
        config = MagicMock()
        config.get.return_value = 0
        repl = REPL(
            session=MagicMock(),
            tool_registry=MagicMock(),
            llm_client=_make_llm(),
            display=MagicMock(),
            workspace=MagicMock(),
            config=config,
        )
        assert repl._max_tool_rounds == _DEFAULT_MAX_TOOL_ROUNDS

    def test_rounds_config_none_falls_back_to_default(self):
        config = MagicMock()
        config.get.return_value = None
        repl = REPL(
            session=MagicMock(),
            tool_registry=MagicMock(),
            llm_client=_make_llm(),
            display=MagicMock(),
            workspace=MagicMock(),
            config=config,
        )
        assert repl._max_tool_rounds == _DEFAULT_MAX_TOOL_ROUNDS

    def test_rounds_persist_failure_shows_error(self, tmp_path):
        # Simulate a write failure deterministically by patching write_text.
        workspace = MagicMock()
        workspace.root = tmp_path
        display = MagicMock()
        repl = _make_repl(workspace=workspace, display=display)
        with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            repl._handle_input("/rounds 4")
        # Value is still updated in memory even if persist failed.
        assert repl._max_tool_rounds == 4
        display.show_error.assert_called_once()


# ---------------------------------------------------------------------------
# /agents command
# ---------------------------------------------------------------------------


class TestAgentsCommand:
    def test_no_agents_configured_shows_status(self):
        display = MagicMock()
        repl = _make_repl(display=display)
        # No agent_registry passed — _agent_registry is None.
        repl._handle_slash_command("agents")
        display.show_status.assert_called_once()
        display.show_agents.assert_not_called()

    def test_empty_agent_registry_shows_status(self):
        from ai_cli.core.agent_registry import AgentRegistry

        display = MagicMock()
        repl = _make_repl(display=display)
        repl._agent_registry = AgentRegistry({})
        repl._handle_slash_command("agents")
        display.show_status.assert_called_once()
        display.show_agents.assert_not_called()

    def test_agents_present_calls_show_agents(self):
        from ai_cli.core.agent import AgentSpec
        from ai_cli.core.agent_registry import AgentRegistry

        spec = AgentSpec(
            name="explore",
            system_message="You explore.",
            tools=["read_file"],
            model="llama3:8b",
            persistence="ephemeral",
            max_tool_rounds=5,
        )
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._agent_registry = AgentRegistry({"explore": spec})
        repl._handle_slash_command("agents")
        display.show_agents.assert_called_once()
        display.show_status.assert_not_called()

    def test_show_agents_rows_contain_expected_fields(self):
        from ai_cli.core.agent import AgentSpec
        from ai_cli.core.agent_registry import AgentRegistry

        spec = AgentSpec(
            name="coder",
            system_message="You write code.",
            tools=["read_file", "write_file"],
            model="llama3:8b",
            persistence="session",
            max_tool_rounds=7,
        )
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._agent_registry = AgentRegistry({"coder": spec})
        repl._handle_slash_command("agents")
        rows = display.show_agents.call_args[0][0]
        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == "coder"
        assert row["model"] == "llama3:8b"
        assert row["persistence"] == "session"
        assert row["max_tool_rounds"] == 7
        assert "read_file" in row["tools"]
        assert "write_file" in row["tools"]

    def test_show_agents_rows_sorted_by_name(self):
        from ai_cli.core.agent import AgentSpec
        from ai_cli.core.agent_registry import AgentRegistry

        specs = {
            "zebra": AgentSpec(name="zebra", system_message="z", tools=[], model="m"),
            "alpha": AgentSpec(name="alpha", system_message="a", tools=[], model="m"),
        }
        display = MagicMock()
        repl = _make_repl(display=display)
        repl._agent_registry = AgentRegistry(specs)
        repl._handle_slash_command("agents")
        rows = display.show_agents.call_args[0][0]
        assert [r["name"] for r in rows] == ["alpha", "zebra"]


# ---------------------------------------------------------------------------
# /tasks slash command
# ---------------------------------------------------------------------------


def _make_task_manager(
    *,
    root_tasks=None,
    detail_map=None,
    goal=None,
):
    """Return a MagicMock TaskManager with sensible defaults."""
    tm = MagicMock()
    tm.list_tasks.return_value = root_tasks or []
    tm.list_task_details.return_value = root_tasks or []
    tm.get_all_task_details_map.return_value = detail_map or {}
    tm.get_goal.return_value = goal
    return tm


def _make_task_detail(
    name="MyTask", status="not_started", priority="medium", parent_id=None
):
    return {
        "id": f"task_{name.lower()}",
        "parent_id": parent_id,
        "name": name,
        "status": status,
        "priority": priority,
        "description": "A description",
        "definition_of_done": "Done criteria",
        "next_action": "",
        "blockers": [],
        "notes": [],
        "subtasks": [],
    }


class TestTasksCommand:
    # ------------------------------------------------------------------
    # No task_manager → error
    # ------------------------------------------------------------------

    def test_no_task_manager_shows_error(self):
        display = MagicMock()
        repl = _make_repl(display=display)  # task_manager=None
        repl._handle_slash_command("tasks")
        display.show_error.assert_called_once()

    # ------------------------------------------------------------------
    # bare /tasks
    # ------------------------------------------------------------------

    def test_bare_tasks_calls_show_tasks_simple(self):
        display = MagicMock()
        detail = _make_task_detail()
        tm = _make_task_manager(root_tasks=[detail])
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks")
        display.show_tasks_simple.assert_called_once()

    def test_bare_tasks_filters_done(self):
        display = MagicMock()
        done = _make_task_detail("Done", status="done")
        active = _make_task_detail("Active", status="in_progress")
        tm = _make_task_manager(root_tasks=[done, active])
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks")
        nodes = display.show_tasks_simple.call_args[0][0]
        assert all(n["status"] != "done" for n in nodes)

    def test_bare_tasks_shows_goal(self):
        display = MagicMock()
        tm = _make_task_manager(goal="My project goal")
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks")
        status_calls = [str(c) for c in display.show_status.call_args_list]
        assert any("My project goal" in c for c in status_calls)

    def test_bare_tasks_validation_error_shows_error(self):
        from ai_cli.core.task_manager import TaskValidationError

        display = MagicMock()
        tm = _make_task_manager()
        tm.list_task_details.side_effect = TaskValidationError("missing subtask")
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks")
        display.show_error.assert_called_once()
        display.show_tasks_simple.assert_not_called()

    # ------------------------------------------------------------------
    # /tasks list
    # ------------------------------------------------------------------

    def test_list_calls_show_tasks_list(self):
        display = MagicMock()
        detail = _make_task_detail()
        tm = _make_task_manager(detail_map={detail["id"]: detail})
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks list")
        display.show_tasks_list.assert_called_once()
        # Single load via detail_map — list_task_details must not be called
        tm.list_task_details.assert_not_called()

    def test_list_extra_args_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks list Root Extra")
        display.show_error.assert_called_once()
        assert "Usage" in display.show_error.call_args[0][0]
        tm.get_all_task_details_map.assert_not_called()

    def test_list_strips_trailing_dot_before_detail_map_lookup(self):
        display = MagicMock()
        parent = _make_task_detail("Root")
        child = _make_task_detail("Child", parent_id=parent["id"])
        tm = _make_task_manager(detail_map={parent["id"]: parent, child["id"]: child})
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks list Root.")
        display.show_tasks_list.assert_called_once()
        nodes = display.show_tasks_list.call_args[0][0]
        assert [n["name"] for n in nodes] == ["Child"]

    def test_list_dot_only_path_shows_validation_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks list .")
        display.show_error.assert_called_once()
        tm.get_all_task_details_map.assert_not_called()

    # ------------------------------------------------------------------
    # /tasks tree
    # ------------------------------------------------------------------

    def test_tree_calls_show_tasks_tree(self):
        display = MagicMock()
        detail = _make_task_detail("X")  # parent_id=None → root task
        tm = _make_task_manager(detail_map={detail["id"]: detail})
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks tree")
        display.show_tasks_tree.assert_called_once()
        # list_tasks should NOT be called — roots come from the detail_map
        tm.list_tasks.assert_not_called()

    def test_tree_depth_flag_missing_value_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks tree --depth")
        display.show_error.assert_called_once()
        assert "Usage" in display.show_error.call_args[0][0]

    def test_tree_depth_invalid_value_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks tree --depth abc")
        display.show_error.assert_called_once()

    def test_tree_depth_zero_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks tree --depth 0")
        display.show_error.assert_called_once()
        assert "positive integer" in display.show_error.call_args[0][0]

    def test_tree_depth_negative_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks tree --depth -1")
        display.show_error.assert_called_once()
        assert "positive integer" in display.show_error.call_args[0][0]

    def test_tree_extra_positional_args_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks tree Root Extra")
        display.show_error.assert_called_once()
        assert "Usage" in display.show_error.call_args[0][0]
        tm.get_all_task_details_map.assert_not_called()

    def test_tree_depth_zero_in_config_falls_back_to_default(self):
        """tree_depth: 0 is invalid and should fall back to default (3) with a warning."""
        repl = _make_repl()
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: (
            {"tree_depth": 0} if key == "tasks" else default
        )
        repl._config = config
        assert repl._get_tree_depth() == 3

    def test_tree_depth_negative_in_config_falls_back_to_default(self):
        repl = _make_repl()
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: (
            {"tree_depth": -5} if key == "tasks" else default
        )
        repl._config = config
        assert repl._get_tree_depth() == 3

    # ------------------------------------------------------------------
    # /tasks info
    # ------------------------------------------------------------------

    def test_info_without_path_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks info")
        display.show_error.assert_called_once()

    def test_info_extra_args_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks info Root Extra")
        display.show_error.assert_called_once()
        assert "Usage" in display.show_error.call_args[0][0]
        tm.find_by_path.assert_not_called()

    def test_info_calls_show_task_info(self):
        display = MagicMock()
        detail = _make_task_detail()
        tm = _make_task_manager()
        tm.find_by_path.return_value = detail
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks info MyTask")
        display.show_task_info.assert_called_once_with(detail)

    def test_info_strips_trailing_dot_before_lookup(self):
        display = MagicMock()
        detail = _make_task_detail()
        tm = _make_task_manager()
        tm.find_by_path.return_value = detail
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks info MyTask.")
        tm.find_by_path.assert_called_once_with("MyTask")
        display.show_task_info.assert_called_once_with(detail)

    def test_info_dot_only_path_shows_validation_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks info .")
        display.show_error.assert_called_once()
        tm.find_by_path.assert_not_called()

    # ------------------------------------------------------------------
    # /tasks edit
    # ------------------------------------------------------------------

    def test_edit_without_path_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks edit")
        display.show_error.assert_called_once()

    def test_edit_extra_args_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks edit Root Extra")
        display.show_error.assert_called_once()
        assert "Usage" in display.show_error.call_args[0][0]
        tm.find_by_path.assert_not_called()

    # ------------------------------------------------------------------
    # /tasks delete
    # ------------------------------------------------------------------

    def test_delete_no_path_confirms_and_clears(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        with patch("ai_cli.cli.repl.REPL._handle_tasks_subcommand") as mock_sub:
            # Bypass the actual confirm prompt — just test dispatch works
            repl._handle_slash_command("tasks delete")
            mock_sub.assert_called_once_with("delete")

    def test_delete_cancels_on_non_yes(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        with patch("prompt_toolkit.prompt", return_value="n"):
            repl._handle_slash_command("tasks delete")
        tm.clear.assert_not_called()
        display.show_status.assert_called()

    def test_delete_clears_on_yes(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        with patch("prompt_toolkit.prompt", return_value="y"):
            repl._handle_slash_command("tasks delete")
        tm.clear.assert_called_once()

    def test_delete_dot_only_path_shows_validation_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks delete .")
        display.show_error.assert_called_once()
        tm.clear.assert_not_called()

    def test_delete_storage_error_shows_storage_error_prefix(self):
        """TaskStorageError from clear() propagates to the top-level handler."""
        from ai_cli.core.task_manager import TaskStorageError

        display = MagicMock()
        tm = _make_task_manager()
        tm.clear.side_effect = TaskStorageError("disk full")
        repl = _make_repl(display=display, task_manager=tm)
        with patch("prompt_toolkit.prompt", return_value="y"):
            repl._handle_slash_command("tasks delete")
        display.show_error.assert_called_once()
        assert "Task storage error:" in display.show_error.call_args[0][0]

    def test_delete_extra_args_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks delete Foo Bar")
        display.show_error.assert_called_once()
        assert "too many arguments" in display.show_error.call_args[0][0]
        tm.find_by_path.assert_not_called()

    def test_add_extra_args_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks add Parent Extra")
        display.show_error.assert_called_once()
        assert "Usage" in display.show_error.call_args[0][0]
        tm.find_by_path.assert_not_called()

    # ------------------------------------------------------------------
    # /tasks close and open (stubs)
    # ------------------------------------------------------------------

    def test_close_calls_close_task(self):
        display = MagicMock()
        detail = _make_task_detail()
        tm = _make_task_manager()
        tm.find_by_path.return_value = detail
        tm.close_task.return_value = detail
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks close MyTask")
        tm.close_task.assert_called_once_with(detail["id"])
        display.show_status.assert_called_once()

    def test_close_not_found_shows_error(self):
        from ai_cli.core.task_manager import TaskNotFoundError

        display = MagicMock()
        tm = _make_task_manager()
        tm.find_by_path.side_effect = TaskNotFoundError("not found")
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks close bad.path")
        display.show_error.assert_called_once()

    def test_close_no_args_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks close")
        display.show_error.assert_called_once()
        assert "requires exactly one argument" in display.show_error.call_args[0][0]

    def test_close_extra_args_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks close Foo Bar")
        display.show_error.assert_called_once()
        assert "requires exactly one argument" in display.show_error.call_args[0][0]

    def test_open_calls_open_task(self):
        display = MagicMock()
        detail = _make_task_detail()
        tm = _make_task_manager()
        tm.find_by_path.return_value = detail
        tm.open_task.return_value = detail
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks open MyTask")
        tm.open_task.assert_called_once_with(detail["id"])
        display.show_status.assert_called_once()

    def test_open_not_found_shows_error(self):
        from ai_cli.core.task_manager import TaskNotFoundError

        display = MagicMock()
        tm = _make_task_manager()
        tm.find_by_path.side_effect = TaskNotFoundError("not found")
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks open bad.path")
        display.show_error.assert_called_once()

    def test_open_no_args_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks open")
        display.show_error.assert_called_once()
        assert "requires exactly one argument" in display.show_error.call_args[0][0]

    def test_open_extra_args_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks open Foo Bar")
        display.show_error.assert_called_once()
        assert "requires exactly one argument" in display.show_error.call_args[0][0]

    # ------------------------------------------------------------------
    # /tasks note obsolete
    # ------------------------------------------------------------------

    def test_note_obsolete_calls_task_manager(self):
        display = MagicMock()
        detail = _make_task_detail()
        tm = _make_task_manager()
        tm.find_by_path.return_value = detail
        tm.obsolete_note.return_value = detail
        repl = _make_repl(display=display, task_manager=tm)

        repl._handle_slash_command("tasks note obsolete MyTask 0")

        tm.find_by_path.assert_called_once_with("MyTask")
        tm.obsolete_note.assert_called_once_with(detail["id"], 0, reason="")
        display.show_status.assert_called_once()

    def test_note_obsolete_accepts_reason_with_spaces(self):
        display = MagicMock()
        detail = _make_task_detail()
        tm = _make_task_manager()
        tm.find_by_path.return_value = detail
        tm.obsolete_note.return_value = detail
        repl = _make_repl(display=display, task_manager=tm)

        repl._handle_slash_command(
            'tasks note obsolete MyTask 1 --reason "fixed in follow-up"'
        )

        tm.obsolete_note.assert_called_once_with(
            detail["id"], 1, reason="fixed in follow-up"
        )

    def test_note_obsolete_usage_error_on_missing_args(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)

        repl._handle_slash_command("tasks note obsolete MyTask")

        display.show_error.assert_called_once()
        assert "Usage:" in display.show_error.call_args[0][0]
        tm.find_by_path.assert_not_called()

    def test_note_obsolete_requires_integer_index(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)

        repl._handle_slash_command("tasks note obsolete MyTask xyz")

        display.show_error.assert_called_once()
        assert "integer" in display.show_error.call_args[0][0]
        tm.find_by_path.assert_not_called()

    def test_note_obsolete_rejects_invalid_flag_shape(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)

        repl._handle_slash_command("tasks note obsolete MyTask 0 --because stale")

        display.show_error.assert_called_once()
        assert "Usage:" in display.show_error.call_args[0][0]

    def test_note_obsolete_unbalanced_quotes_show_parse_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)

        repl._handle_slash_command('tasks note obsolete MyTask 0 --reason "broken')

        display.show_error.assert_called_once()
        assert "Could not parse /tasks arguments" in display.show_error.call_args[0][0]

    # ------------------------------------------------------------------
    # Unknown subcommand
    # ------------------------------------------------------------------

    def test_unknown_subcommand_shows_error(self):
        display = MagicMock()
        tm = _make_task_manager()
        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks frobnicate")
        display.show_error.assert_called_once()
        assert "Unknown" in display.show_error.call_args[0][0]

    # ------------------------------------------------------------------
    # Add wizard — DoD minimum-length validation
    # ------------------------------------------------------------------

    def test_add_wizard_reprompts_when_dod_too_short(self, monkeypatch):
        """Short DoD shows error and re-prompts; valid DoD on third try proceeds."""
        display = MagicMock()
        tm = _make_task_manager()
        created = _make_task_detail("NewTask")
        tm.create_task.return_value = created

        # Simulate: name → description → empty dod → short dod → valid dod → priority
        responses = iter(["NewTask", "", "", "hi", "at least five chars", "medium"])
        monkeypatch.setattr("prompt_toolkit.prompt", lambda *_a, **_kw: next(responses))

        repl = _make_repl(display=display, task_manager=tm)
        repl._handle_slash_command("tasks add")

        # Error shown twice (empty and "hi" are both < 5 non-whitespace chars)
        assert display.show_error.call_count == 2
        # Task eventually created with valid DoD
        tm.create_task.assert_called_once()
        _, kwargs = tm.create_task.call_args
        assert kwargs.get("definition_of_done") == "at least five chars"


# ---------------------------------------------------------------------------
# /mcp — MCPError handling
# ---------------------------------------------------------------------------


class TestMcpCommandErrorHandling:
    """MCPError during /mcp --persist should be shown, not crash the REPL."""

    def test_persist_mcp_error_shown(self):
        from ai_cli.core.mcp_manager import MCPError

        display = MagicMock()
        mcp_mgr = MagicMock()
        mcp_mgr.server_names.return_value = ["srv"]
        mcp_mgr.disable_server.side_effect = MCPError("corrupt mcp.yaml")

        repl = REPL(
            session=MagicMock(),
            tool_registry=MagicMock(),
            llm_client=_make_llm(),
            display=display,
            workspace=MagicMock(),
            mcp_manager=mcp_mgr,
        )
        repl._handle_mcp_subcommand("disable --persist srv")
        display.show_error.assert_called_once()
        assert "corrupt mcp.yaml" in display.show_error.call_args[0][0]
