"""Tests for ai_cli/tools/bash.py — Phase 1: core tool, single command, permission grants."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from ai_cli.tools.bash import BashTool, _normalize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tool(*, permission_required: bool = True) -> BashTool:
    ws = MagicMock()
    pm = MagicMock()
    pm.request.return_value = (True, "")
    return BashTool(
        workspace=ws,
        permission_manager=pm,
        permission_required=permission_required,
        name="bash",
        description="Run an arbitrary shell command on the client computer.",
    )


# ---------------------------------------------------------------------------
# Class attributes
# ---------------------------------------------------------------------------


class TestClassAttributes:
    def test_name(self):
        assert BashTool.NAME == "bash"

    def test_permission_required(self):
        assert BashTool.PERMISSION_REQUIRED is True

    def test_disabled_by_default(self):
        assert BashTool.DISABLED_BY_DEFAULT is True


# ---------------------------------------------------------------------------
# _normalize()
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_collapses_whitespace(self):
        assert _normalize("echo  hello") == "echo hello"

    def test_strips_leading_trailing(self):
        assert _normalize("  ls  ") == "ls"

    def test_handles_quoted_args(self):
        # shlex.join re-quotes tokens that contain shell metacharacters, so
        # the canonical form preserves token boundaries.
        assert _normalize("python3 -c 'print(1)'") == "python3 -c 'print(1)'"

    def test_spaces_in_token_preserved(self):
        assert _normalize('rm "file with spaces.txt"') == "rm 'file with spaces.txt'"

    def test_distinct_from_unquoted_spaces(self):
        assert _normalize('rm "a b"') != _normalize("rm a b")

    def test_invalid_shlex_falls_back_to_strip(self):
        result = _normalize("echo 'unclosed")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# definition()
# ---------------------------------------------------------------------------


class TestDefinition:
    def test_schema_shape(self):
        tool = make_tool()
        d = tool.definition().schema()
        assert d["type"] == "function"
        fn = d["function"]
        assert fn["name"] == "bash"
        params = fn["parameters"]
        props = params["properties"]
        assert "command" in props
        assert params["required"] == ["command"]

    def test_command_is_string_type(self):
        tool = make_tool()
        d = tool.definition().schema()
        assert d["function"]["parameters"]["properties"]["command"]["type"] == "string"

    def test_capture_in_schema(self):
        tool = make_tool()
        props = tool.definition().schema()["function"]["parameters"]["properties"]
        assert "capture" in props
        assert props["capture"]["type"] == "string"
        assert set(props["capture"]["enum"]) == {
            "stdout",
            "stderr",
            "interleaved",
            "separate",
        }

    def test_max_output_chars_in_schema(self):
        tool = make_tool()
        props = tool.definition().schema()["function"]["parameters"]["properties"]
        assert "max_output_chars" in props
        assert props["max_output_chars"]["type"] == "integer"
        assert props["max_output_chars"]["minimum"] == 1

    def test_capture_and_max_output_chars_not_required(self):
        tool = make_tool()
        required = tool.definition().schema()["function"]["parameters"]["required"]
        assert "capture" not in required
        assert "max_output_chars" not in required


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


class TestExecute:
    def test_basic_command_returns_stdout(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run", return_value=_completed("hello\n")
        ):
            result = tool.execute(command="echo hello")
        assert result["status"] == "success"
        assert result["data"]["output"] == "hello\n"

    def test_output_key_present_on_success(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run", return_value=_completed("test\n")
        ):
            result = tool.execute(command="echo test")
        assert "output" in result["data"]

    def test_subprocess_receives_parsed_args(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run", return_value=_completed()
        ) as mock_run:
            tool.execute(command="ls -la ./src")
        mock_run.assert_called_once()
        args_passed = mock_run.call_args[0][0]
        assert args_passed == ["ls", "-la", "./src"]

    def test_nonzero_exit_returns_execution_error(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(returncode=1, stderr="no such file"),
        ):
            result = tool.execute(command="ls missing")
        assert result["status"] == "error"
        assert result["error"] == "execution_error"
        assert "status 1" in result["message"]
        assert "no such file" in result["message"]

    def test_nonzero_exit_without_stderr(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(returncode=2),
        ):
            result = tool.execute(command="false")
        assert result["status"] == "error"
        assert "status 2" in result["message"]

    def test_file_not_found_returns_execution_error(self):
        tool = make_tool(permission_required=False)
        with patch("ai_cli.tools.bash.subprocess.run", side_effect=FileNotFoundError()):
            result = tool.execute(command="no_such_exe arg")
        assert result["status"] == "error"
        assert result["error"] == "execution_error"
        assert "no_such_exe" in result["message"]

    def test_timeout_returns_execution_error(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="sleep", timeout=30),
        ):
            result = tool.execute(command="sleep 999")
        assert result["status"] == "error"
        assert result["error"] == "execution_error"
        assert "timed out" in result["message"]

    def test_invalid_shlex_returns_invalid_command(self):
        tool = make_tool(permission_required=False)
        result = tool.execute(command="echo 'unclosed quote")
        assert result["status"] == "error"
        assert result["error"] == "invalid_command"

    def test_env_var_prefix_returns_clear_error(self):
        tool = make_tool(permission_required=False)
        result = tool.execute(command="A=1 ls -la")
        assert result["status"] == "error"
        assert result["error"] == "invalid_command"
        assert "Phase 3" in result["message"]

    def test_subprocess_called_with_timeout(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run", return_value=_completed()
        ) as mock_run:
            tool.execute(command="true")
        _, kwargs = mock_run.call_args
        assert "timeout" in kwargs
        assert kwargs["timeout"] == 30


# ---------------------------------------------------------------------------
# execute_log()
# ---------------------------------------------------------------------------


class TestExecuteLog:
    def test_short_command_returned_verbatim(self):
        tool = make_tool()
        assert tool.execute_log(command="echo hello world") == "echo hello world"

    def test_command_at_60_chars_not_truncated(self):
        tool = make_tool()
        cmd = "a" * 60
        assert tool.execute_log(command=cmd) == cmd

    def test_command_over_60_chars_truncated_with_ellipsis(self):
        tool = make_tool()
        cmd = "echo " + "x" * 60
        result = tool.execute_log(command=cmd)
        assert result is not None
        assert len(result) == 60
        assert result.endswith("...")

    def test_missing_command_returns_empty_label(self):
        tool = make_tool()
        assert tool.execute_log() == "<empty command>"

    def test_empty_command_returns_empty_label(self):
        tool = make_tool()
        assert tool.execute_log(command="") == "<empty command>"

    def test_unparseable_command_returns_label(self):
        tool = make_tool()
        assert tool.execute_log(command="echo 'unclosed") == "<unparseable command>"

    def test_env_var_prefix_stripped_shows_command(self):
        tool = make_tool()
        result = tool.execute_log(command="SECRET_TOKEN=abc123 python3 script.py")
        assert result == "python3 script.py"
        assert "abc123" not in (result or "")

    def test_multiple_env_vars_stripped(self):
        tool = make_tool()
        result = tool.execute_log(command="A=1 B=2 ls -la")
        assert result == "ls -la"

    def test_only_env_vars_returns_empty_label(self):
        tool = make_tool()
        assert tool.execute_log(command="A=1 B=2") == "<empty command>"


# ---------------------------------------------------------------------------
# extra_permission_options()
# ---------------------------------------------------------------------------


class TestExtraPermissionOptions:
    def test_single_token_command(self):
        tool = make_tool()
        opts = tool.extra_permission_options(command="ls")
        assert "always" in opts
        assert "always: ls *" in opts

    def test_two_token_command(self):
        tool = make_tool()
        opts = tool.extra_permission_options(command="echo hello")
        assert "always" in opts
        assert "always: echo *" in opts

    def test_multi_token_command_leading_args(self):
        tool = make_tool()
        opts = tool.extra_permission_options(command="ls -la ./docs")
        assert "always" in opts
        assert "always: ls -la *" in opts

    def test_empty_command_returns_always_only(self):
        tool = make_tool()
        assert tool.extra_permission_options(command="") == ["always"]

    def test_no_command_kwarg_returns_always_only(self):
        tool = make_tool()
        assert tool.extra_permission_options() == ["always"]

    def test_options_list_has_two_entries(self):
        tool = make_tool()
        opts = tool.extra_permission_options(command="echo hi")
        assert len(opts) == 2

    def test_always_always_present_to_prevent_tool_wide_grant(self):
        tool = make_tool()
        for cmd in ("", "   ", "echo hello", "ls -la"):
            assert "always" in tool.extra_permission_options(command=cmd), (
                f"'always' missing from extra_permission_options for command={cmd!r}"
            )


# ---------------------------------------------------------------------------
# request_permission() — grant matching
# ---------------------------------------------------------------------------


class TestRequestPermission:
    def test_no_grant_delegates_to_permission_manager(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "")
        allowed, _ = tool.request_permission("run echo", command="echo hello")
        assert allowed is True
        tool._permission_manager.request.assert_called_once()

    def test_permission_not_required_always_allowed(self):
        tool = make_tool(permission_required=False)
        allowed, _ = tool.request_permission("run anything", command="echo hi")
        assert allowed is True
        tool._permission_manager.request.assert_not_called()

    def test_exact_grant_skips_permission_manager(self):
        tool = make_tool(permission_required=True)
        tool.on_permission_granted("always", command="echo hello")
        allowed, _ = tool.request_permission("run echo", command="echo hello")
        assert allowed is True
        tool._permission_manager.request.assert_not_called()

    def test_exact_grant_normalises_whitespace(self):
        tool = make_tool(permission_required=True)
        tool.on_permission_granted("always", command="echo  hello")
        allowed, _ = tool.request_permission("run echo", command="echo hello")
        assert allowed is True
        tool._permission_manager.request.assert_not_called()

    def test_exact_grant_does_not_allow_different_command(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (False, "Permission denied.")
        tool.on_permission_granted("always", command="echo hello")
        allowed, _ = tool.request_permission("run echo", command="echo world")
        assert allowed is False

    def test_pattern_grant_allows_matching_command(self):
        tool = make_tool(permission_required=True)
        tool.on_permission_granted("always: echo *", command="echo hello")
        allowed, _ = tool.request_permission("run echo", command="echo world")
        assert allowed is True
        tool._permission_manager.request.assert_not_called()

    def test_pattern_grant_allows_multi_arg_command(self):
        tool = make_tool(permission_required=True)
        tool.on_permission_granted("always: ls *", command="ls ./docs")
        allowed, _ = tool.request_permission("run ls", command="ls -la ./src")
        assert allowed is True

    def test_pattern_grant_does_not_match_different_exe(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (False, "Permission denied.")
        tool.on_permission_granted("always: echo *", command="echo hello")
        allowed, _ = tool.request_permission("run ls", command="ls hello")
        assert allowed is False

    def test_pattern_not_stored_for_unknown_choice_format(self):
        tool = make_tool(permission_required=True)
        tool.on_permission_granted("unknown_choice", command="echo hi")
        assert len(tool._exact_grants) == 0
        assert len(tool._pattern_grants) == 0

    def test_whitespace_only_command_does_not_create_grant(self):
        tool = make_tool(permission_required=True)
        tool.on_permission_granted("always", command="   ")
        assert len(tool._exact_grants) == 0

    def test_whitespace_only_command_extra_options_always_only(self):
        tool = make_tool(permission_required=True)
        assert tool.extra_permission_options(command="   ") == ["always"]


# ---------------------------------------------------------------------------
# reset_session_state()
# ---------------------------------------------------------------------------


class TestResetSessionState:
    def test_clears_exact_grants(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (False, "Permission denied.")
        tool.on_permission_granted("always", command="echo hello")
        tool.reset_session_state()
        allowed, _ = tool.request_permission("run echo", command="echo hello")
        assert allowed is False

    def test_clears_pattern_grants(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (False, "Permission denied.")
        tool.on_permission_granted("always: echo *", command="echo hello")
        tool.reset_session_state()
        allowed, _ = tool.request_permission("run echo", command="echo world")
        assert allowed is False

    def test_reset_grants_are_empty(self):
        tool = make_tool()
        tool.on_permission_granted("always", command="echo hi")
        tool.on_permission_granted("always: ls *", command="ls ./docs")
        tool.reset_session_state()
        assert len(tool._exact_grants) == 0
        assert len(tool._pattern_grants) == 0


# ---------------------------------------------------------------------------
# Phase 2: capture modes
# ---------------------------------------------------------------------------


class TestCapture:
    def test_default_capture_returns_stdout(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout="hello\n", stderr="ignored"),
        ):
            result = tool.execute(command="echo hello")
        assert result["status"] == "success"
        assert result["data"]["output"] == "hello\n"

    def test_capture_stdout_explicit(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout="out\n", stderr="err\n"),
        ):
            result = tool.execute(command="echo out", capture="stdout")
        assert result["data"]["output"] == "out\n"
        assert "stderr" not in result["data"]

    def test_capture_stderr(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout="", stderr="err text\n"),
        ):
            result = tool.execute(command="ls missing", capture="stderr")
        assert result["status"] == "success"
        assert result["data"]["output"] == "err text\n"
        assert "stdout" not in result["data"]

    def test_capture_interleaved(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout="merged\n"),
        ):
            result = tool.execute(command="echo merged", capture="interleaved")
        assert result["status"] == "success"
        assert result["data"]["output"] == "merged\n"

    def test_capture_interleaved_subprocess_kwargs(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run", return_value=_completed()
        ) as mock_run:
            tool.execute(command="echo hi", capture="interleaved")
        kwargs = mock_run.call_args[1]
        assert kwargs["stderr"] == subprocess.STDOUT

    def test_capture_separate_returns_both_fields(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout="out\n", stderr="err\n"),
        ):
            result = tool.execute(command="echo out", capture="separate")
        assert result["status"] == "success"
        assert result["data"]["stdout"] == "out\n"
        assert result["data"]["stderr"] == "err\n"
        assert "output" not in result["data"]

    def test_capture_stdout_subprocess_kwargs(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run", return_value=_completed()
        ) as mock_run:
            tool.execute(command="echo hi", capture="stdout")
        kwargs = mock_run.call_args[1]
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE

    def test_capture_stdout_nonzero_exit_includes_stderr(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stderr="something went wrong", returncode=1),
        ):
            result = tool.execute(command="false", capture="stdout")
        assert result["status"] == "error"
        assert "something went wrong" in result["message"]

    def test_nonzero_exit_error_output_is_truncated(self):
        tool = make_tool(permission_required=False)
        long_stderr = "e" * 200
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stderr=long_stderr, returncode=1),
        ):
            result = tool.execute(command="false", max_output_chars=10)
        assert result["status"] == "error"
        assert len(result["message"]) < 200

    def test_capture_stderr_subprocess_uses_devnull_for_stdout(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run", return_value=_completed()
        ) as mock_run:
            tool.execute(command="echo hi", capture="stderr")
        kwargs = mock_run.call_args[1]
        assert kwargs["stdout"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.PIPE

    def test_invalid_capture_returns_error(self):
        tool = make_tool(permission_required=False)
        result = tool.execute(command="echo hi", capture="bogus")
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_zero_max_output_chars_returns_error(self):
        tool = make_tool(permission_required=False)
        result = tool.execute(command="echo hi", max_output_chars=0)
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_negative_max_output_chars_returns_error(self):
        tool = make_tool(permission_required=False)
        result = tool.execute(command="echo hi", max_output_chars=-5)
        assert result["status"] == "error"
        assert result["error"] == "invalid_arguments"

    def test_capture_interleaved_nonzero_exit_includes_merged_output(self):
        # With stderr=STDOUT, proc.stderr is None; merged output is in proc.stdout.
        tool = make_tool(permission_required=False)
        mock_proc = _completed(stdout="merged error output\n", returncode=1)
        mock_proc.stderr = None
        with patch("ai_cli.tools.bash.subprocess.run", return_value=mock_proc):
            result = tool.execute(command="false", capture="interleaved")
        assert result["status"] == "error"
        assert "merged error output" in result["message"]


# ---------------------------------------------------------------------------
# Phase 2: truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_output_within_limit_has_no_warning(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout="hi\n"),
        ):
            result = tool.execute(command="echo hi", max_output_chars=100)
        assert "warning" not in result["data"]
        assert result["data"]["output"] == "hi\n"

    def test_output_exceeding_limit_is_truncated(self):
        tool = make_tool(permission_required=False)
        long_output = "x" * 200
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout=long_output),
        ):
            result = tool.execute(command="echo x", max_output_chars=10)
        assert result["status"] == "success"
        assert len(result["data"]["output"]) == 10
        assert "warning" in result["data"]
        assert "10" in result["data"]["warning"]

    def test_output_at_exactly_limit_has_no_warning(self):
        tool = make_tool(permission_required=False)
        exact_output = "a" * 10
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout=exact_output),
        ):
            result = tool.execute(command="echo a", max_output_chars=10)
        assert "warning" not in result["data"]

    def test_multibyte_characters_truncated_by_char_count(self):
        # Each '€' is 3 UTF-8 bytes; truncation must count characters, not bytes.
        tool = make_tool(permission_required=False)
        euro_output = "€" * 20
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout=euro_output),
        ):
            result = tool.execute(command="echo euro", max_output_chars=5)
        assert result["status"] == "success"
        assert result["data"]["output"] == "€" * 5
        assert "warning" in result["data"]

    def test_separate_capture_warns_when_stdout_truncated(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout="x" * 20, stderr="ok"),
        ):
            result = tool.execute(
                command="echo x", capture="separate", max_output_chars=5
            )
        assert "warning" in result["data"]

    def test_separate_capture_warns_when_stderr_truncated(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout="ok", stderr="e" * 20),
        ):
            result = tool.execute(
                command="echo ok", capture="separate", max_output_chars=5
            )
        assert "warning" in result["data"]

    def test_separate_capture_no_warning_when_both_within_limit(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout="out", stderr="err"),
        ):
            result = tool.execute(
                command="echo out", capture="separate", max_output_chars=100
            )
        assert "warning" not in result["data"]

    def test_default_max_output_chars_is_1024(self):
        tool = make_tool(permission_required=False)
        # 1024 chars fits, no warning
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout="a" * 1024),
        ):
            result = tool.execute(command="echo a")
        assert "warning" not in result["data"]
        # 1025 chars truncates
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout="a" * 1025),
        ):
            result = tool.execute(command="echo a")
        assert "warning" in result["data"]

    def test_warning_message_says_characters(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash.subprocess.run",
            return_value=_completed(stdout="x" * 200),
        ):
            result = tool.execute(command="echo x", max_output_chars=10)
        assert "characters" in result["data"]["warning"]
