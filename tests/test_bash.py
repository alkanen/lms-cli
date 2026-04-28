"""Tests for ai_cli/tools/bash.py — Phase 1: core tool, single command, permission grants."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ai_cli.tools.bash import (
    BashTool,
    _chain_summary,
    _has_heredoc,
    _parse_chain,
    _parse_redirections,
    _redir_pattern_match,
    _run_popen,
    _split_env_vars,
    _tokenize_segment,
)

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


def _run_result(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    truncated: bool = False,
) -> tuple[int, str, str, bool]:
    """Mock return value for _run_popen: (returncode, stdout, stderr, truncated)."""
    return (returncode, stdout, stderr, truncated)


class TestExecute:
    def test_basic_command_returns_stdout(self):
        tool = make_tool(permission_required=False)
        with patch("ai_cli.tools.bash._run_popen", return_value=_run_result("hello\n")):
            result = tool.execute(command="echo hello")
        assert result["status"] == "success"
        assert result["data"]["output"] == "hello\n"

    def test_output_key_present_on_success(self):
        tool = make_tool(permission_required=False)
        with patch("ai_cli.tools.bash._run_popen", return_value=_run_result("test\n")):
            result = tool.execute(command="echo test")
        assert "output" in result["data"]

    def test_subprocess_receives_parsed_args(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="ls -la ./src")
        mock_run.assert_called_once()
        args_passed = mock_run.call_args[0][0]
        assert args_passed == ["ls", "-la", "./src"]

    def test_nonzero_exit_returns_execution_error(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(returncode=1, stderr="no such file"),
        ):
            result = tool.execute(command="ls missing")
        assert result["status"] == "error"
        assert result["error"] == "execution_error"
        assert "status 1" in result["message"]
        assert "no such file" in result["message"]

    def test_nonzero_exit_without_stderr(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(returncode=2),
        ):
            result = tool.execute(command="false")
        assert result["status"] == "error"
        assert "status 2" in result["message"]

    def test_file_not_found_returns_execution_error(self):
        tool = make_tool(permission_required=False)
        with patch("ai_cli.tools.bash._run_popen", side_effect=FileNotFoundError()):
            result = tool.execute(command="no_such_exe arg")
        assert result["status"] == "error"
        assert result["error"] == "execution_error"
        assert "no_such_exe" in result["message"]

    def test_timeout_returns_timeout(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            side_effect=subprocess.TimeoutExpired(cmd="sleep", timeout=30),
        ):
            result = tool.execute(command="sleep 999")
        assert result["status"] == "error"
        assert result["error"] == "timeout"
        assert result["code"] == 408
        assert "timed out" in result["message"]

    def test_unexpected_exception_returns_internal_error(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            side_effect=RuntimeError("something exploded"),
        ):
            result = tool.execute(command="ls")
        assert result["status"] == "error"
        assert result["error"] == "internal_error"
        assert result["code"] == 500

    def test_unexpected_exception_shell_path_returns_internal_error(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            side_effect=RuntimeError("something exploded"),
        ):
            result = tool.execute(command="ls | cat")
        assert result["status"] == "error"
        assert result["error"] == "internal_error"
        assert result["code"] == 500

    def test_invalid_shlex_returns_invalid_command(self):
        tool = make_tool(permission_required=False)
        result = tool.execute(command="echo 'unclosed quote")
        assert result["status"] == "error"
        assert result["error"] == "invalid_command"

    def test_env_var_prefix_strips_and_executes(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result("out\n")
        ) as mock_run:
            result = tool.execute(command="A=1 ls -la")
        assert result["status"] == "success"
        args_passed = mock_run.call_args[0][0]
        assert args_passed == ["ls", "-la"]

    def test_subprocess_called_with_timeout(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
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

    def test_env_var_values_shown_in_log(self):
        tool = make_tool()
        result = tool.execute_log(command="SECRET_TOKEN=abc123 python3 script.py")
        assert result is not None
        assert "python3" in result
        assert "abc123" in result

    def test_multiple_env_vars_shown_in_log(self):
        tool = make_tool()
        result = tool.execute_log(command="A=1 B=2 ls -la")
        assert result is not None
        assert "A=1" in result
        assert "B=2" in result
        assert "ls" in result

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

    def test_pattern_grant_matches_bare_no_arg_command(self):
        # "ls *" must also match bare "ls" with no arguments.
        tool = make_tool(permission_required=True)
        tool.on_permission_granted("always: ls *", command="ls")
        allowed, _ = tool.request_permission("run ls", command="ls")
        assert allowed is True
        tool._permission_manager.request.assert_not_called()

    def test_pattern_grant_bare_command_does_not_match_different_exe(self):
        # "ls *" must not match "lsblk" — the space position in the pattern is fixed.
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (False, "Permission denied.")
        tool.on_permission_granted("always: ls *", command="ls")
        allowed, _ = tool.request_permission("run lsblk", command="lsblk")
        assert allowed is False

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
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="hello\n", stderr="ignored"),
        ):
            result = tool.execute(command="echo hello")
        assert result["status"] == "success"
        assert result["data"]["output"] == "hello\n"

    def test_capture_stdout_explicit(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="out\n", stderr="err\n"),
        ):
            result = tool.execute(command="echo out", capture="stdout")
        assert result["data"]["output"] == "out\n"
        assert "stderr" not in result["data"]

    def test_capture_stderr(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="", stderr="err text\n"),
        ):
            result = tool.execute(command="ls missing", capture="stderr")
        assert result["status"] == "success"
        assert result["data"]["output"] == "err text\n"
        assert "stdout" not in result["data"]

    def test_capture_interleaved(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="merged\n"),
        ):
            result = tool.execute(command="echo merged", capture="interleaved")
        assert result["status"] == "success"
        assert result["data"]["output"] == "merged\n"

    def test_capture_interleaved_subprocess_kwargs(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="echo hi", capture="interleaved")
        kwargs = mock_run.call_args[1]
        assert kwargs["stream_kwargs"]["stderr"] == subprocess.STDOUT

    def test_capture_separate_returns_both_fields(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="out\n", stderr="err\n"),
        ):
            result = tool.execute(command="echo out", capture="separate")
        assert result["status"] == "success"
        assert result["data"]["stdout"] == "out\n"
        assert result["data"]["stderr"] == "err\n"
        assert "output" not in result["data"]

    def test_capture_stdout_subprocess_kwargs(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="echo hi", capture="stdout")
        kwargs = mock_run.call_args[1]
        assert kwargs["stream_kwargs"]["stdout"] == subprocess.PIPE
        assert kwargs["stream_kwargs"]["stderr"] == subprocess.PIPE

    def test_capture_stdout_nonzero_exit_includes_stderr(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stderr="something went wrong", returncode=1),
        ):
            result = tool.execute(command="false", capture="stdout")
        assert result["status"] == "error"
        assert "something went wrong" in result["message"]

    def test_nonzero_exit_error_output_is_truncated(self):
        tool = make_tool(permission_required=False)
        long_stderr = "e" * 200
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stderr=long_stderr, returncode=1),
        ):
            result = tool.execute(command="false", max_output_chars=10)
        assert result["status"] == "error"
        assert len(result["message"]) < 200

    def test_capture_stderr_subprocess_uses_devnull_for_stdout(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="echo hi", capture="stderr")
        kwargs = mock_run.call_args[1]
        assert kwargs["stream_kwargs"]["stdout"] == subprocess.DEVNULL
        assert kwargs["stream_kwargs"]["stderr"] == subprocess.PIPE

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
        # interleaved merges stderr into stdout; error info is in stdout_text.
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="merged error output\n", returncode=1),
        ):
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
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="hi\n"),
        ):
            result = tool.execute(command="echo hi", max_output_chars=100)
        assert "warning" not in result["data"]
        assert result["data"]["output"] == "hi\n"

    def test_output_exceeding_limit_is_truncated(self):
        # _run_popen truncates and signals via truncated=True; execute() adds warning.
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="x" * 10, truncated=True),
        ):
            result = tool.execute(command="echo x", max_output_chars=10)
        assert result["status"] == "success"
        assert len(result["data"]["output"]) == 10
        assert "warning" in result["data"]
        assert "10" in result["data"]["warning"]

    def test_output_at_exactly_limit_has_no_warning(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="a" * 10),
        ):
            result = tool.execute(command="echo a", max_output_chars=10)
        assert "warning" not in result["data"]

    def test_multibyte_characters_truncated_by_char_count(self):
        # Each '€' is 3 UTF-8 bytes; _run_popen counts characters, not bytes.
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="€" * 5, truncated=True),
        ):
            result = tool.execute(command="echo euro", max_output_chars=5)
        assert result["status"] == "success"
        assert result["data"]["output"] == "€" * 5
        assert "warning" in result["data"]

    def test_separate_capture_warns_when_stdout_truncated(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="x" * 5, stderr="ok", truncated=True),
        ):
            result = tool.execute(
                command="echo x", capture="separate", max_output_chars=5
            )
        assert "warning" in result["data"]

    def test_separate_capture_warns_when_stderr_truncated(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="ok", stderr="e" * 5, truncated=True),
        ):
            result = tool.execute(
                command="echo ok", capture="separate", max_output_chars=5
            )
        assert "warning" in result["data"]

    def test_separate_capture_no_warning_when_both_within_limit(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="out", stderr="err"),
        ):
            result = tool.execute(
                command="echo out", capture="separate", max_output_chars=100
            )
        assert "warning" not in result["data"]

    def test_default_max_output_chars_is_1024(self):
        tool = make_tool(permission_required=False)
        # 1024 chars fits, no warning (truncated=False).
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="a" * 1024),
        ):
            result = tool.execute(command="echo a")
        assert "warning" not in result["data"]
        # 1025 chars: _run_popen would truncate and signal truncated=True.
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="a" * 1024, truncated=True),
        ):
            result = tool.execute(command="echo a")
        assert "warning" in result["data"]

    def test_warning_message_says_characters(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="x" * 10, truncated=True),
        ):
            result = tool.execute(command="echo x", max_output_chars=10)
        assert "characters" in result["data"]["warning"]

    def test_killed_for_truncation_is_not_a_failure(self):
        # A SIGKILL-terminated process (returncode < 0) that was truncated is
        # success with a warning, not an error.
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="x" * 10, returncode=-9, truncated=True),
        ):
            result = tool.execute(command="yes", max_output_chars=10)
        assert result["status"] == "success"
        assert result["data"]["output"] == "x" * 10
        assert "warning" in result["data"]

    def test_nonzero_exit_with_truncation_is_still_failure(self):
        # A process that exits with a positive error code AND happens to have
        # truncated output is a genuine failure, not a kill-for-limit event.
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="x" * 10, returncode=1, truncated=True),
        ):
            result = tool.execute(command="bad-cmd", max_output_chars=10)
        assert result["status"] == "error"
        assert result["error"] == "execution_error"


# ---------------------------------------------------------------------------
# _split_env_vars()
# ---------------------------------------------------------------------------


class TestSplitEnvVars:
    def test_no_env_vars(self):
        env, cmd = _split_env_vars(["ls", "-la"])
        assert env == {}
        assert cmd == ["ls", "-la"]

    def test_single_env_var(self):
        env, cmd = _split_env_vars(["A=1", "ls"])
        assert env == {"A": "1"}
        assert cmd == ["ls"]

    def test_multiple_env_vars(self):
        env, cmd = _split_env_vars(["A=1", "B=hello", "python3", "script.py"])
        assert env == {"A": "1", "B": "hello"}
        assert cmd == ["python3", "script.py"]

    def test_only_env_vars_returns_empty_cmd(self):
        env, cmd = _split_env_vars(["A=1", "B=2"])
        assert env == {"A": "1", "B": "2"}
        assert cmd == []

    def test_empty_tokens_returns_empty(self):
        env, cmd = _split_env_vars([])
        assert env == {}
        assert cmd == []

    def test_value_with_equals_sign(self):
        env, cmd = _split_env_vars(["URL=http://x?a=b", "curl"])
        assert env == {"URL": "http://x?a=b"}
        assert cmd == ["curl"]


# ---------------------------------------------------------------------------
# Phase 3: environment variable support
# ---------------------------------------------------------------------------


class TestEnvVars:
    def test_env_var_passed_to_subprocess(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result("123\n")
        ) as mock_run:
            tool.execute(command="MYVAR=123 python3 -c 'print(1)'")
        kwargs = mock_run.call_args[1]
        assert "env" in kwargs
        assert kwargs["env"] is not None
        assert kwargs["env"]["MYVAR"] == "123"

    def test_env_var_inherits_parent_env(self):
        import os

        tool = make_tool(permission_required=False)
        with (
            patch.dict(
                os.environ,
                {"PATH": "/tmp/bin", "HOME": "/tmp/home", "MY_VAR": "original"},
                clear=True,
            ),
            patch(
                "ai_cli.tools.bash._run_popen", return_value=_run_result()
            ) as mock_run,
        ):
            tool.execute(command="MY_VAR=abc python3 -c 'pass'")
        kwargs = mock_run.call_args[1]
        assert kwargs["env"] is not None
        assert kwargs["env"]["PATH"] == "/tmp/bin"
        assert kwargs["env"]["HOME"] == "/tmp/home"
        assert kwargs["env"]["MY_VAR"] == "abc"

    def test_no_env_var_passes_none_as_env(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="ls -la")
        kwargs = mock_run.call_args[1]
        assert kwargs["env"] is None

    def test_env_var_does_not_modify_parent_process_env(self):
        import os

        tool = make_tool(permission_required=False)
        with (
            patch.dict(os.environ, {"PATH": "/tmp/bin"}, clear=True),
            patch("ai_cli.tools.bash._run_popen", return_value=_run_result()),
        ):
            before = dict(os.environ)
            tool.execute(command="SECRET=leaked ls")
            assert os.environ == before
            assert "SECRET" not in os.environ

    def test_subprocess_called_with_cmd_tokens_not_env_prefix(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="A=1 B=2 ls -la ./docs")
        args_passed = mock_run.call_args[0][0]
        assert args_passed == ["ls", "-la", "./docs"]

    def test_multiple_env_vars_all_passed(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="A=1 B=hello C=world python3 -c 'pass'")
        kwargs = mock_run.call_args[1]
        assert kwargs["env"]["A"] == "1"
        assert kwargs["env"]["B"] == "hello"
        assert kwargs["env"]["C"] == "world"

    def test_only_env_vars_no_command_returns_error(self):
        tool = make_tool(permission_required=False)
        result = tool.execute(command="A=1 B=2")
        assert result["status"] == "error"
        assert result["error"] == "invalid_command"

    def test_grant_key_same_var_name_different_value_matches(self):
        # Same env var NAME, different value → same grant key → should match.
        tool = make_tool(permission_required=True)
        tool.on_permission_granted(
            "always: MYVAR=* python3 *", command="MYVAR=123 python3 script.py"
        )
        allowed, _ = tool.request_permission(
            "run", command="MYVAR=456 python3 other.py"
        )
        assert allowed is True
        tool._permission_manager.request.assert_not_called()

    def test_grant_key_different_env_var_names_do_not_match(self):
        # Different env var NAMES → different grant key → must NOT silently match.
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (False, "Permission denied.")
        tool.on_permission_granted("always", command="MYVAR=123 python3 script.py")
        allowed, _ = tool.request_permission(
            "run", command="PATH=/evil python3 script.py"
        )
        assert allowed is False

    def test_exact_grant_env_var_name_scoped(self):
        tool = make_tool(permission_required=True)
        # Grant stored with MYVAR should match same MYVAR with a different value.
        tool.on_permission_granted("always", command="MYVAR=123 python3 script.py")
        allowed, _ = tool.request_permission(
            "run", command="MYVAR=999 python3 script.py"
        )
        assert allowed is True
        tool._permission_manager.request.assert_not_called()

    def test_extra_permission_options_includes_env_var_names_not_values(self):
        tool = make_tool()
        opts = tool.extra_permission_options(
            command="MYVAR=secret123 python3 script.py"
        )
        assert "always" in opts
        assert any("python3" in o for o in opts)
        assert any("MYVAR=*" in o for o in opts)
        assert not any("secret123" in o for o in opts)

    def test_grant_key_env_var_order_independent(self):
        # A=1 B=2 cmd and B=2 A=1 cmd must produce the same grant key.
        tool = make_tool(permission_required=True)
        tool.on_permission_granted("always", command="A=1 B=2 python3 script.py")
        allowed, _ = tool.request_permission(
            "run", command="B=99 A=42 python3 script.py"
        )
        assert allowed is True
        tool._permission_manager.request.assert_not_called()

    def test_extra_permission_options_env_var_order_stable(self):
        # Options must be identical regardless of env var declaration order.
        tool = make_tool()
        opts_ab = tool.extra_permission_options(command="A=1 B=2 python3 script.py")
        opts_ba = tool.extra_permission_options(command="B=2 A=1 python3 script.py")
        assert opts_ab == opts_ba

    def test_extra_permission_options_env_var_only_returns_always_only(self):
        tool = make_tool()
        opts = tool.extra_permission_options(command="A=1 B=2")
        assert opts == ["always"]


# ---------------------------------------------------------------------------
# Phase 4: _parse_chain()
# ---------------------------------------------------------------------------


class TestParseChain:
    def test_single_command_returns_one_segment(self):
        segs = _parse_chain("ls -la")
        assert len(segs) == 1
        assert segs[0][0] is None
        assert segs[0][1] == "ls -la"

    def test_pipe_returns_two_segments(self):
        segs = _parse_chain("cat foo | grep bar")
        assert len(segs) == 2
        assert segs[0] == (None, "cat foo")
        assert segs[1] == ("|", "grep bar")

    def test_pipe_three_segments(self):
        segs = _parse_chain("cat foo | grep bar | wc -l")
        assert len(segs) == 3
        assert segs[0][0] is None
        assert segs[1][0] == "|"
        assert segs[2][0] == "|"

    def test_and_operator(self):
        segs = _parse_chain("ls && cat foo")
        assert len(segs) == 2
        assert segs[0] == (None, "ls")
        assert segs[1][0] == "&&"

    def test_or_operator(self):
        segs = _parse_chain("ls missing || echo not_found")
        assert len(segs) == 2
        assert segs[0] == (None, "ls missing")
        assert segs[1][0] == "||"

    def test_quoted_pipe_not_treated_as_operator(self):
        segs = _parse_chain("echo 'hello | world'")
        assert len(segs) == 1

    def test_empty_command_returns_empty_list(self):
        segs = _parse_chain("")
        assert segs == []

    def test_invalid_quoting_raises_value_error(self):
        import pytest

        with pytest.raises(ValueError):
            _parse_chain("echo 'unclosed")

    def test_segments_are_raw_substrings(self):
        segs = _parse_chain("ls -la ./docs | grep foo")
        assert segs[0][1] == "ls -la ./docs"
        assert segs[1][1] == "grep foo"

    def test_chain_preserves_redirect_in_segment(self):
        # Redirect operator must NOT be re-quoted by shlex.join — it must be
        # detectable by _parse_redirections in a chained segment.
        segs = _parse_chain("echo hi>out.txt | wc")
        assert segs[0][1] == "echo hi>out.txt"
        assert segs[1][1] == "wc"

    def test_mixed_operators(self):
        segs = _parse_chain("ls && cat foo || echo done")
        assert len(segs) == 3
        assert segs[0][0] is None
        assert segs[1][0] == "&&"
        assert segs[2][0] == "||"

    def test_semicolon_operator(self):
        segs = _parse_chain("ls; echo done")
        assert len(segs) == 2
        assert segs[0] == (None, "ls")
        assert segs[1][0] == ";"

    def test_trailing_pipe_raises(self):
        import pytest

        with pytest.raises(ValueError, match="ends with"):
            _parse_chain("ls |")

    def test_trailing_and_raises(self):
        import pytest

        with pytest.raises(ValueError, match="ends with"):
            _parse_chain("ls &&")

    def test_leading_pipe_raises(self):
        import pytest

        with pytest.raises(ValueError, match="starts with"):
            _parse_chain("| grep foo")

    def test_leading_semicolon_raises(self):
        import pytest

        with pytest.raises(ValueError, match="starts with"):
            _parse_chain("; echo hi")

    def test_unquoted_hash_not_treated_as_comment(self):
        # shlex defaults commenters='#'; disable it so # is a literal token.
        segs = _parse_chain("echo hello#world")
        assert len(segs) == 1
        assert "hello#world" in segs[0][1]

    def test_consecutive_and_operators_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Empty segment"):
            _parse_chain("ls && && echo")

    def test_consecutive_pipe_operators_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Empty segment"):
            _parse_chain("ls | | grep foo")

    def test_consecutive_or_operators_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Empty segment"):
            _parse_chain("ls || || echo done")

    def test_mixed_consecutive_operators_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Empty segment"):
            _parse_chain("ls | && echo")

    def test_quoted_chain_operator_is_not_rejected(self):
        # echo "&&" — the && is inside double-quotes; must NOT be treated as a
        # trailing chain operator (shlex would strip the quotes and misidentify it).
        segs = _parse_chain('echo "&&"')
        assert len(segs) == 1
        assert segs[0][0] is None

    def test_trailing_operator_detected_via_scan(self):
        # echo hello && — trailing && should still be caught via scan state.
        import pytest

        with pytest.raises(ValueError, match="ends with"):
            _parse_chain("echo hello &&")


# ---------------------------------------------------------------------------
# Phase 4: _chain_summary()
# ---------------------------------------------------------------------------


class TestChainSummary:
    def test_single_segment(self):
        segs = [(None, "cat foo")]
        assert _chain_summary(segs) == "cat"

    def test_pipe_two_segments(self):
        segs = _parse_chain("cat foo | grep bar")
        assert _chain_summary(segs) == "cat | grep"

    def test_three_segments(self):
        segs = _parse_chain("cat foo | grep bar | wc -l")
        assert _chain_summary(segs) == "cat | grep | wc"

    def test_and_operator_in_summary(self):
        segs = _parse_chain("ls ./src && cat README")
        assert _chain_summary(segs) == "ls && cat"

    def test_or_operator_in_summary(self):
        segs = _parse_chain("ls missing || echo done")
        assert _chain_summary(segs) == "ls || echo"

    def test_env_var_stripped_from_summary(self):
        segs = _parse_chain("MYVAR=1 python3 script.py | grep ok")
        assert _chain_summary(segs) == "python3 | grep"

    def test_mixed_operators_in_summary(self):
        segs = _parse_chain("ls && cat foo || echo done")
        assert _chain_summary(segs) == "ls && cat || echo"


# ---------------------------------------------------------------------------
# Phase 4: chain permission (acceptance criteria)
# ---------------------------------------------------------------------------


class TestChainPermission:
    def test_pipe_chain_prompts_each_unapproved_segment(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.request_permission("run", command="cat foo | grep bar")
        assert tool._permission_manager.request.call_count == 2

    def test_chain_summary_appears_in_question(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.request_permission("run", command="cat foo | grep bar")
        first_call = tool._permission_manager.request.call_args_list[0]
        question = first_call.kwargs.get("question") or first_call.args[1]
        assert "cat | grep" in question

    def test_all_segments_approved_returns_true(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        allowed, _ = tool.request_permission("run", command="cat foo | grep bar")
        assert allowed is True

    def test_first_segment_denied_returns_false_immediately(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (False, "Permission denied.")
        allowed, _ = tool.request_permission("run", command="cat foo | grep bar")
        assert allowed is False
        assert tool._permission_manager.request.call_count == 1

    def test_second_segment_denied_returns_false(self):
        tool = make_tool(permission_required=True)
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (True, "yes")
            return (False, "Permission denied.")

        tool._permission_manager.request.side_effect = side_effect
        allowed, _ = tool.request_permission("run", command="cat foo | grep bar")
        assert allowed is False
        assert call_count[0] == 2

    def test_always_grant_stored_before_later_denial(self):
        tool = make_tool(permission_required=True)
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (True, "always")
            return (False, "Permission denied.")

        tool._permission_manager.request.side_effect = side_effect
        allowed, _ = tool.request_permission("run", command="cat foo | grep bar")
        assert allowed is False
        # The always grant for the first segment must be stored despite later denial.
        assert len(tool._exact_grants) == 1

    def test_already_granted_segment_skips_prompt(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        # Grant the first segment.
        tool.on_permission_granted("always", command="cat foo")
        tool.request_permission("run", command="cat foo | grep bar")
        # Only the second segment should be prompted.
        assert tool._permission_manager.request.call_count == 1

    def test_all_segments_granted_skips_all_prompts(self):
        tool = make_tool(permission_required=True)
        tool.on_permission_granted("always", command="cat foo")
        tool.on_permission_granted("always", command="grep bar")
        allowed, _ = tool.request_permission("run", command="cat foo | grep bar")
        assert allowed is True
        tool._permission_manager.request.assert_not_called()

    def test_and_chain_permission(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        allowed, _ = tool.request_permission("run", command="ls && cat foo")
        assert allowed is True
        assert tool._permission_manager.request.call_count == 2

    def test_or_chain_permission(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        allowed, _ = tool.request_permission("run", command="ls missing || echo done")
        assert allowed is True
        assert tool._permission_manager.request.call_count == 2

    def test_semicolon_chain_permission(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        allowed, _ = tool.request_permission("run", command="ls; echo done")
        assert allowed is True
        assert tool._permission_manager.request.call_count == 2

    def test_semicolon_bypasses_permission_grant_treated_as_chain(self):
        # "grep bar; rm -rf /" must NOT be covered by a "grep *" pattern grant.
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.on_permission_granted("always: grep *", command="grep bar")
        tool.request_permission("run", command="grep bar; rm -rf /")
        # grep bar is auto-granted; rm -rf / must still prompt.
        assert tool._permission_manager.request.call_count == 1

    def test_malformed_chain_trailing_op_returns_error_from_execute(self):
        tool = make_tool(permission_required=False)
        result = tool.execute(command="ls |")
        assert result["status"] == "error"
        assert result["error"] == "invalid_command"

    def test_malformed_chain_leading_op_returns_error_from_execute(self):
        tool = make_tool(permission_required=False)
        result = tool.execute(command="| grep foo")
        assert result["status"] == "error"
        assert result["error"] == "invalid_command"

    def test_pattern_grant_covers_chain_segment(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.on_permission_granted("always: grep *", command="grep bar")
        tool.request_permission("run", command="cat foo | grep baz")
        # grep baz is covered by pattern — only cat foo needs prompting.
        assert tool._permission_manager.request.call_count == 1

    def test_single_command_still_uses_existing_logic(self):
        tool = make_tool(permission_required=True)
        tool.on_permission_granted("always", command="echo hello")
        allowed, _ = tool.request_permission("run", command="echo hello")
        assert allowed is True
        tool._permission_manager.request.assert_not_called()

    def test_one_time_grant_not_reused_on_next_call(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        # First call: both segments approved one-time.
        tool.request_permission("run", command="cat foo | grep bar")
        call_count_after_first = tool._permission_manager.request.call_count
        # Second call: no always-grant stored, so both segments prompted again.
        tool.request_permission("run", command="cat foo | grep bar")
        assert tool._permission_manager.request.call_count == call_count_after_first * 2

    def test_permission_not_required_skips_all_prompts(self):
        tool = make_tool(permission_required=False)
        allowed, _ = tool.request_permission("run", command="cat foo | grep bar")
        assert allowed is True
        tool._permission_manager.request.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 4: chain execute()
# ---------------------------------------------------------------------------


class TestChainExecute:
    def test_chain_uses_shell_true(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result("result\n")
        ) as mock_run:
            tool.execute(command="cat foo | grep bar")
        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is True

    def test_chain_passes_original_command_string(self):
        tool = make_tool(permission_required=False)
        cmd = "cat foo | grep bar"
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result("result\n")
        ) as mock_run:
            tool.execute(command=cmd)
        args_passed = mock_run.call_args[0][0]
        assert args_passed == cmd

    def test_chain_returns_stdout(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="matched\n"),
        ):
            result = tool.execute(command="cat foo | grep bar")
        assert result["status"] == "success"
        assert result["data"]["output"] == "matched\n"

    def test_chain_nonzero_exit_returns_error(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(returncode=1, stderr="no match"),
        ):
            result = tool.execute(command="cat foo | grep missing")
        assert result["status"] == "error"
        assert result["error"] == "execution_error"
        assert "status 1" in result["message"]

    def test_chain_timeout_returns_timeout(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            side_effect=subprocess.TimeoutExpired(cmd="cat", timeout=30),
        ):
            result = tool.execute(command="cat foo | sleep 999")
        assert result["status"] == "error"
        assert result["error"] == "timeout"
        assert result["code"] == 408
        assert "timed out" in result["message"]

    def test_chain_capture_separate(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="out\n", stderr="err\n"),
        ):
            result = tool.execute(command="cat foo | grep bar", capture="separate")
        assert result["status"] == "success"
        assert result["data"]["stdout"] == "out\n"
        assert result["data"]["stderr"] == "err\n"

    def test_chain_capture_interleaved_kwargs(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="cat foo | grep bar", capture="interleaved")
        kwargs = mock_run.call_args[1]
        assert kwargs["stream_kwargs"]["stderr"] == subprocess.STDOUT

    def test_single_command_does_not_use_shell_true(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="ls -la")
        _, kwargs = mock_run.call_args
        assert not kwargs.get("shell")

    def test_chain_execute_log_shows_raw_command(self):
        tool = make_tool()
        cmd = "cat foo | grep bar"
        result = tool.execute_log(command=cmd)
        assert result == cmd

    def test_chain_execute_log_truncates_long_command(self):
        tool = make_tool()
        cmd = "cat " + "x" * 30 + " | grep " + "y" * 30
        result = tool.execute_log(command=cmd)
        assert result is not None
        assert len(result) == 60
        assert result.endswith("...")

    def test_semicolon_chain_uses_shell_true(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result("done\n")
        ) as mock_run:
            tool.execute(command="ls; echo done")
        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is True


# ---------------------------------------------------------------------------
# Phase 5: _tokenize_segment()
# ---------------------------------------------------------------------------


class TestTokenizeSegment:
    def test_plain_words(self):
        toks = _tokenize_segment("ls -la ./docs")
        assert [v for v, *_ in toks] == ["ls", "-la", "./docs"]

    def test_redirect_op_split_out(self):
        toks = _tokenize_segment("echo hi>out.txt")
        values = [v for v, *_ in toks]
        assert values == ["echo", "hi", ">", "out.txt"]

    def test_append_op(self):
        toks = _tokenize_segment("echo hi>>log.txt")
        assert [v for v, *_ in toks] == ["echo", "hi", ">>", "log.txt"]

    def test_2_adjacent_to_op(self):
        toks = _tokenize_segment("ls 2>&1")
        vals = [v for v, *_ in toks]
        assert vals == ["ls", "2", ">", "&1"]
        # raw_end of "2" == raw_start of ">"
        two_end = toks[1][2]
        gt_start = toks[2][1]
        assert two_end == gt_start

    def test_2_space_separated_from_op(self):
        toks = _tokenize_segment("echo 2 > out.txt")
        vals = [v for v, *_ in toks]
        assert vals == ["echo", "2", ">", "out.txt"]
        two_end = toks[1][2]
        gt_start = toks[2][1]
        assert two_end != gt_start  # NOT adjacent

    def test_single_quoted_string_preserves_content(self):
        toks = _tokenize_segment("echo '> not a redirect'")
        assert toks[1][0] == "> not a redirect"

    def test_raw_positions_are_accurate(self):
        # For an unquoted-only input the raw slice must equal the token value.
        text = "cat<input.txt"
        toks = _tokenize_segment(text)
        for val, start, end in toks:
            assert 0 <= start < end <= len(text)
            assert text[start:end] == val

    def test_dq_backslash_escapes_special_chars(self):
        # Inside double-quotes, backslash escapes \, $, `, ", newline.
        toks = _tokenize_segment(r'echo "\\"')
        assert toks[1][0] == "\\"  # one literal backslash

    def test_dq_backslash_preserved_for_non_special(self):
        # Inside double-quotes, backslash before non-special char is kept literally.
        toks = _tokenize_segment(r'echo "\>"')
        assert toks[1][0] == r"\>"  # backslash preserved — not stripped

    def test_dq_backslash_gt_not_a_redirect(self):
        # echo "\>" — the > is escaped inside double-quotes, so not a redirect.
        cmd, redirs = _parse_redirections(r'echo "\>"')
        assert redirs == []
        assert r"\>" in cmd


# ---------------------------------------------------------------------------
# Phase 5: _parse_redirections()
# ---------------------------------------------------------------------------


class TestParseRedirections:
    def test_no_redirections(self):
        cmd, redirs = _parse_redirections("ls -la")
        assert cmd == "ls -la"
        assert redirs == []

    def test_stdout_redirect(self):
        cmd, redirs = _parse_redirections("cat file > output.txt")
        assert cmd == "cat file"
        assert redirs == ["> output.txt"]

    def test_stdout_append(self):
        cmd, redirs = _parse_redirections("echo hello >> log.txt")
        assert cmd == "echo hello"
        assert redirs == [">> log.txt"]

    def test_stderr_to_stdout(self):
        cmd, redirs = _parse_redirections("ls path 2>&1")
        assert cmd == "ls path"
        assert redirs == ["2>&1"]

    def test_stderr_redirect(self):
        cmd, redirs = _parse_redirections("ls missing 2> err.txt")
        assert cmd == "ls missing"
        assert redirs == ["2> err.txt"]

    def test_stderr_append(self):
        cmd, redirs = _parse_redirections("cmd 2>> err.log")
        assert cmd == "cmd"
        assert redirs == ["2>> err.log"]

    def test_stdin_redirect(self):
        cmd, redirs = _parse_redirections("cat < input.txt")
        assert cmd == "cat"
        assert redirs == ["< input.txt"]

    def test_multiple_redirections(self):
        cmd, redirs = _parse_redirections("cmd > out.txt 2>&1")
        assert cmd == "cmd"
        assert "> out.txt" in redirs
        assert "2>&1" in redirs
        assert len(redirs) == 2

    def test_empty_segment(self):
        cmd, redirs = _parse_redirections("")
        assert cmd == ""
        assert redirs == []

    def test_quoted_redirect_string_not_treated_as_redirect(self):
        # A quoted string whose content starts with ">" but is not a standalone
        # ">" token (e.g. "> not a redirect") is correctly left in the command.
        cmd, redirs = _parse_redirections("echo '> not a redirect'")
        assert redirs == []
        assert "echo" in cmd

    def test_quoted_standalone_operator_is_not_a_redirect(self):
        # A single-quoted ">" is a literal argument, not a redirection operator.
        cmd, redirs = _parse_redirections("echo '>' foo")
        assert redirs == []
        assert ">" in cmd

    def test_dangling_redirect_operator_treated_as_arg(self):
        # A trailing ">" with no following filename is absorbed into command.
        cmd, redirs = _parse_redirections("ls >")
        assert ">" in cmd
        assert redirs == []

    def test_stdout_to_devnull(self):
        cmd, redirs = _parse_redirections("cmd > /dev/null 2>&1")
        assert cmd == "cmd"
        assert "> /dev/null" in redirs
        assert "2>&1" in redirs

    def test_path_with_spaces_in_filename(self):
        # Raw form preserves quoting so the stored grant key reflects the original form.
        cmd, redirs = _parse_redirections("echo hi > 'my file.txt'")
        assert cmd == "echo hi"
        assert redirs == ["> 'my file.txt'"]

    def test_stdout_redirect_no_whitespace(self):
        cmd, redirs = _parse_redirections("echo hi >output.txt")
        assert cmd == "echo hi"
        assert redirs == ["> output.txt"]

    def test_stderr_redirect_no_whitespace(self):
        cmd, redirs = _parse_redirections("ls 2>err.txt")
        assert cmd == "ls"
        assert redirs == ["2> err.txt"]

    def test_append_redirect_no_whitespace(self):
        cmd, redirs = _parse_redirections("echo hi >>log.txt")
        assert cmd == "echo hi"
        assert redirs == [">> log.txt"]

    def test_stdin_redirect_no_whitespace(self):
        cmd, redirs = _parse_redirections("cat <input.txt")
        assert cmd == "cat"
        assert redirs == ["< input.txt"]

    def test_stderr_append_no_whitespace(self):
        cmd, redirs = _parse_redirections("cmd 2>>err.log")
        assert cmd == "cmd"
        assert redirs == ["2>> err.log"]

    def test_no_whitespace_redirect_does_not_affect_2_to_1(self):
        # "2>&1" must still be treated as self-contained, not split as "2>" + "&1".
        cmd, redirs = _parse_redirections("ls 2>&1")
        assert cmd == "ls"
        assert redirs == ["2>&1"]

    def test_operator_adjacent_to_preceding_word(self):
        # "echo hi>out.txt" — operator immediately follows a non-digit word.
        cmd, redirs = _parse_redirections("echo hi>out.txt")
        assert cmd == "echo hi"
        assert redirs == ["> out.txt"]

    def test_stdin_redirect_adjacent_to_preceding_word(self):
        # "cat<input.txt" — "<" immediately follows the command name.
        cmd, redirs = _parse_redirections("cat<input.txt")
        assert cmd == "cat"
        assert redirs == ["< input.txt"]

    def test_append_adjacent_to_preceding_word(self):
        cmd, redirs = _parse_redirections("echo hi>>log.txt")
        assert cmd == "echo hi"
        assert redirs == [">> log.txt"]

    def test_space_separated_digit_is_not_fd_prefix(self):
        # "echo 2 > out.txt" prints "2" to out.txt (stdout redirect); the "2"
        # must NOT be absorbed as an fd prefix — it is a command argument.
        cmd, redirs = _parse_redirections("echo 2 > out.txt")
        assert cmd == "echo 2"
        assert redirs == ["> out.txt"]

    def test_adjacent_digit_is_fd_prefix(self):
        # "ls 2>&1" has no space between "2" and ">", so "2" IS the fd prefix.
        cmd, redirs = _parse_redirections("ls 2>&1")
        assert cmd == "ls"
        assert redirs == ["2>&1"]

    def test_adjacent_digit_fd_redirect_to_file(self):
        cmd, redirs = _parse_redirections("cmd 2>err.txt")
        assert cmd == "cmd"
        assert redirs == ["2> err.txt"]

    def test_quoted_digit_not_absorbed_as_fd_prefix(self):
        # Shell requires an unquoted integer for an IO number: '2'>file means
        # "print '2' to file" (stdout redirect), NOT a stderr redirect.
        cmd, redirs = _parse_redirections("echo '2'>out.txt")
        assert cmd == "echo 2"
        assert redirs == ["> out.txt"]

    def test_backslash_escaped_digit_not_absorbed_as_fd_prefix(self):
        # \2 is an escaped literal "2" — not a bare integer IO number.
        cmd, redirs = _parse_redirections(r"echo \2>out.txt")
        assert cmd == r"echo 2"
        assert redirs == ["> out.txt"]

    def test_space_separated_digit_not_absorbed_even_when_adjacent_op_appears_later(
        self,
    ):
        # "echo 2 > out.txt 2>err.txt": the first ">" has a space before it so
        # "2" must NOT be absorbed as its fd prefix, even though "2>" appears
        # later in the segment (a false-positive from a global substring check).
        cmd, redirs = _parse_redirections("echo 2 > out.txt 2>err.txt")
        assert cmd == "echo 2"
        assert "> out.txt" in redirs
        assert "2> err.txt" in redirs
        assert len(redirs) == 2

    def test_quoted_fd_target_treated_as_filename(self):
        # cmd >'&1' — the &1 is quoted, so not an fd-dup target.
        # Raw form preserves the quoting so grant for "> '&1'" != grant for "> &1".
        cmd, redirs = _parse_redirections("cmd >'&1'")
        assert cmd == "cmd"
        assert redirs == ["> '&1'"]

    def test_backslash_fd_target_treated_as_filename(self):
        # cmd >\&1 — the & is escaped, so not an fd-dup target.
        # Raw form preserves the backslash.
        cmd, redirs = _parse_redirections(r"cmd >\&1")
        assert cmd == "cmd"
        assert redirs == [r"> \&1"]

    def test_space_before_fd_target_treated_as_filename(self):
        # cmd > &1 — space between op and &1, so not fd-dup → file named "&1".
        cmd, redirs = _parse_redirections("cmd > &1")
        assert cmd == "cmd"
        assert redirs == ["> &1"]

    def test_adjacent_unquoted_fd_target_is_fd_dup(self):
        # cmd >&1 — adjacent and unquoted → genuine fd-dup.
        cmd, redirs = _parse_redirections("cmd >&1")
        assert cmd == "cmd"
        assert redirs == [">&1"]

    def test_space_between_op_and_ampersand_digit_is_filename(self):
        # "cmd > &1 other" — &1 is not adjacent to >, so not fd-dup; treated as filename.
        cmd, redirs = _parse_redirections("cmd > &1 other")
        assert redirs == ["> &1"]
        assert "other" in cmd

    def test_single_quoted_gt_is_not_redirect(self):
        # echo '>' foo — the > is quoted, so it is a literal argument.
        cmd, redirs = _parse_redirections("echo '>' foo")
        assert redirs == []
        assert ">" in cmd

    def test_backslash_escaped_gt_is_not_redirect(self):
        # echo \> foo — the > is backslash-escaped, so it is a literal argument.
        cmd, redirs = _parse_redirections(r"echo \> foo")
        assert redirs == []
        assert ">" in cmd

    def test_double_quoted_gt_is_not_redirect(self):
        # echo ">" foo — the > is inside double-quotes, so it is a literal argument.
        cmd, redirs = _parse_redirections('echo ">" foo')
        assert redirs == []
        assert ">" in cmd


# ---------------------------------------------------------------------------
# Phase 5: permission flow for redirections
# ---------------------------------------------------------------------------


class TestRedirectionPermission:
    def test_stdout_redirect_two_prompts(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.request_permission("run", command="cat file > output.txt")
        assert tool._permission_manager.request.call_count == 2

    def test_stderr_to_stdout_two_prompts(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.request_permission("run", command="ls path 2>&1")
        assert tool._permission_manager.request.call_count == 2

    def test_redirect_denial_denies_whole_call(self):
        tool = make_tool(permission_required=True)
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            return (
                (True, "yes") if call_count[0] == 1 else (False, "Permission denied.")
            )

        tool._permission_manager.request.side_effect = side_effect
        allowed, _ = tool.request_permission("run", command="cat file > output.txt")
        assert allowed is False
        assert call_count[0] == 2

    def test_command_denial_skips_redirect_check(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (False, "Permission denied.")
        allowed, _ = tool.request_permission("run", command="cat file > output.txt")
        assert allowed is False
        # Command denied → redirect never prompted.
        assert tool._permission_manager.request.call_count == 1

    def test_redirect_exact_grant_skips_prompt(self):
        tool = make_tool(permission_required=True)
        tool.on_permission_granted("always", command="cat file")
        tool.on_permission_granted("always", redirection="> output.txt")
        tool.request_permission("run", command="cat file > output.txt")
        tool._permission_manager.request.assert_not_called()

    def test_redirect_pattern_grant_skips_prompt(self):
        tool = make_tool(permission_required=True)
        tool.on_permission_granted("always", command="cat file")
        tool.on_permission_granted("always: > *", redirection="> output.txt")
        tool.request_permission("run", command="cat file > other.txt")
        tool._permission_manager.request.assert_not_called()

    def test_redirect_path_pattern_grant(self):
        tool = make_tool(permission_required=True)
        tool.on_permission_granted("always", command="cat file")
        tool.on_permission_granted("always: > ./docs/*", redirection="> ./docs/out.txt")
        allowed, _ = tool.request_permission("run", command="cat file > ./docs/new.txt")
        assert allowed is True
        tool._permission_manager.request.assert_not_called()

    def test_append_not_covered_by_redirect_grant(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.on_permission_granted("always", command="echo hello")
        tool.on_permission_granted("always: > *", redirection="> output.txt")
        # ">>" should NOT be covered by the "> *" pattern.
        tool.request_permission("run", command="echo hello >> log.txt")
        assert tool._permission_manager.request.call_count == 1

    def test_wildcard_grant_does_not_match_absolute_path(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.on_permission_granted("always", command="cat file")
        tool.on_permission_granted("always: > *", redirection="> output.txt")
        # "> *" must NOT cover absolute paths like "> /etc/passwd".
        tool.request_permission("run", command="cat file > /etc/passwd")
        assert tool._permission_manager.request.call_count == 1

    def test_dir_wildcard_grant_does_not_match_subdirectory(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.on_permission_granted("always", command="cat file")
        tool.on_permission_granted("always: > ./docs/*", redirection="> ./docs/out.txt")
        # "> ./docs/*" must NOT cover deeper paths like "> ./docs/sub/file".
        tool.request_permission("run", command="cat file > ./docs/sub/file")
        assert tool._permission_manager.request.call_count == 1

    def test_no_redirect_single_prompt(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.request_permission("run", command="cat file")
        assert tool._permission_manager.request.call_count == 1

    def test_permission_not_required_skips_redirect_check(self):
        tool = make_tool(permission_required=False)
        allowed, _ = tool.request_permission("run", command="cat file > output.txt")
        assert allowed is True
        tool._permission_manager.request.assert_not_called()

    def test_shell_meta_redir_not_auto_approved_by_wildcard_grant(self):
        # A wildcard grant "> *" must NOT auto-approve a target containing '$'.
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.on_permission_granted("always", command="cat file")
        tool.on_permission_granted("always: > *", redirection="> output.txt")
        # Simulate: the model passes a redirection target with a shell metachar.
        # _redir_is_granted must return False for "$(rm -rf /)", forcing a prompt.
        tool.request_permission("run", command="cat file > '$(rm -rf /)'")
        assert tool._permission_manager.request.call_count >= 1

    def test_backtick_redir_not_auto_approved_by_wildcard_grant(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.on_permission_granted("always", command="cat file")
        tool.on_permission_granted("always: > *", redirection="> output.txt")
        tool.request_permission("run", command="cat file > '`id`'")
        assert tool._permission_manager.request.call_count >= 1

    def test_redirect_always_grant_survives_later_denial(self):
        # An always grant on the redirect must be stored even if a later
        # segment in a chain is denied.
        tool = make_tool(permission_required=True)
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (True, "always")  # command: cat file — always grant
            if call_count[0] == 2:
                return (True, "always")  # redirect: > out.txt — always grant
            return (False, "Permission denied.")  # grep: denied

        tool._permission_manager.request.side_effect = side_effect
        allowed, _ = tool.request_permission(
            "run", command="cat file > out.txt | grep foo"
        )
        assert allowed is False
        # Both always grants must be stored.
        assert "cat file" in tool._exact_grants
        assert "> out.txt" in tool._exact_grants


# ---------------------------------------------------------------------------
# Phase 5: extra_permission_options for redirections
# ---------------------------------------------------------------------------


class TestRedirPatternMatch:
    def test_star_matches_plain_filename(self):
        assert _redir_pattern_match("> out.txt", "> *") is True

    def test_star_does_not_match_absolute_path(self):
        assert _redir_pattern_match("> /etc/passwd", "> *") is False

    def test_star_does_not_cross_slash(self):
        assert _redir_pattern_match("> ./docs/out.txt", "> *") is False

    def test_dir_star_matches_direct_child(self):
        assert _redir_pattern_match("> ./docs/out.txt", "> ./docs/*") is True

    def test_dir_star_does_not_match_subdirectory(self):
        assert _redir_pattern_match("> ./docs/sub/file", "> ./docs/*") is False

    def test_dir_star_does_not_match_absolute_path(self):
        assert _redir_pattern_match("> /etc/passwd", "> ./docs/*") is False

    def test_exact_match_no_wildcard(self):
        assert _redir_pattern_match("2>&1", "2>&1") is True

    def test_exact_match_no_wildcard_mismatch(self):
        assert _redir_pattern_match("2>&2", "2>&1") is False

    def test_append_operator(self):
        assert _redir_pattern_match(">> log.txt", ">> *") is True

    def test_stdout_pattern_does_not_match_append(self):
        assert _redir_pattern_match(">> log.txt", "> *") is False

    def test_star_does_not_match_windows_absolute_path(self):
        # "> *" must not cover Windows absolute paths containing backslashes.
        assert _redir_pattern_match(r"> C:\Windows\system32\file", "> *") is False

    def test_star_does_not_cross_backslash(self):
        # "*" must not cross a Windows directory separator.
        assert _redir_pattern_match(r"> dir\file.txt", "> *") is False

    def test_backslash_dir_star_matches_direct_child(self):
        # A pattern with a literal backslash prefix can match its direct children.
        assert _redir_pattern_match(r"> dir\file.txt", r"> dir\*") is True

    def test_backslash_dir_star_does_not_match_subdirectory(self):
        assert _redir_pattern_match(r"> dir\sub\file", r"> dir\*") is False

    def test_question_mark_matches_single_char(self):
        assert _redir_pattern_match("> out.txt", "> out.???") is True

    def test_question_mark_does_not_cross_slash(self):
        assert _redir_pattern_match("> a/b", "> a?b") is False

    def test_bracket_class_matches(self):
        assert _redir_pattern_match("> out.txt", "> out.[tT]xt") is True

    def test_bracket_class_no_match(self):
        assert _redir_pattern_match("> out.txt", "> out.[xyz]xt") is False

    def test_unclosed_bracket_treated_as_literal(self):
        # An unclosed '[' is treated as a literal character, not a class.
        assert _redir_pattern_match("> [out.txt", "> [out.txt") is True

    def test_negated_bracket_class_matches(self):
        # "[!a]" in fnmatch means "not a" — must be translated to "[^a]" in regex.
        assert _redir_pattern_match("> out.txt", "> out.[!x]xt") is True

    def test_negated_bracket_class_no_match(self):
        assert _redir_pattern_match("> out.txt", "> out.[!t]xt") is False

    def test_quoted_target_grant_does_not_match_unquoted_expanding_form(self):
        # Approving "> '$(id)'" must NOT auto-approve "> $(id)" (unquoted).
        # The raw form is stored so the two are different exact grant keys.
        cmd, redirs_quoted = _parse_redirections("cmd > '$(id)'")
        cmd2, redirs_unquoted = _parse_redirections("cmd > $(id)")
        assert redirs_quoted == ["> '$(id)'"]
        assert redirs_unquoted == ["> $(id)"]
        assert redirs_quoted[0] != redirs_unquoted[0]


class TestRedirectionExtraOptions:
    def test_redirect_with_file_has_two_options(self):
        tool = make_tool()
        opts = tool.extra_permission_options(redirection="> output.txt")
        assert "always" in opts
        assert len(opts) == 2

    def test_redirect_wildcard_no_parent_dir(self):
        tool = make_tool()
        opts = tool.extra_permission_options(redirection="> output.txt")
        assert "always: > *" in opts

    def test_redirect_wildcard_uses_parent_dir(self):
        tool = make_tool()
        opts = tool.extra_permission_options(redirection="> ./docs/output.txt")
        assert "always: > ./docs/*" in opts

    def test_redirect_no_file_only_always(self):
        tool = make_tool()
        opts = tool.extra_permission_options(redirection="2>&1")
        assert opts == ["always"]

    def test_append_redirect_wildcard(self):
        tool = make_tool()
        opts = tool.extra_permission_options(redirection=">> /tmp/log.txt")
        assert "always: >> /tmp/*" in opts

    def test_stdin_redirect_wildcard(self):
        tool = make_tool()
        opts = tool.extra_permission_options(redirection="< input.txt")
        assert "always: < *" in opts

    def test_stderr_redirect_wildcard(self):
        tool = make_tool()
        opts = tool.extra_permission_options(redirection="2> err.txt")
        assert "always: 2> *" in opts

    def test_root_level_file_no_double_slash(self):
        # dirname("/out.txt") == "/" — must emit "> /*" not "> //*".
        tool = make_tool()
        opts = tool.extra_permission_options(redirection="> /out.txt")
        assert "always: > /*" in opts
        assert "always: > //*" not in opts

    def test_backslash_separator_parent(self):
        # On POSIX, os.path.dirname("dir\\file.txt") returns "" — we'd lose the
        # parent.  The separator-aware split must give "> dir\\*" instead of "> *".
        tool = make_tool()
        opts = tool.extra_permission_options(redirection=r"> dir\file.txt")
        assert r"always: > dir\*" in opts

    def test_quoted_filename_falls_back_to_exact_only(self):
        # A quoted target like "'./docs/out.txt'" would produce a malformed
        # pattern ">'./docs/*" (unbalanced quote); offer only "always".
        tool = make_tool()
        opts = tool.extra_permission_options(redirection="> './docs/out.txt'")
        assert opts == ["always"]

    def test_quoted_simple_filename_falls_back_to_exact_only(self):
        # Even a simply-quoted filename like "'out.txt'" skips the wildcard option.
        tool = make_tool()
        opts = tool.extra_permission_options(redirection="> 'out.txt'")
        assert opts == ["always"]

    def test_shell_meta_in_filename_suppresses_wildcard_option(self):
        # A target like "> $HOME/file" contains a metachar; wildcard grant would
        # never fire (blocked in _redir_is_granted), so only "always" is offered.
        tool = make_tool()
        opts = tool.extra_permission_options(redirection="> $HOME/file.txt")
        assert opts == ["always"]

    def test_backtick_in_filename_suppresses_wildcard_option(self):
        tool = make_tool()
        opts = tool.extra_permission_options(redirection="> `id`.txt")
        assert opts == ["always"]


# ---------------------------------------------------------------------------
# Phase 5: execute() with redirections
# ---------------------------------------------------------------------------


class TestRedirectionExecute:
    def test_stdout_redirect_uses_shell(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="echo hello > /tmp/out.txt")
        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is True

    def test_stdout_redirect_passes_full_command_string(self):
        tool = make_tool(permission_required=False)
        cmd = "echo hello > /tmp/out.txt"
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command=cmd)
        args_passed = mock_run.call_args[0][0]
        assert args_passed == cmd

    def test_plain_command_no_redirect_does_not_use_shell(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="ls -la")
        _, kwargs = mock_run.call_args
        assert not kwargs.get("shell")

    def test_redirect_execute_log_shows_raw_command(self):
        tool = make_tool()
        cmd = "echo hello > output.txt"
        assert tool.execute_log(command=cmd) == cmd

    def test_redirect_execute_log_truncates_long_command(self):
        tool = make_tool()
        cmd = "echo " + "x" * 30 + " > " + "y" * 30
        result = tool.execute_log(command=cmd)
        assert result is not None
        assert len(result) == 60
        assert result.endswith("...")

    def test_stdout_redirect_no_whitespace_uses_shell(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="echo hi >output.txt")
        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is True

    def test_stderr_redirect_no_whitespace_permission_prompt(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.request_permission("run", command="ls 2>err.txt")
        assert tool._permission_manager.request.call_count == 2

    def test_operator_adjacent_to_word_uses_shell(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="echo hi>output.txt")
        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is True

    def test_operator_adjacent_to_word_two_permission_prompts(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.request_permission("run", command="echo hi>output.txt")
        assert tool._permission_manager.request.call_count == 2


# ---------------------------------------------------------------------------
# Phase 5: redirections inside chained commands
# ---------------------------------------------------------------------------


class TestRedirectionInChain:
    def test_chain_redirect_in_first_segment_prompts_separately(self):
        # "echo hi>out.txt | wc" must produce 3 prompts:
        # 1) echo hi (command), 2) >out.txt (redirect), 3) wc (command)
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.request_permission("run", command="echo hi>out.txt | wc")
        assert tool._permission_manager.request.call_count == 3

    def test_chain_redirect_denial_in_first_segment_denies_chain(self):
        # Denying the redirect in segment 1 aborts the whole chain.
        tool = make_tool(permission_required=True)
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            return (
                (True, "yes") if call_count[0] == 1 else (False, "Permission denied.")
            )

        tool._permission_manager.request.side_effect = side_effect
        allowed, _ = tool.request_permission("run", command="echo hi>out.txt | wc")
        assert allowed is False
        assert call_count[0] == 2  # cmd approved, redirect denied → stop

    def test_chain_redirect_grant_skips_redirect_prompt(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.on_permission_granted("always", command="echo hi")
        # _parse_redirections normalises the redirect with a space: "> out.txt"
        tool.on_permission_granted("always", redirection="> out.txt")
        # Only "wc" needs prompting.
        tool.request_permission("run", command="echo hi>out.txt | wc")
        assert tool._permission_manager.request.call_count == 1

    def test_chain_redirect_and_operator_prompts_each_segment(self):
        # "cmd > out.txt && other" → 3 prompts: cmd, redirect, other
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.request_permission("run", command="cmd > out.txt && other")
        assert tool._permission_manager.request.call_count == 3

    def test_chain_redirect_in_second_segment_prompts_correctly(self):
        # "cat foo | grep bar > out.txt" → 3 prompts: cat, grep, redirect
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.request_permission("run", command="cat foo | grep bar > out.txt")
        assert tool._permission_manager.request.call_count == 3


# ---------------------------------------------------------------------------
# Phase 6: heredoc support
# ---------------------------------------------------------------------------


class TestHasHeredoc:
    def test_heredoc_detected(self):
        assert _has_heredoc("cat <<EOF\nhello\nEOF") is True

    def test_heredoc_with_dash(self):
        assert _has_heredoc("cat <<-EOF\nhello\nEOF") is True

    def test_heredoc_with_space_before_marker(self):
        assert _has_heredoc("cat << EOF\nhello\nEOF") is True

    def test_plain_command_not_heredoc(self):
        assert _has_heredoc("ls -la") is False

    def test_redirect_less_than_not_heredoc(self):
        assert _has_heredoc("cat < input.txt") is False

    def test_append_not_heredoc(self):
        assert _has_heredoc("echo hi >> log.txt") is False

    def test_heredoc_in_chain_segment(self):
        assert _has_heredoc("cat <<EOF\ntest\nEOF") is True

    def test_stdin_redirect_alone_not_heredoc(self):
        assert _has_heredoc("echo 2>&1") is False

    # --- quote-awareness ---

    def test_bitshift_inside_single_quotes_not_heredoc(self):
        # << inside single quotes is a bitshift, not a heredoc operator.
        assert _has_heredoc("python3 -c 'print(1<<2)'") is False

    def test_bitshift_inside_double_quotes_not_heredoc(self):
        assert _has_heredoc('python3 -c "print(1<<2)"') is False

    def test_bitshift_inside_ansi_c_quotes_not_heredoc(self):
        # << inside $'...' (ANSI-C quoting) must not be detected as a heredoc.
        assert _has_heredoc("python3 -c $'print(1<<2)'") is False

    def test_escaped_quote_in_ansi_c_does_not_exit_string(self):
        # $'print(\'<<\')' — the \' sequences are escapes inside $'...', not
        # string terminators.  The << must remain inside the quoted region.
        assert _has_heredoc("python3 -c $'print(\\'<<\\')'") is False

    def test_heredoc_after_ansi_c_quoted_arg_detected(self):
        # The << is outside the $'...' region; it must still be detected.
        assert _has_heredoc("cmd $'arg' <<EOF\ntest\nEOF") is True

    def test_heredoc_after_single_quoted_arg_detected(self):
        # The << is outside the quoted region.
        assert _has_heredoc("cmd 'arg' <<EOF\ntest\nEOF") is True

    def test_here_string_three_less_than_not_heredoc(self):
        # <<< is a here-string (bash), not a heredoc.
        assert _has_heredoc("cat <<<word") is False

    def test_heredoc_with_quoted_marker_detected(self):
        # <<'EOF' uses a quoted marker (suppresses variable expansion in body);
        # the << operator itself is unquoted, so this IS a heredoc.
        assert _has_heredoc("cat <<'EOF'\nhello\nEOF") is True

    def test_heredoc_with_doublequoted_marker_detected(self):
        assert _has_heredoc('cat <<"EOF"\nhello\nEOF') is True

    def test_heredoc_without_quotes_detected(self):
        assert _has_heredoc("cat <<MARKER\ntest\nMARKER") is True

    def test_single_less_than_is_stdin_redirect(self):
        assert _has_heredoc("cat <file.txt") is False

    # --- arithmetic / conditional contexts that must not false-positive ---

    def test_arithmetic_expansion_bitshift_not_heredoc(self):
        # $((1<<2)) — << is inside an arithmetic expansion, not a heredoc.
        assert _has_heredoc("echo $((1<<2))") is False

    def test_compound_assign_shift_not_heredoc(self):
        # ((x<<=1)) — <<= is the compound left-shift-assign operator.
        assert _has_heredoc("((x<<=1))") is False

    def test_conditional_shift_not_heredoc(self):
        # [[ $a << $b ]] — << is inside [[ ]], so depth tracking suppresses
        # heredoc detection in this conditional context.
        assert _has_heredoc("[[ $a << $b ]]") is False

    def test_if_conditional_shift_not_heredoc(self):
        # Common real-world form: the [[ ... ]] command is prefixed by "if".
        # The keyword "if" is recognised as command-introducing, so at_cmd_start
        # becomes True before "[[" and depth tracking correctly suppresses "<<".
        assert _has_heredoc("if [[ $a << $b ]]; then echo ok; fi") is False

    def test_if_arithmetic_shift_not_heredoc(self):
        # "if ((...))": the keyword "if" must reset at_cmd_start so that "(("
        # is treated as an arithmetic compound command, not a literal argument.
        assert _has_heredoc("if ((1<<2)); then echo ok; fi") is False

    def test_if_arithmetic_no_space_not_heredoc(self):
        # "if((" — no space between keyword and compound command.  _end_word()
        # must be called on the first "(" so that "if" is recognised as a
        # keyword before the "((" check fires.
        assert _has_heredoc("if((1<<2)); then echo ok; fi") is False

    def test_quoted_word_then_bracket_heredoc_detected(self):
        # 'echo' is a quoted token, not a command-introducing keyword.
        # at_cmd_start must be cleared so [[ is treated as a literal argument
        # and the heredoc after it is still detected.
        assert _has_heredoc("'echo' [[ foo <<EOF\nhello\nEOF") is True

    def test_heredoc_without_newline_still_detected(self):
        # "cat <<EOF" with no newline is detected as a heredoc — the scanner
        # returns True as soon as it finds a valid letter-start delimiter after
        # "<<", without requiring a newline.
        assert _has_heredoc("cat <<EOF") is True

    def test_heredoc_with_pipe_after_delimiter_detected(self):
        # "cat <<EOF | wc" — extra operator after the delimiter; still a heredoc.
        assert _has_heredoc("cat <<EOF | wc") is True

    def test_heredoc_with_redirect_after_delimiter_detected(self):
        # "cat <<EOF >out.txt" — redirection after the delimiter; still a heredoc.
        assert _has_heredoc("cat <<EOF >out.txt") is True

    def test_heredoc_trailing_whitespace_after_delimiter_detected(self):
        # Shell allows trailing spaces/tabs after the delimiter before the newline.
        assert _has_heredoc("cat <<EOF  \nhello\nEOF") is True

    def test_heredoc_trailing_tab_after_delimiter_detected(self):
        assert _has_heredoc("cat <<EOF\t\nhello\nEOF") is True

    # --- delimiter forms: digit-start and backslash-escaped ---

    def test_heredoc_digit_delimiter_detected(self):
        # <<1 is a valid heredoc in bash/POSIX — digit-start delimiter.
        assert _has_heredoc("cat <<1\nhello\n1") is True

    def test_heredoc_backslash_escaped_delimiter_detected(self):
        # <<\EOF is a valid heredoc — backslash escapes the delimiter.
        assert _has_heredoc("cat <<\\EOF\nhello\nEOF") is True

    def test_heredoc_non_alnum_start_delimiter_detected(self):
        # "<< -EOF" uses "-EOF" as the delimiter (space separates << from -EOF).
        assert _has_heredoc("cat << -EOF\nhello\n-EOF") is True

    def test_heredoc_parameter_expanded_delimiter_detected(self):
        # <<$DELIM is a heredoc — delimiter word begins with $.
        assert _has_heredoc("cat <<$DELIM\nhello\n$DELIM") is True

    def test_heredoc_quoted_parameter_expanded_delimiter_detected(self):
        # <<"$DELIM" — quoted delimiter with parameter expansion inside.
        assert _has_heredoc('cat <<"$DELIM"\nhello\n$DELIM') is True

    # --- depth-tracking: only (( )) and [[ ]] suppress detection ---

    def test_nested_arithmetic_in_command_substitution_not_heredoc(self):
        # << is inside $((...)) nested within $(...); double-bracket depth
        # tracking must suppress it while leaving the outer $(...) transparent.
        assert _has_heredoc("echo $(printf '%s' $((1<<2)))") is False

    def test_heredoc_inside_subshell_detected(self):
        # A genuine heredoc inside a subshell must still be detected — single
        # ( ) does not increment depth, so << is visible at depth zero.
        assert _has_heredoc("(cat <<EOF\nhello\nEOF)") is True

    def test_heredoc_with_quoted_delimiter_inside_subshell_detected(self):
        # Quoted delimiter inside parentheses must also be detected.
        assert _has_heredoc("(cat <<'EOF'\nhello\nEOF)") is True

    def test_heredoc_after_literal_double_bracket_detected(self):
        # "echo [[ foo <<EOF" — [[ is a literal argument to echo, not a
        # compound conditional.  at_cmd_start is False after the word "echo",
        # so [[ must not enter depth-tracking and the heredoc must be found.
        assert _has_heredoc("echo [[ foo <<EOF\nhello\nEOF") is True

    def test_heredoc_after_literal_double_paren_detected(self):
        # "echo ((x)) <<EOF" — (( is a literal argument, not a compound
        # arithmetic command.  Heredoc must still be detected.
        assert _has_heredoc("echo ((x)) <<EOF\nhello\nEOF") is True

    def test_nested_parens_inside_arithmetic_heredoc_detected(self):
        # "echo $(( ((1<<2) ) )) <<EOF" — the inner (( is nested inside the
        # $((…)) arithmetic expansion.  Lone ( inside a tracked construct must
        # NOT increment depth (which would leave depth > 0 after the closing
        # )) and prevent heredoc detection).
        assert _has_heredoc("echo $(( ((1<<2) ) )) <<EOF\nhello\nEOF") is True

    def test_reserved_word_as_argument_heredoc_detected(self):
        # "echo if [[ foo <<EOF" — "if" is an argument to echo, not a
        # command-introducing keyword.  word_started_at_cmd_start is False for
        # "if" (because echo already cleared at_cmd_start), so _end_word()
        # must NOT set at_cmd_start=True.  Consequently [[ is seen at a non-
        # command position and heredoc detection is not suppressed.
        assert _has_heredoc("echo if [[ foo <<EOF\nhello\nEOF") is True

    # --- command prefixes: env assignments and leading redirects ---

    def test_env_assignment_prefix_bracket_not_heredoc(self):
        # "A=1 [[ $a << $b ]]" — A=1 is an env-var prefix, not a command word;
        # at_cmd_start must remain True so [[ enters depth tracking and << is
        # seen as an arithmetic comparison, not a heredoc.
        assert _has_heredoc("A=1 [[ $a << $b ]]") is False

    def test_env_assignment_prefix_heredoc_still_detected(self):
        # Heredoc on an env-prefixed command must still be detected.
        assert _has_heredoc("A=1 cmd <<EOF\nhello\nEOF") is True

    def test_stdin_redirect_prefix_bracket_not_heredoc(self):
        # "< /dev/null [[ $a << $b ]]" — the leading stdin redirect and its
        # target consume two tokens; after them at_cmd_start must be True so
        # [[ enters depth tracking and << is not mistaken for a heredoc.
        assert _has_heredoc("< /dev/null [[ $a << $b ]]") is False

    def test_stdin_redirect_prefix_heredoc_still_detected(self):
        # Heredoc on a command with a leading stdin redirect must be detected.
        assert _has_heredoc("< /dev/null cmd <<EOF\nhello\nEOF") is True

    def test_stdout_redirect_prefix_bracket_not_heredoc(self):
        # "> /dev/null [[ $a << $b ]]" — leading stdout redirect; after the
        # target word at_cmd_start should be True for the same reason.
        assert _has_heredoc("> /dev/null [[ $a << $b ]]") is False

    def test_multiple_env_assignments_bracket_not_heredoc(self):
        # Multiple env-var prefixes before [[ must each preserve at_cmd_start.
        assert _has_heredoc("A=1 B=2 [[ $a << $b ]]") is False


class TestHeredocExtraOptions:
    def test_heredoc_command_returns_always_only(self):
        # Heredoc: only "always" is offered (to intercept the universal always-
        # choice); no wildcard pattern option since the content is dynamic.
        tool = make_tool()
        opts = tool.extra_permission_options(command="cat <<EOF\nhello\nEOF")
        assert opts == ["always"]

    def test_heredoc_with_dash_returns_always_only(self):
        tool = make_tool()
        opts = tool.extra_permission_options(command="python3 <<-EOF\nprint(1)\nEOF")
        assert opts == ["always"]

    def test_non_heredoc_command_returns_options(self):
        tool = make_tool()
        opts = tool.extra_permission_options(command="ls -la")
        assert "always" in opts
        assert len(opts) == 2

    def test_redirection_kwarg_unaffected_by_heredoc(self):
        # The redirection path is independent of heredoc detection.
        tool = make_tool()
        opts = tool.extra_permission_options(redirection="> output.txt")
        assert "always" in opts
        assert len(opts) == 2


class TestHeredocPermission:
    def test_heredoc_prompts_for_permission(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        allowed, _ = tool.request_permission("run", command="cat <<EOF\nhello\nEOF")
        assert allowed is True
        tool._permission_manager.request.assert_called_once()

    def test_heredoc_denial_returns_false(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (False, "Permission denied.")
        allowed, _ = tool.request_permission("run", command="cat <<EOF\nhello\nEOF")
        assert allowed is False

    def test_heredoc_no_grant_stored_after_yes(self):
        # Selecting yes must NOT store any grant — the next identical call must prompt.
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.request_permission("run", command="cat <<EOF\nhello\nEOF")
        call_count_first = tool._permission_manager.request.call_count
        # Second call with the identical command must still prompt.
        tool.request_permission("run", command="cat <<EOF\nhello\nEOF")
        assert tool._permission_manager.request.call_count == call_count_first * 2

    def test_heredoc_no_grant_stored_after_always(self):
        # Even if pm returns "always" (intercepted), no grant must be stored.
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "always")
        tool.request_permission("run", command="cat <<EOF\nhello\nEOF")
        assert len(tool._exact_grants) == 0
        assert len(tool._pattern_grants) == 0
        # Next call must still prompt.
        tool.request_permission("run", command="cat <<EOF\nhello\nEOF")
        assert tool._permission_manager.request.call_count == 2

    def test_heredoc_permission_not_required_skips_prompt(self):
        tool = make_tool(permission_required=False)
        allowed, _ = tool.request_permission("run", command="cat <<EOF\nhello\nEOF")
        assert allowed is True
        tool._permission_manager.request.assert_not_called()

    def test_heredoc_prompt_receives_always_in_extra_options(self):
        # The "always" extra option is passed so pm cannot create a tool-wide grant.
        received_extras: list[list] = []

        def side_effect(**kwargs):
            received_extras.append(kwargs.get("extra_options", []))
            return (True, "yes")

        tool = make_tool(permission_required=True)
        tool._permission_manager.request.side_effect = side_effect
        tool.request_permission("run", command="cat <<EOF\nhello\nEOF")
        assert received_extras[0] == ["always"]

    def test_heredoc_uses_distinct_tool_name(self):
        # Heredoc permission is requested under "bash_heredoc", not "bash", so
        # a pre-existing tool-wide always-grant for bash cannot bypass prompting.
        received_names: list[str] = []

        def side_effect(**kwargs):
            received_names.append(kwargs.get("tool_name", ""))
            return (True, "yes")

        tool = make_tool(permission_required=True)
        tool._permission_manager.request.side_effect = side_effect
        tool.request_permission("run", command="cat <<EOF\nhello\nEOF")
        assert received_names[0] == "bash_heredoc"

    def test_mixed_chain_heredoc_whole_command_one_time(self):
        # Any command containing a heredoc gets a single one-time permission prompt;
        # _parse_chain is bypassed so the body's operators are not misread.
        # No grant is stored — not even for the non-heredoc segment.
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "always")
        tool.request_permission("run", command="echo hello | cat <<EOF\ntest\nEOF")
        assert len(tool._exact_grants) == 0
        assert len(tool._pattern_grants) == 0
        tool._permission_manager.request.assert_called_once()

    def test_mixed_chain_heredoc_one_prompt_with_always_interception(self):
        # The single prompt receives only ["always"] — no wildcard grant option.
        received_extras: list[list] = []

        def side_effect(**kwargs):
            received_extras.append(kwargs.get("extra_options") or [])
            return (True, "yes")

        tool = make_tool(permission_required=True)
        tool._permission_manager.request.side_effect = side_effect
        tool.request_permission("run", command="echo hello | cat <<EOF\ntest\nEOF")
        assert tool._permission_manager.request.call_count == 1
        assert received_extras[0] == ["always"]

    def test_heredoc_denial_in_chain_denies_whole_command(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (False, "Permission denied.")
        allowed, _ = tool.request_permission(
            "run", command="echo hello | cat <<EOF\ntest\nEOF"
        )
        assert allowed is False
        tool._permission_manager.request.assert_called_once()

    def test_heredoc_body_with_pipe_one_time_permission(self):
        # A heredoc whose body contains "|" must NOT be split by _parse_chain.
        # The whole command gets a single one-time permission prompt.
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        allowed, _ = tool.request_permission("run", command="cat <<EOF\na | b\nEOF")
        assert allowed is True
        tool._permission_manager.request.assert_called_once()
        assert len(tool._exact_grants) == 0

    def test_heredoc_body_with_and_op_one_time_permission(self):
        # A heredoc whose body contains "&&" must NOT be split by _parse_chain.
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        allowed, _ = tool.request_permission("run", command="cat <<EOF\na && b\nEOF")
        assert allowed is True
        tool._permission_manager.request.assert_called_once()
        assert len(tool._exact_grants) == 0


class TestHeredocExecute:
    def test_heredoc_uses_shell_true(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="hello\n"),
        ) as mock_run:
            tool.execute(command="cat <<EOF\nhello\nEOF")
        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is True

    def test_heredoc_passes_full_command_string(self):
        tool = make_tool(permission_required=False)
        cmd = "cat <<EOF\nhello\nEOF"
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="hello\n"),
        ) as mock_run:
            tool.execute(command=cmd)
        args_passed = mock_run.call_args[0][0]
        assert args_passed == cmd

    def test_heredoc_success_returns_output(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="hello\n"),
        ):
            result = tool.execute(command="cat <<EOF\nhello\nEOF")
        assert result["status"] == "success"
        assert result["data"]["output"] == "hello\n"

    def test_heredoc_nonzero_exit_returns_error(self):
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(returncode=1, stderr="fail"),
        ):
            result = tool.execute(command="false <<EOF\ntest\nEOF")
        assert result["status"] == "error"
        assert result["error"] == "execution_error"

    def test_plain_command_not_affected_by_heredoc_detection(self):
        # Regression: plain commands without heredoc must still use direct execution.
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen", return_value=_run_result()
        ) as mock_run:
            tool.execute(command="ls -la")
        _, kwargs = mock_run.call_args
        assert not kwargs.get("shell")

    def test_heredoc_body_with_unmatched_quote_uses_shell(self):
        # A heredoc whose body contains an unmatched single quote must NOT return
        # invalid_command — shlex-based parsing is skipped for heredoc commands.
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="hello 'world\n"),
        ) as mock_run:
            result = tool.execute(command="cat <<EOF\nhello 'world\nEOF")
        assert result["status"] == "success"
        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is True

    def test_heredoc_body_with_pipe_uses_shell(self):
        # A heredoc whose body contains a pipe character must NOT be split by
        # the chain parser — the whole command goes to the shell as-is.
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            return_value=_run_result(stdout="a | b\n"),
        ) as mock_run:
            result = tool.execute(command="cat <<EOF\na | b\nEOF")
        assert result["status"] == "success"
        assert mock_run.call_args[0][0] == "cat <<EOF\na | b\nEOF"


class TestHeredocPermissionParseFailure:
    """Heredoc commands with bodies containing unmatched quotes or operators.

    request_permission() bypasses _parse_chain for all heredoc commands, so
    these cases never reach the chain parser regardless of body content.
    """

    def test_parse_failure_still_prompts_once(self):
        # An unmatched quote in the heredoc body causes _parse_chain to fail;
        # request_permission must still prompt once (not raise or silently allow).
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        allowed, _ = tool.request_permission(
            "run", command="cat <<EOF\nhello 'world\nEOF"
        )
        assert allowed is True
        tool._permission_manager.request.assert_called_once()

    def test_parse_failure_no_grant_stored(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.request_permission("run", command="cat <<EOF\nhello 'world\nEOF")
        assert len(tool._exact_grants) == 0
        assert len(tool._pattern_grants) == 0

    def test_parse_failure_always_intercepted(self):
        # Even on parse failure, the "always" choice must NOT create a tool-wide grant.
        received_extras: list[list] = []

        def side_effect(**kwargs):
            received_extras.append(kwargs.get("extra_options", []))
            return (True, "always")

        tool = make_tool(permission_required=True)
        tool._permission_manager.request.side_effect = side_effect
        tool.request_permission("run", command="cat <<EOF\nhello 'world\nEOF")
        assert received_extras[0] == ["always"]
        # No grant must be stored after "always" on a heredoc parse-failure path.
        assert len(tool._exact_grants) == 0

    def test_parse_failure_denial_returns_false(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (False, "Permission denied.")
        allowed, _ = tool.request_permission(
            "run", command="cat <<EOF\nhello 'world\nEOF"
        )
        assert allowed is False

    def test_parse_failure_re_prompt_on_next_call(self):
        tool = make_tool(permission_required=True)
        tool._permission_manager.request.return_value = (True, "yes")
        tool.request_permission("run", command="cat <<EOF\nhello 'world\nEOF")
        first_count = tool._permission_manager.request.call_count
        tool.request_permission("run", command="cat <<EOF\nhello 'world\nEOF")
        assert tool._permission_manager.request.call_count == first_count * 2


# ---------------------------------------------------------------------------
# Phase 7: streaming output capture (_run_popen integration tests)
# ---------------------------------------------------------------------------


class TestStreaming:
    """Integration tests that call _run_popen directly with real subprocesses."""

    def test_large_output_killed_mid_stream(self):
        # The subprocess writes output slowly and only creates a marker file if
        # it reaches normal completion.  _run_popen must kill it at the output
        # limit so the marker is never written.
        max_chars = 100
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = os.path.join(tmpdir, "completed.txt")
            script = (
                "import sys, time\n"
                f"marker = {marker_path!r}\n"
                "for _ in range(1000):\n"
                "    sys.stdout.write('x' * 10)\n"
                "    sys.stdout.flush()\n"
                "    time.sleep(0.05)\n"
                "open(marker, 'w').write('done')\n"
            )
            rc, stdout, stderr, truncated = _run_popen(
                [sys.executable, "-c", script],
                shell=False,
                capture="stdout",
                stream_kwargs={
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                },
                max_output_chars=max_chars,
                cwd=tmpdir,
                env=None,
                timeout=10.0,
            )
            assert truncated is True
            assert len(stdout) == max_chars
            assert not os.path.exists(marker_path)

    def test_output_within_limit_not_truncated(self):
        rc, stdout, stderr, truncated = _run_popen(
            [sys.executable, "-c", "print('hello')"],
            shell=False,
            capture="stdout",
            stream_kwargs={
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            },
            max_output_chars=1024,
            cwd=tempfile.gettempdir(),
            env=None,
            timeout=10.0,
        )
        assert rc == 0
        assert truncated is False
        assert "hello" in stdout

    def test_timeout_raises_timeout_expired(self):
        # A command that produces continuous output; should raise TimeoutExpired.
        # Use sys.executable so this test runs without bash on PATH.
        with pytest.raises(subprocess.TimeoutExpired):
            _run_popen(
                [
                    sys.executable,
                    "-c",
                    "import sys\nwhile True:\n sys.stdout.write('x')\n sys.stdout.flush()",
                ],
                shell=False,
                capture="stdout",
                stream_kwargs={
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                },
                max_output_chars=10_000_000,
                cwd=tempfile.gettempdir(),
                env=None,
                timeout=0.2,
            )

    def test_execute_timeout_returns_timeout(self):
        # Via execute(): a timed-out streaming command returns timeout.
        tool = make_tool(permission_required=False)
        with patch(
            "ai_cli.tools.bash._run_popen",
            side_effect=subprocess.TimeoutExpired(cmd="bash", timeout=30),
        ):
            result = tool.execute(command="bash -c 'while true; do echo x; done'")
        assert result["status"] == "error"
        assert result["error"] == "timeout"
        assert result["code"] == 408
        assert "timed out" in result["message"]

    def test_separate_mode_still_buffered(self):
        # separate uses communicate(); both stdout and stderr returned correctly.
        rc, stdout, stderr, truncated = _run_popen(
            [
                sys.executable,
                "-c",
                "import sys; print('out'); print('err', file=sys.stderr)",
            ],
            shell=False,
            capture="separate",
            stream_kwargs={
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            },
            max_output_chars=1024,
            cwd=tempfile.gettempdir(),
            env=None,
            timeout=10.0,
        )
        assert rc == 0
        assert "out" in stdout
        assert "err" in stderr
        assert truncated is False

    def test_separate_mode_truncation_applied(self):
        # separate mode: output exceeding max_output_chars is truncated.
        max_chars = 5
        rc, stdout, stderr, truncated = _run_popen(
            [sys.executable, "-c", f"print('x' * {max_chars * 10}, end='')"],
            shell=False,
            capture="separate",
            stream_kwargs={
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            },
            max_output_chars=max_chars,
            cwd=tempfile.gettempdir(),
            env=None,
            timeout=10.0,
        )
        assert rc == 0
        assert truncated is True
        assert len(stdout) == max_chars

    def test_stderr_capture_mode_streams_stderr(self):
        rc, stdout, stderr, truncated = _run_popen(
            [sys.executable, "-c", "import sys; print('err', file=sys.stderr)"],
            shell=False,
            capture="stderr",
            stream_kwargs={
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.PIPE,
            },
            max_output_chars=1024,
            cwd=tempfile.gettempdir(),
            env=None,
            timeout=10.0,
        )
        assert rc == 0
        assert stdout == ""
        assert "err" in stderr
        assert truncated is False

    def test_interleaved_capture_mode_merges_streams(self):
        rc, stdout, stderr, truncated = _run_popen(
            [
                sys.executable,
                "-c",
                "import sys; print('out'); print('err', file=sys.stderr)",
            ],
            shell=False,
            capture="interleaved",
            stream_kwargs={
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
            },
            max_output_chars=1024,
            cwd=tempfile.gettempdir(),
            env=None,
            timeout=10.0,
        )
        assert rc == 0
        assert "out" in stdout
        assert "err" in stdout  # merged into stdout
        assert stderr == ""
        assert truncated is False
