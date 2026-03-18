"""Tests for ai_cli.core.permission_manager.PermissionManager."""

from ai_cli.core.permission_manager import (
    PERM_ALWAYS,
    PERM_CUSTOM,
    PERM_NO,
    PERM_YES,
    PermissionManager,
)


def make_manager(choice: str, user_text: str = "") -> PermissionManager:
    """Return a PermissionManager whose prompt_fn always returns (choice, user_text)."""
    return PermissionManager(prompt_fn=lambda q, opts: (choice, user_text))


# ---------------------------------------------------------------------------
# Basic permission choices
# ---------------------------------------------------------------------------


class TestRequest:
    def test_yes_allows(self):
        pm = make_manager(PERM_YES)
        allowed, reason = pm.request("read_file", "Read /etc/hosts?")
        assert allowed is True
        assert reason == ""

    def test_no_denies(self):
        pm = make_manager(PERM_NO)
        allowed, reason = pm.request("read_file", "Read /etc/hosts?")
        assert allowed is False
        assert "denied" in reason.lower()

    def test_custom_denies_with_user_text(self):
        pm = make_manager(PERM_CUSTOM, "Try /tmp instead.")
        allowed, reason = pm.request("read_file", "Read /etc/hosts?")
        assert allowed is False
        assert reason == "Try /tmp instead."

    def test_unrecognised_choice_denies(self):
        pm = make_manager("something_else")
        allowed, _ = pm.request("read_file", "Read /etc/hosts?")
        assert allowed is False

    def test_choice_yes_is_case_insensitive(self):
        pm = make_manager("YES")
        allowed, _ = pm.request("read_file", "Read /etc/hosts?")
        assert allowed is True

    def test_choice_no_is_case_insensitive(self):
        pm = make_manager("NO")
        allowed, reason = pm.request("read_file", "Read /etc/hosts?")
        assert allowed is False
        assert "denied" in reason.lower()

    def test_choice_custom_is_case_insensitive(self):
        pm = make_manager("CUSTOM", "Try /tmp instead.")
        allowed, reason = pm.request("read_file", "Read /etc/hosts?")
        assert allowed is False
        assert reason == "Try /tmp instead."

    def test_choice_always_is_case_insensitive(self):
        pm = make_manager("ALWAYS")
        allowed, reason = pm.request("bash", "Run ls?")
        assert allowed is True
        assert reason == ""


# ---------------------------------------------------------------------------
# Always grant
# ---------------------------------------------------------------------------


class TestAlwaysGrant:
    def test_always_grants_future_requests(self):
        called = []

        def prompt_fn(q: str, opts: list[str]) -> tuple[str, str]:
            if not called:
                called.append(1)
                return (PERM_ALWAYS, "")
            raise AssertionError("prompt called after always grant")

        pm = PermissionManager(prompt_fn=prompt_fn)
        allowed1, reason1 = pm.request("bash", "Run ls?")
        assert allowed1 is True
        assert reason1 == ""
        # Second request for the same tool must skip the prompt entirely.
        allowed2, reason2 = pm.request("bash", "Run ls again?")
        assert (allowed2, reason2) == (True, "")

    def test_always_grant_bypasses_prompt(self):
        called = []
        pm = PermissionManager(
            prompt_fn=lambda q, opts: called.append(1) or (PERM_YES, "")
        )
        pm.grant_always("bash")
        allowed, _ = pm.request("bash", "Run ls?")
        assert allowed is True
        assert called == []  # prompt was never called

    def test_grant_always_direct(self):
        pm = make_manager(PERM_NO)
        pm.grant_always("write_file")
        allowed, _ = pm.request("write_file", "Write foo.txt?")
        assert allowed is True

    def test_always_grant_is_tool_specific(self):
        pm = make_manager(PERM_NO)
        pm.grant_always("read_file")
        # write_file still needs to prompt (and gets NO)
        allowed, _ = pm.request("write_file", "Write foo.txt?")
        assert allowed is False


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_always_grants(self):
        pm = make_manager(PERM_NO)
        pm.grant_always("bash")
        pm.reset()
        allowed, _ = pm.request("bash", "Run ls?")
        assert allowed is False

    def test_reset_is_idempotent(self):
        pm = make_manager(PERM_YES)
        pm.reset()
        pm.reset()
        allowed, _ = pm.request("read_file", "Read file?")
        assert allowed is True


# ---------------------------------------------------------------------------
# Extra options
# ---------------------------------------------------------------------------


class TestExtraOptions:
    def test_extra_options_passed_to_prompt(self):
        # prompt_fn receives only tool-specific extras, not the universal four
        # (yes/no/always/custom) — the prompt implementation renders those itself.
        received_opts: list[list[str]] = []

        def prompt_fn(question: str, opts: list[str]) -> tuple[str, str]:
            received_opts.append(opts)
            return PERM_YES, ""

        pm = PermissionManager(prompt_fn=prompt_fn)
        pm.request("read_file", "Read file?", extra_options=["always_in_folder"])
        assert received_opts[0] == ["always_in_folder"]

    def test_extra_option_selected_allows_and_returns_choice(self):
        pm = PermissionManager(prompt_fn=lambda q, opts: ("always_in_folder", ""))
        allowed, choice = pm.request(
            "read_file", "Read file?", extra_options=["always_in_folder"]
        )
        assert allowed is True
        assert choice == "always_in_folder"

    def test_extra_option_named_no_does_not_allow(self):
        # An extra_option whose name collides with PERM_NO must never allow.
        pm = PermissionManager(prompt_fn=lambda q, opts: ("no", ""))
        allowed, _ = pm.request("read_file", "Read file?", extra_options=["no"])
        assert allowed is False

    def test_extra_option_named_always_does_not_create_always_grant(self):
        calls: list[int] = []

        def prompt_fn(q: str, opts: list[str]) -> tuple[str, str]:
            calls.append(1)
            return ("always", "") if len(calls) == 1 else (PERM_NO, "")

        pm = PermissionManager(prompt_fn=prompt_fn)
        allowed1, choice1 = pm.request("read_file", "Read?", extra_options=["always"])
        assert allowed1 is True
        assert choice1 == "always"
        # No always-grant must have been created — second request should deny.
        allowed2, _ = pm.request("read_file", "Read?")
        assert allowed2 is False

    def test_extra_option_named_yes_behaves_like_extra_option(self):
        pm = PermissionManager(prompt_fn=lambda q, opts: ("yes", ""))
        allowed, choice = pm.request("read_file", "Read?", extra_options=["yes"])
        assert allowed is True
        assert choice == "yes"

    def test_extra_option_named_custom_behaves_like_extra_option(self):
        pm = PermissionManager(prompt_fn=lambda q, opts: ("custom", ""))
        allowed, choice = pm.request("read_file", "Read?", extra_options=["custom"])
        assert allowed is True
        assert choice == "custom"

    def test_mixed_case_extra_option_returns_original_string(self):
        pm = PermissionManager(prompt_fn=lambda q, opts: ("AlwaysInFolder", ""))
        allowed, choice = pm.request(
            "read_file", "Read file?", extra_options=["AlwaysInFolder"]
        )
        assert allowed is True
        assert choice == "AlwaysInFolder"

    def test_no_extra_options_uses_universal_only(self):
        # With no tool-specific extras, prompt_fn receives an empty list — the
        # universal four are rendered by the prompt implementation, not passed here.
        received_opts: list[list[str]] = []

        def prompt_fn(question: str, opts: list[str]) -> tuple[str, str]:
            received_opts.append(opts)
            return PERM_YES, ""

        pm = PermissionManager(prompt_fn=prompt_fn)
        pm.request("read_file", "Read file?")
        assert received_opts[0] == []
