"""
PermissionManager — in-memory tool permission state.

Permissions are scoped to the current process lifetime and never written
to disk.  All grants reset when ``reset()`` is called (e.g. on session
resume) or when the process exits.

The four universal permission choices:

``yes``     Allow this one request only.
``no``      Deny this one request only.
``always``  Allow all future requests from this tool for the rest of the
            session.
``custom``  Deny, but return a user-supplied message/suggestion to the LLM.

Tools may offer additional variants (e.g. "Always in this folder") and
pass them via the ``extra_options`` argument to
:meth:`PermissionManager.request`.  Tool-specific extras are forwarded to
``prompt_fn`` (the universal four are NOT forwarded — the prompt
implementation renders them itself with fixed key bindings) and returned to
the caller as-is; PermissionManager does not interpret them.

The ``prompt_fn`` callable is provided by the REPL layer, keeping UI
concerns (Rich formatting, keyboard input) out of this module.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

PERM_YES = "yes"
PERM_NO = "no"
PERM_ALWAYS = "always"
PERM_CUSTOM = "custom"

# Type alias for the injected prompt function.
# prompt_fn(question, extra_options) -> (choice, user_text)
# extra_options contains only tool-specific choices; the universal four
# (yes/no/always/custom) are always rendered by the prompt implementation itself.
PromptFn = Callable[[str, list[str]], tuple[str, str]]


class PermissionManager:
    """
    Manages per-tool permission grants for the current session.

    Parameters
    ----------
    prompt_fn:
        Called when user input is needed.  Receives the question string
        and the full list of option strings (universal + tool-specific).
        Returns ``(choice, user_text)`` where ``user_text`` is non-empty
        only for the ``custom`` choice.
    """

    def __init__(self, prompt_fn: PromptFn) -> None:
        self._prompt_fn = prompt_fn
        self._always_allowed: set[str] = set()

    @property
    def prompt_fn(self) -> PromptFn:
        return self._prompt_fn

    @prompt_fn.setter
    def prompt_fn(self, fn: PromptFn) -> None:
        self._prompt_fn = fn

    def request(
        self,
        tool_name: str,
        question: str,
        extra_options: list[str] | None = None,
    ) -> tuple[bool, str]:
        """
        Check permission for a tool action.

        If the tool has an active ``always`` grant, allow immediately.
        Otherwise call ``prompt_fn`` and act on the response.

        Parameters
        ----------
        tool_name:
            Name of the tool requesting permission.
        question:
            Human-readable description of the action being requested.
        extra_options:
            Tool-specific option strings beyond the universal four.

        Returns
        -------
        tuple[bool, str]
            ``(allowed, reason_or_choice)`` where the second element is:

            * empty string on ``yes`` / ``always`` / always-grant bypass
            * ``"User denied tool request with message: <text>"`` on ``custom``
              when the user supplied a non-empty message, or
              ``"Permission denied."`` if the message was empty
            * the chosen extra-option string on a tool-specific choice
              (caller is responsible for acting on it, e.g. scoped grant)
            * ``"Permission denied."`` on ``no`` or unrecognised input
        """
        if tool_name in self._always_allowed:
            logger.debug("Permission auto-granted for '%s' (always-allow)", tool_name)
            return True, ""

        universal = {PERM_YES, PERM_NO, PERM_ALWAYS, PERM_CUSTOM}

        # The prompt implementation always renders the universal four (yes / no /
        # always / custom) with their own key bindings.  Pass only the
        # tool-specific extras so they do not appear twice.
        extras_for_prompt = [
            opt for opt in (extra_options or []) if opt.strip().lower() not in universal
        ]
        choice, user_text = self._prompt_fn(question, extras_for_prompt)
        choice = choice.strip().lower()

        # PERM_NO always denies, even if it appears in extra_options.
        if choice == PERM_NO:
            logger.info("Permission denied for '%s'", tool_name)
            return False, "Permission denied."

        # Extra options take precedence over the universal allow/custom choices
        # so that e.g. an "always" extra option doesn't create an always-grant.
        if extra_options:
            for opt in extra_options:
                if opt.lower() == choice:
                    # Return the original (unmodified) option string so callers
                    # can reliably match it against their extra_options list.
                    logger.info(
                        "Permission granted for '%s' (choice=%r)", tool_name, opt
                    )
                    return True, opt

        if choice == PERM_YES:
            logger.info("Permission granted for '%s' (once)", tool_name)
            return True, ""
        if choice == PERM_ALWAYS:
            self.grant_always(tool_name)
            logger.info("Permission granted for '%s' (always)", tool_name)
            return True, ""
        if choice == PERM_CUSTOM:
            if user_text:
                logger.info(
                    "Permission denied for '%s' (custom suggestion provided)", tool_name
                )
            else:
                logger.info(
                    "Permission denied for '%s' (custom, no message)", tool_name
                )
            message = (
                f"User denied tool request with message: {user_text}"
                if user_text.strip()
                else "Permission denied."
            )
            return False, message
        # anything unrecognised
        logger.info(
            "Permission denied for '%s' (unrecognised choice %r)", tool_name, choice
        )
        return False, "Permission denied."

    def grant_always(self, tool_name: str) -> None:
        """Record an in-memory always-allow grant for *tool_name*."""
        self._always_allowed.add(tool_name)
        logger.debug("Always-allow registered for '%s'", tool_name)

    def reset(self) -> None:
        """Clear all grants.  Called on session resume."""
        self._always_allowed.clear()
