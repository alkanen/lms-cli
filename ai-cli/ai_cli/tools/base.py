"""
Tool base class.

Every bundled, user, and project tool must subclass ``Tool`` and implement
``definition()`` and ``execute()``.  The base class handles permission
checking via the injected ``PermissionManager`` and provides sensible
defaults for the optional hook methods.

Canonical tool result shapes (returned by ``execute()``):

  Success:  {"status": "success", "data": {...}}
  Error:    {"status": "error", "error": "<code>", "message": "<text>",
             "code": <int>, "details": {...}}  # "details" is optional

Helper methods ``_ok`` and ``_err`` produce these shapes so subclasses
don't have to construct them by hand.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.workspace import Workspace


class ToolArgument:
    """
    Typed descriptor for a single tool parameter.

    Produces the ``properties.<name>`` entry expected by the OpenAI
    function-calling schema.  Mypy can validate constructor arguments,
    eliminating the need for hand-crafted dicts and the runtime checks they
    require.

    Parameters
    ----------
    name:
        Parameter name as used in the tool's ``execute(**kwargs)`` signature.
    description:
        Human-readable explanation shown to the LLM.
    argument_type:
        JSON Schema primitive type: ``"string"``, ``"integer"``, ``"number"``,
        ``"boolean"``, ``"array"``, or ``"object"``.
    required:
        Whether this argument must be present in every call.  Defaults to
        ``False`` (optional).
    enum:
        Restrict valid values to this list (e.g. ``["list", "enable"]``).
        Only meaningful when *argument_type* is ``"string"``.
    items:
        For ``argument_type="array"``, the JSON Schema for each element
        (e.g. ``{"type": "string"}``).
    """

    def __init__(
        self,
        name: str,
        description: str,
        argument_type: str,
        *,
        required: bool = False,
        enum: list[str] | None = None,
        items: dict | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.argument_type = argument_type
        self.required = required
        self._enum = enum
        self._items = items

    def schema(self) -> dict:
        """Return the JSON Schema fragment for this argument."""
        prop: dict = {"type": self.argument_type, "description": self.description}
        if self._enum is not None:
            prop["enum"] = self._enum
        if self._items is not None:
            prop["items"] = self._items
        return prop


class ToolSchema:
    """
    Typed descriptor for an entire tool schema.

    Produces the ``{"type": "function", "function": {...}}`` dict expected by
    the OpenAI function-calling API, delegating each argument's fragment to the
    corresponding :class:`ToolArgument` instance.

    Parameters
    ----------
    name:
        Tool name — must match ``Tool.NAME`` and the registry key.
    description:
        One-line summary shown to the LLM.
    arguments:
        Ordered list of :class:`ToolArgument` descriptors.  May be empty for
        tools that take no parameters.
    """

    def __init__(
        self,
        name: str,
        description: str,
        arguments: list[ToolArgument] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.arguments: list[ToolArgument] = arguments or []

    def schema(self) -> dict:
        """Return the full OpenAI function-calling schema dict."""
        properties = {arg.name: arg.schema() for arg in self.arguments}
        required = [arg.name for arg in self.arguments if arg.required]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class Tool(ABC):
    """
    Abstract base class for all ai-cli tools.

    Parameters
    ----------
    workspace:
        The active project workspace.  Tools use this for all file I/O so
        that path-escape and ignore-rule enforcement are applied centrally.
    permission_manager:
        The session-scoped permission manager.  ``request_permission()``
        delegates to it; subclasses should not call it directly.
    permission_required:
        Whether this tool must ask the user before executing.  This is the
        tool's own declared default and may be overridden via project
        configuration.
    name:
        Canonical tool name used in schemas, config, and slash commands.
    description:
        One-line human-readable description shown by ``/tools list`` and
        the ``tool_manager`` list action.
    """

    def __init__(
        self,
        workspace: Workspace,
        permission_manager: PermissionManager,
        permission_required: bool,
        name: str,
        description: str,
    ) -> None:
        self._workspace = workspace
        self._permission_manager = permission_manager
        self.permission_required = permission_required
        self.name = name
        self.description = description

    # ------------------------------------------------------------------
    # Methods subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def definition(self) -> dict:
        """
        Return the OpenAI function-calling schema for this tool.

        The returned dict must follow the ``{"type": "function", "function":
        {...}}`` wrapper format documented in docs/technical_requirements.md.
        """

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict:
        """
        Run the tool and return a canonical result dict.

        Subclasses should use ``_ok()`` and ``_err()`` to build the return
        value rather than constructing the dict manually.
        """

    # ------------------------------------------------------------------
    # Methods subclasses MAY override
    # ------------------------------------------------------------------

    def extra_permission_options(self, **kwargs: Any) -> list[str]:
        """
        Return tool-specific permission options to present alongside the
        four universal ones (yes / no / always / custom).

        Example: ``["always_in_this_folder"]``

        The default implementation returns an empty list.
        """
        return []

    def reset_session_state(self) -> None:  # noqa: B027
        """
        Clear all session-scoped state held by this tool.

        Called by the registry on session resume so that any in-memory grants
        or caches accumulated during the previous session do not carry over.
        Tools with session-scoped state (e.g. permission allow-lists) must
        override this method and clear that state.  The default is a no-op.
        """
        ...

    def on_permission_granted(self, choice: str, **kwargs: Any) -> None:  # noqa: B027
        """
        Called by the registry after the user grants permission via a named
        extra option from ``extra_permission_options()``.

        *choice* is the non-empty string the user selected.  Universal
        choices such as a plain "yes" or "always" do not produce a named
        choice string; in those cases ``PermissionManager.request()`` returns
        an empty string and this hook is **not called at all**.

        Tools can override this to react to the chosen extra option, for
        example to remember a folder-scoped "always allow" decision.  The
        default implementation is a no-op.
        """
        ...

    # ------------------------------------------------------------------
    # Permission helper (not normally overridden)
    # ------------------------------------------------------------------

    def request_permission(self, action: str, **kwargs: Any) -> tuple[bool, str]:
        """
        Ask the user for permission to perform *action*.

        If ``permission_required`` is False the call is a no-op and
        ``(True, "")`` is returned immediately.

        Parameters
        ----------
        action:
            Human-readable description of the specific action being
            requested (e.g. ``"Read /etc/hosts"``).
        **kwargs:
            Forwarded to ``extra_permission_options()`` so tools can
            tailor their extra choices based on the current arguments.

        Returns
        -------
        tuple[bool, str]
            ``(allowed, reason_or_choice)`` — see ``PermissionManager.request()``
            for the full description of the second element.
        """
        if not self.permission_required:
            return True, ""

        extra = self.extra_permission_options(**kwargs)
        return self._permission_manager.request(
            tool_name=self.name,
            question=f"[{self.name}] {action}",
            extra_options=extra or None,
        )

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ok(data: dict | None = None) -> dict:
        """Return a canonical success result."""
        return {"status": "success", "data": data or {}}

    @staticmethod
    def _err(
        error: str, message: str, code: int = 400, details: dict | None = None
    ) -> dict:
        """Return a canonical error result."""
        result: dict = {
            "status": "error",
            "error": error,
            "message": message,
            "code": code,
        }
        if details is not None:
            result["details"] = details
        return result
