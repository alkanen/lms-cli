"""
ToolRegistry — three-tier tool discovery, loading, and dispatch.

Tools are loaded in order: bundled → global (get_global_dir()/tools/,
``~/.ai-cli/tools/`` by default) → project (.ai-cli/tools/).  Later tiers
can override earlier ones by name; the user is warned at startup when this
happens.

Tool subclasses must define three class-level attributes so the registry
can instantiate them without knowing their internals:

    class MyTool(Tool):
        NAME = "my_tool"
        DESCRIPTION = "One-line description."
        PERMISSION_REQUIRED = True   # default; overridable via config

When loading from a file, *all* Tool subclasses accessible in that module
are registered — including ones imported from elsewhere.  This is
intentional: it allows wrapper modules to re-export tools defined in other
packages.  Use ``ToolRegistry.register()`` for programmatic registration
without file discovery.

Note: bundled tools (``ai_cli/tools/``) are loaded via
``importlib.import_module`` as part of the ``ai_cli`` package, so they can
use intra-package relative imports freely.  Global (``get_global_dir()/tools/``)
and project (``.ai-cli/tools/``) tool files are loaded via
``importlib.util.spec_from_file_location`` with a synthetic module name and
are therefore *not* part of any package.  Relative imports and sibling-module
imports will fail in those tiers unless the tools directory is installed as a
proper Python package or the relevant modules are importable by absolute name.
Re-exporting tools from already-installed packages works without restriction.

Per-tool config is read from the ``tools`` mapping in config.yaml, keyed by
tool name::

    tools:
      my_tool:
        permission_required: false
        disabled: true
        allowed: false

Configuration hierarchy (lowest to highest precedence):

  1. Hardcoded tool defaults (``PERMISSION_REQUIRED``, ``DISABLED_BY_DEFAULT``)
  2. Global config (``AI_CLI_GLOBAL_DIR/config.yaml``, default ``~/.ai-cli/``)
  3. Project config (``.ai-cli/config.yaml``)
  4. Session-level overrides (in-memory, reset on session resume)
  5. Transient enables (single API call, no persistent state change)

Later entries override earlier ones.  A project-level setting always wins
over the same setting in global config.  Runtime mutations (``enable``,
``disable``, ``allow``, ``disallow``, ``set_permission_required``) are
persisted to the *project* config so that changes are scoped to the current
project and do not affect other projects.

Three enable modes (highest precedence first):

  transient  — schema returned for one API call, no state change
  session    — in-memory override for this session, reset on exit/resume
  persistent — stored in project .ai-cli/config.yaml

Session overrides always win over the persistent enabled state.

Allowed/disallowed is a hard gate that takes precedence over the
enabled/disabled state.  A disallowed tool cannot be executed or
transiently enabled, regardless of its enabled state.  Use ``allowed: false``
in config to permanently block a tool; ``disallow()`` or ``disallow_session()``
to do so at runtime.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ai_cli.core.workspace import _DOT_AI_CLI, get_global_dir
from ai_cli.tools.base import Tool, ToolArgument, ToolSchema

if TYPE_CHECKING:
    from ai_cli.core.config_manager import ConfigManager
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.workspace import Workspace

logger = logging.getLogger(__name__)

_REQUIRED_ATTRS = ("NAME", "DESCRIPTION", "PERMISSION_REQUIRED")

# Valid tool-name pattern: must start with an alphanumeric or underscore,
# followed by up to 63 alphanumeric, underscore, or hyphen characters.
# This is slightly stricter than the OpenAI spec (which allows a leading
# hyphen) but avoids ambiguity with CLI flags like --persist.
TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_-]{0,63}$")

# Maximum length of a string value shown verbatim in debug logs.
# Longer strings are replaced with a ``<str:Nch>`` placeholder to avoid
# leaking file contents or other secrets into session.log.
_LOG_STR_LIMIT = 80


def _args_summary(kwargs: dict) -> str:
    """Return a compact, secret-safe summary of tool call arguments.

    Short scalar values (numbers, booleans, short strings) are shown
    verbatim.  Long strings are replaced with ``<str:Nch>`` and
    collections with ``<list:N>`` / ``<dict:N>`` so that large file
    contents and prompt text are never written to the log.
    """
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, str) and len(v) > _LOG_STR_LIMIT:
            parts.append(f"{k}=<str:{len(v)}ch>")
        elif isinstance(v, list):
            parts.append(f"{k}=<list:{len(v)}>")
        elif isinstance(v, dict):
            parts.append(f"{k}=<dict:{len(v)}>")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts) if parts else "(no args)"


def _validate_tool_definition(defn: object, expected_name: str | None = None) -> str:
    """
    Validate the OpenAI-format schema returned by a tool's ``definition()`` method.

    Returns an empty string on success, or a human-readable error message that
    describes the first structural problem found.

    If *expected_name* is provided, the schema's function name must match it
    exactly.  A mismatch means the LLM would call a name that the registry
    doesn't recognise, making the tool silently unusable.

    Expected shape::

        {
            "type": "function",
            "function": {
                "name": "<non-empty str>",
                "description": "<str>",
                "parameters": {
                    "type": "object",
                    "properties": { ... },
                    "required": ["<param names present in properties>"],
                },
            },
        }
    """
    if not isinstance(defn, dict):
        return "definition() must return a dict"
    if defn.get("type") != "function":
        return "definition()['type'] must be 'function'"
    fn = defn.get("function")
    if not isinstance(fn, dict):
        return "definition()['function'] must be a dict"
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return "definition()['function']['name'] must be a non-empty string"
    if not TOOL_NAME_RE.match(name):
        return (
            f"definition()['function']['name'] {name!r} does not match the "
            f"required pattern {TOOL_NAME_RE.pattern}"
        )
    if expected_name is not None and name != expected_name:
        return (
            f"definition()['function']['name'] is {name!r} but must match "
            f"the registered tool name {expected_name!r}"
        )
    if not isinstance(fn.get("description"), str):
        return "definition()['function']['description'] must be a string"
    params = fn.get("parameters")
    if not isinstance(params, dict):
        return "definition()['function']['parameters'] must be a dict"
    if params.get("type") != "object":
        return "definition()['function']['parameters']['type'] must be 'object'"
    props = params.get("properties")
    if not isinstance(props, dict):
        return "definition()['function']['parameters']['properties'] must be a dict"
    required = params.get("required", [])
    if not isinstance(required, list):
        return "definition()['function']['parameters']['required'] must be a list"
    for r in required:
        if not isinstance(r, str):
            return f"definition()['function']['parameters']['required'] entry {r!r} must be a string"
        if r not in props:
            return f"required parameter {r!r} is not declared in 'properties'"
    return ""


def _validate_tool_class(cls: type) -> str:
    """
    Validate that *cls* declares the required class attributes with correct types.

    Returns an empty string on success, or a human-readable error message.
    """
    for attr in _REQUIRED_ATTRS:
        if not hasattr(cls, attr):
            return f"missing class attribute '{attr}'"
    if not isinstance(cls.NAME, str):  # type: ignore[attr-defined]
        return f"NAME must be str, got {type(cls.NAME).__name__!r}"  # type: ignore[attr-defined]
    if not isinstance(cls.DESCRIPTION, str):  # type: ignore[attr-defined]
        return f"DESCRIPTION must be str, got {type(cls.DESCRIPTION).__name__!r}"  # type: ignore[attr-defined]
    if not isinstance(cls.PERMISSION_REQUIRED, bool):  # type: ignore[attr-defined]
        return f"PERMISSION_REQUIRED must be bool, got {type(cls.PERMISSION_REQUIRED).__name__!r}"  # type: ignore[attr-defined]
    return ""


def _check_type(val: object, arg_type: str) -> str | None:
    """Return ``None`` if *val* matches *arg_type*, or a human-readable reason
    string if it does not.

    Checks JSON Schema primitive types only; ``"array"`` and ``"object"`` are
    accepted without inspecting their contents.
    """
    if arg_type == "integer":
        # bool is a subclass of int in Python — reject it here.
        if isinstance(val, int) and not isinstance(val, bool):
            return None
        return f"expected integer, got {type(val).__name__} ({val!r})"
    if arg_type == "number":
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return None
        return f"expected number, got {type(val).__name__} ({val!r})"
    if arg_type == "boolean":
        if isinstance(val, bool):
            return None
        return f"expected boolean, got {type(val).__name__} ({val!r})"
    if arg_type == "string":
        if isinstance(val, str):
            return None
        return f"expected string, got {type(val).__name__} ({val!r})"
    if arg_type == "array":
        if isinstance(val, list):
            return None
        return f"expected array, got {type(val).__name__}"
    if arg_type == "object":
        if isinstance(val, dict):
            return None
        return f"expected object, got {type(val).__name__}"
    # Unknown type — accept without checking.
    return None


def _check_bounds(val: object, arg: ToolArgument) -> str | None:
    """Return ``None`` if *val* is within *arg*'s declared bounds, or a
    human-readable reason string if it is not.

    Only applies to ``"integer"`` and ``"number"`` arguments that declare
    a ``minimum`` or ``maximum``.  Both bounds are inclusive.

    Applies defensive validation to the bounds themselves — non-numeric or
    inverted bounds are skipped with a warning rather than crashing dispatch.
    """
    if arg.minimum is None and arg.maximum is None:
        return None

    min_bound = arg.minimum
    max_bound = arg.maximum

    for bound_name, bound_val in (("minimum", min_bound), ("maximum", max_bound)):
        if bound_val is not None and (
            isinstance(bound_val, bool) or not isinstance(bound_val, (int, float))
        ):
            logger.warning(
                "Tool argument '%s': ignoring non-numeric %s bound %r",
                arg.name,
                bound_name,
                bound_val,
            )
            if bound_name == "minimum":
                min_bound = None
            else:
                max_bound = None

    if min_bound is not None and max_bound is not None and min_bound > max_bound:
        logger.warning(
            "Tool argument '%s': ignoring inverted bounds (minimum %r > maximum %r)",
            arg.name,
            min_bound,
            max_bound,
        )
        min_bound = None
        max_bound = None

    if min_bound is None and max_bound is None:
        return None

    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return None  # type mismatch already caught by _check_type

    if min_bound is not None and val < min_bound:
        return f"value {val} is below minimum {min_bound}"
    if max_bound is not None and val > max_bound:
        return f"value {val} is above maximum {max_bound}"
    return None


class ToolRegistry:
    """
    Discovers, loads, and dispatches tool calls.

    Parameters
    ----------
    workspace:
        Active project workspace — passed to every tool on instantiation.
    config_manager:
        Used to read per-tool settings from global + project config.
    permission_manager:
        Passed to every tool on instantiation.
    """

    def __init__(
        self,
        workspace: Workspace,
        config_manager: ConfigManager,
        permission_manager: PermissionManager,
    ) -> None:
        self._workspace = workspace
        self._config = config_manager
        self._permission_manager = permission_manager

        # name → Tool instance
        self._tools: dict[str, Tool] = {}
        # name → persistent enabled state (from config + tool default)
        self._enabled: dict[str, bool] = {}
        # name → session-level override (reset on session resume)
        self._session_overrides: dict[str, bool] = {}
        # name → persistent allowed state (default True)
        self._allowed: dict[str, bool] = {}
        # name → session-level allowed overrides (reset on session resume)
        self._session_allowed_overrides: dict[str, bool] = {}
        # name → which tier each tool came from
        self._tiers: dict[str, str] = {}
        # name → ToolSchema cached at registration time (for arg validation)
        self._schemas: dict[str, ToolSchema] = {}

    @property
    def permission_manager(self) -> PermissionManager:
        """The :class:`PermissionManager` that gates tool execution."""
        return self._permission_manager

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Discover and instantiate tools from all three tiers.

        Load order: bundled → global → project.  Name collisions are
        allowed but logged as warnings.  Per-tool config settings are
        applied after all tools are loaded.
        """
        self._tools.clear()
        self._enabled.clear()
        self._session_overrides.clear()
        self._allowed.clear()
        self._session_allowed_overrides.clear()
        self._tiers.clear()
        self._schemas.clear()

        bundled_dir = Path(__file__).parent.parent / "tools"
        self._load_bundled(bundled_dir)

        global_tools = get_global_dir() / "tools"
        if global_tools.is_dir():
            self._load_from_directory(global_tools, tier="global")

        project_tools = self._workspace.root / _DOT_AI_CLI / "tools"
        if project_tools.is_dir():
            self._load_from_directory(project_tools, tier="project")

        self._apply_config()
        logger.info(
            "Tool registry loaded: %d tool(s) registered (%s)",
            len(self._tools),
            ", ".join(sorted(self._tools)),
        )

    def _load_bundled(self, directory: Path) -> None:
        """Load bundled tools via normal package imports (``ai_cli.tools.<name>``).

        Bundled tools are part of the installed ``ai_cli`` package, so they
        must be loaded with ``importlib.import_module`` rather than
        ``spec_from_file_location``.  This preserves the package context and
        allows bundled tool modules to use intra-package imports safely.
        """
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            module_name = f"ai_cli.tools.{path.stem}"
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                logger.warning(
                    "Error loading bundled tool module %s: %s", module_name, exc
                )
                continue
            for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
                if obj is Tool or not issubclass(obj, Tool):
                    continue
                if getattr(obj, "REGISTER_VIA_INSTANCE", False):
                    continue
                error = _validate_tool_class(obj)
                if error:
                    logger.warning(
                        "Skipping tool class %s in %s: %s",
                        obj.__name__,
                        module_name,
                        error,
                    )
                    continue
                self._register(obj, tier="bundled")

    def _load_from_directory(self, directory: Path, tier: str) -> None:
        """Load all Tool subclasses found in *.py files under *directory*."""
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            self._load_from_file(path, tier)

    def _load_from_file(self, path: Path, tier: str) -> None:
        """Import *path* and register any Tool subclasses found in it."""
        module_name = f"_ai_cli_tool_{tier}_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            logger.warning("Could not load tool file: %s", path)
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            logger.warning("Error loading tool file %s: %s", path, exc)
            sys.modules.pop(module_name, None)
            return

        for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is Tool or not issubclass(obj, Tool):
                continue
            if getattr(obj, "REGISTER_VIA_INSTANCE", False):
                continue
            error = _validate_tool_class(obj)
            if error:
                logger.warning(
                    "Skipping tool class %s in %s: %s", obj.__name__, path, error
                )
                continue
            self._register(obj, tier)

    def _register(self, tool_cls: type[Tool], tier: str) -> None:
        """Instantiate *tool_cls* and add it to the registry."""
        name: str = tool_cls.NAME  # type: ignore[attr-defined]
        if name in self._tools:
            logger.warning(
                "Tool '%s' from %s tier overrides an earlier definition.", name, tier
            )
        tool = tool_cls(
            workspace=self._workspace,
            permission_manager=self._permission_manager,
            permission_required=tool_cls.PERMISSION_REQUIRED,  # type: ignore[attr-defined]
            name=name,
            description=tool_cls.DESCRIPTION,  # type: ignore[attr-defined]
        )
        try:
            tool_schema = tool.definition()
        except Exception as exc:
            logger.warning(
                "Skipping tool '%s' from %s tier: definition() raised — %s",
                name,
                tier,
                exc,
            )
            return
        try:
            defn = tool_schema.schema()
        except Exception as exc:
            logger.warning(
                "Skipping tool '%s' from %s tier: ToolSchema.schema() raised — %s",
                name,
                tier,
                exc,
            )
            return
        defn_error = _validate_tool_definition(defn, expected_name=name)
        if defn_error:
            logger.warning(
                "Skipping tool '%s' from %s tier: invalid definition() — %s",
                name,
                tier,
                defn_error,
            )
            return

        self._tools[name] = tool
        self._schemas[name] = tool_schema
        enabled = not getattr(tool_cls, "DISABLED_BY_DEFAULT", False)
        self._enabled[name] = enabled
        self._allowed[name] = not getattr(tool_cls, "DISALLOWED_BY_DEFAULT", False)
        self._tiers[name] = tier
        logger.debug(
            "Registered tool '%s' from %s tier (enabled=%s, permission_required=%s)",
            name,
            tier,
            enabled,
            tool.permission_required,
        )
        # Allow tools that need registry access (e.g. tool_manager) to receive
        # a back-reference after construction.  Guard carefully: the attribute
        # may exist on user-supplied tools with a different shape, and any
        # exception here would abort loading for all subsequent tools.
        set_reg = getattr(tool, "set_registry", None)
        if set_reg is not None:
            if not callable(set_reg):
                logger.warning(
                    "Tool '%s' has a non-callable 'set_registry' attribute — "
                    "registry back-reference skipped.  If this is intentional, "
                    "rename the attribute to avoid the conflict.",
                    name,
                )
            else:
                try:
                    set_reg(self)
                except Exception as exc:
                    logger.warning(
                        "Tool '%s': set_registry() raised an exception — "
                        "registry access may not be available: %s",
                        name,
                        exc,
                    )

    def register(self, tool_cls: type[Tool], tier: str = "programmatic") -> None:
        """
        Programmatically register a Tool subclass without file discovery.

        This is intended for use by wrapper applications or tests that want
        to add tools directly without going through the three-tier loader.
        The tool must declare ``NAME`` (str), ``DESCRIPTION`` (str), and
        ``PERMISSION_REQUIRED`` (bool) as class attributes.

        Raises ``ValueError`` if the class attributes are missing or have
        wrong types.
        """
        error = _validate_tool_class(tool_cls)
        if error:
            raise ValueError(f"Cannot register {tool_cls.__name__}: {error}")
        self._register(tool_cls, tier)

    def apply_config(self) -> None:
        """
        Re-apply per-tool settings from the merged config.

        Useful after programmatic registration (``register`` / ``register_instance``)
        that happens *after* :meth:`load` has already run.  Safe to call multiple
        times — already-correct values are simply overwritten with the same value.
        """
        self._apply_config()

    def register_instance(self, tool: Tool, tier: str = "programmatic") -> None:
        """
        Register a pre-built tool instance directly, bypassing class discovery.

        Use this for tools with non-standard constructor signatures (e.g.
        :class:`~ai_cli.tools.call_agent.CallAgentTool`) that cannot be
        instantiated with the standard ``(workspace, permission_manager, …)``
        signature used by the three-tier loader.

        The instance is validated (``definition()`` is called and the returned
        schema is checked) before registration.  Invalid instances are logged
        and skipped.

        The initial enabled/allowed state is derived from the tool's
        ``DISABLED_BY_DEFAULT`` and ``DISALLOWED_BY_DEFAULT`` class attributes
        (both default to ``False`` when absent).  The merged project config is
        then re-applied, which may further enable, disable, allow, or disallow
        the tool.
        """
        name = tool.name
        try:
            tool_schema = tool.definition()
        except Exception as exc:
            logger.warning(
                "Skipping tool instance '%s' from %s tier: definition() raised — %s",
                name,
                tier,
                exc,
            )
            return
        try:
            defn = tool_schema.schema()
        except Exception as exc:
            logger.warning(
                "Skipping tool instance '%s' from %s tier: ToolSchema.schema() raised — %s",
                name,
                tier,
                exc,
            )
            return
        defn_error = _validate_tool_definition(defn, expected_name=name)
        if defn_error:
            logger.warning(
                "Skipping tool instance '%s' from %s tier: invalid definition() — %s",
                name,
                tier,
                defn_error,
            )
            return
        if name in self._tools:
            logger.warning(
                "Tool '%s' from %s tier overrides an earlier definition.", name, tier
            )
        self._tools[name] = tool
        self._schemas[name] = tool_schema
        tool_cls = type(tool)
        self._enabled[name] = not getattr(tool_cls, "DISABLED_BY_DEFAULT", False)
        self._allowed[name] = not getattr(tool_cls, "DISALLOWED_BY_DEFAULT", False)
        self._tiers[name] = tier
        logger.debug(
            "Registered tool instance '%s' from %s tier (enabled=%s, permission_required=%s)",
            name,
            tier,
            self._enabled[name],
            tool.permission_required,
        )
        # Apply project config so user settings (disabled/allowed/permission_required)
        # take effect even for tools registered after load() has run.
        self._apply_config()
        set_reg = getattr(tool, "set_registry", None)
        if set_reg is not None:
            if not callable(set_reg):
                logger.warning(
                    "Tool '%s' has a non-callable 'set_registry' attribute — "
                    "registry back-reference skipped.",
                    name,
                )
            else:
                try:
                    set_reg(self)
                except Exception as exc:
                    logger.warning(
                        "Tool '%s': set_registry() raised an exception — "
                        "registry access may not be available: %s",
                        name,
                        exc,
                    )

    def unregister(self, name: str) -> None:
        """Remove a registered tool by *name* if present."""
        removed = self._tools.pop(name, None)
        self._schemas.pop(name, None)
        self._enabled.pop(name, None)
        self._allowed.pop(name, None)
        self._session_overrides.pop(name, None)
        self._session_allowed_overrides.pop(name, None)
        self._tiers.pop(name, None)
        if removed is not None:
            logger.debug("Unregistered tool '%s'.", name)

    def _apply_config(self) -> None:
        """Apply per-tool settings from the merged config over the loaded defaults.

        Settings are read from the merged ``tools`` mapping (global config
        overridden by project config — see module docstring for the full
        hierarchy).  Project-level values therefore take precedence over
        global-level values for every key.

        The ``tools`` key in config is a dict keyed by tool name::

            tools:
              read_file:
                permission_required: false
              bash:
                permission_required: true
                disabled: false
        """
        tools_cfg = self._config.get("tools", {})
        if not isinstance(tools_cfg, dict):
            return
        for name, entry in tools_cfg.items():
            if not isinstance(name, str) or not isinstance(entry, dict):
                continue
            if name not in self._tools:
                continue
            tool = self._tools[name]
            if "permission_required" in entry:
                val = entry["permission_required"]
                if not isinstance(val, bool):
                    logger.warning(
                        "Tool '%s': 'permission_required' must be a boolean, got %r — ignored.",
                        name,
                        val,
                    )
                else:
                    if tool.permission_required and not val:
                        logger.warning(
                            "Tool '%s': config lowers 'permission_required' from True "
                            "to False — user prompts will be skipped for this tool.",
                            name,
                        )
                    tool.permission_required = val
            if "disabled" in entry:
                val = entry["disabled"]
                if isinstance(val, bool):
                    self._enabled[name] = not val
                else:
                    logger.warning(
                        "Tool '%s': 'disabled' must be a boolean, got %r — ignored.",
                        name,
                        val,
                    )
            if "allowed" in entry:
                val = entry["allowed"]
                if not isinstance(val, bool):
                    logger.warning(
                        "Tool '%s': 'allowed' must be a boolean, got %r — ignored.",
                        name,
                        val,
                    )
                else:
                    self._allowed[name] = val

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        """Return the named tool, or ``None`` if not registered."""
        return self._tools.get(name)

    def is_allowed(self, name: str) -> bool:
        """Return whether *name* is registered and currently allowed.

        Cheaper than :meth:`tool_info` for callers that only need the
        allowed/unknown state — does not compute the tool's schema.
        Returns ``False`` for unknown tool names.
        """
        if name not in self._tools:
            return False
        return self._is_allowed(name)

    def all_enabled(self) -> list[Tool]:
        """Return all tools that are currently enabled and allowed."""
        return [
            tool
            for name, tool in self._tools.items()
            if self._is_allowed(name) and self._is_enabled(name)
        ]

    def list_all(self) -> list[dict]:
        """Return name, description, and enabled status for every allowed tool."""
        return [
            {
                "name": name,
                "description": tool.description,
                "enabled": self._is_enabled(name),
            }
            for name, tool in self._tools.items()
            if self._is_allowed(name)
        ]

    def all_tools_info(self) -> list[dict]:
        """Return full info for ALL tools including disallowed (for /tools list)."""
        return [
            {
                "name": name,
                "description": tool.description,
                "enabled": self._is_enabled(name),
                "allowed": self._is_allowed(name),
                "permission_required": tool.permission_required,
                "tier": self._tiers.get(name, "unknown"),
            }
            for name, tool in self._tools.items()
        ]

    def tool_info(self, name: str) -> dict | None:
        """Return detailed info for a single tool, or ``None`` if unknown."""
        tool = self._tools.get(name)
        if tool is None:
            return None
        try:
            defn = tool.definition().schema()
            params = defn.get("function", {}).get("parameters", {})
        except Exception as exc:
            logger.warning("tool_info: definition() raised for '%s': %s", name, exc)
            params = {}
        return {
            "name": name,
            "description": tool.description,
            "enabled": self._is_enabled(name),
            "allowed": self._is_allowed(name),
            "permission_required": tool.permission_required,
            "tier": self._tiers.get(name, "unknown"),
            "parameters": params,
        }

    def definitions(self) -> list[dict]:
        """Return OpenAI-format schemas for all enabled tools."""
        return [tool.definition().schema() for tool in self.all_enabled()]

    def _is_enabled(self, name: str) -> bool:
        if name in self._session_overrides:
            return self._session_overrides[name]
        return self._enabled.get(name, False)

    def _is_allowed(self, name: str) -> bool:
        if name in self._session_allowed_overrides:
            return self._session_allowed_overrides[name]
        return self._allowed.get(name, True)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self, name: str, kwargs: dict, *, allow_transient: bool = False
    ) -> dict:
        """
        Look up *name*, request permission, then execute.

        Returns the tool's canonical result dict, or an error dict if the
        tool is unknown, disabled, or permission is denied.

        Parameters
        ----------
        allow_transient:
            When ``True``, the enabled/disabled check is skipped.  Use this
            for tools whose schema was injected for a single API call via
            ``enable_transient()`` without changing the tool's persistent or
            session-level enabled state.
        """
        tool = self._tools.get(name)
        if tool is None:
            return Tool._err("unknown_tool", f"No tool named '{name}'.", 404)

        if not self._is_allowed(name):
            return Tool._err("tool_disallowed", f"Tool '{name}' is not available.", 403)

        if not allow_transient and not self._is_enabled(name):
            return Tool._err("tool_disabled", f"Tool '{name}' is disabled.", 403)

        if not isinstance(kwargs, dict):
            return Tool._err(
                "invalid_arguments",
                f"Tool '{name}' arguments must be a JSON object, got "
                f"{type(kwargs).__name__}.",
                400,
            )

        kwargs, arg_error = self._validate_args(name, kwargs)
        if arg_error is not None:
            return arg_error

        def _fmt(v: object, limit: int = 60) -> str:
            s = repr(v)
            return s if len(s) <= limit else s[:limit] + "…"

        choice = ""
        if tool.permission_required:
            allowed, choice = tool.request_permission(
                f"Execute {name}({', '.join(f'{k}={_fmt(v)}' for k, v in kwargs.items())})",
                **kwargs,
            )
            if not allowed:
                return Tool._err(
                    "permission_denied", choice or "Permission denied.", 403
                )

        try:
            if choice:
                tool.on_permission_granted(choice, **kwargs)
            try:
                log_summary = tool.execute_log(**kwargs)
            except Exception:
                logger.debug(
                    "execute_log raised for '%s' — falling back to summary",
                    name,
                    exc_info=True,
                )
                log_summary = None
            if log_summary is None:
                log_summary = _args_summary(kwargs)
            logger.debug("Executing tool '%s' — %s", name, log_summary)
            result = tool.execute(**kwargs)
            logger.debug(
                "Tool '%s' returned status=%r",
                name,
                result.get("status") if isinstance(result, dict) else "?",
            )
            return result
        except Exception:
            logger.exception(
                "Error executing tool '%s' — %s", name, _args_summary(kwargs)
            )
            return Tool._err(
                "tool_execution_error",
                f"Tool '{name}' failed during execution. See logs for details.",
                500,
            )

    def _validate_args(self, name: str, kwargs: dict) -> tuple[dict, dict | None]:
        """Validate *kwargs* against the cached ToolSchema for *name*.

        Checks that all required arguments are present, that present arguments
        have the correct JSON Schema types, and strips unknown keys (with a
        warning log).  Returns an ``invalid_arguments`` error so the model can
        self-correct rather than silently coercing bad values.

        Returns ``(validated_kwargs, None)`` on success, or ``({}, error_dict)``
        on the first validation failure.
        """
        schema = self._schemas.get(name)
        if schema is None:
            return kwargs, None

        declared = {arg.name: arg for arg in schema.arguments}

        missing = [
            arg.name
            for arg in schema.arguments
            if arg.required and arg.name not in kwargs
        ]
        if missing:
            return {}, Tool._err(
                "invalid_arguments",
                f"Tool '{name}' missing required argument(s): {', '.join(missing)}.",
                400,
            )

        unknown = [k for k in kwargs if k not in declared]
        if unknown:
            logger.warning(
                "Tool '%s': ignoring unknown argument(s) sent by model: %s",
                name,
                ", ".join(unknown),
            )

        result: dict = {}
        for arg in schema.arguments:
            if arg.name not in kwargs:
                continue
            val = kwargs[arg.name]
            type_error = _check_type(val, arg.argument_type)
            if type_error:
                return {}, Tool._err(
                    "invalid_arguments",
                    f"Tool '{name}' argument '{arg.name}': {type_error}.",
                    400,
                )
            bounds_error = _check_bounds(val, arg)
            if bounds_error:
                return {}, Tool._err(
                    "invalid_arguments",
                    f"Tool '{name}' argument '{arg.name}': {bounds_error}.",
                    400,
                )
            result[arg.name] = val

        return result, None

    # ------------------------------------------------------------------
    # Enable / disable (persistent — writes to config)
    # ------------------------------------------------------------------

    def enable(self, name: str) -> None:
        """Enable *name* and persist the change to project .ai-cli/config.yaml.

        Any session-level override for *name* is cleared so the persistent
        state takes effect immediately in the current session.
        """
        if name not in self._tools:
            return
        self._session_overrides.pop(name, None)
        self._set_enabled(name, enabled=True)
        self._persist_tool_setting(name, "disabled", False)
        logger.info("Tool '%s' enabled (persisted to config)", name)

    def disable(self, name: str) -> None:
        """Disable *name* and persist the change to project .ai-cli/config.yaml.

        Any session-level override for *name* is cleared so the persistent
        state takes effect immediately in the current session.
        """
        if name not in self._tools:
            return
        self._session_overrides.pop(name, None)
        self._set_enabled(name, enabled=False)
        self._persist_tool_setting(name, "disabled", True)
        logger.info("Tool '%s' disabled (persisted to config)", name)

    def _set_enabled(self, name: str, enabled: bool) -> None:
        if name in self._tools:
            self._enabled[name] = enabled

    # ------------------------------------------------------------------
    # Enable / disable (session — no config write)
    # ------------------------------------------------------------------

    def enable_session(self, name: str) -> None:
        """Enable *name* for this session only — no config write."""
        if name in self._tools:
            self._session_overrides[name] = True
            logger.info("Tool '%s' enabled for this session only", name)

    def disable_session(self, name: str) -> None:
        """Disable *name* for this session only — no config write."""
        if name in self._tools:
            self._session_overrides[name] = False
            logger.info("Tool '%s' disabled for this session only", name)

    # ------------------------------------------------------------------
    # Allow / disallow (persistent — writes to config)
    # ------------------------------------------------------------------

    def allow(self, name: str) -> None:
        """Re-allow a disallowed tool and persist the change to project config."""
        if name not in self._tools:
            return
        self._session_allowed_overrides.pop(name, None)
        self._allowed[name] = True
        self._persist_tool_setting(name, "allowed", True)

    def disallow(self, name: str) -> None:
        """Disallow a tool and persist the change to project config."""
        if name not in self._tools:
            return
        self._session_allowed_overrides.pop(name, None)
        self._allowed[name] = False
        self._persist_tool_setting(name, "allowed", False)

    # ------------------------------------------------------------------
    # Allow / disallow (session — no config write)
    # ------------------------------------------------------------------

    def allow_session(self, name: str) -> None:
        """Allow for this session only — no config write."""
        if name in self._tools:
            self._session_allowed_overrides[name] = True

    def disallow_session(self, name: str) -> None:
        """Disallow for this session only — no config write."""
        if name in self._tools:
            self._session_allowed_overrides[name] = False

    def reset_session_overrides(self) -> None:
        """Clear all session-level overrides and tool session state. Called on session resume."""
        self._session_overrides.clear()
        self._session_allowed_overrides.clear()
        for tool in self._tools.values():
            try:
                tool.reset_session_state()
            except Exception:
                logger.exception(
                    "Error resetting session state for tool '%s'", tool.name
                )

    # ------------------------------------------------------------------
    # Transient enable (one API call, no state change)
    # ------------------------------------------------------------------

    def enable_transient(self, name: str) -> dict | None:
        """
        Return the named tool's schema for one-call injection without
        changing its enabled state.  Returns ``None`` if unknown or disallowed.
        """
        if not self._is_allowed(name):
            return None
        tool = self._tools.get(name)
        return tool.definition().schema() if tool is not None else None

    # ------------------------------------------------------------------
    # Permission toggle (persistent)
    # ------------------------------------------------------------------

    def set_permission_required(self, name: str, value: bool) -> None:
        """
        Toggle *permission_required* for *name* and persist to project config.
        """
        tool = self._tools.get(name)
        if tool is not None:
            tool.permission_required = value
            self._persist_tool_setting(name, "permission_required", value)

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _persist_tool_setting(self, name: str, key: str, value: object) -> None:
        """Write a single tool setting to project .ai-cli/config.yaml."""
        config_path = self._workspace.root / _DOT_AI_CLI / "config.yaml"
        try:
            if config_path.is_file():
                text = config_path.read_text(encoding="utf-8")
                loaded = yaml.safe_load(text) or {}
                data: dict = loaded if isinstance(loaded, dict) else {}
            else:
                data = {}

            tools_dict: dict = data.get("tools", {})
            if not isinstance(tools_dict, dict):
                tools_dict = {}

            if name not in tools_dict or not isinstance(tools_dict[name], dict):
                tools_dict[name] = {}
            tools_dict[name][key] = value

            data["tools"] = tools_dict
            config_path.write_text(
                yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Could not persist tool setting for '%s': %s", name, exc)
