"""
ToolRegistry — three-tier tool discovery, loading, and dispatch.

Tools are loaded in order: bundled → global (~/.ai-cli/tools/) → project
(.ai-cli/tools/).  Later tiers can override earlier ones by name; the user
is warned at startup when this happens.

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
use intra-package relative imports freely.  Global (``~/.ai-cli/tools/``)
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

Three enable modes (highest precedence first):

  transient  — schema returned for one API call, no state change
  session    — in-memory override for this session, reset on exit/resume
  persistent — stored in project .ai-cli/config.yaml

Session overrides always win over the persistent enabled state.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ai_cli.core.workspace import _DOT_AI_CLI, _GLOBAL_DIR
from ai_cli.tools.base import Tool

if TYPE_CHECKING:
    from ai_cli.core.config_manager import ConfigManager
    from ai_cli.core.permission_manager import PermissionManager
    from ai_cli.core.workspace import Workspace

logger = logging.getLogger(__name__)

_REQUIRED_ATTRS = ("NAME", "DESCRIPTION", "PERMISSION_REQUIRED")


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

        bundled_dir = Path(__file__).parent.parent / "tools"
        self._load_bundled(bundled_dir)

        global_tools = _GLOBAL_DIR / "tools"
        if global_tools.is_dir():
            self._load_from_directory(global_tools, tier="global")

        project_tools = self._workspace.root / _DOT_AI_CLI / "tools"
        if project_tools.is_dir():
            self._load_from_directory(project_tools, tier="project")

        self._apply_config()

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
        self._tools[name] = tool
        self._enabled[name] = not getattr(tool_cls, "DISABLED_BY_DEFAULT", False)

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

    def _apply_config(self) -> None:
        """Apply per-tool settings from config.yaml over the loaded defaults.

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
        # Project-layer tools dict (without global/CLI merge) — used to
        # determine whether a security-lowering setting came from the
        # untrusted project config or from the trusted global config.
        project_tools_cfg = self._config.get_project("tools") or {}
        if not isinstance(project_tools_cfg, dict):
            project_tools_cfg = {}
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
                elif not val and tool.permission_required:
                    # Lowering permission_required from the project config layer
                    # is untrusted (a cloned repo could silently disable prompts).
                    # It is only honoured when the entry carries a
                    # 'user_confirmed: true' marker written by
                    # set_permission_required().  Global config is trusted
                    # (it is the user's own ~/.ai-cli/config.yaml), so entries
                    # that don't appear in the project layer bypass this check.
                    project_entry = project_tools_cfg.get(name, {})
                    from_project = isinstance(project_entry, dict) and (
                        "permission_required" in project_entry
                    )
                    if not from_project or project_entry.get("user_confirmed") is True:
                        tool.permission_required = val
                    else:
                        logger.warning(
                            "Tool '%s': project config attempts to disable "
                            "'permission_required' without explicit user "
                            "confirmation — ignored.",
                            name,
                        )
                else:
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

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        """Return the named tool, or ``None`` if not registered."""
        return self._tools.get(name)

    def all_enabled(self) -> list[Tool]:
        """Return all tools that are currently enabled (session + persistent)."""
        return [tool for name, tool in self._tools.items() if self._is_enabled(name)]

    def definitions(self) -> list[dict]:
        """Return OpenAI-format schemas for all enabled tools."""
        return [tool.definition() for tool in self.all_enabled()]

    def _is_enabled(self, name: str) -> bool:
        if name in self._session_overrides:
            return self._session_overrides[name]
        return self._enabled.get(name, False)

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

        if not allow_transient and not self._is_enabled(name):
            return Tool._err("tool_disabled", f"Tool '{name}' is disabled.", 403)

        choice = ""
        if tool.permission_required:

            def _fmt(v: object, limit: int = 60) -> str:
                s = repr(v)
                return s if len(s) <= limit else s[:limit] + "…"

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
            return tool.execute(**kwargs)
        except Exception:
            logger.exception("Error executing tool '%s' with kwargs=%r", name, kwargs)
            return Tool._err(
                "tool_execution_error",
                f"Tool '{name}' failed during execution. See logs for details.",
                500,
            )

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

    def disable_session(self, name: str) -> None:
        """Disable *name* for this session only — no config write."""
        if name in self._tools:
            self._session_overrides[name] = False

    def reset_session_overrides(self) -> None:
        """Clear all session-level overrides and tool session state. Called on session resume."""
        self._session_overrides.clear()
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
        changing its enabled state.  Returns ``None`` if unknown.
        """
        tool = self._tools.get(name)
        return tool.definition() if tool is not None else None

    # ------------------------------------------------------------------
    # Permission toggle (persistent)
    # ------------------------------------------------------------------

    def set_permission_required(self, name: str, value: bool) -> None:
        """
        Toggle *permission_required* for *name* and persist to config.

        When *value* is ``False`` (lowering security), a ``user_confirmed: true``
        marker is also written so that ``_apply_config()`` can distinguish an
        explicit user action from an untrusted project config entry.
        """
        tool = self._tools.get(name)
        if tool is not None:
            tool.permission_required = value
            self._persist_tool_setting(name, "permission_required", value)
            if not value:
                self._persist_tool_setting(name, "user_confirmed", True)

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
