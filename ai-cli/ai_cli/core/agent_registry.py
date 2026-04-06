"""
AgentRegistry — parse agent specs from config and provide lookup.

Reads the ``agents:`` and ``agent_defaults:`` sections from
:class:`~ai_cli.core.config_manager.ConfigManager`, validates each entry,
and exposes the resulting ``AgentSpec`` objects via a simple registry.

Agent *instantiation* (lazy ``get_or_create``) is added in a later PR
once ``Agent.run()`` exists.
"""

from __future__ import annotations

import logging
from typing import Any

from ai_cli.core.agent import AgentSpec, BackendConfig
from ai_cli.core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Fields on AgentSpec that have defaults and are therefore optional in config.
_OPTIONAL_FIELDS: dict[str, Any] = {
    "max_response_tokens": 4096,
    "persistence": "ephemeral",
    "tool_permission_overrides": {},
    "max_tool_rounds": 10,
    "context_limit_threshold": 0.90,
}

_REQUIRED_FIELDS = ("system_message", "tools", "model")

_KNOWN_KEYS = {
    "system_message",
    "tools",
    "model",
    "max_response_tokens",
    "persistence",
    "backend",
    "tool_permission_overrides",
    "max_tool_rounds",
    "context_limit_threshold",
}


def _parse_backend(raw: Any) -> BackendConfig | None:
    """Parse a ``backend:`` mapping into a ``BackendConfig``, or ``None``."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"'backend' must be a mapping, got {type(raw).__name__}")
    base_url = raw.get("base_url")
    if not base_url or not isinstance(base_url, str):
        raise ValueError("'backend.base_url' is required and must be a string")
    api_key_env = raw.get("api_key_env")
    if api_key_env is not None and not isinstance(api_key_env, str):
        raise ValueError("'backend.api_key_env' must be a string")
    return BackendConfig(base_url=base_url, api_key_env=api_key_env)


def _parse_agent_spec(name: str, raw: dict, defaults: dict) -> AgentSpec:
    """Build an ``AgentSpec`` from a single agent config entry.

    *defaults* is the ``agent_defaults:`` mapping — its values are used as
    fallbacks for any key not present in *raw*.

    Raises ``ValueError`` on missing required fields or invalid types.
    """
    # Merge: agent-level overrides defaults.
    merged = {**defaults, **raw}

    # Warn on unknown keys (could be a typo).
    for key in merged:
        if key not in _KNOWN_KEYS:
            logger.warning("Agent '%s': unknown config key '%s' — ignored", name, key)

    # Required fields.
    for field_name in _REQUIRED_FIELDS:
        if field_name not in merged:
            raise ValueError(f"Agent '{name}': missing required field '{field_name}'")

    system_message = merged["system_message"]
    if not isinstance(system_message, str):
        raise ValueError(
            f"Agent '{name}': 'system_message' must be a string, "
            f"got {type(system_message).__name__}"
        )

    tools = merged["tools"]
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        raise ValueError(f"Agent '{name}': 'tools' must be a list of strings")

    model = merged["model"]
    if not isinstance(model, str):
        raise ValueError(
            f"Agent '{name}': 'model' must be a string, got {type(model).__name__}"
        )

    # Optional fields with defaults.
    max_response_tokens = merged.get(
        "max_response_tokens", _OPTIONAL_FIELDS["max_response_tokens"]
    )
    if not isinstance(max_response_tokens, int) or isinstance(
        max_response_tokens, bool
    ):
        raise ValueError(f"Agent '{name}': 'max_response_tokens' must be an integer")

    persistence = merged.get("persistence", _OPTIONAL_FIELDS["persistence"])
    if persistence not in ("ephemeral", "session"):
        raise ValueError(
            f"Agent '{name}': 'persistence' must be 'ephemeral' or 'session', "
            f"got {persistence!r}"
        )

    tool_permission_overrides = merged.get(
        "tool_permission_overrides", _OPTIONAL_FIELDS["tool_permission_overrides"]
    )
    if not isinstance(tool_permission_overrides, dict):
        raise ValueError(
            f"Agent '{name}': 'tool_permission_overrides' must be a mapping"
        )
    for tool_name, is_allowed in tool_permission_overrides.items():
        if not isinstance(tool_name, str):
            raise ValueError(
                f"Agent '{name}': 'tool_permission_overrides' keys must be strings, "
                f"got {type(tool_name).__name__}"
            )
        if not isinstance(is_allowed, bool):
            raise ValueError(
                f"Agent '{name}': 'tool_permission_overrides[{tool_name!r}]' must "
                f"be a boolean, got {type(is_allowed).__name__}"
            )

    max_tool_rounds = merged.get("max_tool_rounds", _OPTIONAL_FIELDS["max_tool_rounds"])
    if not isinstance(max_tool_rounds, int) or isinstance(max_tool_rounds, bool):
        raise ValueError(f"Agent '{name}': 'max_tool_rounds' must be an integer")

    context_limit_threshold = merged.get(
        "context_limit_threshold", _OPTIONAL_FIELDS["context_limit_threshold"]
    )
    if not isinstance(context_limit_threshold, (int, float)) or isinstance(
        context_limit_threshold, bool
    ):
        raise ValueError(f"Agent '{name}': 'context_limit_threshold' must be a number")
    if not 0 < context_limit_threshold <= 1:
        raise ValueError(
            f"Agent '{name}': 'context_limit_threshold' must be > 0 and <= 1"
        )

    backend = _parse_backend(merged.get("backend"))

    return AgentSpec(
        name=name,
        system_message=system_message,
        tools=tools,
        model=model,
        max_response_tokens=max_response_tokens,
        persistence=persistence,
        backend=backend,
        tool_permission_overrides=dict(tool_permission_overrides),
        max_tool_rounds=max_tool_rounds,
        context_limit_threshold=float(context_limit_threshold),
    )


def load_agent_specs(config: ConfigManager) -> dict[str, AgentSpec]:
    """Parse all agent specs from *config*.

    Returns an empty dict when ``agents:`` is absent or empty.
    Logs a warning and skips individual agents that fail validation.
    """
    agents_raw = config.get("agents")
    if agents_raw is None:
        return {}
    if not isinstance(agents_raw, dict):
        logger.warning("'agents' config must be a mapping — ignoring")
        return {}

    defaults = config.get("agent_defaults")
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        logger.warning("'agent_defaults' config must be a mapping — ignoring")
        defaults = {}

    specs: dict[str, AgentSpec] = {}
    for name, entry in agents_raw.items():
        if not isinstance(name, str):
            logger.warning("Agent name must be a string, got %r — skipping", name)
            continue
        if not isinstance(entry, dict):
            logger.warning("Agent '%s': config must be a mapping — skipping", name)
            continue
        try:
            specs[name] = _parse_agent_spec(name, entry, defaults)
        except ValueError as exc:
            logger.warning("%s — skipping", exc)
    return specs


class AgentRegistry:
    """Registry of available agent types.

    Constructed with the output of :func:`load_agent_specs`.  Provides
    read-only access to specs and a ``has_agents`` convenience flag.

    Agent instantiation (``get_or_create``) is added in a later PR.
    """

    def __init__(self, specs: dict[str, AgentSpec]) -> None:
        self._specs = dict(specs)

    @property
    def specs(self) -> dict[str, AgentSpec]:
        """Return a copy of the name → spec mapping."""
        return dict(self._specs)

    @property
    def has_agents(self) -> bool:
        """True when at least one agent spec is loaded."""
        return bool(self._specs)
