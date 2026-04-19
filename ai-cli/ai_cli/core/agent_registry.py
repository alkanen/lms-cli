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
import os
import threading
from typing import TYPE_CHECKING, Any

from ai_cli.core.agent import AgentSpec, BackendConfig
from ai_cli.core.config_manager import ConfigManager

if TYPE_CHECKING:
    from ai_cli.cli.display import Display
    from ai_cli.core.agent import Agent
    from ai_cli.core.llm_client import LLMClient
    from ai_cli.core.tool_registry import ToolRegistry
    from ai_cli.core.workspace import Workspace

logger = logging.getLogger(__name__)

# Fields on AgentSpec that have defaults and are therefore optional in config.
_OPTIONAL_FIELDS: dict[str, Any] = {
    "max_response_tokens": 4096,
    "persistence": "ephemeral",
    "tool_permission_overrides": {},
    "max_tool_rounds": 10,
    "context_limit_threshold": 0.90,
    "context_window": None,
    "skills": None,
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
    "context_window",
    "skills",
}


def _parse_backend(raw: Any) -> BackendConfig | None:
    """Parse a ``backend:`` mapping into a ``BackendConfig``, or ``None``."""
    if raw is None:
        logger.debug("No agent-specific backend configured; using coordinator backend")
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"'backend' must be a mapping, got {type(raw).__name__}")
    base_url = raw.get("base_url")
    if not base_url or not isinstance(base_url, str):
        raise ValueError("'backend.base_url' is required and must be a string")
    api_key_env = raw.get("api_key_env")
    if api_key_env is not None and not isinstance(api_key_env, str):
        raise ValueError("'backend.api_key_env' must be a string")
    logger.debug(
        "Parsed agent backend config for base_url=%r api_key_env=%r",
        base_url,
        api_key_env,
    )
    return BackendConfig(base_url=base_url, api_key_env=api_key_env)


def _parse_agent_spec(name: str, raw: dict, defaults: dict) -> AgentSpec:
    """Build an ``AgentSpec`` from a single agent config entry.

    *defaults* is the ``agent_defaults:`` mapping — its values are used as
    fallbacks for any key not present in *raw*.

    Raises ``ValueError`` on missing required fields or invalid types.
    """
    logger.debug(
        "Agent '%s': parsing spec with raw keys=%s default keys=%s",
        name,
        sorted(raw.keys()),
        sorted(defaults.keys()),
    )
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

    raw_context_window = merged.get(
        "context_window", _OPTIONAL_FIELDS["context_window"]
    )
    context_window: int | None = None
    if raw_context_window is not None:
        if isinstance(raw_context_window, bool) or not isinstance(
            raw_context_window, int
        ):
            raise ValueError(
                f"Agent '{name}': 'context_window' must be a positive integer"
            )
        if raw_context_window <= 0:
            raise ValueError(
                f"Agent '{name}': 'context_window' must be a positive integer, "
                f"got {raw_context_window}"
            )
        context_window = raw_context_window

    raw_skills = merged.get("skills", _OPTIONAL_FIELDS["skills"])
    skills: list[str] | None = None
    if raw_skills is not None:
        if not isinstance(raw_skills, list) or not all(
            isinstance(s, str) for s in raw_skills
        ):
            raise ValueError(f"Agent '{name}': 'skills' must be a list of strings")
        # Canonicalized list (trimmed, de-duplicated, order-preserving).
        seen: set[str] = set()
        canonical: list[str] = []
        for skill_name in raw_skills:
            normalized = skill_name.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            canonical.append(normalized)
        skills = canonical

    backend = _parse_backend(merged.get("backend"))

    logger.info(
        "Agent '%s': parsed spec (tools=%d, persistence=%s, backend=%s, context_window=%r, max_tool_rounds=%d)",
        name,
        len(tools),
        persistence,
        "custom" if backend is not None else "coordinator",
        context_window,
        max_tool_rounds,
    )

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
        context_window=context_window,
        skills=skills,
    )


def load_agent_specs(config: ConfigManager) -> dict[str, AgentSpec]:
    """Parse all agent specs from *config*.

    Returns an empty dict when ``agents:`` is absent or empty.
    Logs a warning and skips individual agents that fail validation.
    """
    agents_raw = config.get("agents")
    if agents_raw is None:
        logger.info("No 'agents' config found; agent registry will be empty")
        return {}
    if not isinstance(agents_raw, dict):
        logger.warning("'agents' config must be a mapping — ignoring")
        return {}

    defaults = config.get("agent_defaults")
    if defaults is None:
        defaults = {}
        logger.debug("No 'agent_defaults' config found; using built-in defaults")
    if not isinstance(defaults, dict):
        logger.warning("'agent_defaults' config must be a mapping — ignoring")
        defaults = {}

    logger.info(
        "Loading agent specs for %d configured agents with %d default keys",
        len(agents_raw),
        len(defaults),
    )
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
    logger.info(
        "Loaded %d valid agent specs (from %d configured entries)",
        len(specs),
        len(agents_raw),
    )
    return specs


class AgentRegistry:
    """Registry of available agent types.

    Constructed with the output of :func:`load_agent_specs`.  Provides
    read-only access to specs, a ``has_agents`` convenience flag, and
    lazy agent instantiation via :meth:`get_or_create`.
    """

    def __init__(
        self,
        specs: dict[str, AgentSpec],
        *,
        parent_display: Display | None = None,
    ) -> None:
        self._specs = dict(specs)
        self._parent_display = parent_display
        # Cache for session-persistent agent instances.
        self._instances: dict[str, Agent] = {}
        # Protects _instances cache mutations for thread-safe parallel dispatch.
        self._lock = threading.Lock()
        logger.info(
            "AgentRegistry initialised with %d specs: %s",
            len(self._specs),
            sorted(self._specs),
        )

    @property
    def specs(self) -> dict[str, AgentSpec]:
        """Return a copy of the name → spec mapping."""
        return dict(self._specs)

    @property
    def has_agents(self) -> bool:
        """True when at least one agent spec is loaded."""
        return bool(self._specs)

    def has(self, name: str) -> bool:
        """Return ``True`` when a spec named *name* is registered."""
        return name in self._specs

    def reset(self, name: str) -> None:
        """Clear the cached session instance for *name*.

        The next :meth:`get_or_create` call for a ``persistence == "session"``
        agent with this name will build a fresh instance, discarding the prior
        conversation history.  Has no effect for ephemeral agents or unknown
        names.
        """
        with self._lock:
            removed = self._instances.pop(name, None)
        if removed is None:
            logger.debug("AgentRegistry reset ignored for uncached agent %r", name)
        else:
            logger.info("AgentRegistry reset cached session agent %r", name)

    def get_or_create(
        self,
        name: str,
        *,
        workspace: Workspace,
        config: ConfigManager,
        coordinator_llm: LLMClient,
        global_tool_registry: ToolRegistry,
    ) -> Agent:
        """Return an ``Agent`` for *name*, building it if necessary.

        For ``persistence == "session"`` specs the agent is cached and
        returned on subsequent calls after calling ``agent.reset()`` (which
        clears the display buffer and pending transient schemas).  For
        ``persistence == "ephemeral"`` specs a fresh agent is built on every
        call.

        Raises ``KeyError`` if *name* is not in the registry.
        """
        spec = self._specs.get(name)
        if spec is None:
            logger.warning("AgentRegistry lookup failed for unknown agent %r", name)
            raise KeyError(f"No agent spec named {name!r}")

        logger.info(
            "AgentRegistry get_or_create for agent %r (persistence=%s)",
            name,
            spec.persistence,
        )

        if spec.persistence == "session":
            # Fast path: already cached — acquire lock only briefly.
            with self._lock:
                if name in self._instances:
                    agent = self._instances[name]
                    logger.info("AgentRegistry cache hit for session agent %r", name)
                    agent.reset()
                    return agent

            # Build outside the lock so expensive construction doesn't
            # serialise concurrent parallel dispatches.
            logger.info("AgentRegistry cache miss for session agent %r; building", name)
            candidate = self._build_agent(
                spec, workspace, config, coordinator_llm, global_tool_registry
            )

            # Re-check under lock: another thread may have built and cached
            # the agent while we were constructing ours.
            with self._lock:
                if name in self._instances:
                    cached = self._instances[name]
                    logger.info(
                        "AgentRegistry concurrent cache fill won race for session agent %r; reusing cached instance",
                        name,
                    )
                    cached.reset()
                    return cached
                self._instances[name] = candidate
                logger.info("AgentRegistry cached new session agent %r", name)
                return candidate

        # ephemeral — build a fresh agent every time
        logger.info("AgentRegistry building fresh ephemeral agent %r", name)
        return self._build_agent(
            spec, workspace, config, coordinator_llm, global_tool_registry
        )

    def _build_agent(
        self,
        spec: AgentSpec,
        workspace: Workspace,
        config: ConfigManager,
        coordinator_llm: LLMClient,
        global_tool_registry: ToolRegistry,
    ) -> Agent:
        """Construct a new ``Agent`` from *spec*."""
        from ai_cli.cli.display import SubAgentDisplay
        from ai_cli.core.agent import Agent, build_agent_tool_registry
        from ai_cli.core.llm_client import OpenAIClient
        from ai_cli.core.session_manager import InMemorySession

        logger.info(
            "AgentRegistry building agent %r (model=%s, backend=%s, tools=%d)",
            spec.name,
            spec.model,
            "custom" if spec.backend is not None else "coordinator",
            len(spec.tools),
        )
        if spec.backend is not None:
            meta = coordinator_llm.get_model_metadata()
            llm_cfg: dict = {
                "model": spec.model,
                # Prefer the per-agent context_window spec; fall back to the
                # coordinator's value so existing configs still work.
                "context_window": (
                    spec.context_window
                    if spec.context_window is not None
                    else meta.get("context_window", 8192)
                ),
                "max_response_tokens": spec.max_response_tokens,
                "base_url": spec.backend.base_url,
            }
            if spec.backend.api_key_env:
                api_key = os.environ.get(spec.backend.api_key_env)
                if api_key is None:
                    logger.warning(
                        "Agent '%s': environment variable '%s' (api_key_env) is not set"
                        " — connecting without an API key.",
                        spec.name,
                        spec.backend.api_key_env,
                    )
                    api_key = "no-key"
                llm_cfg["api_key"] = api_key
            logger.debug(
                "AgentRegistry creating dedicated OpenAIClient for agent %r with context_window=%r",
                spec.name,
                llm_cfg.get("context_window"),
            )
            llm_client: LLMClient = OpenAIClient(llm_cfg)
        else:
            logger.debug(
                "AgentRegistry reusing coordinator LLM for agent %r",
                spec.name,
            )
            llm_client = coordinator_llm

        if self._parent_display is not None:
            display = SubAgentDisplay(
                verbose=self._parent_display.verbose,
                markdown_enabled=self._parent_display.markdown_enabled,
                parent_display=self._parent_display,
                agent_name=spec.name,
            )
        else:
            display = SubAgentDisplay()
        session = InMemorySession(llm_client, system_message=spec.system_message)
        registry = build_agent_tool_registry(
            spec, workspace, config, display, global_tool_registry
        )
        logger.info(
            "AgentRegistry finished building agent %r with scoped tools=%d",
            spec.name,
            len(registry.definitions()),
        )
        return Agent(spec, session, llm_client, registry, display)
