"""
Agent data structures — declarative specs and result types.

This module defines the data classes used by the multi-agent system.
``AgentSpec`` describes an agent type (parsed from config); ``AgentResult``
carries the outcome of a single ``Agent.run()`` invocation.

The ``Agent`` class itself (the runtime loop) is added in a later PR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BackendConfig:
    """Connection details for an LLM backend.

    When ``None`` on an ``AgentSpec``, the agent inherits the
    coordinator's backend.  Authentication follows the same contract as
    ``ConfigManager``: config stores the environment variable name
    (``api_key_env``), and the actual key is resolved from the environment
    at agent instantiation time.
    """

    base_url: str
    api_key_env: str | None = None


@dataclass
class AgentSpec:
    """Declarative description of an agent type, parsed from config."""

    name: str
    system_message: str
    tools: list[str]
    model: str
    max_response_tokens: int = 4096
    persistence: Literal["ephemeral", "session"] = "ephemeral"
    backend: BackendConfig | None = None
    tool_permission_overrides: dict[str, bool] = field(default_factory=dict)
    max_tool_rounds: int = 10
    context_limit_threshold: float = 0.90


@dataclass
class AgentResult:
    """Returned by ``Agent.run()`` when the send/tool/repeat loop ends."""

    text: str
    status: Literal["ok", "context_limit", "tool_limit", "error"]
    partial: bool = False
    error_message: str = ""
