"""Tests for ai_cli.core.agent_registry — spec parsing and registry."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_cli.core.agent import AgentSpec, BackendConfig
from ai_cli.core.agent_registry import (
    AgentRegistry,
    _parse_agent_spec,
    load_agent_specs,
)
from ai_cli.core.config_manager import ConfigManager
from ai_cli.core.workspace import _DOT_AI_CLI

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_global_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_global = tmp_path_factory.mktemp("fake_global")
    monkeypatch.setattr(
        "ai_cli.core.config_manager.get_global_dir", lambda: fake_global
    )


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / _DOT_AI_CLI).mkdir()
    return tmp_path


def _make_config(project: Path, yaml_text: str) -> ConfigManager:
    (project / _DOT_AI_CLI / "config.yaml").write_text(yaml_text)
    return ConfigManager(project, {})


# Minimal valid agent entry for use as a base.
_MINIMAL = {
    "system_message": "You are a helper.",
    "tools": ["read_file"],
    "model": "llama3.2:3b",
}


# ---------------------------------------------------------------------------
# _parse_agent_spec
# ---------------------------------------------------------------------------


class TestParseAgentSpec:
    def test_minimal(self):
        spec = _parse_agent_spec("explore", dict(_MINIMAL), {})
        assert spec.name == "explore"
        assert spec.system_message == "You are a helper."
        assert spec.tools == ["read_file"]
        assert spec.model == "llama3.2:3b"
        assert spec.persistence == "ephemeral"
        assert spec.backend is None

    def test_all_fields(self):
        raw = {
            **_MINIMAL,
            "max_response_tokens": 2048,
            "persistence": "session",
            "backend": {
                "base_url": "http://localhost:11435/v1",
                "api_key_env": "TEST_API_KEY",
            },
            "tool_permission_overrides": {"write_file": False},
            "max_tool_rounds": 5,
            "context_limit_threshold": 0.80,
        }
        spec = _parse_agent_spec("coder", raw, {})
        assert spec.max_response_tokens == 2048
        assert spec.persistence == "session"
        assert spec.backend == BackendConfig(
            base_url="http://localhost:11435/v1", api_key_env="TEST_API_KEY"
        )
        assert spec.tool_permission_overrides == {"write_file": False}
        assert spec.max_tool_rounds == 5
        assert spec.context_limit_threshold == 0.80

    def test_defaults_merge(self):
        defaults = {"model": "default-model", "max_tool_rounds": 20}
        raw = {"system_message": "Hello.", "tools": ["find_files"]}
        spec = _parse_agent_spec("x", raw, defaults)
        assert spec.model == "default-model"
        assert spec.max_tool_rounds == 20

    def test_skills_trim_and_dedupe(self):
        raw = {**_MINIMAL, "skills": [" planner ", "review", "planner", ""]}
        spec = _parse_agent_spec("x", raw, {})
        assert spec.skills == ["planner", "review"]

    def test_agent_overrides_defaults(self):
        defaults = {"model": "default-model", "max_tool_rounds": 20}
        raw = {**_MINIMAL, "max_tool_rounds": 5}
        spec = _parse_agent_spec("x", raw, defaults)
        assert spec.model == "llama3.2:3b"  # from raw, not defaults
        assert spec.max_tool_rounds == 5

    def test_backend_none_when_absent(self):
        spec = _parse_agent_spec("x", dict(_MINIMAL), {})
        assert spec.backend is None

    def test_backend_api_key_env_none_when_absent(self):
        raw = {**_MINIMAL, "backend": {"base_url": "http://example.com"}}
        spec = _parse_agent_spec("x", raw, {})
        assert spec.backend is not None
        assert spec.backend.api_key_env is None

    # --- Validation errors ---

    def test_missing_system_message(self):
        raw = {"tools": ["read_file"], "model": "m"}
        with pytest.raises(ValueError, match="missing required field 'system_message'"):
            _parse_agent_spec("bad", raw, {})

    def test_missing_tools(self):
        raw = {"system_message": "hi", "model": "m"}
        with pytest.raises(ValueError, match="missing required field 'tools'"):
            _parse_agent_spec("bad", raw, {})

    def test_missing_model(self):
        raw = {"system_message": "hi", "tools": []}
        with pytest.raises(ValueError, match="missing required field 'model'"):
            _parse_agent_spec("bad", raw, {})

    def test_tools_not_list(self):
        raw = {**_MINIMAL, "tools": "read_file"}
        with pytest.raises(ValueError, match="'tools' must be a list"):
            _parse_agent_spec("bad", raw, {})

    def test_tools_items_not_strings(self):
        raw = {**_MINIMAL, "tools": [123]}
        with pytest.raises(ValueError, match="'tools' must be a list of strings"):
            _parse_agent_spec("bad", raw, {})

    def test_model_not_string(self):
        raw = {**_MINIMAL, "model": 42}
        with pytest.raises(ValueError, match="'model' must be a string"):
            _parse_agent_spec("bad", raw, {})

    def test_system_message_not_string(self):
        raw = {**_MINIMAL, "system_message": ["a", "b"]}
        with pytest.raises(ValueError, match="'system_message' must be a string"):
            _parse_agent_spec("bad", raw, {})

    def test_invalid_persistence(self):
        raw = {**_MINIMAL, "persistence": "forever"}
        with pytest.raises(ValueError, match="'persistence' must be"):
            _parse_agent_spec("bad", raw, {})

    def test_max_response_tokens_not_int(self):
        raw = {**_MINIMAL, "max_response_tokens": "big"}
        with pytest.raises(
            ValueError, match="'max_response_tokens' must be an integer"
        ):
            _parse_agent_spec("bad", raw, {})

    def test_max_response_tokens_bool_rejected(self):
        raw = {**_MINIMAL, "max_response_tokens": True}
        with pytest.raises(
            ValueError, match="'max_response_tokens' must be an integer"
        ):
            _parse_agent_spec("bad", raw, {})

    def test_max_tool_rounds_not_int(self):
        raw = {**_MINIMAL, "max_tool_rounds": 3.5}
        with pytest.raises(ValueError, match="'max_tool_rounds' must be an integer"):
            _parse_agent_spec("bad", raw, {})

    def test_context_limit_threshold_not_number(self):
        raw = {**_MINIMAL, "context_limit_threshold": "high"}
        with pytest.raises(
            ValueError, match="'context_limit_threshold' must be a number"
        ):
            _parse_agent_spec("bad", raw, {})

    def test_context_limit_threshold_bool_rejected(self):
        raw = {**_MINIMAL, "context_limit_threshold": True}
        with pytest.raises(
            ValueError, match="'context_limit_threshold' must be a number"
        ):
            _parse_agent_spec("bad", raw, {})

    @pytest.mark.parametrize("value", [0, -0.1, 1.5])
    def test_context_limit_threshold_out_of_range(self, value):
        raw = {**_MINIMAL, "context_limit_threshold": value}
        with pytest.raises(ValueError, match="must be > 0 and <= 1"):
            _parse_agent_spec("bad", raw, {})

    def test_context_limit_threshold_one_is_valid(self):
        raw = {**_MINIMAL, "context_limit_threshold": 1}
        spec = _parse_agent_spec("ok", raw, {})
        assert spec.context_limit_threshold == 1.0

    def test_tool_permission_overrides_not_dict(self):
        raw = {**_MINIMAL, "tool_permission_overrides": [True]}
        with pytest.raises(
            ValueError, match="'tool_permission_overrides' must be a mapping"
        ):
            _parse_agent_spec("bad", raw, {})

    def test_tool_permission_overrides_non_string_key(self):
        raw = {**_MINIMAL, "tool_permission_overrides": {123: True}}
        with pytest.raises(ValueError, match="keys must be strings"):
            _parse_agent_spec("bad", raw, {})

    def test_tool_permission_overrides_non_bool_value(self):
        raw = {**_MINIMAL, "tool_permission_overrides": {"write_file": "yes"}}
        with pytest.raises(ValueError, match="must be a boolean"):
            _parse_agent_spec("bad", raw, {})

    def test_skills_not_list(self):
        raw = {**_MINIMAL, "skills": "planner"}
        with pytest.raises(ValueError, match="'skills' must be a list of strings"):
            _parse_agent_spec("bad", raw, {})

    def test_skills_non_string_entries(self):
        raw = {**_MINIMAL, "skills": ["planner", 1]}
        with pytest.raises(ValueError, match="'skills' must be a list of strings"):
            _parse_agent_spec("bad", raw, {})

    def test_backend_not_dict(self):
        raw = {**_MINIMAL, "backend": "http://example.com"}
        with pytest.raises(ValueError, match="'backend' must be a mapping"):
            _parse_agent_spec("bad", raw, {})

    def test_backend_missing_base_url(self):
        raw = {**_MINIMAL, "backend": {"api_key_env": "K"}}
        with pytest.raises(ValueError, match="'backend.base_url' is required"):
            _parse_agent_spec("bad", raw, {})

    def test_unknown_key_warns(self, caplog):
        raw = {**_MINIMAL, "flavour": "vanilla"}
        with caplog.at_level("WARNING"):
            spec = _parse_agent_spec("x", raw, {})
        assert spec.name == "x"
        assert "unknown config key 'flavour'" in caplog.text


# ---------------------------------------------------------------------------
# load_agent_specs
# ---------------------------------------------------------------------------


class TestLoadAgentSpecs:
    def test_empty_config(self, project):
        cm = _make_config(project, "model: test\n")
        specs = load_agent_specs(cm)
        assert specs == {}

    def test_agents_is_none(self, project):
        cm = _make_config(project, "agents:\n")
        specs = load_agent_specs(cm)
        assert specs == {}

    def test_single_agent(self, project):
        yaml_text = """\
agents:
  explore:
    system_message: "Search files."
    tools:
      - read_file
      - find_files
    model: llama3.2:3b
"""
        cm = _make_config(project, yaml_text)
        specs = load_agent_specs(cm)
        assert "explore" in specs
        assert specs["explore"].tools == ["read_file", "find_files"]

    def test_multiple_agents(self, project):
        yaml_text = """\
agents:
  explore:
    system_message: "Search."
    tools: [read_file]
    model: m1
  coder:
    system_message: "Code."
    tools: [read_file, write_file]
    model: m2
"""
        cm = _make_config(project, yaml_text)
        specs = load_agent_specs(cm)
        assert set(specs.keys()) == {"explore", "coder"}

    def test_agent_defaults_applied(self, project):
        yaml_text = """\
agent_defaults:
  persistence: session
  max_tool_rounds: 20
agents:
  explore:
    system_message: "Search."
    tools: [read_file]
    model: m1
"""
        cm = _make_config(project, yaml_text)
        specs = load_agent_specs(cm)
        assert specs["explore"].persistence == "session"
        assert specs["explore"].max_tool_rounds == 20

    def test_agent_overrides_defaults(self, project):
        yaml_text = """\
agent_defaults:
  max_tool_rounds: 20
agents:
  explore:
    system_message: "Search."
    tools: [read_file]
    model: m1
    max_tool_rounds: 5
"""
        cm = _make_config(project, yaml_text)
        specs = load_agent_specs(cm)
        assert specs["explore"].max_tool_rounds == 5

    def test_agent_defaults_skills_applied(self, project):
        yaml_text = """\
agent_defaults:
  skills: [planner, reviewer]
agents:
  explore:
    system_message: "Search."
    tools: [read_file, skills]
    model: m1
"""
        cm = _make_config(project, yaml_text)
        specs = load_agent_specs(cm)
        assert specs["explore"].skills == ["planner", "reviewer"]

    def test_agent_skills_override_defaults_replace(self, project):
        yaml_text = """\
agent_defaults:
  skills: [planner, reviewer]
agents:
  explore:
    system_message: "Search."
    tools: [read_file, skills]
    model: m1
    skills: [writer]
"""
        cm = _make_config(project, yaml_text)
        specs = load_agent_specs(cm)
        assert specs["explore"].skills == ["writer"]

    def test_agent_skills_empty_list_disables_inherited_defaults(self, project):
        yaml_text = """\
agent_defaults:
  skills: [planner, reviewer]
agents:
  explore:
    system_message: "Search."
    tools: [read_file, skills]
    model: m1
    skills: []
"""
        cm = _make_config(project, yaml_text)
        specs = load_agent_specs(cm)
        assert specs["explore"].skills == []

    def test_invalid_agent_skipped_with_warning(self, project, caplog):
        yaml_text = """\
agents:
  good:
    system_message: "ok."
    tools: [read_file]
    model: m1
  bad:
    tools: [read_file]
    model: m1
"""
        cm = _make_config(project, yaml_text)
        with caplog.at_level("WARNING"):
            specs = load_agent_specs(cm)
        assert "good" in specs
        assert "bad" not in specs
        assert "missing required field 'system_message'" in caplog.text

    def test_agent_entry_not_dict_skipped(self, project, caplog):
        yaml_text = """\
agents:
  bad: "just a string"
"""
        cm = _make_config(project, yaml_text)
        with caplog.at_level("WARNING"):
            specs = load_agent_specs(cm)
        assert specs == {}
        assert "config must be a mapping" in caplog.text

    def test_agents_not_dict_warns(self, project, caplog):
        yaml_text = "agents: [a, b, c]\n"
        cm = _make_config(project, yaml_text)
        with caplog.at_level("WARNING"):
            specs = load_agent_specs(cm)
        assert specs == {}
        assert "'agents' config must be a mapping" in caplog.text

    @pytest.mark.parametrize("value", ["false", "0", "[]"])
    def test_agents_falsey_non_none_warns(self, project, caplog, value):
        yaml_text = f"agents: {value}\n"
        cm = _make_config(project, yaml_text)
        with caplog.at_level("WARNING"):
            specs = load_agent_specs(cm)
        assert specs == {}
        assert "'agents' config must be a mapping" in caplog.text

    def test_agent_defaults_not_dict_warns(self, project, caplog):
        yaml_text = """\
agent_defaults: "bad"
agents:
  explore:
    system_message: "Search."
    tools: [read_file]
    model: m1
"""
        cm = _make_config(project, yaml_text)
        with caplog.at_level("WARNING"):
            specs = load_agent_specs(cm)
        assert "explore" in specs
        assert "'agent_defaults' config must be a mapping" in caplog.text

    def test_backend_parsed(self, project):
        yaml_text = """\
agents:
  remote:
    system_message: "Remote."
    tools: [read_file]
    model: m1
    backend:
      base_url: http://other:11434/v1
      api_key_env: REMOTE_API_KEY
"""
        cm = _make_config(project, yaml_text)
        specs = load_agent_specs(cm)
        assert specs["remote"].backend == BackendConfig(
            base_url="http://other:11434/v1", api_key_env="REMOTE_API_KEY"
        )


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_empty(self):
        reg = AgentRegistry({})
        assert reg.has_agents is False
        assert reg.specs == {}

    def test_with_specs(self):
        spec = AgentSpec(
            name="explore",
            system_message="Search.",
            tools=["read_file"],
            model="m1",
        )
        reg = AgentRegistry({"explore": spec})
        assert reg.has_agents is True
        assert reg.specs == {"explore": spec}

    def test_specs_returns_copy(self):
        spec = AgentSpec(name="x", system_message="m", tools=[], model="m")
        reg = AgentRegistry({"x": spec})
        returned = reg.specs
        returned["y"] = spec  # mutate the copy
        assert "y" not in reg.specs  # original unaffected

    # ------------------------------------------------------------------
    # get_or_create
    # ------------------------------------------------------------------

    def _registry_deps(self):
        """Return keyword args for get_or_create() with mocked dependencies."""
        coordinator_llm = MagicMock()
        coordinator_llm.get_model_metadata.return_value = {
            "model": "test",
            "context_window": 4096,
            "max_response_tokens": 512,
        }
        return {
            "workspace": MagicMock(),
            "config": MagicMock(),
            "coordinator_llm": coordinator_llm,
            "global_tool_registry": MagicMock(get=lambda _name: None),
        }

    def test_get_or_create_unknown_name_raises(self):
        reg = AgentRegistry({})
        with pytest.raises(KeyError):
            reg.get_or_create("nonexistent", **self._registry_deps())

    def test_get_or_create_ephemeral_returns_agent(self):
        from ai_cli.core.agent import Agent

        spec = AgentSpec(
            name="explore",
            system_message="Search.",
            tools=[],
            model="m1",
            persistence="ephemeral",
        )
        reg = AgentRegistry({"explore": spec})
        agent = reg.get_or_create("explore", **self._registry_deps())
        assert isinstance(agent, Agent)

    def test_get_or_create_ephemeral_builds_new_each_call(self):
        """Ephemeral agents are never cached — each call produces a new instance."""

        spec = AgentSpec(
            name="explore",
            system_message="Search.",
            tools=[],
            model="m1",
            persistence="ephemeral",
        )
        reg = AgentRegistry({"explore": spec})
        deps = self._registry_deps()
        agent1 = reg.get_or_create("explore", **deps)
        agent2 = reg.get_or_create("explore", **deps)
        assert agent1 is not agent2

    def test_get_or_create_session_caches_instance(self):
        """Session-persistent agents are returned from cache on the second call."""
        spec = AgentSpec(
            name="helper",
            system_message="Help.",
            tools=[],
            model="m1",
            persistence="session",
        )
        reg = AgentRegistry({"helper": spec})
        deps = self._registry_deps()
        agent1 = reg.get_or_create("helper", **deps)
        agent2 = reg.get_or_create("helper", **deps)
        assert agent1 is agent2

    def test_get_or_create_session_calls_display_reset_on_cache_hit(self):
        """On a cache hit the agent's SubAgentDisplay.reset() is called."""
        from unittest.mock import patch

        spec = AgentSpec(
            name="helper",
            system_message="Help.",
            tools=[],
            model="m1",
            persistence="session",
        )
        reg = AgentRegistry({"helper": spec})
        deps = self._registry_deps()
        agent = reg.get_or_create("helper", **deps)
        # Patch reset on the actual SubAgentDisplay instance so we can count calls.
        with patch.object(agent._display, "reset") as mock_reset:
            reg.get_or_create("helper", **deps)
        mock_reset.assert_called_once()

    def test_get_or_create_session_agent_display_is_sub_agent_display(self):
        from ai_cli.cli.display import SubAgentDisplay

        spec = AgentSpec(
            name="helper",
            system_message="Help.",
            tools=[],
            model="m1",
            persistence="session",
        )
        reg = AgentRegistry({"helper": spec})
        agent = reg.get_or_create("helper", **self._registry_deps())
        assert isinstance(agent._display, SubAgentDisplay)

    def test_get_or_create_uses_coordinator_llm_when_no_backend(self):
        """When spec has no backend, the coordinator's LLM is reused."""
        spec = AgentSpec(
            name="explore",
            system_message="Search.",
            tools=[],
            model="m1",
            persistence="ephemeral",
            backend=None,
        )
        reg = AgentRegistry({"explore": spec})
        deps = self._registry_deps()
        agent = reg.get_or_create("explore", **deps)
        assert agent._llm is deps["coordinator_llm"]

    def test_get_or_create_creates_new_llm_when_backend_set(self, monkeypatch):
        """When spec has a backend, a new OpenAIClient is constructed."""
        from ai_cli.core.agent import BackendConfig
        from ai_cli.core.llm_client import OpenAIClient

        spec = AgentSpec(
            name="remote",
            system_message="Remote.",
            tools=[],
            model="remote-model",
            persistence="ephemeral",
            backend=BackendConfig(base_url="http://remote:11434/v1"),
        )
        reg = AgentRegistry({"remote": spec})
        deps = self._registry_deps()
        agent = reg.get_or_create("remote", **deps)
        assert isinstance(agent._llm, OpenAIClient)
        assert agent._llm is not deps["coordinator_llm"]

    def test_get_or_create_session_seeded_with_system_message(self):
        """The agent's session should contain the spec's system_message."""
        from ai_cli.core.session_manager import InMemorySession

        spec = AgentSpec(
            name="helper",
            system_message="You are a specialist.",
            tools=[],
            model="m1",
            persistence="ephemeral",
        )
        reg = AgentRegistry({"helper": spec})
        agent = reg.get_or_create("helper", **self._registry_deps())
        assert isinstance(agent._session, InMemorySession)
        messages = agent._session.get_messages()
        assert any(
            m.get("role") == "system"
            and "You are a specialist." in m.get("content", "")
            for m in messages
        )
