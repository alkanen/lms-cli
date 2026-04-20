"""Tests for ai_cli.core.config_manager.ConfigManager."""

from pathlib import Path

import pytest

from ai_cli.core.config_manager import ConfigError, ConfigManager
from ai_cli.core.workspace import _DOT_AI_CLI


@pytest.fixture(autouse=True)
def isolate_global_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirect get_global_dir to an empty tmp dir so real ~/.ai-cli/config.yaml is never read."""
    fake_global = tmp_path_factory.mktemp("fake_global")
    monkeypatch.setattr(
        "ai_cli.core.config_manager.get_global_dir", lambda: fake_global
    )


@pytest.fixture()
def global_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A writable fake global dir, already patched into ConfigManager."""
    fake_global = tmp_path_factory.mktemp("global_ai_cli")
    monkeypatch.setattr(
        "ai_cli.core.config_manager.get_global_dir", lambda: fake_global
    )
    return fake_global


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A temporary project root with an empty .ai-cli/ folder."""
    (tmp_path / _DOT_AI_CLI).mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Construction / YAML loading
# ---------------------------------------------------------------------------


class TestInit:
    def test_no_files_loads_empty_config(self, project):
        cm = ConfigManager(project, {})
        assert cm.get("anything") is None

    def test_global_config_loaded(self, global_dir, project):
        (global_dir / "config.yaml").write_text("backend: openai\n")
        cm = ConfigManager(project, {})
        assert cm.get("backend") == "openai"

    def test_project_config_loaded(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text("backend: lmstudio\n")
        cm = ConfigManager(project, {})
        assert cm.get("backend") == "lmstudio"

    def test_project_overrides_global(self, global_dir, project):
        (global_dir / "config.yaml").write_text("backend: openai\nmodel: gpt-4o\n")
        (project / _DOT_AI_CLI / "config.yaml").write_text("backend: lmstudio\n")
        cm = ConfigManager(project, {})
        assert cm.get("backend") == "lmstudio"
        assert cm.get("model") == "gpt-4o"  # global key preserved

    def test_cli_overrides_project(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text("backend: lmstudio\n")
        cm = ConfigManager(project, {"backend": "openai"})
        assert cm.get("backend") == "openai"

    def test_no_project_root(self, global_dir):
        (global_dir / "config.yaml").write_text("backend: openai\n")
        cm = ConfigManager(None, {})
        assert cm.get("backend") == "openai"

    def test_get_default(self, project):
        cm = ConfigManager(project, {})
        assert cm.get("missing_key", "fallback") == "fallback"

    def test_invalid_yaml_raises(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text("key: [unclosed\n")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            ConfigManager(project, {})

    def test_non_dict_yaml_treated_as_empty(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text("- item1\n- item2\n")
        cm = ConfigManager(project, {})
        assert cm.get("anything") is None

    def test_nested_merge(self, global_dir, project):
        (global_dir / "config.yaml").write_text(
            "model_params:\n  temperature: 0.7\n  max_response_tokens: 512\n"
        )
        (project / _DOT_AI_CLI / "config.yaml").write_text(
            "model_params:\n  max_response_tokens: 1024\n"
        )
        cm = ConfigManager(project, {})
        params = cm.get("model_params")
        assert params["temperature"] == 0.7
        assert params["max_response_tokens"] == 1024


# ---------------------------------------------------------------------------
# get_agents_config / get_agent_defaults
# ---------------------------------------------------------------------------


class TestGetAgentsConfig:
    def test_returns_mapping_when_present(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text(
            "agents:\n  explore:\n    model: m1\n"
        )
        cm = ConfigManager(project, {})
        assert cm.get_agents_config() == {"explore": {"model": "m1"}}

    def test_returns_empty_when_absent(self, project):
        cm = ConfigManager(project, {})
        assert cm.get_agents_config() == {}

    def test_returns_empty_when_null(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text("agents:\n")
        cm = ConfigManager(project, {})
        assert cm.get_agents_config() == {}

    def test_returns_empty_when_not_dict(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text("agents: [a, b]\n")
        cm = ConfigManager(project, {})
        assert cm.get_agents_config() == {}


class TestGetAgentDefaults:
    def test_returns_mapping_when_present(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text(
            "agent_defaults:\n  persistence: session\n"
        )
        cm = ConfigManager(project, {})
        assert cm.get_agent_defaults() == {"persistence": "session"}

    def test_returns_empty_when_absent(self, project):
        cm = ConfigManager(project, {})
        assert cm.get_agent_defaults() == {}

    def test_returns_empty_when_null(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text("agent_defaults:\n")
        cm = ConfigManager(project, {})
        assert cm.get_agent_defaults() == {}

    def test_returns_empty_when_not_dict(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text("agent_defaults: bad\n")
        cm = ConfigManager(project, {})
        assert cm.get_agent_defaults() == {}


# ---------------------------------------------------------------------------
# get_backend
# ---------------------------------------------------------------------------


class TestGetBackend:
    def test_defaults_to_openai(self, project):
        cm = ConfigManager(project, {})
        assert cm.get_backend() == "openai"

    def test_null_backend_defaults_to_openai(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text("backend: null\n")
        cm = ConfigManager(project, {})
        assert cm.get_backend() == "openai"

    def test_returns_configured_backend(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text("backend: lmstudio\n")
        cm = ConfigManager(project, {})
        assert cm.get_backend() == "lmstudio"

    def test_non_string_backend_raises(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text("backend: 123\n")
        cm = ConfigManager(project, {})
        with pytest.raises(ConfigError, match="must be a string"):
            cm.get_backend()


# ---------------------------------------------------------------------------
# get_model_config
# ---------------------------------------------------------------------------


class TestGetModelConfig:
    def test_raises_when_no_model_or_base_url(self, project):
        cm = ConfigManager(project, {})
        with pytest.raises(ConfigError, match="No model or base_url"):
            cm.get_model_config()

    def test_returns_config_with_model(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text(
            "model: gpt-4o\nbackend: openai\n"
        )
        cm = ConfigManager(project, {})
        cfg = cm.get_model_config()
        assert cfg["model"] == "gpt-4o"

    def test_resolves_api_key_from_env(self, project, monkeypatch):
        (project / _DOT_AI_CLI / "config.yaml").write_text(
            "model: gpt-4o\napi_key_env: MY_TEST_KEY\n"
        )
        monkeypatch.setenv("MY_TEST_KEY", "sk-secret")
        cm = ConfigManager(project, {})
        cfg = cm.get_model_config()
        assert cfg["api_key"] == "sk-secret"
        assert "api_key_env" not in cfg  # env var name is stripped

    def test_raises_when_api_key_env_missing(self, project, monkeypatch):
        (project / _DOT_AI_CLI / "config.yaml").write_text(
            "model: gpt-4o\napi_key_env: MISSING_KEY\n"
        )
        monkeypatch.delenv("MISSING_KEY", raising=False)
        cm = ConfigManager(project, {})
        with pytest.raises(ConfigError, match="MISSING_KEY"):
            cm.get_model_config()

    def test_no_api_key_env_no_api_key_in_result(self, project):
        (project / _DOT_AI_CLI / "config.yaml").write_text("model: gpt-4o\n")
        cm = ConfigManager(project, {})
        cfg = cm.get_model_config()
        assert "api_key" not in cfg
        assert "api_key_env" not in cfg
