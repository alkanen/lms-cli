"""
ConfigManager — layered YAML configuration loader.

Resolution order (later levels override earlier ones):

1. Global user config:   ~/.ai-cli/config.yaml
2. Project config:       <project>/.ai-cli/config.yaml   (if project_root given)
3. CLI overrides:        passed as a dict at construction time

API keys must never be stored directly in config files.  Instead, store
the name of the environment variable that holds the key
(e.g. ``api_key_env: OPENAI_API_KEY``).  A future ``get_model_config()``
method will resolve the actual key value from the environment at call time;
it is not yet implemented.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from ai_cli.core.workspace import _DOT_AI_CLI, _GLOBAL_DIR


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict.

    Returns an empty dict if the file does not exist.
    Raises ``ConfigError`` on parse or I/O errors.
    """
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Cannot read config file '{path}': {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in '{path}': {exc}") from exc
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with *override* merged on top of *base*.

    Nested dicts are merged recursively; all other values are replaced.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ConfigManager:
    """
    Loads and merges configuration from global, project, and CLI sources.

    Parameters
    ----------
    project_root:
        Absolute path to the project workspace root (the directory that
        *contains* the ``.ai-cli/`` folder).  ``None`` when running without
        a project workspace.
    cli_overrides:
        Dict of key/value pairs provided via CLI flags.  These take
        highest priority and override everything else.
    """

    def __init__(
        self,
        project_root: Path | None,
        cli_overrides: dict,
    ) -> None:
        global_cfg = _load_yaml(_GLOBAL_DIR / "config.yaml")
        project_cfg: dict = {}
        if project_root is not None:
            project_cfg = _load_yaml(
                project_root / _DOT_AI_CLI / "config.yaml"
            )

        # Merge layers: global → project → CLI overrides.
        self._config: dict = _deep_merge(
            _deep_merge(global_cfg, project_cfg),
            cli_overrides,
        )
        self._project_root = project_root

    def get(self, key: str, default=None):
        """Layered lookup: cli_overrides > project config > global config > default."""
        return self._config.get(key, default)

    def get_model_config(self) -> dict:
        """Return the effective model/backend configuration.

        Resolves ``api_key_env`` to the actual API key from the environment.
        The raw key value is never read from config files — only the env-var
        name is stored there.

        Raises ``ConfigError`` if:
        - Neither ``model`` nor ``base_url`` is configured (nothing to connect to).
        - ``api_key_env`` is set but the named environment variable is missing.
        """
        model = self._config.get("model")
        base_url = self._config.get("base_url")
        if not model and not base_url:
            raise ConfigError(
                "No model or base_url configured. "
                "Add at least one to ~/.ai-cli/config.yaml or "
                "<project>/.ai-cli/config.yaml."
            )

        cfg = dict(self._config)

        api_key_env = cfg.pop("api_key_env", None)
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ConfigError(
                    f"api_key_env is set to '{api_key_env}' but that "
                    f"environment variable is not set. "
                    f"Add it to your .env file or export it in your shell."
                )
            cfg["api_key"] = api_key

        return cfg

    def get_backend(self) -> str:
        """Return the configured backend name ('openai' or 'lmstudio').

        Defaults to 'openai' if not explicitly set.
        """
        return self._config.get("backend", "openai")
