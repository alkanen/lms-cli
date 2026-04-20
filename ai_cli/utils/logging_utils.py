"""
logging_utils — structured JSONL logging for ai-cli.

Usage
-----
Call ``setup_logging(config, session_dir)`` once at startup (after the
session directory is known) to activate file logging::

    from ai_cli.utils.logging_utils import setup_logging
    setup_logging(config, session.session_dir)

All ``ai_cli.*`` loggers write to ``<session_dir>/session.log`` as
newline-delimited JSON.  Third-party library loggers are not modified by
``setup_logging`` itself, but the ``modules`` override table accepts any
logger name — including third-party ones — so callers can adjust their
levels when needed.

Configuration (in ``config.yaml``)
-----------------------------------
::

    logging:
      level: WARNING   # global level for all ai_cli.* loggers (default: WARNING)
      modules:         # per-module overrides (module name → level string)
        ai_cli.tools: DEBUG
        ai_cli.core.llm_client: INFO

Module names follow Python's logger hierarchy, so ``ai_cli.tools`` covers
``ai_cli.tools.read_file``, ``ai_cli.tools.write_file``, etc.

Log format
----------
Each line in ``session.log`` is a JSON object::

    {"ts": "2026-03-25T12:08:11.628Z", "level": "INFO",
     "logger": "ai_cli.core.session_manager", "msg": "Session created: …"}

``exc`` and ``stack`` fields are added when an exception or stack trace is
present.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_cli.core.config_manager import ConfigManager

# Logger hierarchy root for this package.
_AI_CLI_ROOT = "ai_cli"

# Default level applied to the ai_cli root logger.
_DEFAULT_LEVEL = "WARNING"

# Logger names explicitly overridden by the most recent setup_logging() call.
# Used to reset third-party (non-ai_cli.*) overrides that are absent from a
# subsequent call's modules table.
_last_module_overrides: set[str] = set()


class JsonlFormatter(logging.Formatter):
    """Format log records as compact single-line JSON objects (JSONL)."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            entry["stack"] = self.formatStack(record.stack_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(config: ConfigManager, session_dir: Path) -> None:
    """Configure JSONL file logging for the ai_cli package.

    Sets up a single ``FileHandler`` on the ``ai_cli`` root logger that
    writes structured JSONL to ``<session_dir>/session.log``.  The root
    (third-party) logger is left untouched so external libraries stay quiet.

    **Startup logging gap:** This function must be called after the session
    directory is known (i.e. after session selection).  Log records emitted
    during earlier startup phases — workspace discovery, config loading, LLM
    client construction — are therefore not captured in ``session.log``.

    Parameters
    ----------
    config:
        Application config.  Reads the ``logging`` section for level and
        per-module overrides.
    session_dir:
        Path to the current session directory.  The log file is created
        here as ``session.log``.

    Safe to call multiple times — if the new *session_dir* matches the
    existing handler it is kept as-is; if it differs, the old handler is
    closed and replaced so no duplicate output or leaked file handles occur.
    """
    log_cfg = config.get("logging", {})
    if not isinstance(log_cfg, dict):
        log_cfg = {}

    level = _parse_level(log_cfg.get("level", _DEFAULT_LEVEL))

    ai_cli_logger = logging.getLogger(_AI_CLI_ROOT)
    ai_cli_logger.setLevel(level)
    # Prevent records from bubbling to the root logger (which may write to
    # stderr and produce duplicate output).
    ai_cli_logger.propagate = False

    log_file = session_dir / "session.log"
    log_file_resolved = log_file.resolve()

    # Close and remove any FileHandlers that target a *different* log file so
    # that a call with a new session_dir replaces the old handler rather than
    # accumulating duplicates and leaking an open file handle.
    for h in list(ai_cli_logger.handlers):
        if (
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() != log_file_resolved
        ):
            h.close()
            ai_cli_logger.removeHandler(h)

    # Add a handler for the current log file if one doesn't already exist.
    has_handler = any(
        isinstance(h, logging.FileHandler)
        and Path(h.baseFilename).resolve() == log_file_resolved
        for h in ai_cli_logger.handlers
    )
    if not has_handler:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # let logger-level do the filtering
        file_handler.setFormatter(JsonlFormatter())
        ai_cli_logger.addHandler(file_handler)

    # Determine the new set of explicitly overridden logger names.
    modules_raw = log_cfg.get("modules", {})
    modules: dict[str, object] = (
        {k: v for k, v in modules_raw.items() if isinstance(k, str)}
        if isinstance(modules_raw, dict)
        else {}
    )
    new_overrides = set(modules.keys())

    # Reset stale overrides so that names absent from the new table return to
    # their inherited level.  This covers both ai_cli.* and third-party loggers
    # that were overridden by a previous call but are no longer in the new table.
    global _last_module_overrides
    for name in _last_module_overrides - new_overrides:
        logging.getLogger(name).setLevel(logging.NOTSET)
    # Also sweep all existing ai_cli.* child loggers so that loggers that were
    # set by a previous call (but not tracked because they were registered after
    # that call) are reset too.
    for name, child in list(logging.root.manager.loggerDict.items()):
        if (
            name.startswith(_AI_CLI_ROOT + ".")
            and isinstance(child, logging.Logger)
            and name not in new_overrides
        ):
            child.setLevel(logging.NOTSET)

    # Apply per-module level overrides.
    for module_name, module_level_name in modules.items():
        logging.getLogger(module_name).setLevel(_parse_level(module_level_name))

    _last_module_overrides = new_overrides


def _parse_level(name: object) -> int:
    """Convert a level name to a :mod:`logging` level int.

    Returns ``logging.WARNING`` for any unrecognised or non-string input so
    a bad config value never silently enables verbose output.
    """
    if not isinstance(name, str):
        return logging.WARNING
    level = getattr(logging, name.upper(), None)
    if not isinstance(level, int):
        return logging.WARNING
    return level
