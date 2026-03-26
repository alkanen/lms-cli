"""Tests for ai_cli.utils.logging_utils."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

from ai_cli.utils.logging_utils import JsonlFormatter, _parse_level, setup_logging

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(level: str = "DEBUG", modules: dict | None = None) -> MagicMock:
    log_cfg: dict = {"level": level}
    if modules is not None:
        log_cfg["modules"] = modules

    cfg = MagicMock()
    cfg.get.side_effect = lambda key, default=None: (
        log_cfg if key == "logging" else default
    )
    return cfg


# ---------------------------------------------------------------------------
# _parse_level
# ---------------------------------------------------------------------------


class TestParseLevel:
    def test_debug(self):
        assert _parse_level("debug") == logging.DEBUG

    def test_info_uppercase(self):
        assert _parse_level("INFO") == logging.INFO

    def test_warning_mixed_case(self):
        assert _parse_level("Warning") == logging.WARNING

    def test_error(self):
        assert _parse_level("ERROR") == logging.ERROR

    def test_unknown_string_falls_back_to_warning(self):
        assert _parse_level("VERBOSE") == logging.WARNING

    def test_non_string_falls_back_to_warning(self):
        assert _parse_level(42) == logging.WARNING
        assert _parse_level(None) == logging.WARNING
        assert _parse_level([]) == logging.WARNING


# ---------------------------------------------------------------------------
# JsonlFormatter
# ---------------------------------------------------------------------------


class TestJsonlFormatter:
    def _record(
        self, msg: str, level: int = logging.INFO, name: str = "ai_cli.test"
    ) -> logging.LogRecord:
        record = logging.LogRecord(
            name=name,
            level=level,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        return record

    def test_output_is_valid_json(self):
        fmt = JsonlFormatter()
        line = fmt.format(self._record("hello"))
        data = json.loads(line)
        assert isinstance(data, dict)

    def test_required_fields_present(self):
        fmt = JsonlFormatter()
        data = json.loads(fmt.format(self._record("hello")))
        assert "ts" in data
        assert "level" in data
        assert "logger" in data
        assert "msg" in data

    def test_msg_content(self):
        fmt = JsonlFormatter()
        data = json.loads(fmt.format(self._record("test message")))
        assert data["msg"] == "test message"

    def test_level_name(self):
        fmt = JsonlFormatter()
        data = json.loads(fmt.format(self._record("x", level=logging.WARNING)))
        assert data["level"] == "WARNING"

    def test_logger_name(self):
        fmt = JsonlFormatter()
        data = json.loads(fmt.format(self._record("x", name="ai_cli.core.repl")))
        assert data["logger"] == "ai_cli.core.repl"

    def test_timestamp_ends_with_Z(self):
        fmt = JsonlFormatter()
        data = json.loads(fmt.format(self._record("x")))
        assert data["ts"].endswith("Z")

    def test_no_exc_field_without_exception(self):
        fmt = JsonlFormatter()
        data = json.loads(fmt.format(self._record("x")))
        assert "exc" not in data

    def test_exc_field_with_exception(self):
        fmt = JsonlFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
        record = self._record("error occurred")
        record.exc_info = exc_info
        data = json.loads(fmt.format(record))
        assert "exc" in data
        assert "ValueError" in data["exc"]

    def test_output_is_single_line(self):
        fmt = JsonlFormatter()
        line = fmt.format(self._record("single line message"))
        assert "\n" not in line

    def test_unicode_preserved(self):
        fmt = JsonlFormatter()
        data = json.loads(fmt.format(self._record("héllo wörld")))
        assert data["msg"] == "héllo wörld"


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def teardown_method(self, _method: object) -> None:
        """Remove any handlers added to the ai_cli logger by tests."""
        ai_cli_logger = logging.getLogger("ai_cli")
        for handler in list(ai_cli_logger.handlers):
            handler.close()
            ai_cli_logger.removeHandler(handler)
        ai_cli_logger.propagate = True
        ai_cli_logger.setLevel(logging.NOTSET)
        # Reset logger levels only for ai_cli (and its children) plus any
        # loggers explicitly overridden by setup_logging.
        import ai_cli.utils.logging_utils as _lu

        overridden = getattr(_lu, "_last_module_overrides", set()) or set()
        for name, child in list(logging.root.manager.loggerDict.items()):
            if not isinstance(child, logging.Logger):
                continue
            if name == "ai_cli" or name.startswith("ai_cli.") or name in overridden:
                child.setLevel(logging.NOTSET)
        # Clear the module-level tracking set so it doesn't bleed across tests.
        _lu._last_module_overrides = set()

    def test_creates_log_file(self, tmp_path: Path):
        cfg = _make_config(level="DEBUG")
        setup_logging(cfg, tmp_path)
        assert (tmp_path / "session.log").exists()

    def test_log_file_receives_records(self, tmp_path: Path):
        cfg = _make_config(level="DEBUG")
        setup_logging(cfg, tmp_path)
        logging.getLogger("ai_cli.test_setup").debug("written by test")
        log_file = tmp_path / "session.log"
        lines = [ln for ln in log_file.read_text().splitlines() if ln.strip()]
        assert any("written by test" in ln for ln in lines)

    def test_log_file_is_jsonl(self, tmp_path: Path):
        cfg = _make_config(level="DEBUG")
        setup_logging(cfg, tmp_path)
        logging.getLogger("ai_cli.test_jsonl").info("jsonl test")
        log_file = tmp_path / "session.log"
        for line in log_file.read_text().splitlines():
            if line.strip():
                json.loads(line)  # must not raise

    def test_ai_cli_level_set(self, tmp_path: Path):
        cfg = _make_config(level="INFO")
        setup_logging(cfg, tmp_path)
        assert logging.getLogger("ai_cli").level == logging.INFO

    def test_debug_messages_filtered_at_info_level(self, tmp_path: Path):
        cfg = _make_config(level="INFO")
        setup_logging(cfg, tmp_path)
        logging.getLogger("ai_cli.test_filter").debug("should not appear")
        log_file = tmp_path / "session.log"
        content = log_file.read_text()
        assert "should not appear" not in content

    def test_per_module_override(self, tmp_path: Path):
        cfg = _make_config(level="WARNING", modules={"ai_cli.test_override": "DEBUG"})
        setup_logging(cfg, tmp_path)
        # ai_cli root is WARNING, but the override module should be DEBUG
        assert logging.getLogger("ai_cli.test_override").level == logging.DEBUG

    def test_repeat_call_clears_stale_third_party_override(self, tmp_path: Path):
        # First call overrides a third-party (non-ai_cli.*) logger.
        cfg1 = _make_config(level="WARNING", modules={"urllib3": "DEBUG"})
        setup_logging(cfg1, tmp_path)
        assert logging.getLogger("urllib3").level == logging.DEBUG

        # Second call omits that module — the third-party level must be cleared.
        cfg2 = _make_config(level="WARNING")
        setup_logging(cfg2, tmp_path)
        assert logging.getLogger("urllib3").level == logging.NOTSET

    def test_repeat_call_clears_stale_module_overrides(self, tmp_path: Path):
        # First call sets ai_cli.test_stale to DEBUG.
        cfg1 = _make_config(level="WARNING", modules={"ai_cli.test_stale": "DEBUG"})
        setup_logging(cfg1, tmp_path)
        assert logging.getLogger("ai_cli.test_stale").level == logging.DEBUG

        # Second call omits that module — the override must be cleared.
        cfg2 = _make_config(level="WARNING")
        setup_logging(cfg2, tmp_path)
        assert logging.getLogger("ai_cli.test_stale").level == logging.NOTSET

    def test_new_session_dir_replaces_old_handler(self, tmp_path: Path):
        dir1 = tmp_path / "session1"
        dir2 = tmp_path / "session2"
        dir1.mkdir()
        dir2.mkdir()
        cfg = _make_config(level="DEBUG")

        setup_logging(cfg, dir1)
        ai_cli_logger = logging.getLogger("ai_cli")
        assert any(
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() == (dir1 / "session.log").resolve()
            for h in ai_cli_logger.handlers
        )

        setup_logging(cfg, dir2)
        handlers = [
            h for h in ai_cli_logger.handlers if isinstance(h, logging.FileHandler)
        ]
        # Only one FileHandler should remain, pointing to the new path.
        assert len(handlers) == 1
        assert (
            Path(handlers[0].baseFilename).resolve() == (dir2 / "session.log").resolve()
        )

    def test_no_duplicate_handler_on_repeat_call(self, tmp_path: Path):
        cfg = _make_config(level="DEBUG")
        setup_logging(cfg, tmp_path)
        setup_logging(cfg, tmp_path)
        ai_cli_logger = logging.getLogger("ai_cli")
        file_handlers = [
            h for h in ai_cli_logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

    def test_propagation_disabled(self, tmp_path: Path):
        cfg = _make_config(level="DEBUG")
        setup_logging(cfg, tmp_path)
        assert logging.getLogger("ai_cli").propagate is False

    def test_invalid_level_string_falls_back_to_warning(self, tmp_path: Path):
        cfg = _make_config(level="NOTAREAL_LEVEL")
        setup_logging(cfg, tmp_path)
        assert logging.getLogger("ai_cli").level == logging.WARNING

    def test_non_dict_logging_config_is_ignored(self, tmp_path: Path):
        cfg = MagicMock()
        cfg.get.side_effect = lambda key, default=None: (
            "bad_value" if key == "logging" else default
        )
        setup_logging(cfg, tmp_path)
        # Should not raise; falls back to defaults
        assert (tmp_path / "session.log").exists()
