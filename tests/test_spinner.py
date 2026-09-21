"""Tests for ai_cli.utils.spinner."""

import io
import time

from ai_cli.utils.spinner import (
    _ASCII_FRAMES,
    _BRAILLE_FRAMES,
    Spinner,
    _pick_frames,
    _stream_is_tty,
)


class FakeTTY:
    """A writable text stream that claims to be a terminal.

    Not an ``io.StringIO`` subclass: ``encoding`` is read-only on the real
    thing, and these tests need to vary it.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self._buf = io.StringIO()
        self.encoding = encoding
        self.closed = False

    def write(self, text: str) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return self._buf.write(text)

    def flush(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed file")

    def isatty(self) -> bool:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return True

    def close(self) -> None:
        self.closed = True

    def getvalue(self) -> str:
        return self._buf.getvalue()


def _run_briefly(spinner: Spinner, frames: int = 3) -> None:
    """Start the spinner, let it draw a few frames, then stop it."""
    spinner.start()
    time.sleep(spinner._interval * frames)
    spinner.stop()


class TestEnablement:
    def test_disabled_when_stream_is_not_a_tty(self):
        stream = io.StringIO()
        spinner = Spinner(stream=stream, interval=0.01)
        assert spinner.enabled is False
        _run_briefly(spinner)
        assert stream.getvalue() == ""

    def test_enabled_when_stream_is_a_tty(self):
        spinner = Spinner(stream=FakeTTY(), interval=0.01)
        assert spinner.enabled is True

    def test_enabled_can_be_forced_off_for_a_tty(self):
        stream = FakeTTY()
        spinner = Spinner(stream=stream, interval=0.01, enabled=False)
        assert spinner.enabled is False
        _run_briefly(spinner)
        assert stream.getvalue() == ""

    def test_enabled_can_be_forced_on_for_a_non_tty(self):
        stream = io.StringIO()
        spinner = Spinner(stream=stream, interval=0.01, enabled=True)
        _run_briefly(spinner)
        assert stream.getvalue() != ""

    def test_stream_without_isatty_counts_as_not_a_tty(self):
        class NoIsatty:
            pass

        assert _stream_is_tty(NoIsatty()) is False

    def test_closed_stream_counts_as_not_a_tty(self):
        stream = io.StringIO()
        stream.close()
        assert _stream_is_tty(stream) is False


class TestDrawing:
    def test_draws_frames_and_erases_on_stop(self):
        stream = FakeTTY()
        spinner = Spinner(stream=stream, interval=0.01)
        _run_briefly(spinner)
        output = stream.getvalue()

        assert any(frame in output for frame in _BRAILLE_FRAMES)
        # Ends erased: carriage return, blanks, carriage return.
        assert output.endswith("\r  \r")
        # Every write starts with \r, so nothing is left on the line.
        assert output.startswith("\r")

    def test_first_frame_is_drawn_immediately(self):
        # A slow interval must not delay the first frame.
        stream = FakeTTY()
        spinner = Spinner(stream=stream, interval=10.0)
        spinner.start()
        deadline = time.monotonic() + 2.0
        while not stream.getvalue() and time.monotonic() < deadline:
            time.sleep(0.01)
        drawn = stream.getvalue()
        spinner.stop()
        assert drawn != ""

    def test_thread_is_not_running_after_stop(self):
        spinner = Spinner(stream=FakeTTY(), interval=0.01)
        spinner.start()
        thread = spinner._thread
        assert thread is not None and thread.is_alive()
        spinner.stop()
        assert spinner._thread is None
        assert not thread.is_alive()

    def test_nothing_is_written_after_stop_returns(self):
        """stop() joins before erasing, so no frame can land after the erase."""
        stream = FakeTTY()
        spinner = Spinner(stream=stream, interval=0.01)
        _run_briefly(spinner)
        settled = stream.getvalue()
        time.sleep(0.05)  # a live thread would draw several more frames here
        assert stream.getvalue() == settled


class TestIdempotence:
    def test_stop_without_start_is_a_no_op(self):
        stream = FakeTTY()
        Spinner(stream=stream, interval=0.01).stop()
        assert stream.getvalue() == ""

    def test_stop_is_idempotent(self):
        stream = FakeTTY()
        spinner = Spinner(stream=stream, interval=0.01)
        _run_briefly(spinner)
        after_first_stop = stream.getvalue()
        spinner.stop()
        spinner.stop()
        assert stream.getvalue() == after_first_stop

    def test_double_start_runs_a_single_thread(self):
        spinner = Spinner(stream=FakeTTY(), interval=0.01)
        spinner.start()
        thread = spinner._thread
        spinner.start()
        assert spinner._thread is thread
        spinner.stop()

    def test_can_be_restarted_after_stop(self):
        stream = FakeTTY()
        spinner = Spinner(stream=stream, interval=0.01)
        _run_briefly(spinner)
        first = stream.getvalue()
        _run_briefly(spinner)
        assert len(stream.getvalue()) > len(first)


class TestEncodingFallback:
    def test_ascii_stream_falls_back_to_ascii_frames(self):
        assert _pick_frames(FakeTTY(encoding="ascii")) == _ASCII_FRAMES

    def test_utf8_stream_uses_braille_frames(self):
        assert _pick_frames(FakeTTY(encoding="utf-8")) == _BRAILLE_FRAMES

    def test_unknown_encoding_falls_back_to_ascii_frames(self):
        assert _pick_frames(FakeTTY(encoding="not-a-real-codec")) == _ASCII_FRAMES

    def test_stream_without_encoding_falls_back_to_ascii_frames(self):
        stream = io.StringIO()  # StringIO has no .encoding
        assert _pick_frames(stream) == _ASCII_FRAMES

    def test_ascii_frames_are_actually_drawn(self):
        stream = FakeTTY(encoding="ascii")
        spinner = Spinner(stream=stream, interval=0.01)
        _run_briefly(spinner)
        assert any(frame in stream.getvalue() for frame in _ASCII_FRAMES)


class TestRobustness:
    def test_closed_stream_does_not_raise(self):
        stream = FakeTTY()
        spinner = Spinner(stream=stream, interval=0.01, enabled=True)
        spinner.start()
        stream.close()
        time.sleep(0.03)
        spinner.stop()  # must not raise

    def test_broken_pipe_does_not_raise(self):
        class BrokenPipe(FakeTTY):
            def write(self, _text):
                raise BrokenPipeError("downstream is gone")

        spinner = Spinner(stream=BrokenPipe(), interval=0.01)
        _run_briefly(spinner)  # must not raise

    def test_context_manager_stops_on_exit(self):
        stream = FakeTTY()
        spinner = Spinner(stream=stream, interval=0.01)
        with spinner:
            assert spinner._thread is not None
        assert spinner._thread is None
        assert stream.getvalue().endswith("\r  \r")

    def test_context_manager_stops_on_exception(self):
        spinner = Spinner(stream=FakeTTY(), interval=0.01)
        try:
            with spinner:
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert spinner._thread is None
