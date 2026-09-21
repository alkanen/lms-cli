"""
spinner.py — minimal progress indicator for non-interactive commands.

One-shot commands such as ``--ask`` stream their answer to **stdout**, which is
routinely captured by another program.  The spinner therefore writes to
**stderr** only, and only when stderr is a terminal, so redirected output,
pipelines and captured stdout stay byte-for-byte clean.

Usage::

    spinner = Spinner()
    spinner.start()
    try:
        ...                 # slow work
        spinner.stop()      # before the first byte of real output
        ...
    finally:
        spinner.stop()      # idempotent

The animation runs on a daemon thread because the work it covers (a blocking
network read) never yields control to the main thread.  :meth:`stop` joins that
thread *before* erasing, so no stray frame can land after the erase — and
because the caller stops the spinner before writing, nothing it draws can
overwrite real output on a shared terminal.
"""

from __future__ import annotations

import sys
import threading
from types import TracebackType
from typing import IO

# Preferred frames, and the fallback for terminals that cannot encode them.
_BRAILLE_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_ASCII_FRAMES = "|/-\\"

# Characters written per frame ("<frame><space>"), and so the width to erase.
_WIDTH = 2

# Seconds to wait for the animation thread to exit before erasing anyway.
_JOIN_TIMEOUT = 2.0


def _stream_is_tty(stream: IO[str]) -> bool:
    """Return True when *stream* is an interactive terminal.

    Streams that are closed, or that do not implement ``isatty`` at all (test
    doubles, some captured streams), count as not a terminal.
    """
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _pick_frames(stream: IO[str]) -> str:
    """Return the braille frames when *stream* can encode them, else ASCII."""
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        _BRAILLE_FRAMES.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return _ASCII_FRAMES
    return _BRAILLE_FRAMES


class Spinner:
    """An animated stderr spinner that erases itself when stopped.

    Parameters
    ----------
    stream:
        Where to draw.  Defaults to ``sys.stderr`` — never stdout, which
        belongs to the command's real output.
    interval:
        Seconds between frames.
    enabled:
        Force the spinner on or off.  ``None`` (default) enables it only when
        *stream* is a terminal, which is what keeps redirected stderr clean.
    """

    def __init__(
        self,
        stream: IO[str] | None = None,
        interval: float = 0.1,
        enabled: bool | None = None,
    ) -> None:
        self._stream: IO[str] = sys.stderr if stream is None else stream
        self._interval = interval
        self._enabled = _stream_is_tty(self._stream) if enabled is None else enabled
        self._frames = _pick_frames(self._stream)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        """True when this spinner will actually draw anything."""
        return self._enabled

    def start(self) -> None:
        """Begin animating.  No-op when disabled or already running."""
        if not self._enabled or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop animating and erase the spinner.

        Idempotent, and safe to call when the spinner was never started, so it
        can sit in a ``finally`` and still be called on the fast path.
        """
        thread = self._thread
        if thread is None:
            return
        self._thread = None
        self._stop_event.set()
        thread.join(timeout=_JOIN_TIMEOUT)
        self._write(f"\r{' ' * _WIDTH}\r")

    def __enter__(self) -> Spinner:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spin(self) -> None:
        """Draw a frame, then wait — exiting promptly once stopped."""
        index = 0
        while True:
            self._write(f"\r{self._frames[index % len(self._frames)]} ")
            if self._stop_event.wait(self._interval):
                return
            index += 1

    def _write(self, text: str) -> None:
        """Write *text* to the stream, ignoring a closed or broken stream.

        A progress indicator must never be the reason a command fails.
        """
        try:
            self._stream.write(text)
            self._stream.flush()
        except (OSError, ValueError):
            pass
