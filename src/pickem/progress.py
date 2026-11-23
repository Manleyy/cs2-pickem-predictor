"""Tiny dependency-free progress bar (prints to stderr, redraws ~10x/second)."""
from __future__ import annotations

import shutil
import sys
import time


class Progress:
    def __init__(self, total: int, label: str, width: int = 30):
        self.total, self.label, self.width = max(int(total), 1), label, width
        self.n, self.start, self._last = 0, time.time(), 0.0
        if sys.platform == "win32":  # make Windows consoles understand the line-clear code
            import os
            os.system("")
        enc = getattr(sys.stderr, "encoding", None) or "ascii"
        try:
            "█░".encode(enc)
            self.full, self.empty = "█", "░"
        except (UnicodeEncodeError, LookupError):
            self.full, self.empty = "#", "-"
        self._draw()

    def update(self, n: int = 1):
        self.n += n
        now = time.time()
        if now - self._last >= 0.1 or self.n >= self.total:
            self._last = now
            self._draw()

    @staticmethod
    def _num(n: int) -> str:
        """Compact count: 950, 12.3k, 4.1M."""
        if n < 10_000:
            return f"{n:,}"
        if n < 1_000_000:
            return f"{n / 1000:.1f}k"
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")

    @staticmethod
    def _time(sec: float) -> str:
        sec = int(round(sec))
        h, r = divmod(sec, 3600)
        m, s = divmod(r, 60)
        return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")

    def _draw(self):
        frac = min(self.n / self.total, 1.0)
        el = time.time() - self.start
        eta = el / frac - el if frac > 0 else 0
        info = f" {frac:6.1%} {self._num(self.n)}/{self._num(self.total)} ETA {self._time(eta)}"
        # Kept short (<= 72 chars) and fitted to the current terminal width on every
        # redraw: a line wider than the window wraps, and "\r" can then no longer
        # overwrite it. Narrowing the window mid-run can still leave one stale line,
        # because the terminal rewraps text that is already on screen.
        cols = min(72, shutil.get_terminal_size((80, 20)).columns - 1)
        width = max(5, min(self.width, cols - len(self.label) - len(info) - 2))
        filled = int(frac * width)
        line = f"{self.label} {self.full * filled}{self.empty * (width - filled)}{info}"[:cols]
        sys.stderr.write("\r\x1b[2K" + line)  # \x1b[2K clears the whole line first
        sys.stderr.flush()

    def close(self):
        self.n = self.total
        self._draw()
        sys.stderr.write("\n")
        sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
