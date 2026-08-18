"""Lightweight terminal progress reporting for long-running pipeline stages."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TextIO


@dataclass
class ProgressTask:
    """Single progress bar task that updates in-place in TTY sessions."""

    label: str
    total: int
    stream: TextIO
    current: int = 0
    width: int = 28
    _last_render_length: int = 0
    _last_render_time: float = 0.0

    def render(self, note: str = "", *, force: bool = False) -> None:
        """Render the current task state to the terminal."""
        now = time.time()
        if not force and now - self._last_render_time < 0.1 and self.current < self.total:
            return
        total = max(1, self.total)
        ratio = min(1.0, self.current / total)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = ratio * 100.0
        suffix = f" | {note}" if note else ""
        line = f"[{bar}] {self.current:>5}/{self.total:<5} {percent:6.2f}% {self.label}{suffix}"
        padded = line.ljust(self._last_render_length)
        self.stream.write("\r" + padded)
        self.stream.flush()
        self._last_render_length = max(self._last_render_length, len(line))
        self._last_render_time = now
        if self.current >= self.total:
            self.stream.write("\n")
            self.stream.flush()

    def advance(self, amount: int = 1, note: str = "") -> None:
        """Advance the task and refresh the display."""
        self.current = min(self.total, self.current + amount)
        self.render(note=note)

    def refresh(self, note: str = "") -> None:
        """Refresh the task without changing the completed count."""
        self.render(note=note, force=True)

    def complete(self, note: str = "") -> None:
        """Mark the task complete and force a final render."""
        self.current = self.total
        self.render(note=note, force=True)


class ProgressReporter:
    """Stage and progress-bar reporting helper for terminal pipeline runs."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr

    def stage(self, message: str) -> None:
        """Emit a one-line stage message."""
        self.stream.write(f"\n==> {message}\n")
        self.stream.flush()

    def task(self, label: str, total: int) -> ProgressTask:
        """Create and prime a new progress task."""
        task = ProgressTask(label=label, total=total, stream=self.stream)
        task.render(force=True)
        return task
