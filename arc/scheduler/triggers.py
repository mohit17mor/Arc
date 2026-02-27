"""
Trigger implementations — compute the next fire time for a job.

Usage:
    trigger = make_trigger({"type": "cron", "expression": "0 9 * * *"})
    next_ts = trigger.next_fire_time(last_run=0, now=time.time())
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod


class Trigger(ABC):
    """Computes the next fire timestamp for a job."""

    @abstractmethod
    def next_fire_time(self, last_run: int, now: float | None = None) -> int:
        """
        Return the next unix timestamp at which this job should fire.

        Args:
            last_run: Unix timestamp of last execution (0 = never run).
            now:      Current time (defaults to time.time()).

        Returns:
            Unix timestamp, or 0 if the trigger has expired (one-shot, past).
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description, e.g. 'every day at 09:00'."""
        ...


class CronTrigger(Trigger):
    """
    Fires on a cron schedule.

    expression: standard 5-field cron string, e.g. "0 9 * * 1-5"

    Requires the `croniter` package.
    """

    def __init__(self, expression: str) -> None:
        self._expression = expression

    def next_fire_time(self, last_run: int, now: float | None = None) -> int:
        from croniter import croniter
        base = float(last_run) if last_run > 0 else (now or time.time())
        it = croniter(self._expression, base)
        return int(it.get_next(float))

    @property
    def description(self) -> str:
        try:
            from croniter import croniter
            return f"cron({self._expression})"
        except ImportError:
            return f"cron({self._expression})"


class IntervalTrigger(Trigger):
    """
    Fires every N seconds.

    On first run (last_run=0) fires immediately (next_fire = now).
    """

    def __init__(self, seconds: int) -> None:
        if seconds < 1:
            raise ValueError("Interval must be at least 1 second")
        self._seconds = seconds

    def next_fire_time(self, last_run: int, now: float | None = None) -> int:
        t = now or time.time()
        if last_run == 0:
            return int(t)  # fire immediately on first run
        return last_run + self._seconds

    @property
    def description(self) -> str:
        s = self._seconds
        if s % 3600 == 0:
            return f"every {s // 3600}h"
        if s % 60 == 0:
            return f"every {s // 60}m"
        return f"every {s}s"


class OneshotTrigger(Trigger):
    """
    Fires once at a specific unix timestamp, then deactivates the job.
    Returns 0 after the scheduled time has passed.
    """

    def __init__(self, at: int) -> None:
        self._at = at

    def next_fire_time(self, last_run: int, now: float | None = None) -> int:
        t = now or time.time()
        if last_run > 0 or t > self._at:
            return 0  # already fired or past due — deactivate
        return self._at

    @property
    def description(self) -> str:
        import datetime
        dt = datetime.datetime.fromtimestamp(self._at).strftime("%Y-%m-%d %H:%M")
        return f"once at {dt}"


def make_trigger(trigger_dict: dict) -> Trigger:
    """
    Build a Trigger from the serialised dict stored in a Job.

    Raises ValueError for unknown trigger types.
    """
    t = trigger_dict.get("type", "")
    if t == "cron":
        return CronTrigger(trigger_dict["expression"])
    elif t == "interval":
        return IntervalTrigger(int(trigger_dict["seconds"]))
    elif t == "oneshot":
        return OneshotTrigger(int(trigger_dict["at"]))
    else:
        raise ValueError(f"Unknown trigger type: {t!r}")
