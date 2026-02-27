"""Tests for arc/scheduler/triggers.py"""
from __future__ import annotations

import time

import pytest

from arc.scheduler.triggers import (
    CronTrigger,
    IntervalTrigger,
    OneshotTrigger,
    make_trigger,
)


# ── CronTrigger ──────────────────────────────────────────────────────────────

class TestCronTrigger:
    def test_next_fire_time_after_last_run(self):
        # Cron: every minute.  next should be ~60 s after last_run.
        trigger = CronTrigger("* * * * *")
        last = int(time.time()) - 10
        nxt = trigger.next_fire_time(last_run=last)
        assert nxt > last

    def test_next_fire_time_no_last_run_uses_now(self):
        trigger = CronTrigger("* * * * *")
        before = int(time.time())
        nxt = trigger.next_fire_time(last_run=0)
        assert nxt >= before

    def test_description_contains_expression(self):
        trigger = CronTrigger("0 9 * * 1-5")
        assert "0 9 * * 1-5" in trigger.description

    def test_make_trigger_cron(self):
        t = make_trigger({"type": "cron", "expression": "0 0 * * *"})
        assert isinstance(t, CronTrigger)

    def test_explicit_now_parameter(self):
        trigger = CronTrigger("* * * * *")
        fixed_now = 1_700_000_000.0
        nxt = trigger.next_fire_time(last_run=0, now=fixed_now)
        assert nxt > fixed_now


# ── IntervalTrigger ──────────────────────────────────────────────────────────

class TestIntervalTrigger:
    def test_first_run_fires_now(self):
        trigger = IntervalTrigger(seconds=300)
        now = time.time()
        nxt = trigger.next_fire_time(last_run=0, now=now)
        assert nxt == int(now)

    def test_subsequent_run_adds_interval(self):
        trigger = IntervalTrigger(seconds=300)
        last = 1_000_000
        nxt = trigger.next_fire_time(last_run=last)
        assert nxt == last + 300

    def test_description_seconds(self):
        assert "30s" in IntervalTrigger(30).description

    def test_description_minutes(self):
        assert "5m" in IntervalTrigger(300).description

    def test_description_hours(self):
        assert "2h" in IntervalTrigger(7200).description

    def test_invalid_interval_raises(self):
        with pytest.raises(ValueError):
            IntervalTrigger(0)

    def test_make_trigger_interval(self):
        t = make_trigger({"type": "interval", "seconds": "60"})
        assert isinstance(t, IntervalTrigger)


# ── OneshotTrigger ───────────────────────────────────────────────────────────

class TestOneshotTrigger:
    def test_future_oneshot_returns_at(self):
        future = int(time.time()) + 3600
        trigger = OneshotTrigger(at=future)
        nxt = trigger.next_fire_time(last_run=0)
        assert nxt == future

    def test_past_oneshot_returns_zero(self):
        past = int(time.time()) - 3600
        trigger = OneshotTrigger(at=past)
        nxt = trigger.next_fire_time(last_run=0)
        assert nxt == 0

    def test_already_run_returns_zero(self):
        at = int(time.time()) + 3600
        trigger = OneshotTrigger(at=at)
        nxt = trigger.next_fire_time(last_run=int(time.time()))
        assert nxt == 0

    def test_description_contains_datetime(self):
        at = 1_700_000_000
        desc = OneshotTrigger(at=at).description
        assert "once at" in desc

    def test_make_trigger_oneshot(self):
        at = int(time.time()) + 60
        t = make_trigger({"type": "oneshot", "at": str(at)})
        assert isinstance(t, OneshotTrigger)


# ── make_trigger ─────────────────────────────────────────────────────────────

class TestMakeTrigger:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown trigger type"):
            make_trigger({"type": "weekly"})
