"""Tests for arc/scheduler/store.py"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import pytest_asyncio

from arc.scheduler.job import Job
from arc.scheduler.store import SchedulerStore


@pytest.fixture
def tmp_store(tmp_path):
    return SchedulerStore(db_path=tmp_path / "test_scheduler.db")


@pytest.mark.asyncio
class TestSchedulerStore:
    async def test_initialize_creates_table(self, tmp_store):
        await tmp_store.initialize()
        jobs = await tmp_store.get_all()
        assert isinstance(jobs, list)
        await tmp_store.close()

    async def test_save_and_get_all(self, tmp_store):
        await tmp_store.initialize()
        job = Job(name="daily", prompt="say hi", trigger={"type": "cron", "expression": "0 9 * * *"})
        await tmp_store.save(job)
        jobs = await tmp_store.get_all()
        assert len(jobs) == 1
        assert jobs[0].name == "daily"
        await tmp_store.close()

    async def test_upsert_updates_existing(self, tmp_store):
        await tmp_store.initialize()
        job = Job(name="daily", prompt="say hi", trigger={"type": "cron", "expression": "0 9 * * *"})
        await tmp_store.save(job)
        job.prompt = "say hello"
        await tmp_store.save(job)
        jobs = await tmp_store.get_all()
        assert len(jobs) == 1
        assert jobs[0].prompt == "say hello"
        await tmp_store.close()

    async def test_get_by_name(self, tmp_store):
        await tmp_store.initialize()
        job = Job(name="hourly", prompt="check stuff", trigger={"type": "interval", "seconds": 3600})
        await tmp_store.save(job)
        found = await tmp_store.get_by_name("hourly")
        assert found is not None
        assert found.id == job.id
        await tmp_store.close()

    async def test_get_by_name_missing_returns_none(self, tmp_store):
        await tmp_store.initialize()
        result = await tmp_store.get_by_name("ghost")
        assert result is None
        await tmp_store.close()

    async def test_delete(self, tmp_store):
        await tmp_store.initialize()
        job = Job(name="del_me", prompt="x", trigger={"type": "interval", "seconds": 60})
        await tmp_store.save(job)
        deleted = await tmp_store.delete(job.id)
        assert deleted is True
        assert await tmp_store.get_by_name("del_me") is None
        await tmp_store.close()

    async def test_delete_missing_returns_false(self, tmp_store):
        await tmp_store.initialize()
        result = await tmp_store.delete("nonexistent_id")
        assert result is False
        await tmp_store.close()

    async def test_get_due_jobs(self, tmp_store):
        await tmp_store.initialize()
        now = int(time.time())
        due_job = Job(
            name="due",
            prompt="fire",
            trigger={"type": "interval", "seconds": 60},
            next_run=now - 10,
        )
        future_job = Job(
            name="future",
            prompt="wait",
            trigger={"type": "interval", "seconds": 60},
            next_run=now + 9999,
        )
        await tmp_store.save(due_job)
        await tmp_store.save(future_job)

        due = await tmp_store.get_due_jobs(now=float(now))
        names = [j.name for j in due]
        assert "due" in names
        assert "future" not in names
        await tmp_store.close()

    async def test_update_after_run(self, tmp_store):
        await tmp_store.initialize()
        now = int(time.time())
        job = Job(
            name="runner",
            prompt="run",
            trigger={"type": "interval", "seconds": 300},
            next_run=now - 5,
        )
        await tmp_store.save(job)
        new_next = now + 300
        await tmp_store.update_after_run(job.id, next_run=new_next, last_run=now)
        updated = await tmp_store.get_by_name("runner")
        assert updated is not None
        assert updated.next_run == new_next
        assert updated.last_run == now
        assert updated.active is True
        await tmp_store.close()

    async def test_update_after_run_zero_deactivates(self, tmp_store):
        await tmp_store.initialize()
        job = Job(name="once", prompt="x", trigger={"type": "oneshot", "at": int(time.time()) + 60})
        await tmp_store.save(job)
        await tmp_store.update_after_run(job.id, next_run=0)
        updated = await tmp_store.get_by_name("once")
        assert updated is not None
        assert updated.active is False
        await tmp_store.close()

    async def test_get_all_active_only(self, tmp_store):
        await tmp_store.initialize()
        j1 = Job(name="active_job", prompt="a", trigger={"type": "interval", "seconds": 60}, active=True)
        j2 = Job(name="inactive_job", prompt="b", trigger={"type": "interval", "seconds": 60}, active=False)
        await tmp_store.save(j1)
        await tmp_store.save(j2)
        active = await tmp_store.get_all(active_only=True)
        names = [j.name for j in active]
        assert "active_job" in names
        assert "inactive_job" not in names
        await tmp_store.close()
