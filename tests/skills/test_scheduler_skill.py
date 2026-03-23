from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from arc.scheduler.job import Job
from arc.skills.builtin.scheduler import SchedulerSkill


@pytest.fixture
def skill() -> SchedulerSkill:
    return SchedulerSkill()


@pytest.fixture
def store():
    return SimpleNamespace(
        save=AsyncMock(),
        get_all=AsyncMock(return_value=[]),
        get_by_name=AsyncMock(return_value=None),
        delete=AsyncMock(),
    )


class TestSchedulerSkillManifest:
    def test_manifest_exposes_expected_tools(self, skill: SchedulerSkill):
        manifest = skill.manifest()

        assert manifest.name == "scheduler"
        assert {tool.name for tool in manifest.tools} == {
            "schedule_job",
            "list_jobs",
            "cancel_job",
        }


@pytest.mark.asyncio
class TestSchedulerSkillExecution:
    async def test_execute_tool_rejects_unknown_tool(self, skill: SchedulerSkill):
        result = await skill.execute_tool("unknown", {})

        assert result.success is False
        assert "Unknown tool" in result.error

    async def test_schedule_job_requires_store(self, skill: SchedulerSkill):
        result = await skill._schedule_job("daily", "check mail", "interval", interval_seconds=60)

        assert result.success is False
        assert "not initialised" in result.error

    async def test_schedule_job_validates_cron_expression(self, skill: SchedulerSkill, store, monkeypatch):
        skill.set_store(store)
        monkeypatch.setitem(
            sys.modules,
            "croniter",
            SimpleNamespace(croniter=SimpleNamespace(is_valid=lambda expr: False)),
        )

        result = await skill._schedule_job("daily", "check mail", "cron", cron_expression="bad cron")

        assert result.success is False
        assert "Invalid cron expression" in result.error

    async def test_schedule_job_validates_interval_seconds(self, skill: SchedulerSkill, store):
        skill.set_store(store)

        result = await skill._schedule_job("daily", "check mail", "interval", interval_seconds=0)

        assert result.success is False
        assert "interval_seconds must be >= 1" in result.error

    async def test_schedule_job_validates_oneshot_fire_at(self, skill: SchedulerSkill, store, monkeypatch):
        skill.set_store(store)
        monkeypatch.setattr("arc.skills.builtin.scheduler.time.time", lambda: 100)

        missing = await skill._schedule_job("daily", "check mail", "oneshot")
        past = await skill._schedule_job("daily", "check mail", "oneshot", fire_at=50)

        assert "fire_at" in missing.error
        assert "future" in past.error

    async def test_schedule_job_rejects_unknown_trigger_type(self, skill: SchedulerSkill, store):
        skill.set_store(store)

        result = await skill._schedule_job("daily", "check mail", "weird")

        assert result.success is False
        assert "Unknown trigger_type" in result.error

    async def test_schedule_job_uses_fire_after_seconds_and_saves_job(self, skill: SchedulerSkill, store, monkeypatch):
        saved: list[Job] = []
        store.save = AsyncMock(side_effect=lambda job: saved.append(job))
        skill.set_store(store)
        monkeypatch.setattr("arc.skills.builtin.scheduler.time.time", lambda: 100)
        monkeypatch.setattr(
            "arc.skills.builtin.scheduler.make_trigger",
            lambda trigger: SimpleNamespace(
                next_fire_time=lambda last_run, now: 200,
                description="once at 2026-03-23 10:00",
            ),
        )
        monkeypatch.setattr(
            "arc.skills.builtin.scheduler.datetime.datetime",
            SimpleNamespace(fromtimestamp=lambda ts: SimpleNamespace(strftime=lambda fmt: "2026-03-23 10:00:00")),
        )

        result = await skill._schedule_job(
            "reminder",
            "check releases",
            "oneshot",
            fire_after_seconds=25,
            use_tools=True,
        )

        assert result.success is True
        assert "Job 'reminder' scheduled" in result.output
        assert "Mode: task (with tools)." in result.output
        assert saved[0].trigger == {"type": "oneshot", "at": 125}
        assert saved[0].next_run == 200
        assert saved[0].use_tools is True

    async def test_list_jobs_requires_store(self, skill: SchedulerSkill):
        result = await skill._list_jobs()

        assert result.success is False
        assert "not initialised" in result.error

    async def test_list_jobs_reports_empty_store(self, skill: SchedulerSkill, store):
        skill.set_store(store)

        result = await skill._list_jobs()

        assert result.success is True
        assert result.output == "No scheduled jobs."

    async def test_list_jobs_formats_active_and_inactive_jobs(self, skill: SchedulerSkill, store, monkeypatch):
        store.get_all = AsyncMock(
            return_value=[
                Job(
                    id="job-1",
                    name="daily",
                    prompt="x" * 90,
                    trigger={"type": "interval", "seconds": 3600},
                    next_run=200,
                    last_run=150,
                    active=True,
                    use_tools=True,
                ),
                Job(
                    id="job-2",
                    name="once",
                    prompt="hello",
                    trigger={"type": "oneshot", "at": 300},
                    next_run=0,
                    last_run=0,
                    active=False,
                    use_tools=False,
                ),
            ]
        )
        skill.set_store(store)
        monkeypatch.setattr(
            "arc.skills.builtin.scheduler.make_trigger",
            lambda trigger: SimpleNamespace(description=f"{trigger['type']} trigger"),
        )
        monkeypatch.setattr(
            "arc.skills.builtin.scheduler.datetime.datetime",
            SimpleNamespace(fromtimestamp=lambda ts: SimpleNamespace(strftime=lambda fmt: f"ts:{ts}")),
        )

        result = await skill._list_jobs()

        assert result.success is True
        assert "[job-1] daily  (active, task+tools)" in result.output
        assert "prompt:  " in result.output
        assert "..." in result.output
        assert "[job-2] once  (inactive, simple)" in result.output
        assert "last:    never" in result.output

    async def test_cancel_job_requires_store(self, skill: SchedulerSkill):
        result = await skill._cancel_job("daily")

        assert result.success is False
        assert "not initialised" in result.error

    async def test_cancel_job_finds_job_by_name(self, skill: SchedulerSkill, store):
        job = Job(id="job-1", name="daily", prompt="hello", trigger={"type": "interval", "seconds": 60})
        store.get_by_name = AsyncMock(return_value=job)
        skill.set_store(store)

        result = await skill._cancel_job("daily")

        assert result.success is True
        store.delete.assert_awaited_once_with("job-1")

    async def test_cancel_job_finds_job_by_id_when_name_lookup_misses(self, skill: SchedulerSkill, store):
        store.get_all = AsyncMock(
            return_value=[
                Job(id="job-1", name="daily", prompt="hello", trigger={"type": "interval", "seconds": 60})
            ]
        )
        skill.set_store(store)

        result = await skill._cancel_job("job-1")

        assert result.success is True
        assert "job-1" in result.output

    async def test_cancel_job_reports_missing_job(self, skill: SchedulerSkill, store):
        skill.set_store(store)

        result = await skill._cancel_job("missing")

        assert result.success is False
        assert "No job found" in result.error
