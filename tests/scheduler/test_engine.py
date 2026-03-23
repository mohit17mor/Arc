"""Tests for the background scheduler engine."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from arc.core.events import EventType
from arc.scheduler.engine import SchedulerEngine
from arc.scheduler.job import Job


def _make_job(**kwargs) -> Job:
    defaults = {
        "id": "job-1",
        "name": "daily-news",
        "prompt": "Summarize AI news",
        "trigger": {"type": "interval", "seconds": 3600},
        "next_run": 100,
        "last_run": 0,
        "use_tools": False,
    }
    defaults.update(kwargs)
    return Job(**defaults)


class FakeLLM:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        for chunk in self._chunks:
            yield SimpleNamespace(text=chunk)


@pytest.mark.asyncio
class TestSchedulerEngine:
    async def test_run_prompt_collects_chunks_and_uses_guardrails(self):
        llm = FakeLLM(["Hello", "", None, " world  "])
        engine = SchedulerEngine(
            store=AsyncMock(),
            llm=llm,
            agent_factory=Mock(),
            router=AsyncMock(),
        )

        result = await engine._run_prompt("Write a short digest")

        assert result == "Hello world"
        assert len(llm.calls) == 1
        call = llm.calls[0]
        assert call["tools"] is None
        assert call["temperature"] == 0.5
        assert "Do not ask follow-up questions." in call["messages"][0].content
        assert call["messages"][1].content == "Write a short digest"

    async def test_compute_initial_next_runs_only_updates_zero_or_past_jobs(self):
        zero_job = _make_job(id="job-zero", name="zero", next_run=0, last_run=10)
        past_job = _make_job(id="job-past", name="past", next_run=50, last_run=20)
        future_job = _make_job(id="job-future", name="future", next_run=999999, last_run=30)

        store = AsyncMock()
        store.get_all.return_value = [zero_job, past_job, future_job]

        trigger = Mock()
        trigger.next_fire_time.return_value = 1234

        engine = SchedulerEngine(
            store=store,
            llm=AsyncMock(),
            agent_factory=Mock(),
            router=AsyncMock(),
        )

        with patch("arc.scheduler.engine.time.time", return_value=100):
            with patch("arc.scheduler.engine.make_trigger", return_value=trigger):
                await engine._compute_initial_next_runs()

        assert store.update_after_run.await_count == 2
        store.update_after_run.assert_any_await("job-zero", next_run=1234, last_run=10)
        store.update_after_run.assert_any_await("job-past", next_run=1234, last_run=20)

    async def test_tick_creates_tasks_and_registers_workers_for_due_jobs(self):
        job = _make_job(id="job-1", name="digest")
        store = AsyncMock()
        store.get_due_jobs.return_value = [job]
        registry = Mock()

        engine = SchedulerEngine(
            store=store,
            llm=AsyncMock(),
            agent_factory=Mock(),
            router=AsyncMock(),
            agent_registry=registry,
        )
        engine._fire_job = AsyncMock()

        created_tasks = []

        def fake_create_task(coro, name=None):
            task = Mock(name="task")
            created_tasks.append((coro, name, task))
            coro.close()
            return task

        with patch("arc.scheduler.engine.time.time", return_value=100):
            with patch("arc.scheduler.engine.asyncio.create_task", side_effect=fake_create_task):
                await engine._tick()

        assert "job-1" in engine._in_flight
        assert len(created_tasks) == 1
        registry.register_worker.assert_called_once_with("scheduler:digest", created_tasks[0][2])

    async def test_tick_skips_jobs_already_in_flight(self):
        job = _make_job(id="job-1", name="digest")
        store = AsyncMock()
        store.get_due_jobs.return_value = [job]

        engine = SchedulerEngine(
            store=store,
            llm=AsyncMock(),
            agent_factory=Mock(),
            router=AsyncMock(),
        )
        engine._in_flight.add("job-1")

        with patch("arc.scheduler.engine.asyncio.create_task") as create_task:
            await engine._tick()

        create_task.assert_not_called()

    async def test_fire_job_plain_prompt_updates_store_and_routes_notification(self):
        job = _make_job(id="job-plain", name="daily-news")
        store = AsyncMock()
        router = AsyncMock()
        kernel = AsyncMock()
        engine = SchedulerEngine(
            store=store,
            llm=AsyncMock(),
            agent_factory=Mock(),
            router=router,
            kernel=kernel,
        )
        engine._in_flight.add(job.id)
        engine._run_prompt = AsyncMock(return_value="Here is your digest")

        trigger = Mock()
        trigger.next_fire_time.return_value = 777

        with patch("arc.scheduler.engine.make_trigger", return_value=trigger):
            with patch("arc.scheduler.engine.time.time", return_value=200):
                await engine._fire_job(job)

        engine._run_prompt.assert_awaited_once_with(job.prompt)
        store.update_after_run.assert_awaited_once_with(job.id, next_run=777, last_run=200)
        router.route.assert_awaited_once()
        notification = router.route.await_args.args[0]
        assert notification.job_id == job.id
        assert notification.job_name == job.name
        assert notification.content == "Here is your digest"
        assert job.id not in engine._in_flight

        emitted = [call.args[0] for call in kernel.emit.await_args_list]
        assert emitted[0].type == EventType.AGENT_SPAWNED
        assert emitted[0].data["use_tools"] is False
        assert emitted[1].type == EventType.AGENT_TASK_COMPLETE
        assert emitted[1].data["success"] is True

    async def test_fire_job_deletes_oneshot_when_trigger_returns_zero(self):
        job = _make_job(id="job-once", name="one-shot")
        store = AsyncMock()
        router = AsyncMock()
        engine = SchedulerEngine(
            store=store,
            llm=AsyncMock(),
            agent_factory=Mock(),
            router=router,
        )
        engine._in_flight.add(job.id)
        engine._run_prompt = AsyncMock(return_value="done")

        trigger = Mock()
        trigger.next_fire_time.return_value = 0

        with patch("arc.scheduler.engine.make_trigger", return_value=trigger):
            with patch("arc.scheduler.engine.time.time", return_value=300):
                await engine._fire_job(job)

        store.delete.assert_awaited_once_with(job.id)
        store.update_after_run.assert_not_called()
        assert job.id not in engine._in_flight

    async def test_fire_job_tool_mode_turns_runner_error_into_failure_notification(self):
        job = _make_job(id="job-tools", name="tool-job", use_tools=True)
        store = AsyncMock()
        router = AsyncMock()
        kernel = AsyncMock()
        agent_factory = Mock(return_value=Mock())
        engine = SchedulerEngine(
            store=store,
            llm=AsyncMock(),
            agent_factory=agent_factory,
            router=router,
            kernel=kernel,
        )
        engine._in_flight.add(job.id)

        trigger = Mock()
        trigger.next_fire_time.return_value = 999

        with patch("arc.scheduler.engine.make_trigger", return_value=trigger):
            with patch("arc.scheduler.engine.time.time", return_value=400):
                with patch(
                    "arc.agent.runner.run_agent_on_virtual_platform",
                    AsyncMock(return_value=("ignored", "search failed")),
                ):
                    await engine._fire_job(job)

        agent_factory.assert_called_once_with("scheduler:tool-job")
        store.update_after_run.assert_awaited_once_with(job.id, next_run=999, last_run=400)
        notification = router.route.await_args.args[0]
        assert notification.content == "(job failed: search failed)"

        emitted = [call.args[0] for call in kernel.emit.await_args_list]
        assert emitted[-1].type == EventType.AGENT_TASK_COMPLETE
        assert emitted[-1].data["success"] is False

    async def test_start_and_stop_manage_background_task(self):
        store = AsyncMock()
        engine = SchedulerEngine(
            store=store,
            llm=AsyncMock(),
            agent_factory=Mock(),
            router=AsyncMock(),
        )

        started = asyncio.Event()
        released = asyncio.Event()

        async def fake_loop():
            started.set()
            await released.wait()

        engine._compute_initial_next_runs = AsyncMock()
        engine._loop = fake_loop

        await engine.start()
        await asyncio.wait_for(started.wait(), timeout=1)
        assert engine._running is True
        assert engine._task is not None

        released.set()
        await engine.stop()

        engine._compute_initial_next_runs.assert_awaited_once()
        assert engine._running is False
