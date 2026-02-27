"""Tests for AgentRegistry."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.agent.registry import AgentRegistry


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_mock_entry(name: str):
    """Return (mock_loop, mock_platform, asyncio.Task) for an expert entry."""
    loop_mock = MagicMock()
    platform_mock = AsyncMock()
    platform_mock.send_message = AsyncMock(return_value=f"response from {name}")

    async def _noop():
        await asyncio.sleep(100)  # stays alive until cancelled

    task = asyncio.create_task(_noop())
    return loop_mock, platform_mock, task


# ---------------------------------------------------------------------------
# Expert management
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_and_get_expert():
    reg = AgentRegistry()
    loop, platform, task = make_mock_entry("research")

    reg.register_expert("research", loop, platform, task, specialty="web research")

    entry = reg.get_expert("research")
    assert entry is not None
    assert entry.name == "research"
    assert entry.specialty == "web research"

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_get_expert_missing_returns_none():
    reg = AgentRegistry()
    assert reg.get_expert("nonexistent") is None


@pytest.mark.asyncio
async def test_has_expert():
    reg = AgentRegistry()
    loop, platform, task = make_mock_entry("code")
    reg.register_expert("code", loop, platform, task)

    assert reg.has_expert("code")
    assert not reg.has_expert("other")

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_list_experts():
    reg = AgentRegistry()
    for name in ["alpha", "beta", "gamma"]:
        loop, platform, task = make_mock_entry(name)
        reg.register_expert(name, loop, platform, task)

    names = [e.name for e in reg.list_experts()]
    assert set(names) == {"alpha", "beta", "gamma"}

    await reg.shutdown_all()


@pytest.mark.asyncio
async def test_remove_expert():
    reg = AgentRegistry()
    loop, platform, task = make_mock_entry("temp")
    reg.register_expert("temp", loop, platform, task)

    removed = await reg.remove_expert("temp")
    assert removed
    assert not reg.has_expert("temp")


@pytest.mark.asyncio
async def test_remove_nonexistent_expert_returns_false():
    reg = AgentRegistry()
    assert not await reg.remove_expert("ghost")


@pytest.mark.asyncio
async def test_send_to_expert():
    reg = AgentRegistry()
    loop, platform, task = make_mock_entry("writer")
    reg.register_expert("writer", loop, platform, task)

    response = await reg.send_to_expert("writer", "write a poem")
    assert response == "response from writer"

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_send_to_missing_expert_returns_none():
    reg = AgentRegistry()
    result = await reg.send_to_expert("nobody", "hello")
    assert result is None


# ---------------------------------------------------------------------------
# Worker task management
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_and_list_workers():
    reg = AgentRegistry()

    async def _work():
        await asyncio.sleep(100)

    t1 = asyncio.create_task(_work())
    t2 = asyncio.create_task(_work())
    reg.register_worker("task-1", t1)
    reg.register_worker("task-2", t2)

    assert set(reg.list_workers()) == {"task-1", "task-2"}

    t1.cancel()
    t2.cancel()
    await asyncio.gather(t1, t2, return_exceptions=True)


@pytest.mark.asyncio
async def test_worker_auto_removes_on_completion():
    reg = AgentRegistry()

    async def _quick():
        return "done"

    task = asyncio.create_task(_quick())
    reg.register_worker("quick", task)
    await task  # let it complete naturally

    # give the done_callback a chance to fire
    await asyncio.sleep(0)
    assert "quick" not in reg.list_workers()


@pytest.mark.asyncio
async def test_cancel_worker():
    reg = AgentRegistry()

    async def _long():
        await asyncio.sleep(100)

    task = asyncio.create_task(_long())
    reg.register_worker("long-job", task)

    cancelled = reg.cancel_worker("long-job")
    assert cancelled
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_missing_worker_returns_false():
    reg = AgentRegistry()
    assert not reg.cancel_worker("ghost")


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_shutdown_all_cancels_workers():
    reg = AgentRegistry()

    async def _infinite():
        await asyncio.sleep(1000)

    tasks = [asyncio.create_task(_infinite()) for _ in range(3)]
    for i, t in enumerate(tasks):
        reg.register_worker(f"w{i}", t)

    await reg.shutdown_all()

    for t in tasks:
        assert t.done()


@pytest.mark.asyncio
async def test_shutdown_all_stops_experts():
    reg = AgentRegistry()

    async def _infinite():
        await asyncio.sleep(1000)

    for name in ["e1", "e2"]:
        loop, platform, task = make_mock_entry(name)
        # Use real task so it can be cancelled
        real_task = asyncio.create_task(_infinite())
        reg.register_expert(name, loop, platform, real_task)

    await reg.shutdown_all()
    assert reg.list_experts() == []


@pytest.mark.asyncio
async def test_shutdown_all_idempotent():
    """Calling shutdown_all twice should not raise."""
    reg = AgentRegistry()
    await reg.shutdown_all()
    await reg.shutdown_all()  # should not raise
