"""Tests for EscalationBus."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from arc.core.escalation import EscalationBus
from arc.core.events import EventType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_kernel():
    kernel = MagicMock()
    kernel.emit = AsyncMock()
    return kernel


@pytest.fixture
def bus(mock_kernel):
    return EscalationBus(mock_kernel, timeout=5.0)


# ---------------------------------------------------------------------------
# Basic ask/resolve cycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_manager_emits_escalation_event(bus, mock_kernel):
    async def _resolver():
        await asyncio.sleep(0.05)
        pending = bus.pending
        assert len(pending) == 1
        bus.resolve_escalation(pending[0].escalation_id, "my answer")

    asyncio.create_task(_resolver())
    answer = await bus.ask_manager("worker-1", "What should I do?")

    mock_kernel.emit.assert_called_once()
    event = mock_kernel.emit.call_args[0][0]
    assert event.type == EventType.AGENT_ESCALATION
    assert event.data["from_agent"] == "worker-1"
    assert "What should I do?" in event.data["question"]


@pytest.mark.asyncio
async def test_ask_manager_returns_answer(bus):
    async def _resolver():
        await asyncio.sleep(0.02)
        pending = bus.pending
        bus.resolve_escalation(pending[0].escalation_id, "42")

    asyncio.create_task(_resolver())
    answer = await bus.ask_manager("worker-x", "What is the answer?")
    assert answer == "42"


@pytest.mark.asyncio
async def test_resolve_escalation_returns_true(bus):
    async def _ask():
        await bus.ask_manager("a", "q")

    task = asyncio.create_task(_ask())
    await asyncio.sleep(0.02)  # let ask_manager register pending

    pending = bus.pending
    assert len(pending) == 1
    result = bus.resolve_escalation(pending[0].escalation_id, "answer")
    assert result is True
    await task


@pytest.mark.asyncio
async def test_resolve_nonexistent_returns_false(bus):
    assert bus.resolve_escalation("nonexistent_id", "answer") is False


# ---------------------------------------------------------------------------
# Pending tracking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pending_lists_waiting_escalations(bus):
    assert not bus.has_pending

    # Fire two concurrent asks
    t1 = asyncio.create_task(bus.ask_manager("a1", "question 1"))
    t2 = asyncio.create_task(bus.ask_manager("a2", "question 2"))
    await asyncio.sleep(0.02)

    assert bus.has_pending
    assert len(bus.pending) == 2
    agents = {e.from_agent for e in bus.pending}
    assert agents == {"a1", "a2"}

    # Resolve both
    for req in list(bus.pending):
        bus.resolve_escalation(req.escalation_id, "ok")

    await asyncio.gather(t1, t2)
    assert not bus.has_pending


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_manager_times_out_with_fallback():
    mock_kernel = MagicMock()
    mock_kernel.emit = AsyncMock()

    short_bus = EscalationBus(mock_kernel, timeout=0.1)
    answer = await short_bus.ask_manager("slow-worker", "waiting forever")

    # Should return a fallback string, not raise
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "No answer" in answer or "proceeding" in answer.lower()


# ---------------------------------------------------------------------------
# Multiple resolves / edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_double_resolve_is_safe(bus):
    """Resolving twice should not raise."""
    t = asyncio.create_task(bus.ask_manager("a", "q"))
    await asyncio.sleep(0.02)

    req_id = bus.pending[0].escalation_id
    bus.resolve_escalation(req_id, "first")
    bus.resolve_escalation(req_id, "second")  # should not raise

    await t


@pytest.mark.asyncio
async def test_unique_escalation_ids(bus):
    tasks = []
    for i in range(5):
        tasks.append(asyncio.create_task(bus.ask_manager(f"a{i}", f"q{i}")))
    await asyncio.sleep(0.02)

    ids = [r.escalation_id for r in bus.pending]
    assert len(ids) == len(set(ids))  # all unique

    for req in list(bus.pending):
        bus.resolve_escalation(req.escalation_id, "ok")
    await asyncio.gather(*tasks)
