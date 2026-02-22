"""Tests for cost tracking middleware."""

import pytest
from arc.core.events import Event, EventType
from arc.middleware.cost import CostTracker


@pytest.mark.asyncio
async def test_cost_tracker_tracks_tokens():
    """CostTracker accumulates token counts."""
    tracker = CostTracker()

    async def next_handler(event):
        return event

    # Simulate LLM response event
    event = Event(
        type=EventType.LLM_RESPONSE,
        data={"input_tokens": 100, "output_tokens": 50},
    )

    await tracker.middleware(event, next_handler)

    assert tracker.input_tokens == 100
    assert tracker.output_tokens == 50
    assert tracker.total_tokens == 150
    assert tracker.request_count == 1


@pytest.mark.asyncio
async def test_cost_tracker_multiple_calls():
    """CostTracker accumulates across multiple calls."""
    tracker = CostTracker()

    async def next_handler(event):
        return event

    for i in range(3):
        event = Event(
            type=EventType.LLM_RESPONSE,
            data={"input_tokens": 100, "output_tokens": 50},
        )
        await tracker.middleware(event, next_handler)

    assert tracker.input_tokens == 300
    assert tracker.output_tokens == 150
    assert tracker.request_count == 3


@pytest.mark.asyncio
async def test_cost_tracker_ignores_other_events():
    """CostTracker ignores non-LLM events."""
    tracker = CostTracker()

    async def next_handler(event):
        return event

    event = Event(
        type=EventType.AGENT_THINKING,
        data={"iteration": 1},
    )

    await tracker.middleware(event, next_handler)

    assert tracker.total_tokens == 0
    assert tracker.request_count == 0


@pytest.mark.asyncio
async def test_cost_tracker_cost_calculation():
    """CostTracker calculates costs correctly."""
    tracker = CostTracker(
        cost_per_input_token=0.001,
        cost_per_output_token=0.002,
    )

    tracker.input_tokens = 1000
    tracker.output_tokens = 500

    assert tracker.session_cost == 1000 * 0.001 + 500 * 0.002  # $2.00


def test_cost_tracker_reset():
    """CostTracker reset clears all counters."""
    tracker = CostTracker()
    tracker.input_tokens = 100
    tracker.output_tokens = 50
    tracker.request_count = 5

    tracker.reset()

    assert tracker.input_tokens == 0
    assert tracker.output_tokens == 0
    assert tracker.request_count == 0


def test_cost_tracker_summary():
    """CostTracker summary returns all stats."""
    tracker = CostTracker(
        cost_per_input_token=0.001,
        cost_per_output_token=0.002,
    )
    tracker.input_tokens = 100
    tracker.output_tokens = 50
    tracker.request_count = 2

    summary = tracker.summary()

    assert summary["requests"] == 2
    assert summary["input_tokens"] == 100
    assert summary["output_tokens"] == 50
    assert summary["total_tokens"] == 150
    assert "cost_usd" in summary