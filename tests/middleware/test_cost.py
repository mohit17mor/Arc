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
        source="main",
    )

    await tracker.middleware(event, next_handler)

    assert tracker.input_tokens == 100
    assert tracker.output_tokens == 50
    assert tracker.main_total_tokens == 150
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
            source="main",
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
    """CostTracker calculates costs correctly (main + workers)."""
    tracker = CostTracker(
        cost_per_input_token=0.001,
        cost_per_output_token=0.002,
    )

    tracker.input_tokens = 1000
    tracker.output_tokens = 500
    tracker.worker_input_tokens = 200
    tracker.worker_output_tokens = 100

    # Total: (1000+200)*0.001 + (500+100)*0.002 = 1.2 + 1.2 = 2.4
    assert tracker.session_cost == pytest.approx(2.4)


def test_cost_tracker_reset():
    """CostTracker reset clears all counters."""
    tracker = CostTracker()
    tracker.input_tokens = 100
    tracker.output_tokens = 50
    tracker.request_count = 5
    tracker.worker_input_tokens = 200
    tracker.worker_output_tokens = 100
    tracker.worker_request_count = 3
    tracker.turn_input_tokens = 50
    tracker.turn_output_tokens = 25
    tracker.turn_peak_input = 50

    tracker.reset()

    assert tracker.input_tokens == 0
    assert tracker.output_tokens == 0
    assert tracker.request_count == 0
    assert tracker.worker_input_tokens == 0
    assert tracker.worker_output_tokens == 0
    assert tracker.worker_request_count == 0
    assert tracker.turn_input_tokens == 0
    assert tracker.turn_output_tokens == 0
    assert tracker.turn_peak_input == 0


def test_cost_tracker_summary():
    """CostTracker summary returns all stats including worker breakdown."""
    tracker = CostTracker(
        cost_per_input_token=0.001,
        cost_per_output_token=0.002,
    )
    tracker.input_tokens = 100
    tracker.output_tokens = 50
    tracker.request_count = 2
    tracker.worker_input_tokens = 400
    tracker.worker_output_tokens = 200
    tracker.worker_request_count = 5

    summary = tracker.summary()

    assert summary["requests"] == 2
    assert summary["input_tokens"] == 100
    assert summary["output_tokens"] == 50
    assert summary["total_tokens"] == 150  # main only
    assert summary["worker_requests"] == 5
    assert summary["worker_input_tokens"] == 400
    assert summary["worker_output_tokens"] == 200
    assert summary["worker_total_tokens"] == 600
    assert summary["grand_total_tokens"] == 750
    assert summary["context_window"] == 0  # not set in this test
    assert "cost_usd" in summary


# ---------------------------------------------------------------------------
# Worker token separation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_tokens_tracked_separately():
    """Events from worker agents go to worker counters, not main."""
    tracker = CostTracker()

    async def noop(event):
        return event

    # Main agent call
    await tracker.middleware(
        Event(type=EventType.LLM_RESPONSE,
              data={"input_tokens": 100, "output_tokens": 50},
              source="main"),
        noop,
    )

    # Worker call
    await tracker.middleware(
        Event(type=EventType.LLM_RESPONSE,
              data={"input_tokens": 300, "output_tokens": 150},
              source="worker:research_abc123"),
        noop,
    )

    # Main counters
    assert tracker.input_tokens == 100
    assert tracker.output_tokens == 50
    assert tracker.request_count == 1

    # Worker counters
    assert tracker.worker_input_tokens == 300
    assert tracker.worker_output_tokens == 150
    assert tracker.worker_request_count == 1

    # Grand total
    assert tracker.total_tokens == 600


@pytest.mark.asyncio
async def test_scheduler_tokens_go_to_workers():
    """Scheduler sub-agent tokens are counted as worker tokens."""
    tracker = CostTracker()

    async def noop(event):
        return event

    await tracker.middleware(
        Event(type=EventType.LLM_RESPONSE,
              data={"input_tokens": 200, "output_tokens": 100},
              source="scheduler"),
        noop,
    )

    assert tracker.worker_input_tokens == 200
    assert tracker.worker_output_tokens == 100
    assert tracker.input_tokens == 0  # main untouched


# ---------------------------------------------------------------------------
# Per-turn tracking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_turn_tracking():
    """Per-turn counters track only the current turn's main-agent usage."""
    tracker = CostTracker()

    async def noop(event):
        return event

    # Turn 1
    tracker.start_turn()
    await tracker.middleware(
        Event(type=EventType.LLM_RESPONSE,
              data={"input_tokens": 100, "output_tokens": 50},
              source="main"),
        noop,
    )
    await tracker.middleware(
        Event(type=EventType.LLM_RESPONSE,
              data={"input_tokens": 80, "output_tokens": 40},
              source="main"),
        noop,
    )

    assert tracker.turn_input_tokens == 180
    assert tracker.turn_output_tokens == 90
    assert tracker.turn_total_tokens == 270
    assert tracker.turn_request_count == 2

    # Turn 2 — resets
    tracker.start_turn()
    assert tracker.turn_input_tokens == 0
    assert tracker.turn_output_tokens == 0
    assert tracker.turn_request_count == 0

    await tracker.middleware(
        Event(type=EventType.LLM_RESPONSE,
              data={"input_tokens": 50, "output_tokens": 25},
              source="main"),
        noop,
    )

    assert tracker.turn_total_tokens == 75
    assert tracker.turn_request_count == 1

    # Session totals still accumulate
    assert tracker.input_tokens == 230
    assert tracker.output_tokens == 115


@pytest.mark.asyncio
async def test_worker_tokens_not_in_per_turn():
    """Worker tokens do NOT count toward per-turn totals."""
    tracker = CostTracker()

    async def noop(event):
        return event

    tracker.start_turn()
    await tracker.middleware(
        Event(type=EventType.LLM_RESPONSE,
              data={"input_tokens": 100, "output_tokens": 50},
              source="main"),
        noop,
    )
    await tracker.middleware(
        Event(type=EventType.LLM_RESPONSE,
              data={"input_tokens": 500, "output_tokens": 250},
              source="worker:task_abc"),
        noop,
    )

    assert tracker.turn_total_tokens == 150  # main only
    assert tracker.total_tokens == 900  # everything


@pytest.mark.asyncio
async def test_turn_peak_input_tracks_largest_call():
    """turn_peak_input tracks the largest single-call input_tokens in a turn."""
    tracker = CostTracker()

    async def noop(event):
        return event

    tracker.start_turn()

    # First call: 2,000 input
    await tracker.middleware(
        Event(type=EventType.LLM_RESPONSE,
              data={"input_tokens": 2000, "output_tokens": 100},
              source="main"),
        noop,
    )
    assert tracker.turn_peak_input == 2000

    # Second call: 3,500 input (new peak)
    await tracker.middleware(
        Event(type=EventType.LLM_RESPONSE,
              data={"input_tokens": 3500, "output_tokens": 200},
              source="main"),
        noop,
    )
    assert tracker.turn_peak_input == 3500

    # Third call: 3,000 input (no change — lower than peak)
    await tracker.middleware(
        Event(type=EventType.LLM_RESPONSE,
              data={"input_tokens": 3000, "output_tokens": 150},
              source="main"),
        noop,
    )
    assert tracker.turn_peak_input == 3500  # unchanged

    # New turn resets peak
    tracker.start_turn()
    assert tracker.turn_peak_input == 0


def test_context_window_in_summary():
    """Context window appears in summary when set."""
    tracker = CostTracker()
    tracker.context_window = 128_000
    tracker.turn_peak_input = 5_000

    summary = tracker.summary()
    assert summary["context_window"] == 128_000
    assert summary["turn_peak_input"] == 5_000