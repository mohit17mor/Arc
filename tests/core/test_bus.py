"""Tests for the Event Bus."""

import asyncio
import pytest
from arc.core.bus import EventBus
from arc.core.events import Event, EventType


@pytest.mark.asyncio
async def test_emit_and_subscribe(bus: EventBus):
    """Basic pub/sub works."""
    received = []

    async def handler(event: Event):
        received.append(event)

    bus.on(EventType.AGENT_THINKING, handler)
    await bus.emit(Event(type=EventType.AGENT_THINKING, data={"x": 1}))

    assert len(received) == 1
    assert received[0].data == {"x": 1}


@pytest.mark.asyncio
async def test_wildcard_subscription(bus: EventBus):
    """Wildcard 'agent:*' matches 'agent:thinking'."""
    received = []

    async def handler(event: Event):
        received.append(event.type)

    bus.on("agent:*", handler)

    await bus.emit(Event(type=EventType.AGENT_THINKING))
    await bus.emit(Event(type=EventType.AGENT_RESPONSE))
    await bus.emit(Event(type=EventType.LLM_REQUEST))  # should NOT match

    assert len(received) == 2
    assert "agent:thinking" in received
    assert "agent:response" in received


@pytest.mark.asyncio
async def test_catch_all_subscription(bus: EventBus):
    """Wildcard '*' matches everything."""
    received = []

    async def handler(event: Event):
        received.append(event.type)

    bus.on("*", handler)

    await bus.emit(Event(type=EventType.AGENT_THINKING))
    await bus.emit(Event(type=EventType.LLM_REQUEST))
    await bus.emit(Event(type=EventType.SKILL_TOOL_CALL))

    assert len(received) == 3


@pytest.mark.asyncio
async def test_multiple_subscribers(bus: EventBus):
    """Multiple subscribers all receive the event."""
    results = {"a": False, "b": False}

    async def handler_a(event: Event):
        results["a"] = True

    async def handler_b(event: Event):
        results["b"] = True

    bus.on(EventType.AGENT_THINKING, handler_a)
    bus.on(EventType.AGENT_THINKING, handler_b)

    await bus.emit(Event(type=EventType.AGENT_THINKING))

    assert results["a"] is True
    assert results["b"] is True


@pytest.mark.asyncio
async def test_unsubscribe(bus: EventBus):
    """off() removes a handler."""
    received = []

    async def handler(event: Event):
        received.append(event)

    bus.on(EventType.AGENT_THINKING, handler)
    await bus.emit(Event(type=EventType.AGENT_THINKING))
    assert len(received) == 1

    bus.off(EventType.AGENT_THINKING, handler)
    await bus.emit(Event(type=EventType.AGENT_THINKING))
    assert len(received) == 1  # no new events


@pytest.mark.asyncio
async def test_middleware_chain(bus: EventBus):
    """Middleware executes in order and can modify events."""
    order = []

    async def mw_first(event, next_handler):
        order.append("first_before")
        event.metadata["first"] = True
        result = await next_handler(event)
        order.append("first_after")
        return result

    async def mw_second(event, next_handler):
        order.append("second_before")
        event.metadata["second"] = True
        result = await next_handler(event)
        order.append("second_after")
        return result

    bus.use(mw_first)
    bus.use(mw_second)

    received_events = []

    async def handler(event):
        received_events.append(event)

    bus.on(EventType.AGENT_THINKING, handler)
    await bus.emit(Event(type=EventType.AGENT_THINKING))

    # Middleware executes in order
    assert order == ["first_before", "second_before", "second_after", "first_after"]

    # Subscriber received modified event
    assert received_events[0].metadata["first"] is True
    assert received_events[0].metadata["second"] is True


@pytest.mark.asyncio
async def test_middleware_can_block(bus: EventBus):
    """Middleware can prevent event delivery by not calling next."""
    received = []

    async def blocking_middleware(event, next_handler):
        if event.data.get("block"):
            return event  # don't call next
        return await next_handler(event)

    bus.use(blocking_middleware)

    async def handler(event):
        received.append(event)

    bus.on(EventType.AGENT_THINKING, handler)

    # This should be blocked
    await bus.emit(Event(type=EventType.AGENT_THINKING, data={"block": True}))
    assert len(received) == 0

    # This should pass through
    await bus.emit(Event(type=EventType.AGENT_THINKING, data={"block": False}))
    assert len(received) == 1


@pytest.mark.asyncio
async def test_subscriber_error_isolated(bus: EventBus):
    """One bad subscriber doesn't break others."""
    results = {"good": False}

    async def bad_handler(event: Event):
        raise ValueError("I'm broken")

    async def good_handler(event: Event):
        results["good"] = True

    bus.on(EventType.AGENT_THINKING, bad_handler)
    bus.on(EventType.AGENT_THINKING, good_handler)

    # Should not raise even though bad_handler throws
    await bus.emit(Event(type=EventType.AGENT_THINKING))

    # Good handler still ran
    assert results["good"] is True


@pytest.mark.asyncio
async def test_no_subscribers(bus: EventBus):
    """Emitting with no subscribers doesn't raise."""
    event = await bus.emit(Event(type="some:random:event"))
    assert event.type == "some:random:event"


@pytest.mark.asyncio
async def test_subscriber_count(bus: EventBus):
    async def handler(e):
        pass

    assert bus.subscriber_count == 0

    bus.on("a", handler)
    bus.on("b", handler)
    bus.on("b", handler)  # duplicate on same type

    assert bus.subscriber_count == 3