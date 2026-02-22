"""Tests for the Kernel."""

import pytest
from arc.core.kernel import Kernel
from arc.core.config import ArcConfig
from arc.core.events import Event, EventType


@pytest.mark.asyncio
async def test_kernel_start_stop(kernel: Kernel):
    """Kernel starts and stops cleanly."""
    assert kernel.running is False

    await kernel.start()
    assert kernel.running is True

    await kernel.stop()
    assert kernel.running is False


@pytest.mark.asyncio
async def test_kernel_emits_lifecycle_events(kernel: Kernel):
    """Kernel emits system:start and system:stop events."""
    events = []

    async def handler(event: Event):
        events.append(event.type)

    kernel.on(EventType.SYSTEM_START, handler)
    kernel.on(EventType.SYSTEM_STOP, handler)

    await kernel.start()
    await kernel.stop()

    assert EventType.SYSTEM_START in events
    assert EventType.SYSTEM_STOP in events


@pytest.mark.asyncio
async def test_kernel_register_and_get(kernel: Kernel):
    """Kernel proxies to registry."""
    kernel.register("llm", "test", "test_provider")

    assert kernel.has("llm", "test")
    assert kernel.get("llm", "test") == "test_provider"
    assert kernel.get_all("llm") == ["test_provider"]


@pytest.mark.asyncio
async def test_kernel_event_bus(kernel: Kernel):
    """Kernel proxies to event bus."""
    received = []

    async def handler(event: Event):
        received.append(event)

    kernel.on("test:event", handler)
    await kernel.emit(Event(type="test:event", data={"key": "value"}))

    assert len(received) == 1
    assert received[0].data == {"key": "value"}


@pytest.mark.asyncio
async def test_kernel_middleware(kernel: Kernel):
    """Kernel proxies middleware to event bus."""
    log = []

    async def test_middleware(event, next_handler):
        log.append(f"before:{event.type}")
        result = await next_handler(event)
        log.append(f"after:{event.type}")
        return result

    kernel.use(test_middleware)
    await kernel.emit(Event(type="test:event"))

    assert log == ["before:test:event", "after:test:event"]


@pytest.mark.asyncio
async def test_kernel_double_start(kernel: Kernel):
    """Starting twice is a no-op."""
    events = []

    async def handler(event: Event):
        events.append(event.type)

    kernel.on(EventType.SYSTEM_START, handler)

    await kernel.start()
    await kernel.start()  # should be no-op

    assert events.count(EventType.SYSTEM_START) == 1

    await kernel.stop()


@pytest.mark.asyncio
async def test_kernel_config(kernel: Kernel):
    """Kernel has config accessible."""
    assert kernel.config.llm.default_provider == "ollama"
    assert kernel.config.agent.max_iterations == 25