"""
Arc Kernel — the central coordinator.

Composes the event bus, registry, and config into a single entry point.
The kernel is the only object that components need to interact with
the rest of the system.

This is intentionally small (~100 lines). All intelligence lives
in the subsystems (memory, skills, agent, etc.).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from arc.core.bus import EventBus, EventHandler, MiddlewareFunc
from arc.core.config import ArcConfig
from arc.core.events import Event, EventType
from arc.core.registry import Registry

logger = logging.getLogger(__name__)


class Kernel:
    """
    The Arc Kernel — central nervous system.

    Provides:
    1. Event bus (publish/subscribe + middleware)
    2. Provider registry (service locator / DI)
    3. Configuration
    4. Lifecycle management

    Usage:
        kernel = Kernel()

        # Register providers
        kernel.register("llm", "ollama", ollama_provider)
        kernel.register("skill", "filesystem", fs_skill)

        # Subscribe to events
        kernel.on("agent:thinking", my_handler)

        # Add middleware
        kernel.use(logging_middleware)

        # Lifecycle
        await kernel.start()
        ...
        await kernel.stop()
    """

    def __init__(self, config: ArcConfig | None = None) -> None:
        self.config = config or ArcConfig.load()
        self.bus = EventBus()
        self.registry = Registry()
        self._running = False
        self._background_tasks: list[asyncio.Task] = []

    # ━━━ Registry Shortcuts ━━━

    def register(self, category: str, name: str, provider: Any) -> None:
        """Register a provider with the registry."""
        self.registry.register(category, name, provider)

    def get(self, category: str, name: str | None = None) -> Any:
        """Get a provider from the registry."""
        return self.registry.get(category, name)

    def get_all(self, category: str) -> list[Any]:
        """Get all providers in a category."""
        return self.registry.get_all(category)

    def has(self, category: str, name: str | None = None) -> bool:
        """Check if a provider exists."""
        return self.registry.has(category, name)

    # ━━━ Event Bus Shortcuts ━━━

    def on(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to an event type."""
        self.bus.on(event_type, handler)

    def off(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from an event type."""
        self.bus.off(event_type, handler)

    async def emit(self, event: Event) -> Event:
        """Emit an event through the bus."""
        return await self.bus.emit(event)

    def emit_nowait(self, event: Event) -> None:
        """Emit an event without waiting."""
        self.bus.emit_nowait(event)

    # ━━━ Middleware ━━━

    def use(self, middleware: MiddlewareFunc) -> None:
        """Add middleware to the event processing pipeline."""
        self.bus.use(middleware)

    # ━━━ Lifecycle ━━━

    async def start(self) -> None:
        """Start the kernel. Emits system:start event."""
        if self._running:
            return
        self._running = True
        logger.info("Arc kernel starting")
        await self.emit(Event(type=EventType.SYSTEM_START, source="kernel"))

    async def stop(self) -> None:
        """Stop the kernel. Cancels background tasks, emits system:stop."""
        if not self._running:
            return
        self._running = False
        logger.info("Arc kernel stopping")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        await self.emit(Event(type=EventType.SYSTEM_STOP, source="kernel"))

    @property
    def running(self) -> bool:
        """Whether the kernel is currently running."""
        return self._running

    def spawn(self, coro: Awaitable[Any]) -> asyncio.Task:
        """Spawn a background task tracked by the kernel."""
        task = asyncio.ensure_future(coro)
        self._background_tasks.append(task)
        task.add_done_callback(lambda t: self._background_tasks.remove(t))
        return task