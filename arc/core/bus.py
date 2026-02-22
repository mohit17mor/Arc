"""
Arc Event Bus — the central nervous system.

Combines two patterns:
1. Observer (pub/sub): Components subscribe to event types
2. Middleware chain: Events pass through middleware before delivery

This is the HEART of the system. Everything communicates through events.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from typing import Any, Awaitable, Callable

from arc.core.events import Event

logger = logging.getLogger(__name__)

# Type aliases
EventHandler = Callable[[Event], Awaitable[None]]
MiddlewareNext = Callable[[Event], Awaitable[Event]]
MiddlewareFunc = Callable[[Event, MiddlewareNext], Awaitable[Event]]


class EventBus:
    """
    Publish/subscribe event bus with middleware pipeline.

    Usage:
        bus = EventBus()

        # Subscribe
        bus.on("agent:thinking", my_handler)
        bus.on("agent:*", my_wildcard_handler)
        bus.on("*", my_catch_all_handler)

        # Add middleware
        bus.use(logging_middleware)
        bus.use(cost_tracking_middleware)

        # Emit
        await bus.emit(Event(type="agent:thinking", data={...}))
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = {}
        self._middleware: list[MiddlewareFunc] = []

    # ━━━ Subscription ━━━

    def on(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to an event type. Supports wildcards: 'agent:*', '*'."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def off(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                h for h in self._subscribers[event_type] if h is not handler
            ]
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

    # ━━━ Middleware ━━━

    def use(self, middleware: MiddlewareFunc) -> None:
        """
        Add middleware to the processing pipeline.

        Middleware signature:
            async def my_middleware(event: Event, next: MiddlewareNext) -> Event:
                # pre-processing
                result = await next(event)  # continue chain
                # post-processing
                return result
        """
        self._middleware.append(middleware)

    # ━━━ Emission ━━━

    async def emit(self, event: Event) -> Event:
        """
        Emit an event through the middleware chain, then to subscribers.

        Middleware executes in registration order.
        Subscribers execute concurrently.
        Returns the (possibly modified) event.
        """
        # Build the middleware chain ending with subscriber dispatch
        chain = self._build_chain()
        return await chain(event)

    def emit_nowait(self, event: Event) -> None:
        """
        Emit an event without waiting for processing.

        Useful for fire-and-forget events (logging, metrics).
        Errors are logged but not raised.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_safe(event))
        except RuntimeError:
            # No running loop — just log and skip
            logger.debug(f"No event loop for nowait emit: {event.type}")

    # ━━━ Internals ━━━

    def _build_chain(self) -> MiddlewareNext:
        """Build the middleware chain ending with subscriber dispatch."""

        async def dispatch(event: Event) -> Event:
            """Final handler — dispatch to all matching subscribers."""
            handlers = self._find_handlers(event.type)
            if handlers:
                results = await asyncio.gather(
                    *(self._call_handler(h, event) for h in handlers),
                    return_exceptions=True,
                )
                # Log any subscriber errors (don't propagate)
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(
                            f"Subscriber error for {event.type}: {result}",
                            exc_info=result,
                        )
            return event

        # Wrap dispatch with middleware (innermost to outermost)
        handler: MiddlewareNext = dispatch
        for mw in reversed(self._middleware):
            next_handler = handler

            # Use default parameter to capture the closure correctly
            async def make_handler(
                event: Event,
                *,
                _mw: MiddlewareFunc = mw,
                _next: MiddlewareNext = next_handler,
            ) -> Event:
                return await _mw(event, _next)

            handler = make_handler

        return handler

    def _find_handlers(self, event_type: str) -> list[EventHandler]:
        """Find all handlers matching an event type, including wildcards."""
        handlers: list[EventHandler] = []

        for pattern, subs in self._subscribers.items():
            if pattern == event_type:
                # Exact match
                handlers.extend(subs)
            elif pattern == "*":
                # Catch-all wildcard
                handlers.extend(subs)
            elif "*" in pattern:
                # Pattern matching (e.g., "agent:*" matches "agent:thinking")
                if fnmatch.fnmatch(event_type, pattern):
                    handlers.extend(subs)

        return handlers

    @staticmethod
    async def _call_handler(handler: EventHandler, event: Event) -> None:
        """Call a handler, catching any exceptions."""
        await handler(event)

    async def _emit_safe(self, event: Event) -> None:
        """Emit with error catching for fire-and-forget."""
        try:
            await self.emit(event)
        except Exception as e:
            logger.error(f"Error in nowait emit for {event.type}: {e}")

    @property
    def subscriber_count(self) -> int:
        """Total number of subscriptions (for debugging)."""
        return sum(len(subs) for subs in self._subscribers.values())