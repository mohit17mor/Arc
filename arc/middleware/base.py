"""
Middleware base — already covered by event bus.

This file provides helper utilities for creating middleware.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from arc.core.events import Event

# Type aliases (same as in bus.py, re-exported for convenience)
MiddlewareNext = Callable[[Event], Awaitable[Event]]
MiddlewareFunc = Callable[[Event, MiddlewareNext], Awaitable[Event]]


def create_middleware(
    event_types: list[str] | None = None,
    handler: Callable[[Event], Awaitable[None]] | None = None,
) -> MiddlewareFunc:
    """
    Helper to create simple middleware.

    Args:
        event_types: Only process these event types (None = all)
        handler: Async function to call for matching events

    Usage:
        @create_middleware(event_types=["llm:response"])
        async def cost_logger(event):
            print(f"Tokens: {event.data.get('tokens')}")
    """

    def decorator(func: Callable[[Event], Awaitable[None]]) -> MiddlewareFunc:
        async def middleware(event: Event, next_handler: MiddlewareNext) -> Event:
            # Check if we should process this event
            if event_types is None or event.type in event_types:
                await func(event)
            return await next_handler(event)

        return middleware

    if handler:
        return decorator(handler)
    return decorator