"""Tests for middleware helper construction."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from arc.core.events import Event
from arc.middleware.base import create_middleware


@pytest.mark.asyncio
async def test_create_middleware_filters_event_types():
    seen = []

    async def handler(event: Event) -> None:
        seen.append(event.type)

    middleware = create_middleware(event_types=["llm:response"], handler=handler)
    next_handler = AsyncMock(side_effect=lambda event: event)

    event = Event(type="llm:response", source="test")
    result = await middleware(event, next_handler)

    assert result is event
    assert seen == ["llm:response"]
    next_handler.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_create_middleware_skips_non_matching_events():
    handler = AsyncMock()
    middleware = create_middleware(event_types=["llm:response"], handler=handler)
    next_handler = AsyncMock(side_effect=lambda event: event)

    event = Event(type="agent:thinking", source="test")
    await middleware(event, next_handler)

    handler.assert_not_awaited()
    next_handler.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_create_middleware_supports_decorator_form():
    seen = []

    @create_middleware()
    async def handler(event: Event) -> None:
        seen.append(event.source)

    event = Event(type="agent:thinking", source="worker:test")
    next_handler = AsyncMock(side_effect=lambda ev: ev)

    result = await handler(event, next_handler)

    assert result is event
    assert seen == ["worker:test"]
