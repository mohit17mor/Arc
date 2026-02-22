"""
Logging Middleware — logs all events.
"""

from __future__ import annotations

import logging
from arc.core.events import Event
from arc.middleware.base import MiddlewareNext

logger = logging.getLogger("arc.events")


async def logging_middleware(event: Event, next_handler: MiddlewareNext) -> Event:
    """Log all events passing through the system."""
    logger.debug(
        f"[{event.type}] source={event.source} "
        f"data_keys={list(event.data.keys()) if event.data else []}"
    )
    return await next_handler(event)