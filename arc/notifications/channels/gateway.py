"""
GatewayChannel — delivers notifications to all connected WebSocket clients.

When a scheduled job or worker completes, the notification appears in
WebChat as a system message and is also queued for injection into the
agent's next conversation turn (so the agent can summarize it naturally).

This mirrors CLIChannel's dual-delivery pattern:
    - Immediate: push to WebSocket clients for real-time display
    - Deferred: queue for agent injection on next user turn
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from arc.notifications.base import Notification, NotificationChannel

logger = logging.getLogger(__name__)


class GatewayChannel(NotificationChannel):
    """
    Delivers notifications to WebChat via WebSocket broadcast + agent queue.

    Usage::

        queue: asyncio.Queue[Notification] = asyncio.Queue()
        channel = GatewayChannel(broadcast_fn=gateway.broadcast_notification, queue=queue)
        notification_router.register(channel)
    """

    def __init__(
        self,
        broadcast_fn: Any = None,
        queue: asyncio.Queue | None = None,
    ) -> None:
        self._broadcast_fn = broadcast_fn
        self._queue = queue
        self._active = False

    @property
    def name(self) -> str:
        return "gateway"

    @property
    def is_active(self) -> bool:
        return self._active and self._broadcast_fn is not None

    def set_active(self, active: bool) -> None:
        self._active = active

    async def deliver(self, notification: Notification) -> bool:
        if not self.is_active:
            return False

        delivered = False

        # 1. Broadcast to all WebSocket clients for immediate display
        if self._broadcast_fn is not None:
            try:
                await self._broadcast_fn(notification)
                delivered = True
            except Exception as e:
                logger.warning(f"GatewayChannel broadcast failed: {e}")

        # 2. Queue for agent injection on next turn
        if self._queue is not None:
            try:
                self._queue.put_nowait(notification)
            except Exception as e:
                logger.warning(f"GatewayChannel queue put failed: {e}")

        return delivered
