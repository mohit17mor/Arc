"""
CLIChannel — queues job-completion notifications for injection into the
next main-agent conversation turn.

Instead of printing directly to the terminal (which would interleave with
a streaming response), results are put on an asyncio.Queue that the
CLIPlatform drains before processing each user message.  The main agent
then sees the result as injected context and surfaces it naturally.
"""

from __future__ import annotations

import asyncio

from arc.notifications.base import Notification, NotificationChannel


class CLIChannel(NotificationChannel):
    """
    Queues notifications so they are injected between conversation turns.

    Usage:
        queue: asyncio.Queue[Notification] = asyncio.Queue()
        channel = CLIChannel(queue)
        channel.set_active(True)   # called by CLIPlatform.run()
        channel.set_active(False)  # called on shutdown
    """

    def __init__(self, queue: "asyncio.Queue[Notification]") -> None:
        self._queue = queue
        self._active = False

    @property
    def name(self) -> str:
        return "cli"

    @property
    def is_active(self) -> bool:
        return self._active

    def set_active(self, active: bool) -> None:
        self._active = active

    async def deliver(self, notification: Notification) -> bool:
        if not self._active:
            return False
        await self._queue.put(notification)
        return True
