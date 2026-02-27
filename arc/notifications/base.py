"""
Notification primitives — Notification dataclass and NotificationChannel ABC.

Every delivery target (CLI, Telegram, WhatsApp, file log) implements
NotificationChannel. The NotificationRouter decides which ones fire.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Notification:
    """A single proactive message produced by a scheduled job."""

    job_id: str
    job_name: str
    content: str
    fired_at: int = field(default_factory=lambda: int(time.time()))


class NotificationChannel(ABC):
    """
    Abstract delivery target.

    Implement this to add a new platform.  The router calls
    is_active first — if False the channel is skipped entirely.
    deliver() should return True if the message was actually sent.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'cli', 'telegram', 'file'."""
        ...

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """
        Whether this channel can currently receive notifications.

        For CLI: True while the chat session is running.
        For Telegram: True while the bot is connected.
        For file: always True.
        """
        ...

    @property
    def is_external(self) -> bool:
        """
        External platforms (Telegram, WhatsApp, …) have higher routing
        priority than the local CLI.  Override to True in those channels.
        """
        return False

    @abstractmethod
    async def deliver(self, notification: Notification) -> bool:
        """
        Attempt to deliver the notification.

        Returns True if successfully delivered so the router can stop
        trying further channels at the same priority level.
        """
        ...
