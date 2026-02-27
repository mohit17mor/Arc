"""
FileChannel — always-on fallback that appends to ~/.arc/notifications.log.

This channel is always is_active=True and is always registered last.
It ensures every notification is persisted even when no interactive
platform is running.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

from arc.notifications.base import Notification, NotificationChannel

logger = logging.getLogger(__name__)


class FileChannel(NotificationChannel):
    """
    Appends notifications to a plain-text log file.

    Always active — acts as a silent fallback and permanent record.
    """

    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path or (Path.home() / ".arc" / "notifications.log")

    @property
    def name(self) -> str:
        return "file"

    @property
    def is_active(self) -> bool:
        return True  # always available

    async def deliver(self, notification: Notification) -> bool:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.fromtimestamp(notification.fired_at).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            entry = (
                f"[{ts}] [{notification.job_name}]\n"
                f"{notification.content}\n"
                f"{'─' * 60}\n"
            )
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(entry)
            return True
        except Exception as e:
            logger.warning(f"FileChannel write failed: {e}")
            return False
