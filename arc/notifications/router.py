"""
NotificationRouter — decides which channels receive each notification.

Routing logic (matches user spec exactly):

    1. Try all active EXTERNAL channels (Telegram, WhatsApp, …).
       If at least one delivers → done for interactive channels.
    2. If no external channel delivered AND CLI is active → deliver to CLI.
    3. ALWAYS write to file log (silent, non-interactive record).

This means:
- Telegram configured + CLI open → Telegram only (not both)
- Only CLI open → CLI
- Nothing active → file log only
"""

from __future__ import annotations

import logging

from arc.notifications.base import Notification, NotificationChannel

logger = logging.getLogger(__name__)


class NotificationRouter:
    """
    Routes notifications to the appropriate channel(s).

    Usage:
        router = NotificationRouter()
        router.register(CLIChannel(console))
        router.register(FileChannel())
        router.register(TelegramChannel(token, chat_id))

        await router.route(notification)
    """

    def __init__(self) -> None:
        self._channels: list[NotificationChannel] = []

    def register(self, channel: NotificationChannel) -> None:
        """Register a channel. Order of registration doesn't affect routing."""
        self._channels.append(channel)
        logger.debug(f"Notification channel registered: {channel.name}")

    def unregister(self, name: str) -> None:
        """Remove a channel by name."""
        self._channels = [c for c in self._channels if c.name != name]

    @property
    def channel_names(self) -> list[str]:
        return [c.name for c in self._channels]

    async def route(self, notification: Notification) -> None:
        """
        Deliver the notification according to the priority rules above.
        Never raises — failures are logged and swallowed.
        """
        external_channels = [c for c in self._channels if c.is_external]
        cli_channels      = [c for c in self._channels if not c.is_external and c.name != "file"]
        file_channels     = [c for c in self._channels if c.name == "file"]

        # ── Step 1: Try external platforms ────────────────────────────────────
        external_delivered = False
        for channel in external_channels:
            if not channel.is_active:
                continue
            try:
                ok = await channel.deliver(notification)
                if ok:
                    external_delivered = True
                    logger.debug(f"Notification delivered via {channel.name}")
            except Exception as e:
                logger.warning(f"Channel {channel.name} delivery failed: {e}")

        # ── Step 2: CLI fallback (only if no external delivery) ───────────────
        if not external_delivered:
            for channel in cli_channels:
                if not channel.is_active:
                    continue
                try:
                    await channel.deliver(notification)
                    logger.debug(f"Notification delivered via {channel.name}")
                except Exception as e:
                    logger.warning(f"Channel {channel.name} delivery failed: {e}")

        # ── Step 3: Always log to file ─────────────────────────────────────────
        for channel in file_channels:
            try:
                await channel.deliver(notification)
            except Exception as e:
                logger.warning(f"File channel delivery failed: {e}")
