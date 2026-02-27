"""
TelegramChannel — delivers notifications via a Telegram bot.

Requires config:
    [telegram]
    token   = "BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"

When token + chat_id are set, this channel is_external=True and
takes routing priority over the CLI.

To get your chat_id:
    1. Create a bot via @BotFather, copy the token.
    2. Send your bot any message.
    3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates
       and read the "chat.id" field.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from arc.notifications.base import Notification, NotificationChannel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramChannel(NotificationChannel):
    """
    Sends notifications as Telegram messages.

    is_external = True  →  higher routing priority than CLI.
    is_active   = True only when token + chat_id are configured.
    """

    def __init__(self, token: str = "", chat_id: str = "") -> None:
        self._token = token.strip()
        self._chat_id = chat_id.strip()
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def is_external(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return bool(self._token and self._chat_id)

    async def deliver(self, notification: Notification) -> bool:
        if not self.is_active:
            return False
        text = f"⏰ *{notification.job_name}*\n\n{notification.content}"
        url = _TELEGRAM_API.format(token=self._token)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    url,
                    json={
                        "chat_id": self._chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                    },
                )
                resp.raise_for_status()
                logger.debug(f"Telegram notification sent to {self._chat_id}")
                return True
        except Exception as e:
            logger.warning(f"Telegram delivery failed: {e}")
            return False
