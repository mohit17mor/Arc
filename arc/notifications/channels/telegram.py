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
_MAX_LENGTH = 4096


def _split_text(text: str, max_length: int = _MAX_LENGTH) -> list[str]:
    """Split text into chunks that fit Telegram's message limit."""
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break
        split_at = max_length
        for sep in ("\n\n", "\n", " "):
            pos = remaining.rfind(sep, 0, max_length)
            if pos > max_length // 2:
                split_at = pos + len(sep)
                break
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    return chunks


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

    async def _send(self, client: httpx.AsyncClient, url: str, text: str) -> None:
        """Send a single message, falling back to plain text on Markdown error."""
        try:
            resp = await client.post(
                url,
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                # Markdown parse failure — retry without parse_mode
                resp = await client.post(
                    url,
                    json={"chat_id": self._chat_id, "text": text},
                )
                resp.raise_for_status()
            else:
                raise

    async def deliver(self, notification: Notification) -> bool:
        if not self.is_active:
            return False
        header = f"⏰ {notification.job_name}\n\n"
        content = notification.content or ""
        full_text = header + content
        url = _TELEGRAM_API.format(token=self._token)

        try:
            chunks = _split_text(full_text)
            async with httpx.AsyncClient(timeout=15) as client:
                for chunk in chunks:
                    await self._send(client, url, chunk)
            logger.debug(f"Telegram notification sent to {self._chat_id}")
            return True
        except Exception as e:
            logger.warning(f"Telegram delivery failed: {e}")
            return False
