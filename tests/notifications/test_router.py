"""Tests for arc/notifications/router.py and channel basics."""
from __future__ import annotations

import pytest

from arc.notifications.base import Notification, NotificationChannel
from arc.notifications.router import NotificationRouter


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_notification(**kwargs) -> Notification:
    import time
    defaults = dict(
        job_id="abc12345",
        job_name="test-job",
        content="Job output here",
        fired_at=int(time.time()),
    )
    defaults.update(kwargs)
    return Notification(**defaults)


class FakeChannel(NotificationChannel):
    """Controllable test channel."""

    def __init__(
        self,
        name: str,
        *,
        active: bool = True,
        external: bool = False,
        should_succeed: bool = True,
    ) -> None:
        self._name = name
        self._active = active
        self._external = external
        self._should_succeed = should_succeed
        self.delivered: list[Notification] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_external(self) -> bool:
        return self._external

    async def deliver(self, notification: Notification) -> bool:
        if self._should_succeed:
            self.delivered.append(notification)
        return self._should_succeed


# ── NotificationRouter ───────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestNotificationRouter:
    async def test_register_and_channel_names(self):
        router = NotificationRouter()
        router.register(FakeChannel("cli"))
        router.register(FakeChannel("file"))
        assert set(router.channel_names) == {"cli", "file"}

    async def test_unregister(self):
        router = NotificationRouter()
        router.register(FakeChannel("cli"))
        router.register(FakeChannel("file"))
        router.unregister("cli")
        assert "cli" not in router.channel_names

    async def test_external_channel_used_first(self):
        """When external channel is active, it gets the notification."""
        ext = FakeChannel("telegram", active=True, external=True)
        cli = FakeChannel("cli", active=True, external=False)
        file_ch = FakeChannel("file", active=True, external=False)

        router = NotificationRouter()
        router.register(ext)
        router.register(cli)
        router.register(file_ch)

        notif = _make_notification()
        await router.route(notif)

        assert len(ext.delivered) == 1
        assert len(cli.delivered) == 0  # external delivered → CLI not used

    async def test_cli_fallback_when_no_external(self):
        """When no external channel delivers, CLI gets used."""
        cli = FakeChannel("cli", active=True, external=False)
        file_ch = FakeChannel("file", active=True, external=False)

        router = NotificationRouter()
        router.register(cli)
        router.register(file_ch)

        notif = _make_notification()
        await router.route(notif)

        assert len(cli.delivered) == 1

    async def test_file_always_receives(self):
        """File channel always logs regardless of other channels."""
        file_ch = FakeChannel("file", active=True, external=False)
        ext = FakeChannel("telegram", active=True, external=True)

        router = NotificationRouter()
        router.register(ext)
        router.register(file_ch)

        notif = _make_notification()
        await router.route(notif)

        assert len(ext.delivered) == 1
        assert len(file_ch.delivered) == 1  # always logs

    async def test_inactive_external_falls_back_to_cli(self):
        """Inactive external → CLI should receive."""
        ext = FakeChannel("telegram", active=False, external=True)
        cli = FakeChannel("cli", active=True, external=False)
        file_ch = FakeChannel("file", active=True, external=False)

        router = NotificationRouter()
        router.register(ext)
        router.register(cli)
        router.register(file_ch)

        notif = _make_notification()
        await router.route(notif)

        assert len(ext.delivered) == 0
        assert len(cli.delivered) == 1

    async def test_failing_external_falls_back_to_cli(self):
        """External channel that fails (returns False) → CLI fallback."""
        ext = FakeChannel("telegram", active=True, external=True, should_succeed=False)
        cli = FakeChannel("cli", active=True, external=False)
        file_ch = FakeChannel("file", active=True, external=False)

        router = NotificationRouter()
        router.register(ext)
        router.register(cli)
        router.register(file_ch)

        notif = _make_notification()
        await router.route(notif)

        assert len(ext.delivered) == 0
        assert len(cli.delivered) == 1

    async def test_no_channels_no_error(self):
        """Empty router should not raise."""
        router = NotificationRouter()
        notif = _make_notification()
        await router.route(notif)  # must not raise

    async def test_inactive_cli_not_used(self):
        """Inactive CLI not called even as fallback."""
        cli = FakeChannel("cli", active=False, external=False)
        file_ch = FakeChannel("file", active=True, external=False)

        router = NotificationRouter()
        router.register(cli)
        router.register(file_ch)

        notif = _make_notification()
        await router.route(notif)

        assert len(cli.delivered) == 0
        assert len(file_ch.delivered) == 1

    async def test_channel_exception_does_not_crash_router(self):
        """A channel that raises should not crash routing."""

        class BrokenChannel(FakeChannel):
            async def deliver(self, notification: Notification) -> bool:
                raise RuntimeError("boom")

        broken = BrokenChannel("broken", active=True, external=True)
        cli = FakeChannel("cli", active=True, external=False)
        file_ch = FakeChannel("file", active=True, external=False)

        router = NotificationRouter()
        router.register(broken)
        router.register(cli)
        router.register(file_ch)

        notif = _make_notification()
        await router.route(notif)  # must not raise

        # broken failed → cli fallback
        assert len(cli.delivered) == 1


# ── Notification dataclass ────────────────────────────────────────────────────

class TestNotification:
    def test_fields(self):
        import time
        n = _make_notification(content="hello")
        assert n.job_name == "test-job"
        assert n.content == "hello"
        assert n.job_id == "abc12345"
        assert n.fired_at > 0


# ── TelegramChannel ──────────────────────────────────────────────────────────

from arc.notifications.channels.telegram import TelegramChannel, _split_text


class TestSplitText:
    def test_short_text_not_split(self):
        assert _split_text("hello", 100) == ["hello"]

    def test_split_at_paragraph_boundary(self):
        text = "A" * 3000 + "\n\n" + "B" * 3000
        chunks = _split_text(text, 4096)
        assert len(chunks) == 2
        assert chunks[0].strip() == "A" * 3000
        assert chunks[1] == "B" * 3000

    def test_split_respects_limit(self):
        text = "x" * 10000
        chunks = _split_text(text, 4096)
        assert all(len(c) <= 4096 for c in chunks)
        assert "".join(chunks) == text


@pytest.mark.asyncio
class TestTelegramChannel:
    def test_inactive_without_config(self):
        ch = TelegramChannel()
        assert not ch.is_active
        assert ch.is_external

    def test_active_with_config(self):
        ch = TelegramChannel(token="tok", chat_id="123")
        assert ch.is_active

    async def test_deliver_inactive_returns_false(self):
        ch = TelegramChannel()
        notif = _make_notification()
        assert await ch.deliver(notif) is False

    async def test_deliver_success(self):
        """Successful send with Markdown."""
        from unittest.mock import AsyncMock, patch, MagicMock
        import httpx

        ch = TelegramChannel(token="tok", chat_id="123")
        notif = _make_notification(content="simple text")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        with patch("arc.notifications.channels.telegram.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await ch.deliver(notif)

        assert result is True
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["parse_mode"] == "Markdown"

    async def test_deliver_markdown_fallback_plain(self):
        """400 on Markdown → retries without parse_mode."""
        from unittest.mock import AsyncMock, patch, MagicMock
        import httpx

        ch = TelegramChannel(token="tok", chat_id="123")
        notif = _make_notification(content="text with _bad_ *markdown")

        # First call raises 400, second succeeds
        error_resp = MagicMock()
        error_resp.status_code = 400
        error_resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "400", request=MagicMock(), response=error_resp,
            )
        )

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.raise_for_status = MagicMock()

        with patch("arc.notifications.channels.telegram.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[error_resp, ok_resp])
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await ch.deliver(notif)

        assert result is True
        assert mock_client.post.call_count == 2
        # Second call should not have parse_mode
        retry_json = mock_client.post.call_args_list[1][1]["json"]
        assert "parse_mode" not in retry_json
