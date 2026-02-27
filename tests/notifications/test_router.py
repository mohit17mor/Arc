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
