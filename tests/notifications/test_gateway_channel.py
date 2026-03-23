"""Tests for the WebChat notification channel."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from arc.notifications.base import Notification
from arc.notifications.channels.gateway import GatewayChannel


def _make_notification(**kwargs) -> Notification:
    defaults = {
        "job_id": "job-456",
        "job_name": "webhook-sync",
        "content": "Sync complete",
        "fired_at": 1704067200,
    }
    defaults.update(kwargs)
    return Notification(**defaults)


class BrokenQueue:
    def put_nowait(self, item):
        raise RuntimeError("queue full")


class TestGatewayChannel:
    def test_name_and_inactive_state_without_activation(self):
        channel = GatewayChannel(broadcast_fn=AsyncMock())

        assert channel.name == "gateway"
        assert channel.is_active is False

    def test_activation_requires_broadcast_function(self):
        channel = GatewayChannel(broadcast_fn=None)

        channel.set_active(True)

        assert channel.is_active is False

    @pytest.mark.asyncio
    async def test_deliver_returns_false_when_inactive(self):
        broadcast = AsyncMock()
        queue = Mock()
        channel = GatewayChannel(broadcast_fn=broadcast, queue=queue)

        delivered = await channel.deliver(_make_notification())

        assert delivered is False
        broadcast.assert_not_awaited()
        queue.put_nowait.assert_not_called()

    @pytest.mark.asyncio
    async def test_deliver_broadcasts_and_enqueues_when_active(self):
        broadcast = AsyncMock()
        queue: asyncio.Queue[Notification] = asyncio.Queue()
        channel = GatewayChannel(broadcast_fn=broadcast, queue=queue)
        channel.set_active(True)
        notification = _make_notification()

        delivered = await channel.deliver(notification)

        assert delivered is True
        broadcast.assert_awaited_once_with(notification)
        assert queue.get_nowait() is notification

    @pytest.mark.asyncio
    async def test_broadcast_failure_still_queues_for_later_summary(self):
        broadcast = AsyncMock(side_effect=RuntimeError("socket closed"))
        queue: asyncio.Queue[Notification] = asyncio.Queue()
        channel = GatewayChannel(broadcast_fn=broadcast, queue=queue)
        channel.set_active(True)
        notification = _make_notification()

        with patch("arc.notifications.channels.gateway.logger.warning") as warning:
            delivered = await channel.deliver(notification)

        assert delivered is False
        assert queue.get_nowait() is notification
        warning.assert_called_once()
        assert "GatewayChannel broadcast failed" in warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_queue_failure_does_not_hide_successful_broadcast(self):
        broadcast = AsyncMock()
        channel = GatewayChannel(broadcast_fn=broadcast, queue=BrokenQueue())
        channel.set_active(True)
        notification = _make_notification()

        with patch("arc.notifications.channels.gateway.logger.warning") as warning:
            delivered = await channel.deliver(notification)

        assert delivered is True
        broadcast.assert_awaited_once_with(notification)
        warning.assert_called_once()
        assert "GatewayChannel queue put failed" in warning.call_args[0][0]
