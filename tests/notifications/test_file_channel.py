"""Tests for the persistent notification file channel."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from arc.notifications.base import Notification
from arc.notifications.channels.file import FileChannel


def _make_notification(**kwargs) -> Notification:
    defaults = {
        "job_id": "job-123",
        "job_name": "daily-summary",
        "content": "Build completed successfully",
        "fired_at": 1704067200,  # 2024-01-01 00:00:00 local time
    }
    defaults.update(kwargs)
    return Notification(**defaults)


class TestFileChannel:
    def test_name_and_availability_contract(self):
        channel = FileChannel()

        assert channel.name == "file"
        assert channel.is_active is True

    @pytest.mark.asyncio
    async def test_deliver_creates_parent_and_writes_formatted_entry(self, tmp_path):
        log_path = tmp_path / "nested" / "notifications.log"
        channel = FileChannel(log_path=log_path)

        notification = _make_notification(content="First line\nSecond line")

        delivered = await channel.deliver(notification)

        assert delivered is True
        assert log_path.exists()

        text = log_path.read_text(encoding="utf-8")
        expected_ts = datetime.datetime.fromtimestamp(notification.fired_at).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        assert "[daily-summary]" in text
        assert "First line\nSecond line" in text
        assert expected_ts in text
        assert "─" * 60 in text

    @pytest.mark.asyncio
    async def test_deliver_appends_instead_of_overwriting(self, tmp_path):
        log_path = tmp_path / "notifications.log"
        channel = FileChannel(log_path=log_path)

        first = _make_notification(job_name="job-one", content="first result")
        second = _make_notification(
            job_name="job-two",
            content="second result",
            fired_at=1704067260,
        )

        assert await channel.deliver(first) is True
        assert await channel.deliver(second) is True

        text = log_path.read_text(encoding="utf-8")
        assert text.count("─" * 60) == 2
        assert "[job-one]" in text
        assert "[job-two]" in text
        assert text.index("[job-one]") < text.index("[job-two]")

    @pytest.mark.asyncio
    async def test_deliver_returns_false_and_warns_when_write_fails(self, tmp_path):
        log_path = tmp_path / "notifications.log"
        channel = FileChannel(log_path=log_path)
        notification = _make_notification()

        with patch.object(Path, "mkdir", side_effect=OSError("disk unavailable")):
            with patch("arc.notifications.channels.file.logger.warning") as warning:
                delivered = await channel.deliver(notification)

        assert delivered is False
        warning.assert_called_once()
        assert "FileChannel write failed" in warning.call_args[0][0]
