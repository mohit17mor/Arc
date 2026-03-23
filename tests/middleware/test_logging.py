"""Tests for the logging middleware and logger setup."""

from __future__ import annotations

import builtins
import json
import logging
from unittest.mock import AsyncMock, patch

import pytest

from arc.core.events import Event
from arc.middleware.logging import EventLogger, setup_logging


class UnserializableValue:
    def __str__(self) -> str:
        return "<custom-object>"


class TestSetupLogging:
    def test_setup_logging_creates_handlers_and_writes_file(self, tmp_path):
        logger = setup_logging(log_dir=tmp_path)

        assert logger.name == "arc"
        assert len(logger.handlers) == 2

        logger.info("hello logger")

        log_files = list(tmp_path.glob("arc_*.log"))
        assert len(log_files) == 1
        text = log_files[0].read_text(encoding="utf-8")
        assert "Logging initialized." in text
        assert "hello logger" in text

    def test_setup_logging_replaces_existing_handlers(self, tmp_path):
        logger = logging.getLogger("arc")
        stale_handler = logging.NullHandler()
        logger.handlers = [stale_handler]

        configured = setup_logging(log_dir=tmp_path)

        assert stale_handler not in configured.handlers
        assert len(configured.handlers) == 2


class TestEventLogger:
    @pytest.mark.asyncio
    async def test_middleware_logs_event_and_calls_next_handler(self, tmp_path):
        event_logger = EventLogger(log_dir=tmp_path)
        event = Event(
            type="agent:thinking",
            source="worker:tester",
            data={"answer": 42, "payload": UnserializableValue()},
        )
        next_handler = AsyncMock(return_value=event)

        result = await event_logger.middleware(event, next_handler)

        assert result is event
        next_handler.assert_awaited_once_with(event)

        events_files = list(tmp_path.glob("events_*.jsonl"))
        assert len(events_files) == 1
        lines = events_files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["id"] == event.id
        assert record["type"] == "agent:thinking"
        assert record["source"] == "worker:tester"
        assert record["data"]["answer"] == 42
        assert record["data"]["payload"] == "<custom-object>"

    def test_safe_serialize_preserves_json_values_and_stringifies_others(self):
        data = {
            "text": "ok",
            "count": 3,
            "nested": {"a": 1},
            "custom": UnserializableValue(),
        }

        result = EventLogger._safe_serialize(data)

        assert result["text"] == "ok"
        assert result["count"] == 3
        assert result["nested"] == {"a": 1}
        assert result["custom"] == "<custom-object>"

    def test_write_event_logs_warning_when_file_write_fails(self, tmp_path):
        event_logger = EventLogger(log_dir=tmp_path)
        event = Event(type="system:error", source="main", data={"message": "boom"})

        with patch.object(builtins, "open", side_effect=OSError("read-only fs")):
            with patch.object(event_logger._logger, "warning") as warning:
                event_logger._write_event(event)

        warning.assert_called_once()
        assert "Failed to write event log" in warning.call_args[0][0]
