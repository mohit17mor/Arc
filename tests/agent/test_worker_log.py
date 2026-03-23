"""Tests for worker activity logging."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from arc.agent.worker_log import WorkerActivityLog, _truncate, _worker_label
from arc.core.events import Event, EventType


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestHelpers:
    def test_truncate_leaves_short_text_untouched(self):
        assert _truncate("short", 10) == "short"

    def test_truncate_adds_ellipsis_for_long_text(self):
        assert _truncate("abcdefgh", 5) == "abcde…"

    def test_worker_label_formats_known_sources(self):
        assert _worker_label("worker:research_ai").startswith("research_ai")
        assert _worker_label("scheduler:morning_news").startswith("morning_news")
        assert _worker_label("scheduler").startswith("[scheduler]")


class TestWorkerActivityLog:
    def test_open_rotates_previous_log_and_writes_session_markers(self, tmp_path):
        log_path = tmp_path / "worker_activity.log"
        log_path.write_text("old run\n", encoding="utf-8")

        activity_log = WorkerActivityLog(log_path)
        activity_log.open()
        activity_log.close()

        prev_path = tmp_path / "worker_activity.prev.log"
        assert prev_path.exists()
        assert prev_path.read_text(encoding="utf-8") == "old run\n"

        text = _read_text(log_path)
        assert "session start" in text
        assert "session end" in text

    @pytest.mark.asyncio
    async def test_handle_ignores_main_source_and_closed_log(self, tmp_path):
        log_path = tmp_path / "worker_activity.log"
        activity_log = WorkerActivityLog(log_path)

        await activity_log.handle(Event(type=EventType.AGENT_SPAWNED, source="main", data={"task_name": "noop"}))
        assert not log_path.exists()

        activity_log.open()
        await activity_log.handle(Event(type=EventType.AGENT_SPAWNED, source="main", data={"task_name": "noop"}))
        activity_log.close()

        text = _read_text(log_path)
        assert "noop" not in text

    @pytest.mark.asyncio
    async def test_handle_formats_spawn_and_thinking_events(self, tmp_path):
        log_path = tmp_path / "worker_activity.log"
        activity_log = WorkerActivityLog(log_path)
        activity_log.open()

        with patch("arc.agent.worker_log._now", return_value="12:34:56"):
            await activity_log.handle(
                Event(
                    type=EventType.AGENT_SPAWNED,
                    source="worker:research_ai_news",
                    data={"task_name": "summarize headlines"},
                )
            )
            await activity_log.handle(
                Event(
                    type=EventType.AGENT_THINKING,
                    source="worker:research_ai_news",
                    data={"iteration": 3},
                )
            )

        activity_log.close()
        text = _read_text(log_path)
        assert "12:34:56 | research_ai_ne | SPAWNED" in text
        assert "summarize headlines" in text
        assert "12:34:56 | research_ai_ne | THINKING" in text
        assert "iter=3" in text

    @pytest.mark.asyncio
    async def test_handle_formats_tool_call_and_result_events(self, tmp_path):
        log_path = tmp_path / "worker_activity.log"
        activity_log = WorkerActivityLog(log_path)
        activity_log.open()

        with patch("arc.agent.worker_log._now", return_value="09:00:00"):
            await activity_log.handle(
                Event(
                    type=EventType.SKILL_TOOL_CALL,
                    source="worker:browser_agent",
                    data={
                        "tool": "web_search",
                        "arguments": {
                            "query": "latest AI agent benchmarks and model comparisons for 2026",
                            "limit": 10,
                            "ignored": "third argument should be omitted",
                        },
                    },
                )
            )
            await activity_log.handle(
                Event(
                    type=EventType.SKILL_TOOL_RESULT,
                    source="worker:browser_agent",
                    data={
                        "success": True,
                        "output_preview": "First line\nSecond line",
                    },
                )
            )

        activity_log.close()
        text = _read_text(log_path)
        assert 'TOOL CALL  | web_search(query="latest AI agent benchmarks and…"' in text
        assert 'limit="10"' in text
        assert "third argument should be omitted" not in text
        assert "TOOL DONE  | ✓ First line Second line" in text

    @pytest.mark.asyncio
    async def test_handle_formats_plan_states_and_error(self, tmp_path):
        log_path = tmp_path / "worker_activity.log"
        activity_log = WorkerActivityLog(log_path)
        activity_log.open()

        with patch("arc.agent.worker_log._now", return_value="15:45:00"):
            await activity_log.handle(
                Event(
                    type=EventType.AGENT_PLAN_UPDATE,
                    source="scheduler:morning_news",
                    data={
                        "plan": [
                            {"step": "Collect feeds", "status": "completed"},
                            {"step": "Summarize the highest-signal stories from multiple long sources", "status": "in_progress"},
                        ]
                    },
                )
            )
            await activity_log.handle(
                Event(
                    type=EventType.AGENT_PLAN_UPDATE,
                    source="scheduler:morning_news",
                    data={
                        "plan": [{"step": "Collect feeds", "status": "completed"}],
                        "all_completed": True,
                    },
                )
            )
            await activity_log.handle(
                Event(
                    type=EventType.AGENT_PLAN_UPDATE,
                    source="scheduler:morning_news",
                    data={
                        "plan": [{"step": "Collect feeds", "status": "completed"}],
                        "lifecycle_status": "interrupted",
                    },
                )
            )
            await activity_log.handle(
                Event(
                    type=EventType.AGENT_ERROR,
                    source="scheduler",
                    data={"error": "network timeout while fetching source"},
                )
            )

        activity_log.close()
        text = _read_text(log_path)
        assert "15:45:00 | morning_news   | PLAN" in text
        assert "[1/2] Summarize the highest-signal stories" in text
        assert "✓ all 1 steps done" in text
        assert "[1/1] interrupted" in text
        assert "15:45:00 | [scheduler]" in text
        assert "ERROR" in text
        assert "network timeout while fetching source" in text

    @pytest.mark.asyncio
    async def test_handle_formats_completion_and_failed_tool_result(self, tmp_path):
        log_path = tmp_path / "worker_activity.log"
        activity_log = WorkerActivityLog(log_path)
        activity_log.open()

        with patch("arc.agent.worker_log._now", return_value="18:00:00"):
            await activity_log.handle(
                Event(
                    type=EventType.SKILL_TOOL_RESULT,
                    source="worker:researcher",
                    data={"success": False},
                )
            )
            await activity_log.handle(
                Event(
                    type=EventType.AGENT_TASK_COMPLETE,
                    source="worker:researcher",
                    data={"success": False},
                )
            )

        activity_log.close()
        text = _read_text(log_path)
        assert "TOOL DONE  | ✗ done" in text
        assert "COMPLETE   | ✗" in text

    def test_write_logs_warning_on_file_write_failure(self, tmp_path):
        log_path = tmp_path / "worker_activity.log"
        activity_log = WorkerActivityLog(log_path)
        activity_log._file = Mock()
        activity_log._file.write.side_effect = OSError("disk full")

        with patch("arc.agent.worker_log.logger.warning") as warning:
            activity_log._write("10:00:00", "worker", "ERROR", "boom")

        warning.assert_called_once()
        assert "WorkerActivityLog write failed" in warning.call_args[0][0]
