"""
WorkerActivityLog — writes worker agent events to a dedicated log file.

The main chat window stays completely clean. To watch worker activity,
run ``arc workers`` in a second terminal — it live-tails this file.

Format of each line (plain text, easy to tail/grep):
    2026-02-27 14:30:15 | research_ai  | SPAWNED
    2026-02-27 14:30:16 | research_ai  | THINKING   iter=1
    2026-02-27 14:30:17 | research_ai  | TOOL CALL  web_search(query="AI news...")
    2026-02-27 14:30:18 | research_ai  | TOOL DONE  ✓ Found 10 results about...
    2026-02-27 14:30:22 | research_ai  | COMPLETE   ✓

The file is rotated at start-up (previous run's log is kept as worker_activity.prev.log).
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

from arc.core.events import Event, EventType

logger = logging.getLogger(__name__)

# Column widths for alignment
_W_WORKER = 14   # worker name column
_W_EVENT  = 10   # event type column


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _truncate(s: str, n: int) -> str:
    return s[:n] + "…" if len(s) > n else s


def _worker_label(source: str) -> str:
    """'worker:research_ai_news' → 'research_ai_n'  (fits _W_WORKER)"""
    label = source[7:] if source.startswith("worker:") else source
    return label[:_W_WORKER].ljust(_W_WORKER)


class WorkerActivityLog:
    """
    Receives kernel events and appends formatted lines to a log file.

    Only processes events whose ``source`` starts with ``"worker:"``.
    """

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._file = None

    def open(self) -> None:
        """Open the log file, rotating any previous log."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            prev = self._path.with_suffix(".prev.log")
            try:
                self._path.replace(prev)
            except Exception:
                pass  # non-fatal
        self._file = self._path.open("a", encoding="utf-8", buffering=1)  # line-buffered
        self._write_separator("session start")

    def close(self) -> None:
        if self._file:
            self._write_separator("session end")
            self._file.close()
            self._file = None

    # ------------------------------------------------------------------ #
    # Event handler — wire this to kernel.on(...)                         #
    # ------------------------------------------------------------------ #

    async def handle(self, event: Event) -> None:
        """Async event handler — write a formatted line for every worker event."""
        if not event.source.startswith("worker:"):
            return
        if self._file is None:
            return

        label = _worker_label(event.source)
        ts = _now()

        if event.type == EventType.AGENT_SPAWNED:
            task_name = event.data.get("task_name", event.data.get("task_id", ""))
            self._write(ts, label, "SPAWNED", task_name)

        elif event.type == EventType.AGENT_THINKING:
            iteration = event.data.get("iteration", "?")
            self._write(ts, label, "THINKING", f"iter={iteration}")

        elif event.type == EventType.SKILL_TOOL_CALL:
            tool = event.data.get("tool", "?")
            args = event.data.get("arguments", {})
            args_parts = []
            for k, v in list(args.items())[:2]:
                v_str = str(v)
                short = _truncate(v_str, 30)
                args_parts.append(f'{k}="{short}"')
            args_str = ", ".join(args_parts)
            self._write(ts, label, "TOOL CALL", f"{tool}({args_str})")

        elif event.type == EventType.SKILL_TOOL_RESULT:
            success = event.data.get("success", False)
            preview = event.data.get("output_preview", "")
            icon = "✓" if success else "✗"
            detail = _truncate(preview.replace("\n", " ").strip(), 60) if preview else "done"
            self._write(ts, label, "TOOL DONE", f"{icon} {detail}")

        elif event.type == EventType.AGENT_TASK_COMPLETE:
            success = event.data.get("success", True)
            icon = "✓" if success else "✗"
            self._write(ts, label, "COMPLETE", icon)

        elif event.type == EventType.AGENT_ERROR:
            error = event.data.get("error", "unknown error")
            self._write(ts, label, "ERROR", _truncate(str(error), 60))

        # Ignore other event types

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _write(self, ts: str, label: str, event_col: str, detail: str = "") -> None:
        """Write one aligned line."""
        ec = event_col.ljust(_W_EVENT)
        line = f"{ts} | {label} | {ec} | {detail}\n"
        try:
            self._file.write(line)
        except Exception as exc:
            logger.warning(f"WorkerActivityLog write failed: {exc}")

    def _write_separator(self, label: str) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self._file.write(f"{'─' * 72}\n")
            self._file.write(f"  {ts}  —  {label}\n")
            self._file.write(f"{'─' * 72}\n")
        except Exception:
            pass
