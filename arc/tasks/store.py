"""
TaskStore — SQLite persistence for the task queue.

DB: ~/.arc/tasks.db

Tables:
    tasks          — task metadata + workflow state
    task_comments  — audit trail / inter-agent communication

All blocking DB ops run in a thread-pool executor.
Status + comment writes happen in a single transaction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path

from arc.tasks.types import Task, TaskComment, TaskStatus, TaskStep

logger = logging.getLogger(__name__)


class TaskStore:
    """
    Thread-safe SQLite store for tasks and comments.

    Usage:
        store = TaskStore()
        await store.initialize()

        await store.save(task)
        pending = await store.get_actionable_tasks()
        await store.update_status_with_comment(task_id, status, comment)
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or (Path.home() / ".arc" / "tasks.db")
        self._db: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._init_sync)

    def _init_sync(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        db = self._get_db()
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id              TEXT PRIMARY KEY,
                title           TEXT NOT NULL,
                instruction     TEXT NOT NULL,
                steps           TEXT NOT NULL DEFAULT '[]',
                current_step    INTEGER NOT NULL DEFAULT 0,
                assigned_agent  TEXT NOT NULL DEFAULT '',
                bounce_count    INTEGER NOT NULL DEFAULT 0,
                max_bounces     INTEGER NOT NULL DEFAULT 3,
                status          TEXT NOT NULL DEFAULT 'queued',
                priority        INTEGER NOT NULL DEFAULT 1,
                result          TEXT NOT NULL DEFAULT '',
                depends_on      TEXT,
                created_at      INTEGER NOT NULL,
                started_at      INTEGER NOT NULL DEFAULT 0,
                completed_at    INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS task_comments (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id     TEXT NOT NULL REFERENCES tasks(id),
                step_index  INTEGER NOT NULL DEFAULT 0,
                agent_name  TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_comments_task
                ON task_comments(task_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON tasks(status, priority, created_at);
        """)
        db.commit()
        logger.debug(f"TaskStore initialised at {self._db_path}")

    def _get_db(self) -> sqlite3.Connection:
        if self._db is None:
            db = sqlite3.connect(str(self._db_path), check_same_thread=False)
            db.row_factory = sqlite3.Row
            self._db = db
        return self._db

    # ── Save / Create ────────────────────────────────────────────────────────

    async def save(self, task: Task) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_sync, task)

    def _save_sync(self, task: Task) -> None:
        db = self._get_db()
        db.execute(
            """
            INSERT INTO tasks (
                id, title, instruction, steps, current_step, assigned_agent,
                bounce_count, max_bounces, status, priority, result,
                depends_on, created_at, started_at, completed_at
            ) VALUES (
                :id, :title, :instruction, :steps, :current_step, :assigned_agent,
                :bounce_count, :max_bounces, :status, :priority, :result,
                :depends_on, :created_at, :started_at, :completed_at
            ) ON CONFLICT(id) DO UPDATE SET
                status=excluded.status, current_step=excluded.current_step,
                bounce_count=excluded.bounce_count, result=excluded.result,
                started_at=excluded.started_at, completed_at=excluded.completed_at
            """,
            {
                **task.to_dict(),
                "steps": json.dumps([s.to_dict() for s in task.steps]),
                "status": task.status.value,
            },
        )
        db.commit()

    # ── Queries ──────────────────────────────────────────────────────────────

    async def get_by_id(self, task_id: str) -> Task | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_by_id_sync, task_id)

    def _get_by_id_sync(self, task_id: str) -> Task | None:
        db = self._get_db()
        row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return self._row_to_task(row) if row else None

    async def get_actionable_tasks(self, agent_names: list[str] | None = None) -> list[Task]:
        """Return tasks ready to be picked up (queued or revision_needed).

        Respects dependencies — only returns tasks whose depends_on is
        either NULL or points to a completed task.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._get_actionable_sync, agent_names
        )

    def _get_actionable_sync(self, agent_names: list[str] | None) -> list[Task]:
        db = self._get_db()
        rows = db.execute(
            """
            SELECT * FROM tasks
            WHERE status IN ('queued', 'revision_needed')
              AND (depends_on IS NULL
                   OR depends_on IN (SELECT id FROM tasks WHERE status='done'))
            ORDER BY priority ASC, created_at ASC
            """,
        ).fetchall()
        tasks = [self._row_to_task(r) for r in rows]
        if agent_names is not None:
            tasks = [t for t in tasks if t.current_agent in agent_names]
        return tasks

    async def get_blocked_task(self, task_id: str) -> Task | None:
        """Get a task only if it is in blocked or awaiting_human status."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_blocked_sync, task_id)

    def _get_blocked_sync(self, task_id: str) -> Task | None:
        db = self._get_db()
        row = db.execute(
            "SELECT * FROM tasks WHERE id=? AND status IN ('blocked', 'awaiting_human')",
            (task_id,),
        ).fetchone()
        return self._row_to_task(row) if row else None

    async def get_all(self, status: str | None = None, limit: int = 50) -> list[Task]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_all_sync, status, limit)

    def _get_all_sync(self, status: str | None, limit: int) -> list[Task]:
        db = self._get_db()
        if status:
            rows = db.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    async def count_in_progress(self, agent_name: str) -> int:
        """Count tasks currently being processed by a specific agent."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._count_in_progress_sync, agent_name
        )

    def _count_in_progress_sync(self, agent_name: str) -> int:
        db = self._get_db()
        # Check all tasks that are in_progress or in_review for this agent
        rows = db.execute(
            """SELECT * FROM tasks
               WHERE status IN ('in_progress', 'in_review')""",
        ).fetchall()
        count = 0
        for row in rows:
            t = self._row_to_task(row)
            if t.current_agent == agent_name:
                count += 1
        return count

    # ── Status transitions (always with comment, single transaction) ─────────

    async def update_status_with_comment(
        self,
        task_id: str,
        new_status: TaskStatus,
        comment_agent: str,
        comment_text: str,
        step_index: int = 0,
        extra_updates: dict | None = None,
    ) -> None:
        """Atomically update task status and add a comment."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._update_status_with_comment_sync,
            task_id, new_status, comment_agent, comment_text,
            step_index, extra_updates or {},
        )

    def _update_status_with_comment_sync(
        self,
        task_id: str,
        new_status: TaskStatus,
        comment_agent: str,
        comment_text: str,
        step_index: int,
        extra_updates: dict,
    ) -> None:
        db = self._get_db()
        now = int(time.time())

        # Build dynamic SET clause for extra fields
        set_parts = ["status=?"]
        params: list = [new_status.value]

        for col, val in extra_updates.items():
            set_parts.append(f"{col}=?")
            params.append(val)

        params.append(task_id)
        set_clause = ", ".join(set_parts)

        db.execute(f"UPDATE tasks SET {set_clause} WHERE id=?", params)
        db.execute(
            """INSERT INTO task_comments (task_id, step_index, agent_name, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (task_id, step_index, comment_agent, comment_text, now),
        )
        db.commit()

    # ── Comments ─────────────────────────────────────────────────────────────

    async def get_comments(self, task_id: str) -> list[TaskComment]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_comments_sync, task_id)

    def _get_comments_sync(self, task_id: str) -> list[TaskComment]:
        db = self._get_db()
        rows = db.execute(
            "SELECT * FROM task_comments WHERE task_id=? ORDER BY created_at ASC",
            (task_id,),
        ).fetchall()
        return [
            TaskComment(
                id=r["id"],
                task_id=r["task_id"],
                step_index=r["step_index"],
                agent_name=r["agent_name"],
                content=r["content"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def add_comment(
        self, task_id: str, agent_name: str, content: str, step_index: int = 0
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._add_comment_sync, task_id, agent_name, content, step_index
        )

    def _add_comment_sync(
        self, task_id: str, agent_name: str, content: str, step_index: int
    ) -> None:
        db = self._get_db()
        db.execute(
            """INSERT INTO task_comments (task_id, step_index, agent_name, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (task_id, step_index, agent_name, content, int(time.time())),
        )
        db.commit()

    # ── Delete / Cancel ──────────────────────────────────────────────────────

    async def cancel(self, task_id: str) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._cancel_sync, task_id)

    def _cancel_sync(self, task_id: str) -> bool:
        db = self._get_db()
        cur = db.execute(
            "UPDATE tasks SET status='cancelled', completed_at=? WHERE id=? AND status NOT IN ('done', 'cancelled')",
            (int(time.time()), task_id),
        )
        db.commit()
        return cur.rowcount > 0

    async def close(self) -> None:
        if self._db:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._db.close)
            self._db = None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        d = dict(row)
        # Steps are stored as JSON string — parse to raw dicts.
        # Task.from_dict will convert them to TaskStep objects.
        d["steps"] = json.loads(d.get("steps", "[]"))
        d["status"] = d["status"]  # Task.from_dict handles string → enum
        return Task.from_dict(d)
