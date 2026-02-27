"""
SchedulerStore — SQLite persistence for scheduled jobs.

DB: ~/.arc/scheduler.db  (separate from memory.db to keep concerns clean)

Table: jobs
    id         TEXT  PK
    name       TEXT  UNIQUE
    prompt     TEXT
    trigger    TEXT  (JSON)
    next_run   INT
    last_run   INT
    active     INT   (0/1)
    created_at INT
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path

from arc.scheduler.job import Job

logger = logging.getLogger(__name__)


class SchedulerStore:
    """
    Thread-safe SQLite store for jobs.  All blocking ops run in executor.

    Usage:
        store = SchedulerStore()
        await store.initialize()

        await store.save(job)
        due = await store.get_due_jobs(now=time.time())
        await store.update_after_run(job_id, next_run)
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or (Path.home() / ".arc" / "scheduler.db")
        self._db: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_sync)

    def _init_sync(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        db = self._get_db()
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id         TEXT PRIMARY KEY,
                name       TEXT UNIQUE NOT NULL,
                prompt     TEXT NOT NULL,
                trigger    TEXT NOT NULL,
                next_run   INTEGER NOT NULL DEFAULT 0,
                last_run   INTEGER NOT NULL DEFAULT 0,
                active     INTEGER NOT NULL DEFAULT 1,
                use_tools  INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            )
        """)
        # Safe migration for existing DBs that predate use_tools column
        try:
            db.execute("ALTER TABLE jobs ADD COLUMN use_tools INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass  # column already exists
        db.commit()
        logger.debug(f"SchedulerStore initialised at {self._db_path}")

    def _get_db(self) -> sqlite3.Connection:
        if self._db is None:
            db = sqlite3.connect(str(self._db_path), check_same_thread=False)
            db.row_factory = sqlite3.Row
            self._db = db
        return self._db

    # ── CRUD ─────────────────────────────────────────────────────────────────

    async def save(self, job: Job) -> None:
        """Insert or update a job."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_sync, job)

    def _save_sync(self, job: Job) -> None:
        db = self._get_db()
        db.execute(
            """
            INSERT INTO jobs (id, name, prompt, trigger, next_run, last_run, active, use_tools, created_at)
            VALUES (:id, :name, :prompt, :trigger, :next_run, :last_run, :active, :use_tools, :created_at)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, prompt=excluded.prompt,
                trigger=excluded.trigger, next_run=excluded.next_run,
                last_run=excluded.last_run, active=excluded.active,
                use_tools=excluded.use_tools
            """,
            {
                **job.to_dict(),
                "trigger": json.dumps(job.trigger),
                "active": int(job.active),
                "use_tools": int(job.use_tools),
            },
        )
        db.commit()

    async def get_all(self, active_only: bool = False) -> list[Job]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_all_sync, active_only)

    def _get_all_sync(self, active_only: bool) -> list[Job]:
        db = self._get_db()
        q = "SELECT * FROM jobs"
        if active_only:
            q += " WHERE active=1"
        q += " ORDER BY created_at ASC"
        return [self._row_to_job(r) for r in db.execute(q).fetchall()]

    async def get_due_jobs(self, now: float | None = None) -> list[Job]:
        """Return active jobs whose next_run <= now."""
        t = int(now or time.time())
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_due_sync, t)

    def _get_due_sync(self, now: int) -> list[Job]:
        db = self._get_db()
        rows = db.execute(
            "SELECT * FROM jobs WHERE active=1 AND next_run > 0 AND next_run <= ?",
            (now,),
        ).fetchall()
        return [self._row_to_job(r) for r in rows]

    async def update_after_run(
        self, job_id: str, next_run: int, last_run: int | None = None
    ) -> None:
        """Update next_run and last_run after a job fires. Deactivate if next_run=0."""
        t = last_run or int(time.time())
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._update_after_run_sync, job_id, next_run, t)

    def _update_after_run_sync(self, job_id: str, next_run: int, last_run: int) -> None:
        db = self._get_db()
        active = 1 if next_run > 0 else 0
        db.execute(
            "UPDATE jobs SET next_run=?, last_run=?, active=? WHERE id=?",
            (next_run, last_run, active, job_id),
        )
        db.commit()

    async def delete(self, job_id: str) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._delete_sync, job_id)

    def _delete_sync(self, job_id: str) -> bool:
        db = self._get_db()
        cur = db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        db.commit()
        return cur.rowcount > 0

    async def get_by_name(self, name: str) -> Job | None:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_by_name_sync, name)

    def _get_by_name_sync(self, name: str) -> Job | None:
        db = self._get_db()
        row = db.execute("SELECT * FROM jobs WHERE name=?", (name,)).fetchone()
        return self._row_to_job(row) if row else None

    async def close(self) -> None:
        if self._db:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._db.close)
            self._db = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        d = dict(row)
        d["trigger"] = json.loads(d["trigger"])
        d["active"] = bool(d["active"])
        d["use_tools"] = bool(d.get("use_tools", 0))
        return Job.from_dict(d)
