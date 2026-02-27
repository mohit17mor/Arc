"""
SchedulerSkill — lets the LLM create, list, and cancel scheduled jobs.

The LLM calls these tools in response to user requests like:
  "remind me every day at 9am to check my downloads"
  "cancel the morning_check job"
  "what jobs are scheduled?"

Tools:
    schedule_job(name, prompt, trigger_type, ...)
    list_jobs()
    cancel_job(name_or_id)
"""

from __future__ import annotations

import datetime
import time
from typing import Any

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec
from arc.scheduler.job import Job
from arc.scheduler.store import SchedulerStore
from arc.scheduler.triggers import CronTrigger, IntervalTrigger, OneshotTrigger, make_trigger
from arc.skills.base import Skill


class SchedulerSkill(Skill):
    """
    Skill for managing scheduled jobs.

    The SchedulerStore must be injected via set_store() before use
    (called from main.py after the store is initialised).
    """

    def __init__(self) -> None:
        self._store: SchedulerStore | None = None
        self._kernel: Any = None
        self._config: dict = {}

    def set_store(self, store: SchedulerStore) -> None:
        """Inject the store. Called from main.py."""
        self._store = store

    # ── Skill ABC ─────────────────────────────────────────────────────────────

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="scheduler",
            version="1.0.0",
            description="Schedule recurring or one-time tasks for Arc to run proactively",
            capabilities=frozenset([Capability.FILE_READ]),  # lowest, no destructive ops
            tools=[
                ToolSpec(
                    name="schedule_job",
                    description=(
                        "Create a scheduled job. Arc will run the prompt automatically "
                        "and notify the user with the result. "
                        "Use trigger_type='cron' with a cron_expression for recurring schedules "
                        "(e.g. '0 9 * * 1-5' = weekdays at 9am). "
                        "Use trigger_type='interval' with interval_seconds for a repeat interval. "
                        "Use trigger_type='oneshot' with fire_at (unix timestamp) for a single future alert. "
                        "Set use_tools=true only when the task genuinely needs live data or file access "
                        "(e.g. fetching news, reading a file). Leave false for reminders, tips, or "
                        "anything the LLM can answer from its own knowledge."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Short unique name for the job, e.g. 'morning_summary'",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "What the agent should do / say when this job fires",
                            },
                            "trigger_type": {
                                "type": "string",
                                "enum": ["cron", "interval", "oneshot"],
                                "description": "Type of trigger",
                            },
                            "cron_expression": {
                                "type": "string",
                                "description": "5-field cron expression (required for trigger_type='cron')",
                            },
                            "interval_seconds": {
                                "type": "integer",
                                "description": "Seconds between runs (required for trigger_type='interval')",
                            },
                            "fire_at": {
                                "type": "integer",
                                "description": "Unix timestamp to fire once (required for trigger_type='oneshot')",
                            },
                            "use_tools": {
                                "type": "boolean",
                                "description": (
                                    "If true, a full sub-agent with tool access runs the prompt. "
                                    "Only set true when the job needs live data or file operations "
                                    "(e.g. web search, read file). Default false."
                                ),
                            },
                        },
                        "required": ["name", "prompt", "trigger_type"],
                    },
                ),
                ToolSpec(
                    name="list_jobs",
                    description="List all scheduled jobs with their trigger, next run time, and status.",
                    parameters={"type": "object", "properties": {}, "required": []},
                ),
                ToolSpec(
                    name="cancel_job",
                    description="Cancel (delete) a scheduled job by its name or id.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "name_or_id": {
                                "type": "string",
                                "description": "Job name or job id to cancel",
                            }
                        },
                        "required": ["name_or_id"],
                    },
                ),
            ],
        )

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if tool_name == "schedule_job":
            return await self._schedule_job(**arguments)
        elif tool_name == "list_jobs":
            return await self._list_jobs()
        elif tool_name == "cancel_job":
            return await self._cancel_job(arguments.get("name_or_id", ""))
        return ToolResult(
            success=False, output="", error=f"Unknown tool: {tool_name}"
        )

    # ── Tool implementations ───────────────────────────────────────────────────

    async def _schedule_job(
        self,
        name: str,
        prompt: str,
        trigger_type: str,
        cron_expression: str = "",
        interval_seconds: int = 0,
        fire_at: int = 0,
        use_tools: bool = False,
    ) -> ToolResult:
        if not self._store:
            return ToolResult(success=False, output="", error="Scheduler not initialised")

        # Build trigger dict
        if trigger_type == "cron":
            if not cron_expression:
                return ToolResult(
                    success=False, output="",
                    error="cron_expression is required for trigger_type='cron'",
                )
            try:
                from croniter import croniter
                if not croniter.is_valid(cron_expression):
                    return ToolResult(
                        success=False, output="",
                        error=f"Invalid cron expression: {cron_expression!r}",
                    )
            except ImportError:
                pass  # croniter not installed — validated at runtime
            trigger_dict = {"type": "cron", "expression": cron_expression}

        elif trigger_type == "interval":
            if interval_seconds < 1:
                return ToolResult(
                    success=False, output="",
                    error="interval_seconds must be >= 1",
                )
            trigger_dict = {"type": "interval", "seconds": interval_seconds}

        elif trigger_type == "oneshot":
            if fire_at <= 0:
                return ToolResult(
                    success=False, output="",
                    error="fire_at (unix timestamp) is required for trigger_type='oneshot'",
                )
            if fire_at <= time.time():
                return ToolResult(
                    success=False, output="",
                    error="fire_at must be in the future",
                )
            trigger_dict = {"type": "oneshot", "at": fire_at}

        else:
            return ToolResult(
                success=False, output="",
                error=f"Unknown trigger_type: {trigger_type!r}",
            )

        # Compute initial next_run
        trigger_obj = make_trigger(trigger_dict)
        now = int(time.time())
        next_run = trigger_obj.next_fire_time(last_run=0, now=now)

        job = Job(
            name=name,
            prompt=prompt,
            trigger=trigger_dict,
            next_run=next_run,
            use_tools=use_tools,
        )
        await self._store.save(job)

        next_dt = datetime.datetime.fromtimestamp(next_run).strftime("%Y-%m-%d %H:%M:%S")
        desc = trigger_obj.description
        mode = "task (with tools)" if use_tools else "simple (text only)"
        return ToolResult(
            success=True,
            output=(
                f"Job '{name}' scheduled ({desc})."
                f" Mode: {mode}.\n"
                f"Next run: {next_dt}\n"
                f"Job id: {job.id}"
            ),
        )

    async def _list_jobs(self) -> ToolResult:
        if not self._store:
            return ToolResult(success=False, output="", error="Scheduler not initialised")

        jobs = await self._store.get_all()
        if not jobs:
            return ToolResult(success=True, output="No scheduled jobs.")

        lines = []
        now = time.time()
        for job in jobs:
            status = "active" if job.active else "inactive"
            trigger = make_trigger(job.trigger)
            next_dt = (
                datetime.datetime.fromtimestamp(job.next_run).strftime("%Y-%m-%d %H:%M")
                if job.next_run > 0
                else "—"
            )
            last_dt = (
                datetime.datetime.fromtimestamp(job.last_run).strftime("%Y-%m-%d %H:%M")
                if job.last_run > 0
                else "never"
            )
            mode = "task+tools" if job.use_tools else "simple"
            lines.append(
                f"[{job.id}] {job.name}  ({status}, {mode})\n"
                f"  trigger: {trigger.description}\n"
                f"  next:    {next_dt}\n"
                f"  last:    {last_dt}\n"
                f"  prompt:  {job.prompt[:80]}{'...' if len(job.prompt) > 80 else ''}"
            )
        return ToolResult(success=True, output="\n\n".join(lines))

    async def _cancel_job(self, name_or_id: str) -> ToolResult:
        if not self._store:
            return ToolResult(success=False, output="", error="Scheduler not initialised")

        # Try by name first, then by id
        job = await self._store.get_by_name(name_or_id)
        if job is None:
            # Try get_all and match by id
            all_jobs = await self._store.get_all()
            job = next((j for j in all_jobs if j.id == name_or_id), None)

        if job is None:
            return ToolResult(
                success=False, output="",
                error=f"No job found with name or id: {name_or_id!r}",
            )

        await self._store.delete(job.id)
        return ToolResult(success=True, output=f"Job '{job.name}' (id={job.id}) cancelled.")
