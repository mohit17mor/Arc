"""
TaskSkill — lets the LLM create, list, and manage queued tasks.

This is the conversational interface to the task queue.  The user says
"queue these for the researcher" and the LLM calls queue_task().

Tools:
    queue_task      — create a new task (simple or multi-step)
    list_tasks      — show task statuses
    task_detail     — show full detail + comments for a task
    cancel_task     — cancel a queued/running task
    clear_tasks     — permanently delete completed/cancelled tasks
    reply_to_task   — answer a blocked task's question or approve/revise
    list_agents     — show available named agents
"""

from __future__ import annotations

import logging
from typing import Any

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec
from arc.skills.base import Skill
from arc.tasks.store import TaskStore
from arc.tasks.types import AgentDef, Task, TaskStatus, TaskStep

logger = logging.getLogger(__name__)


class TaskSkill(Skill):
    """
    Skill for managing the persistent task queue via LLM tools.

    Store + agents + processor are injected via set_dependencies().
    """

    def __init__(self) -> None:
        self._store: TaskStore | None = None
        self._agents: dict[str, AgentDef] = {}
        self._processor: Any = None  # TaskProcessor
        self._kernel: Any = None
        self._config: dict = {}

    def set_dependencies(
        self,
        store: TaskStore,
        agents: dict[str, AgentDef],
        processor: Any,
    ) -> None:
        self._store = store
        self._agents = agents
        self._processor = processor

    # ── Skill ABC ────────────────────────────────────────────────────────────

    def manifest(self) -> SkillManifest:
        agent_list = ", ".join(self._agents.keys()) if self._agents else "none configured"
        return SkillManifest(
            name="task_board",
            version="1.0.0",
            description=(
                "Queue persistent tasks for named agents. Tasks survive across sessions "
                "and are processed in the background. "
                f"Available agents: {agent_list}"
            ),
            capabilities=frozenset([Capability.FILE_READ]),
            always_available=True,
            tools=[
                ToolSpec(
                    name="queue_task",
                    description=(
                        "Create a task and queue it for processing by a named agent. "
                        "The task persists even if the user closes the terminal. "
                        "Results are delivered via notifications (Telegram, CLI). "
                        "For multi-step workflows, provide steps as a list of agent assignments. "
                        "For simple single-agent tasks, just set assigned_agent."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short descriptive title for the task.",
                            },
                            "instruction": {
                                "type": "string",
                                "description": (
                                    "Full instructions for the agent. Include all context — "
                                    "the agent has no conversation history."
                                ),
                            },
                            "assigned_agent": {
                                "type": "string",
                                "description": (
                                    "Name of the agent to assign this task to. "
                                    "Use list_agents to see available agents."
                                ),
                            },
                            "steps": {
                                "type": "array",
                                "description": (
                                    "Optional: define a multi-step workflow. Each step "
                                    "specifies an agent and optionally a reviewer. "
                                    "Output flows from step to step. "
                                    "Omit for simple single-agent tasks."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "agent": {
                                            "type": "string",
                                            "description": "Agent name for this step.",
                                        },
                                        "review_by": {
                                            "type": "string",
                                            "description": (
                                                "Optional: agent name or 'human' to review "
                                                "this step's output before advancing."
                                            ),
                                        },
                                    },
                                    "required": ["agent"],
                                },
                            },
                            "priority": {
                                "type": "integer",
                                "description": "Priority (1=highest). Default 1.",
                            },
                            "max_bounces": {
                                "type": "integer",
                                "description": (
                                    "Max review iterations before auto-completing. Default 3."
                                ),
                            },
                            "depends_on": {
                                "type": "string",
                                "description": (
                                    "Optional: task ID that must complete before this one starts. "
                                    "The completed task's result is injected as context."
                                ),
                            },
                        },
                        "required": ["title", "instruction"],
                    },
                    required_capabilities=frozenset([Capability.FILE_READ]),
                ),
                ToolSpec(
                    name="list_tasks",
                    description=(
                        "List tasks in the queue with their current status. "
                        "Optionally filter by status."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "description": (
                                    "Filter by status: queued, in_progress, in_review, "
                                    "awaiting_human, blocked, done, failed, cancelled. "
                                    "Omit to show all."
                                ),
                            },
                        },
                    },
                    required_capabilities=frozenset(),
                ),
                ToolSpec(
                    name="task_detail",
                    description="Show full details and comment history for a specific task.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "The task ID (e.g. 't-a1b2c3d4').",
                            },
                        },
                        "required": ["task_id"],
                    },
                    required_capabilities=frozenset(),
                ),
                ToolSpec(
                    name="cancel_task",
                    description="Cancel a task that hasn't completed yet.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "The task ID to cancel.",
                            },
                        },
                        "required": ["task_id"],
                    },
                    required_capabilities=frozenset(),
                ),
                ToolSpec(
                    name="clear_tasks",
                    description=(
                        "Permanently delete completed, failed, or cancelled tasks from the queue. "
                        "Use this to clean up finished tasks and reclaim storage."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "task_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "One or more task IDs to permanently delete.",
                            },
                        },
                        "required": ["task_ids"],
                    },
                    required_capabilities=frozenset(),
                ),
                ToolSpec(
                    name="reply_to_task",
                    description=(
                        "Reply to a task that is blocked or awaiting human review. "
                        "Use action='approve' to approve and advance, or "
                        "action='revise' to send back with feedback."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "The task ID to reply to.",
                            },
                            "reply": {
                                "type": "string",
                                "description": "Your response text.",
                            },
                            "action": {
                                "type": "string",
                                "enum": ["approve", "revise"],
                                "description": (
                                    "For awaiting_human tasks: 'approve' to advance, "
                                    "'revise' to send back. For blocked tasks: ignored "
                                    "(reply is always delivered as an answer)."
                                ),
                            },
                        },
                        "required": ["task_id", "reply"],
                    },
                    required_capabilities=frozenset(),
                ),
                ToolSpec(
                    name="list_agents",
                    description="List all configured named agents and their roles.",
                    parameters={
                        "type": "object",
                        "properties": {},
                    },
                    required_capabilities=frozenset(),
                ),
            ],
        )

    async def initialize(self, kernel: Any, config: dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = config

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> ToolResult:
        if tool_name == "queue_task":
            return await self._queue_task(**arguments)
        if tool_name == "list_tasks":
            return await self._list_tasks(arguments.get("status"))
        if tool_name == "task_detail":
            return await self._task_detail(arguments.get("task_id", ""))
        if tool_name == "cancel_task":
            return await self._cancel_task(arguments.get("task_id", ""))
        if tool_name == "clear_tasks":
            return await self._clear_tasks(arguments.get("task_ids", []))
        if tool_name == "reply_to_task":
            return await self._reply_to_task(
                arguments.get("task_id", ""),
                arguments.get("reply", ""),
                arguments.get("action", "approve"),
            )
        if tool_name == "list_agents":
            return self._list_agents()
        return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")

    # ── Tool implementations ─────────────────────────────────────────────────

    async def _queue_task(
        self,
        title: str,
        instruction: str,
        assigned_agent: str = "",
        steps: list[dict] | None = None,
        priority: int = 1,
        max_bounces: int = 3,
        depends_on: str | None = None,
    ) -> ToolResult:
        if not self._store:
            return ToolResult(success=False, output="", error="Task store not initialised")

        # Build steps
        task_steps: list[TaskStep] = []
        if steps:
            for i, s in enumerate(steps):
                agent = s.get("agent", "")
                if agent and agent not in self._agents:
                    return ToolResult(
                        success=False, output="",
                        error=f"Unknown agent '{agent}' in step {i+1}. "
                              f"Available: {', '.join(self._agents.keys())}",
                    )
                task_steps.append(TaskStep(
                    step_index=i,
                    agent_name=agent,
                    review_by=s.get("review_by"),
                ))

        # For single-agent tasks
        if not task_steps:
            if not assigned_agent:
                return ToolResult(
                    success=False, output="",
                    error="Either 'assigned_agent' or 'steps' is required.",
                )
            if assigned_agent not in self._agents:
                return ToolResult(
                    success=False, output="",
                    error=f"Unknown agent '{assigned_agent}'. "
                          f"Available: {', '.join(self._agents.keys())}",
                )
            # Create a single-step workflow for consistency
            task_steps = [TaskStep(step_index=0, agent_name=assigned_agent)]

        # Validate reviewers
        for s in task_steps:
            if s.review_by and s.review_by != "human" and s.review_by not in self._agents:
                return ToolResult(
                    success=False, output="",
                    error=f"Unknown reviewer '{s.review_by}' in step {s.step_index+1}. "
                          f"Available: {', '.join(self._agents.keys())} or 'human'",
                )

        # Validate dependency
        if depends_on:
            dep = await self._store.get_by_id(depends_on)
            if not dep:
                return ToolResult(
                    success=False, output="",
                    error=f"Dependency task '{depends_on}' not found.",
                )

        task = Task(
            title=title,
            instruction=instruction,
            steps=task_steps,
            assigned_agent=assigned_agent or task_steps[0].agent_name,
            priority=max(1, min(priority, 10)),
            max_bounces=max(1, min(max_bounces, 10)),
            depends_on=depends_on,
        )

        await self._store.save(task)

        step_desc = ""
        if len(task_steps) > 1:
            parts = []
            for s in task_steps:
                desc = s.agent_name
                if s.review_by:
                    desc += f" (reviewed by {s.review_by})"
                parts.append(desc)
            step_desc = f"\nWorkflow: {' → '.join(parts)}"

        dep_desc = f"\nWaiting for: {depends_on}" if depends_on else ""

        return ToolResult(
            success=True,
            output=(
                f"Task queued: {task.title} (id: {task.id})\n"
                f"First agent: {task.current_agent}\n"
                f"Priority: {task.priority}"
                f"{step_desc}{dep_desc}\n"
                f"The task will be processed in the background."
            ),
        )

    async def _list_tasks(self, status: str | None) -> ToolResult:
        if not self._store:
            return ToolResult(success=False, output="", error="Task store not initialised")

        tasks = await self._store.get_all(status=status, limit=30)
        if not tasks:
            msg = "No tasks found."
            if status:
                msg = f"No tasks with status '{status}'."
            return ToolResult(success=True, output=msg)

        lines = []
        for t in tasks:
            agent = t.current_agent
            line = f"  {t.id}  [{t.status.value:<16}]  {t.title}  (agent: {agent})"
            if t.depends_on:
                line += f"  [depends on {t.depends_on}]"
            lines.append(line)

        return ToolResult(
            success=True,
            output=f"{len(tasks)} task(s):\n" + "\n".join(lines),
        )

    async def _task_detail(self, task_id: str) -> ToolResult:
        if not self._store:
            return ToolResult(success=False, output="", error="Task store not initialised")

        task = await self._store.get_by_id(task_id)
        if not task:
            return ToolResult(success=False, output="", error=f"Task '{task_id}' not found.")

        comments = await self._store.get_comments(task_id)

        lines = [
            f"Task: {task.title}",
            f"ID: {task.id}",
            f"Status: {task.status.value}",
            f"Instruction: {task.instruction}",
            f"Current agent: {task.current_agent}",
            f"Step: {task.current_step + 1}/{len(task.steps) or 1}",
            f"Bounces: {task.bounce_count}/{task.max_bounces}",
            f"Priority: {task.priority}",
        ]
        if task.depends_on:
            lines.append(f"Depends on: {task.depends_on}")
        if task.result:
            lines.append(f"\nFinal result:\n{task.result[:500]}")

        if comments:
            lines.append(f"\nComments ({len(comments)}):")
            for c in comments:
                lines.append(f"  [{c.agent_name}] {c.content[:200]}")

        return ToolResult(success=True, output="\n".join(lines))

    async def _cancel_task(self, task_id: str) -> ToolResult:
        if not self._store:
            return ToolResult(success=False, output="", error="Task store not initialised")

        ok = await self._store.cancel(task_id)
        if ok:
            return ToolResult(success=True, output=f"Task {task_id} cancelled.")
        return ToolResult(
            success=False, output="",
            error=f"Task {task_id} not found or already completed.",
        )

    async def _clear_tasks(self, task_ids: list[str]) -> ToolResult:
        if not self._store:
            return ToolResult(success=False, output="", error="Task store not initialised")
        if not task_ids:
            return ToolResult(success=False, output="", error="At least one task_id is required.")

        deleted = await self._store.clear_tasks(task_ids, only_terminal=True)
        if deleted == 0:
            return ToolResult(
                success=False,
                output="",
                error="No matching completed, failed, or cancelled tasks could be cleared.",
            )

        return ToolResult(
            success=True,
            output=f"Cleared {deleted} task(s) from the queue.",
        )

    async def _reply_to_task(
        self, task_id: str, reply: str, action: str,
    ) -> ToolResult:
        if not self._processor:
            return ToolResult(success=False, output="", error="Task processor not available")

        msg = await self._processor.handle_human_reply(task_id, reply, action)
        return ToolResult(success=True, output=msg)

    def _list_agents(self) -> ToolResult:
        if not self._agents:
            return ToolResult(
                success=True,
                output="No agents configured. Create one with: arc agent create <name>",
            )
        lines = []
        for a in self._agents.values():
            llm = f"{a.llm_provider}/{a.llm_model}" if a.has_llm_override else "default"
            lines.append(
                f"  {a.name:<20} {a.role:<40} (LLM: {llm}, max_concurrent: {a.max_concurrent})"
            )
        return ToolResult(
            success=True,
            output=f"{len(self._agents)} agent(s):\n" + "\n".join(lines),
        )
