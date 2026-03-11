"""
Task types — data models for the task queue system.

All types are plain dataclasses with to_dict/from_dict for SQLite
serialization.  No ORM, no magic — same pattern as scheduler.job.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Lifecycle states for a task."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    REVISION_NEEDED = "revision_needed"
    AWAITING_HUMAN = "awaiting_human"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Lifecycle states for an individual workflow step."""

    PENDING = "pending"
    ACTIVE = "active"
    IN_REVIEW = "in_review"
    DONE = "done"


@dataclass
class TaskStep:
    """One step in a multi-agent workflow."""

    step_index: int
    agent_name: str
    review_by: str | None = None  # agent name or "human"

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "agent_name": self.agent_name,
            "review_by": self.review_by,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskStep:
        return cls(
            step_index=d["step_index"],
            agent_name=d["agent_name"],
            review_by=d.get("review_by"),
        )


@dataclass
class TaskComment:
    """An entry in a task's audit trail / communication log."""

    id: int
    task_id: str
    step_index: int
    agent_name: str  # agent name, "human", or "system"
    content: str
    created_at: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "step_index": self.step_index,
            "agent_name": self.agent_name,
            "content": self.content,
            "created_at": self.created_at,
        }


@dataclass
class Task:
    """A single queued task with optional multi-step workflow."""

    title: str
    instruction: str

    # Workflow (empty = single-agent, uses assigned_agent directly)
    steps: list[TaskStep] = field(default_factory=list)
    current_step: int = 0

    # Assignment (used when steps is empty)
    assigned_agent: str = ""

    # Review loop
    bounce_count: int = 0
    max_bounces: int = 3

    # State
    id: str = field(default_factory=lambda: f"t-{uuid.uuid4().hex[:8]}")
    status: TaskStatus = TaskStatus.QUEUED
    priority: int = 1  # lower = higher priority
    result: str = ""

    # Dependencies
    depends_on: str | None = None  # task ID that must complete first

    # Timestamps
    created_at: int = field(default_factory=lambda: int(time.time()))
    started_at: int = 0
    completed_at: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "instruction": self.instruction,
            "steps": [s.to_dict() for s in self.steps],
            "current_step": self.current_step,
            "assigned_agent": self.assigned_agent,
            "bounce_count": self.bounce_count,
            "max_bounces": self.max_bounces,
            "status": self.status.value,
            "priority": self.priority,
            "result": self.result,
            "depends_on": self.depends_on,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Task:
        return cls(
            id=d["id"],
            title=d["title"],
            instruction=d["instruction"],
            steps=[TaskStep.from_dict(s) for s in d.get("steps", [])],
            current_step=d.get("current_step", 0),
            assigned_agent=d.get("assigned_agent", ""),
            bounce_count=d.get("bounce_count", 0),
            max_bounces=d.get("max_bounces", 3),
            status=TaskStatus(d["status"]),
            priority=d.get("priority", 1),
            result=d.get("result", ""),
            depends_on=d.get("depends_on"),
            created_at=d["created_at"],
            started_at=d.get("started_at", 0),
            completed_at=d.get("completed_at", 0),
        )

    @property
    def current_agent(self) -> str:
        """The agent that should handle the current step."""
        if self.steps:
            return self.steps[self.current_step].agent_name
        return self.assigned_agent

    @property
    def current_reviewer(self) -> str | None:
        """The reviewer for the current step, or None."""
        if self.steps and self.current_step < len(self.steps):
            return self.steps[self.current_step].review_by
        return None

    @property
    def is_terminal(self) -> bool:
        """Whether the task is in a final state."""
        return self.status in (
            TaskStatus.DONE,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )


@dataclass
class AgentDef:
    """Definition of a named agent loaded from TOML.

    The system prompt can be provided in two ways (highest priority wins):
      1. ``system_prompt`` field in the TOML (supports multi-line triple-quoted strings)
      2. Auto-generated from ``role`` + ``personality`` fields
    """

    name: str
    role: str
    personality: str = ""
    system_prompt: str = ""                  # from TOML system_prompt field
    skills: list[str] | None = None        # whitelist (None = all)
    exclude_skills: list[str] | None = None  # blacklist
    max_concurrent: int = 1

    # Per-agent LLM override
    llm_provider: str = ""
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""

    @property
    def has_llm_override(self) -> bool:
        return bool(self.llm_provider and self.llm_model)

    def build_system_prompt(self) -> str:
        """Return the system prompt for this agent.

        If a dedicated prompt was loaded from ``<name>.md``, use it verbatim.
        Otherwise auto-generate a basic one from role + personality.
        """
        if self.system_prompt:
            return self.system_prompt

        parts = [f"You are {self.name}, a specialized agent."]
        if self.role:
            parts.append(f"Your role: {self.role}")
        if self.personality:
            parts.append(f"Personality: {self.personality}")
        parts.append(
            "\nYou are executing a task from a task queue. "
            "Focus entirely on the task instruction. "
            "Be THOROUGH and PERSISTENT:\n"
            "- If a tool call fails or returns poor results, TRY AGAIN with different inputs.\n"
            "- Rephrase search queries if the first attempt doesn't find what you need.\n"
            "- Try at least 2-3 different approaches before giving up on any sub-goal.\n"
            "- If one source doesn't have the answer, look for alternative sources.\n"
            "- Do NOT give up after a single failed attempt — always have a Plan B.\n"
            "- When done, provide a clear, structured, comprehensive result."
        )
        return "\n".join(parts)
