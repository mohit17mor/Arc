"""
Workflow data models — frozen dataclasses for workflow definitions and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING_USER = "waiting_user"


class OnFail(str, Enum):
    STOP = "stop"
    CONTINUE = "continue"


@dataclass
class WorkflowStep:
    """A single step in a workflow.

    Minimal form — just a plain-English instruction::

        steps:
          - search the web for NVIDIA news

    Extended form — with control options::

        steps:
          - do: search the web for NVIDIA news
            retry: 2
            on_fail: continue

    Explicit form — bypass the agent, call tool directly::

        steps:
          - do: get the jira ticket
            tool: mcp_call
            args: {server: jira, tool: get_issue, arguments: {key: "PROJ-123"}}
    """

    instruction: str
    """Plain-English description of what this step should do."""

    index: int = 0
    """0-based position in the workflow."""

    retry: int = 0
    """Number of retries on failure (0 = no retry)."""

    on_fail: OnFail = OnFail.STOP
    """What to do when the step fails after all retries."""

    ask_if_unclear: bool = True
    """If True, the agent should ask the user when it lacks info."""

    # Optional explicit tool call (bypasses agent)
    tool: str | None = None
    args: dict[str, Any] | None = None
    shell: str | None = None

    # Runtime state
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    error: str = ""
    attempts: int = 0


@dataclass
class Workflow:
    """A complete workflow definition loaded from YAML."""

    name: str
    """Human-readable workflow name."""

    steps: list[WorkflowStep]
    """Ordered list of steps to execute."""

    trigger_patterns: list[str] = field(default_factory=list)
    """Regex patterns to match against user input for auto-triggering."""

    description: str = ""
    """Optional description of what this workflow does."""

    source_path: str = ""
    """File path this workflow was loaded from."""


@dataclass
class StepResult:
    """Result of executing a single step."""

    step_index: int
    status: StepStatus
    output: str = ""
    error: str = ""
    attempts: int = 1
    user_questions: list[str] = field(default_factory=list)
    """Questions asked to the user during this step."""


@dataclass
class WorkflowResult:
    """Result of executing a complete workflow."""

    workflow_name: str
    success: bool
    step_results: list[StepResult] = field(default_factory=list)
    final_output: str = ""
    steps_completed: int = 0
    steps_total: int = 0
    error: str = ""
