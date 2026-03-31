"""
Arc Event System — types and constants.

Every action in the system produces an event.
Events flow through the middleware chain, then to subscribers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time
import uuid


class EventType:
    """
    Event type constants.

    Hierarchical naming: "category:action"
    Supports wildcard matching: "agent:*" matches "agent:thinking"
    """

    # System lifecycle
    SYSTEM_START = "system:start"
    SYSTEM_STOP = "system:stop"
    SYSTEM_ERROR = "system:error"

    # Agent states
    AGENT_START = "agent:start"
    AGENT_THINKING = "agent:thinking"
    AGENT_RESPONSE = "agent:response"
    AGENT_ERROR = "agent:error"
    AGENT_PAUSED = "agent:paused"
    AGENT_RESUMED = "agent:resumed"
    AGENT_COMPLETE = "agent:complete"

    # LLM interactions
    LLM_REQUEST = "llm:request"
    LLM_CHUNK = "llm:chunk"
    LLM_RESPONSE = "llm:response"
    LLM_ERROR = "llm:error"
    LLM_COST = "llm:cost"

    # Skill/tool interactions
    SKILL_LOADED = "skill:loaded"
    SKILL_ACTIVATED = "skill:activated"
    SKILL_TOOL_CALL = "skill:tool_call"
    SKILL_TOOL_RESULT = "skill:tool_result"
    SKILL_ERROR = "skill:error"
    SKILL_DEACTIVATED = "skill:deactivated"

    # Security
    SECURITY_CHECK = "security:check"
    SECURITY_ALLOWED = "security:allowed"
    SECURITY_DENIED = "security:denied"
    SECURITY_APPROVAL = "security:approval"
    SECURITY_AUDIT = "security:audit"

    # Memory
    MEMORY_STORE = "memory:store"
    MEMORY_RETRIEVE = "memory:retrieve"
    MEMORY_COMPOSE = "memory:compose"

    # User interactions
    USER_MESSAGE = "user:message"
    USER_INTERRUPT = "user:interrupt"
    USER_APPROVAL = "user:approval"
    USER_CORRECTION = "user:correction"

    # Multi-agent coordination
    AGENT_SPAWNED = "agent:spawned"           # new worker/expert created
    AGENT_TERMINATED = "agent:terminated"     # worker/expert finished or cancelled
    AGENT_TASK_COMPLETE = "agent:task_complete"  # worker finished, result ready
    AGENT_ESCALATION = "agent:escalation"     # worker/expert needs user input via main
    AGENT_PLAN_UPDATE = "agent:plan_update"   # agent created/updated execution plan

    # Task board
    TASK_QUEUED = "task:queued"               # new task created
    TASK_START = "task:start"                 # agent picked up a task
    TASK_REVIEW = "task:review"               # task submitted for review
    TASK_BOUNCED = "task:bounced"             # reviewer sent task back
    TASK_BLOCKED = "task:blocked"             # agent needs human input
    TASK_COMPLETE = "task:complete"           # task finished successfully
    TASK_FAILED = "task:failed"              # task failed

    # Workflow execution
    WORKFLOW_START = "workflow:start"          # workflow begun
    WORKFLOW_STEP_START = "workflow:step_start"    # step N starting
    WORKFLOW_STEP_COMPLETE = "workflow:step_complete"  # step N done
    WORKFLOW_STEP_FAILED = "workflow:step_failed"  # step N failed
    WORKFLOW_COMPLETE = "workflow:complete"    # workflow finished successfully
    WORKFLOW_FAILED = "workflow:failed"        # workflow stopped due to failure
    WORKFLOW_PAUSED = "workflow:paused"        # waiting for user input
    WORKFLOW_WAITING_INPUT = "workflow:waiting_input"  # step needs user response to continue

    # Workspace rendering
    WORKSPACE_UPDATE = "workspace:update"

    # Wildcard
    ALL = "*"


@dataclass(slots=True)
class Event:
    """
    A single event in the Arc system.

    Every action produces an event. Events are:
    - Typed (hierarchical string)
    - Timestamped
    - Traceable (source + parent_id for causal chains)
    - Extensible (data dict for event-specific payload)
    - Enrichable (metadata dict for middleware annotations)
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def child(self, event_type: str, data: dict[str, Any] | None = None) -> Event:
        """Create a child event linked to this one."""
        return Event(
            type=event_type,
            data=data or {},
            source=self.source,
            parent_id=self.id,
        )
