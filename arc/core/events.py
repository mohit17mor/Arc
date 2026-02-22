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