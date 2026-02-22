"""
Arc shared types — every data object in the system.

All types are dataclasses. Frozen where immutability makes sense.
These types are used across ALL layers — they live in Layer 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time
import uuid


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class StopReason(str, Enum):
    """Why the LLM stopped generating."""

    COMPLETE = "complete"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    CANCELLED = "cancelled"


class AgentStatus(str, Enum):
    """Current state of the agent loop."""

    IDLE = "idle"
    COMPOSING = "composing"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    COMPLETE = "complete"
    ERROR = "error"


class Capability(str, Enum):
    """What a tool needs permission to do."""

    FILE_READ = "file:read"
    FILE_WRITE = "file:write"
    FILE_DELETE = "file:delete"
    SHELL_EXEC = "shell:exec"
    NETWORK_HTTP = "network:http"
    NETWORK_SOCKET = "network:socket"
    BROWSER = "browser"
    SYSTEM_ENV = "system:env"
    SYSTEM_PROCESS = "system:process"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core Message Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(slots=True)
class ToolCall:
    """A tool/function call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]

    @staticmethod
    def new(name: str, arguments: dict[str, Any]) -> ToolCall:
        return ToolCall(id=uuid.uuid4().hex[:12], name=name, arguments=arguments)


@dataclass(slots=True)
class ToolResult:
    """Result from executing a tool."""

    tool_call_id: str
    success: bool
    output: str
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass(slots=True)
class Message:
    """A single message in a conversation.

    This is the universal message format used across all layers.
    Provider adapters convert to/from their specific formats.
    """

    role: str  # "system", "user", "assistant", "tool"
    content: str | None = None
    name: str | None = None  # for tool messages: tool name
    tool_calls: list[ToolCall] | None = None  # for assistant messages
    tool_call_id: str | None = None  # for tool result messages
    timestamp: float = field(default_factory=time.time)

    @staticmethod
    def system(content: str) -> Message:
        return Message(role="system", content=content)

    @staticmethod
    def user(content: str) -> Message:
        return Message(role="user", content=content)

    @staticmethod
    def assistant(
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> Message:
        return Message(role="assistant", content=content, tool_calls=tool_calls)

    @staticmethod
    def tool_result(tool_call_id: str, content: str, name: str = "") -> Message:
        return Message(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Static metadata about an LLM model."""

    provider: str  # "ollama", "anthropic", "openai"
    model: str  # "llama3.1", "claude-sonnet-4-20250514"
    context_window: int  # max input tokens
    max_output_tokens: int  # max output tokens
    cost_per_input_token: float = 0.0  # USD, 0 for free/local
    cost_per_output_token: float = 0.0  # USD, 0 for free/local
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True


@dataclass(slots=True)
class LLMChunk:
    """A single chunk from a streaming LLM response."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason | None = None
    input_tokens: int = 0
    output_tokens: int = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shell Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(slots=True)
class ShellSession:
    """A persistent shell session."""

    id: str
    provider: str  # "powershell", "bash", etc.
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    is_alive: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class ShellOutput:
    """Result from a shell command execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    duration_ms: int = 0
    timed_out: bool = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Skill Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Tool specification — everything the LLM needs to call it."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    required_capabilities: frozenset[Capability] = frozenset()


@dataclass(frozen=True, slots=True)
class SkillManifest:
    """Metadata about a skill — its identity, tools, and requirements."""

    name: str
    version: str
    description: str
    author: str = ""
    dependencies: tuple[str, ...] = ()
    capabilities: frozenset[Capability] = frozenset()
    tools: tuple[ToolSpec, ...] = ()
    config_schema: dict[str, Any] = field(default_factory=dict)
    platforms: tuple[str, ...] = ("all",)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Memory Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(slots=True)
class MemoryEntry:
    """A single unit of memory."""

    id: str
    content: str
    entry_type: str  # "fact", "preference", "episode", "message", "correction"
    source: str  # "identity", "agent:coder", "skill:git"
    timestamp: float = field(default_factory=time.time)
    embedding: list[float] | None = None
    relevance_score: float = 0.0  # set during retrieval
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ComposedContext:
    """The assembled working memory for one LLM call."""

    messages: list[Message]
    token_count: int
    token_budget: int
    breakdown: dict[str, int] = field(default_factory=dict)
    retrieved_memories: list[MemoryEntry] = field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Security Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(slots=True)
class SecurityDecision:
    """Result of a security check."""

    allowed: bool
    reason: str  # "policy:auto_allow", "user:approved", "policy:denied"
    requires_approval: bool = False
    user_response: str | None = None  # "allow_once", "allow_always", "deny"
    remembered: bool = False


@dataclass(slots=True)
class AuditEntry:
    """A single entry in the security audit trail."""

    id: str
    timestamp: float
    agent: str
    skill: str
    tool: str
    capability: str
    arguments: dict[str, Any]
    decision: SecurityDecision
    result: ToolResult | None = None
    user_id: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(slots=True)
class AgentState:
    """Complete state of a running agent — for checkpointing."""

    agent_id: str
    recipe_name: str | None = None
    conversation: list[Message] = field(default_factory=list)
    iteration: int = 0
    status: AgentStatus = AgentStatus.IDLE
    current_task: str | None = None
    skill_states: dict[str, Any] = field(default_factory=dict)
    cost_so_far: float = 0.0
    tokens_used: int = 0
    started_at: float = field(default_factory=time.time)