"""
Arc — Micro-agents you can teach, share, and compose.

Public API:
    from arc import Agent, Skill, tool, Kernel
"""

__version__ = "0.1.0"

# Core
from arc.core.kernel import Kernel
from arc.core.config import ArcConfig
from arc.core.events import Event, EventType
from arc.core.types import Message, ToolCall, ToolResult

# Skills
from arc.skills.base import Skill, tool, FunctionSkill
from arc.skills.manager import SkillManager

# Agent
from arc.agent.loop import AgentLoop, AgentConfig

__all__ = [
    # Core
    "Kernel",
    "ArcConfig",
    "Event",
    "EventType",
    "Message",
    "ToolCall",
    "ToolResult",
    # Skills
    "Skill",
    "tool",
    "FunctionSkill",
    "SkillManager",
    # Agent
    "AgentLoop",
    "AgentConfig",
]