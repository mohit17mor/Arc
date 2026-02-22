"""
Session Memory — tracks conversation for current session.

Simple in-memory storage of messages and tool results.
Context composition builds the messages list for each LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arc.core.types import Message, ToolResult


@dataclass
class SessionMemory:
    """
    Holds conversation history for a single session.

    Usage:
        memory = SessionMemory()
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi there!")
        memory.add_tool_result(tool_result)

        messages = memory.get_messages()
    """

    messages: list[Message] = field(default_factory=list)
    _system_prompt: str = ""

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt (called once at start)."""
        self._system_prompt = prompt

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.messages.append(Message.user(content))

    def add_assistant_message(
        self,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
    ) -> None:
        """Add an assistant message."""
        self.messages.append(Message.assistant(content, tool_calls))

    def add_tool_result(self, result: ToolResult, tool_name: str = "") -> None:
        """Add a tool result message."""
        self.messages.append(
            Message.tool_result(
                tool_call_id=result.tool_call_id,
                content=result.output if result.success else f"Error: {result.error}",
                name=tool_name,
            )
        )

    def get_messages(self, include_system: bool = True) -> list[Message]:
        """Get all messages, optionally including system prompt."""
        result = []
        if include_system and self._system_prompt:
            result.append(Message.system(self._system_prompt))
        result.extend(self.messages)
        return result

    def get_recent_messages(
        self,
        n: int,
        include_system: bool = True,
    ) -> list[Message]:
        """Get the last N messages."""
        result = []
        if include_system and self._system_prompt:
            result.append(Message.system(self._system_prompt))
        result.extend(self.messages[-n:] if n > 0 else [])
        return result

    def clear(self) -> None:
        """Clear all messages but keep system prompt."""
        self.messages.clear()

    @property
    def message_count(self) -> int:
        """Number of messages (excluding system prompt)."""
        return len(self.messages)