"""Tests for session memory."""

import pytest
from arc.core.types import ToolCall, ToolResult
from arc.memory.session import SessionMemory


def test_empty_memory():
    """New memory is empty."""
    memory = SessionMemory()
    assert memory.message_count == 0
    assert memory.get_messages(include_system=False) == []


def test_system_prompt():
    """System prompt is included when requested."""
    memory = SessionMemory()
    memory.set_system_prompt("You are helpful.")

    with_system = memory.get_messages(include_system=True)
    assert len(with_system) == 1
    assert with_system[0].role == "system"
    assert with_system[0].content == "You are helpful."

    without_system = memory.get_messages(include_system=False)
    assert len(without_system) == 0


def test_add_user_message():
    """User messages are added correctly."""
    memory = SessionMemory()
    memory.add_user_message("Hello")
    memory.add_user_message("World")

    messages = memory.get_messages(include_system=False)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Hello"
    assert messages[1].content == "World"


def test_add_assistant_message():
    """Assistant messages are added correctly."""
    memory = SessionMemory()
    memory.add_assistant_message("Hi there!")

    messages = memory.get_messages(include_system=False)
    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].content == "Hi there!"


def test_add_assistant_with_tool_calls():
    """Assistant messages can include tool calls."""
    memory = SessionMemory()
    tool_calls = [ToolCall(id="1", name="read_file", arguments={"path": "x.py"})]
    memory.add_assistant_message(content="Let me read that.", tool_calls=tool_calls)

    messages = memory.get_messages(include_system=False)
    assert len(messages) == 1
    assert messages[0].tool_calls == tool_calls


def test_add_tool_result():
    """Tool results are added correctly."""
    memory = SessionMemory()
    result = ToolResult(
        tool_call_id="123",
        success=True,
        output="file contents here",
    )
    memory.add_tool_result(result, tool_name="read_file")

    messages = memory.get_messages(include_system=False)
    assert len(messages) == 1
    assert messages[0].role == "tool"
    assert messages[0].content == "file contents here"
    assert messages[0].tool_call_id == "123"


def test_add_tool_result_error():
    """Failed tool results include error message."""
    memory = SessionMemory()
    result = ToolResult(
        tool_call_id="456",
        success=False,
        output="",
        error="File not found",
    )
    memory.add_tool_result(result)

    messages = memory.get_messages(include_system=False)
    assert "Error:" in messages[0].content
    assert "File not found" in messages[0].content


def test_get_recent_messages():
    """get_recent_messages returns last N messages."""
    memory = SessionMemory()
    memory.set_system_prompt("System")
    for i in range(5):
        memory.add_user_message(f"Message {i}")

    recent = memory.get_recent_messages(2, include_system=True)
    assert len(recent) == 3  # system + 2 recent
    assert recent[0].role == "system"
    assert recent[1].content == "Message 3"
    assert recent[2].content == "Message 4"


def test_clear():
    """clear removes messages but keeps system prompt."""
    memory = SessionMemory()
    memory.set_system_prompt("System")
    memory.add_user_message("Hello")
    memory.add_assistant_message("Hi")

    memory.clear()

    assert memory.message_count == 0
    messages = memory.get_messages(include_system=True)
    assert len(messages) == 1
    assert messages[0].role == "system"


def test_conversation_flow():
    """Full conversation flow works correctly."""
    memory = SessionMemory()
    memory.set_system_prompt("You are a helpful assistant.")

    # User asks
    memory.add_user_message("What files are here?")

    # Assistant calls tool
    tc = ToolCall(id="1", name="list_directory", arguments={"path": "."})
    memory.add_assistant_message("Let me check.", tool_calls=[tc])

    # Tool result
    memory.add_tool_result(
        ToolResult(tool_call_id="1", success=True, output="file1.py\nfile2.py"),
        tool_name="list_directory",
    )

    # Assistant responds
    memory.add_assistant_message("I found file1.py and file2.py.")

    messages = memory.get_messages()
    assert len(messages) == 5  # system + user + assistant + tool + assistant
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert messages[2].role == "assistant"
    assert messages[3].role == "tool"
    assert messages[4].role == "assistant"