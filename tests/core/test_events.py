"""Tests for the Event system."""

from arc.core.events import Event, EventType


def test_event_creation():
    event = Event(type=EventType.AGENT_THINKING, data={"iteration": 1})

    assert event.type == "agent:thinking"
    assert event.data == {"iteration": 1}
    assert event.id  # auto-generated
    assert event.timestamp > 0
    assert event.parent_id is None
    assert event.metadata == {}
    assert event.source == ""


def test_event_child():
    parent = Event(type=EventType.AGENT_THINKING, source="agent:coder")
    child = parent.child(EventType.LLM_REQUEST, {"model": "llama3"})

    assert child.type == EventType.LLM_REQUEST
    assert child.parent_id == parent.id
    assert child.source == "agent:coder"
    assert child.data == {"model": "llama3"}
    assert child.id != parent.id


def test_event_type_constants():
    assert EventType.SYSTEM_START == "system:start"
    assert EventType.AGENT_THINKING == "agent:thinking"
    assert EventType.SKILL_TOOL_CALL == "skill:tool_call"
    assert EventType.SECURITY_APPROVAL == "security:approval"
    assert EventType.ALL == "*"