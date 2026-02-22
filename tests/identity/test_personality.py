"""Tests for personality system."""

from arc.identity.personality import (
    get_personality,
    list_personalities,
    PERSONALITIES,
)


def test_list_personalities():
    """All predefined personalities are listed."""
    personalities = list_personalities()
    assert len(personalities) >= 5
    names = [p.name for p in personalities]
    assert "Helpful Assistant" in names
    assert "Sarcastic Sidekick" in names


def test_get_personality_valid():
    """Can get a personality by ID."""
    p = get_personality("sarcastic")
    assert p.id == "sarcastic"
    assert p.name == "Sarcastic Sidekick"
    assert p.emoji == "😏"
    assert "sarcastic" in p.system_prompt.lower()


def test_get_personality_invalid():
    """Invalid ID returns default personality."""
    p = get_personality("nonexistent")
    assert p.id == "helpful"


def test_personality_has_system_prompt():
    """All personalities have system prompts."""
    for p in PERSONALITIES.values():
        assert p.system_prompt
        assert len(p.system_prompt) > 50