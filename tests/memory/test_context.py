"""Tests for context composer."""

import pytest
from pathlib import Path
from arc.core.types import Message
from arc.memory.context import ContextComposer
from arc.memory.session import SessionMemory
from arc.memory.manager import MemoryManager
from arc.memory.embedding import MockEmbeddingProvider


async def mock_token_counter(messages: list[Message]) -> int:
    """Simple token counter: 10 tokens per message."""
    return len(messages) * 10


DIM = 8


@pytest.fixture
async def memory_manager(tmp_path: Path) -> MemoryManager:
    embedder = MockEmbeddingProvider(dimension=DIM)
    mm = MemoryManager(
        db_path=str(tmp_path / "ctx_test.db"),
        embedding_provider=embedder,
        embed_dim=DIM,
    )
    await mm.initialize()
    return mm


@pytest.mark.asyncio
async def test_compose_simple():
    """Simple composition with all messages fitting."""
    composer = ContextComposer(
        token_counter=mock_token_counter,
        max_tokens=1000,
        reserve_output=100,
    )

    memory = SessionMemory()
    memory.set_system_prompt("You are helpful.")
    memory.add_user_message("Hello")
    memory.add_assistant_message("Hi!")

    context = await composer.compose(memory)

    assert len(context.messages) == 3
    assert context.token_count == 30
    assert context.token_budget == 900


@pytest.mark.asyncio
async def test_compose_truncation():
    """Messages are truncated when exceeding budget."""
    composer = ContextComposer(
        token_counter=mock_token_counter,
        max_tokens=50,  # Very small
        reserve_output=10,
    )

    memory = SessionMemory()
    memory.set_system_prompt("System")
    for i in range(10):
        memory.add_user_message(f"Message {i}")

    context = await composer.compose(memory, recent_window=5)

    # Should have system + some recent messages
    assert len(context.messages) < 11
    assert context.token_count <= 40  # max - reserve


@pytest.mark.asyncio
async def test_compose_keeps_system():
    """System prompt is always kept."""
    composer = ContextComposer(
        token_counter=mock_token_counter,
        max_tokens=30,
        reserve_output=10,
    )

    memory = SessionMemory()
    memory.set_system_prompt("Important system prompt")
    for i in range(10):
        memory.add_user_message(f"Message {i}")

    context = await composer.compose(memory)

    assert context.messages[0].role == "system"
    assert "Important" in context.messages[0].content


@pytest.mark.asyncio
async def test_token_budget():
    """Token budget is calculated correctly."""
    composer = ContextComposer(
        token_counter=mock_token_counter,
        max_tokens=128000,
        reserve_output=8192,
    )

    assert composer.token_budget == 128000 - 8192


# ── With memory_manager ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compose_with_empty_memory_manager(memory_manager):
    """Should behave like backward compat mode when memory is empty."""
    composer = ContextComposer(
        token_counter=mock_token_counter,
        max_tokens=1000,
        reserve_output=100,
    )
    memory = SessionMemory()
    memory.set_system_prompt("Base system prompt.")
    memory.add_user_message("Hello")

    context = await composer.compose(
        memory,
        query="Hello",
        memory_manager=memory_manager,
    )

    assert len(context.messages) >= 2
    assert context.messages[0].role == "system"
    assert "Base system prompt" in context.messages[0].content


@pytest.mark.asyncio
async def test_compose_injects_core_facts(memory_manager):
    """Core facts should appear in the system prompt."""
    await memory_manager.upsert_core("user_name", "User's name is Zara")

    composer = ContextComposer(
        token_counter=mock_token_counter,
        max_tokens=1000,
        reserve_output=100,
    )
    memory = SessionMemory()
    memory.set_system_prompt("Base prompt.")
    memory.add_user_message("Hi")

    context = await composer.compose(
        memory,
        query="Hi",
        memory_manager=memory_manager,
    )

    system_content = context.messages[0].content
    assert "Zara" in system_content


@pytest.mark.asyncio
async def test_compose_without_memory_manager_unmodified():
    """Without memory_manager, system prompt content is not augmented."""
    composer = ContextComposer(
        token_counter=mock_token_counter,
        max_tokens=1000,
        reserve_output=100,
    )
    memory = SessionMemory()
    memory.set_system_prompt("Clean system prompt.")
    memory.add_user_message("Question")

    context = await composer.compose(memory, query="Question", memory_manager=None)

    assert context.messages[0].content == "Clean system prompt."
