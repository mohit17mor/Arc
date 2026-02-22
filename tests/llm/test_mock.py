"""Tests for the Mock LLM provider."""

import pytest
from arc.core.types import Message, StopReason, LLMChunk
from arc.llm.mock import MockLLMProvider


@pytest.mark.asyncio
async def test_default_response():
    """Mock returns default response when no responses queued."""
    mock = MockLLMProvider()
    chunks = []

    async for chunk in mock.generate([Message.user("hello")]):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert "mock AI" in chunks[0].text
    assert chunks[1].stop_reason == StopReason.COMPLETE


@pytest.mark.asyncio
async def test_set_response():
    """Mock returns configured text response."""
    mock = MockLLMProvider()
    mock.set_response("Hello, Alex!")

    text = ""
    async for chunk in mock.generate([Message.user("hi")]):
        text += chunk.text

    assert text == "Hello, Alex!"


@pytest.mark.asyncio
async def test_set_multiple_responses():
    """Multiple responses are returned in order."""
    mock = MockLLMProvider()
    mock.set_responses(["First response", "Second response"])

    # First call
    text1 = ""
    async for chunk in mock.generate([Message.user("1")]):
        text1 += chunk.text
    assert text1 == "First response"

    # Second call
    text2 = ""
    async for chunk in mock.generate([Message.user("2")]):
        text2 += chunk.text
    assert text2 == "Second response"


@pytest.mark.asyncio
async def test_tool_call_response():
    """Mock can return tool calls."""
    mock = MockLLMProvider()
    mock.set_tool_call("read_file", {"path": "test.py"})

    chunks = []
    async for chunk in mock.generate([Message.user("read test.py")]):
        chunks.append(chunk)

    last = chunks[-1]
    assert last.stop_reason == StopReason.TOOL_USE
    assert len(last.tool_calls) == 1
    assert last.tool_calls[0].name == "read_file"
    assert last.tool_calls[0].arguments == {"path": "test.py"}


@pytest.mark.asyncio
async def test_tool_call_with_text():
    """Mock can return text before a tool call."""
    mock = MockLLMProvider()
    mock.set_tool_call("read_file", {"path": "x.py"}, text_before="Let me read that...")

    texts = []
    tool_calls = []
    async for chunk in mock.generate([Message.user("read x.py")]):
        if chunk.text:
            texts.append(chunk.text)
        tool_calls.extend(chunk.tool_calls)

    assert "".join(texts) == "Let me read that..."
    assert len(tool_calls) == 1


@pytest.mark.asyncio
async def test_call_tracking():
    """Mock tracks all calls made."""
    mock = MockLLMProvider()
    mock.set_responses(["a", "b"])

    msgs1 = [Message.user("first")]
    msgs2 = [Message.user("second")]

    async for _ in mock.generate(msgs1):
        pass
    async for _ in mock.generate(msgs2):
        pass

    assert mock.call_count == 2
    assert mock.last_messages[0].content == "second"
    assert len(mock.all_calls) == 2
    assert mock.all_calls[0]["call_number"] == 1
    assert mock.all_calls[1]["call_number"] == 2


@pytest.mark.asyncio
async def test_count_tokens():
    """Token counting works with rough estimate."""
    mock = MockLLMProvider()
    count = await mock.count_tokens([
        Message.user("Hello, this is a test message"),
    ])
    assert count > 0


def test_model_info():
    """Model info returns correct metadata."""
    mock = MockLLMProvider(model="test-model", context_window=4096)
    info = mock.get_model_info()

    assert info.provider == "mock"
    assert info.model == "test-model"
    assert info.context_window == 4096
    assert info.cost_per_input_token == 0.0


def test_reset():
    """Reset clears all state."""
    mock = MockLLMProvider()
    mock.set_response("test")
    mock.call_count = 5

    mock.reset()

    assert mock.call_count == 0
    assert mock.all_calls == []