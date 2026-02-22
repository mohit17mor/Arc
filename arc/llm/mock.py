"""
Mock LLM Provider — for testing.

Returns configurable responses without making any API calls.
Tracks all calls for test assertions.
"""

from __future__ import annotations

from typing import AsyncIterator

from arc.core.types import (
    LLMChunk,
    Message,
    ModelInfo,
    StopReason,
    ToolCall,
    ToolSpec,
)
from arc.llm.base import LLMProvider


class MockLLMProvider(LLMProvider):
    """
    Mock LLM that returns pre-configured responses.

    Usage in tests:
        mock = MockLLMProvider()
        mock.set_response("Hello, world!")

        async for chunk in mock.generate([...]):
            print(chunk.text)
        # prints "Hello, world!"

        # Check what was sent
        assert mock.last_messages[0].role == "user"

    For tool call testing:
        mock.set_tool_call("read_file", {"path": "test.py"})
    """

    def __init__(
        self,
        model: str = "mock-model",
        context_window: int = 8192,
    ) -> None:
        self._model = model
        self._context_window = context_window

        # Response queue — each generate() call pops the first one
        self._responses: list[list[LLMChunk]] = []

        # Default response if queue is empty
        self._default_response = "I'm a mock AI. Configure me with set_response()."

        # Call tracking
        self.call_count: int = 0
        self.last_messages: list[Message] = []
        self.last_tools: list[ToolSpec] | None = None
        self.all_calls: list[dict] = []

    def set_response(self, text: str) -> None:
        """Queue a text response for the next generate() call."""
        chunks = [
            LLMChunk(text=text),
            LLMChunk(
                stop_reason=StopReason.COMPLETE,
                input_tokens=len(text) // 4,
                output_tokens=len(text) // 4,
            ),
        ]
        self._responses.append(chunks)

    def set_responses(self, texts: list[str]) -> None:
        """Queue multiple text responses for successive generate() calls."""
        for text in texts:
            self.set_response(text)

    def set_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        text_before: str = "",
    ) -> None:
        """Queue a tool call response for the next generate() call."""
        tc = ToolCall.new(name=tool_name, arguments=arguments)
        chunks = []
        if text_before:
            chunks.append(LLMChunk(text=text_before))
        chunks.append(
            LLMChunk(
                tool_calls=[tc],
                stop_reason=StopReason.TOOL_USE,
                input_tokens=50,
                output_tokens=25,
            )
        )
        self._responses.append(chunks)

    def set_chunks(self, chunks: list[LLMChunk]) -> None:
        """Queue raw chunks for fine-grained control."""
        self._responses.append(chunks)

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Return queued response or default."""
        # Track the call
        self.call_count += 1
        self.last_messages = list(messages)
        self.last_tools = tools
        self.all_calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "call_number": self.call_count,
            }
        )

        # Get response chunks
        if self._responses:
            chunks = self._responses.pop(0)
        else:
            # Default response
            chunks = [
                LLMChunk(text=self._default_response),
                LLMChunk(
                    stop_reason=StopReason.COMPLETE,
                    input_tokens=10,
                    output_tokens=len(self._default_response) // 4,
                ),
            ]

        for chunk in chunks:
            yield chunk

    async def count_tokens(self, messages: list[Message]) -> int:
        """Rough estimate: 4 chars per token."""
        total = 0
        for msg in messages:
            if msg.content:
                total += len(msg.content) // 4
        return max(total, 1)

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            provider="mock",
            model=self._model,
            context_window=self._context_window,
            max_output_tokens=self._context_window // 4,
            cost_per_input_token=0.0,
            cost_per_output_token=0.0,
            supports_tools=True,
            supports_vision=False,
            supports_streaming=True,
        )

    def reset(self) -> None:
        """Reset all state. Useful between tests."""
        self._responses.clear()
        self.call_count = 0
        self.last_messages = []
        self.last_tools = None
        self.all_calls = []