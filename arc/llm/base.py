"""
LLM Provider interface — the contract every LLM must implement.

The agent loop calls these methods. It never knows which specific
LLM is behind the interface. Swap Ollama for Claude by changing
one config line.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, AsyncIterator, Awaitable, Callable

from arc.core.types import LLMChunk, Message, ModelInfo, ToolSpec

LLMRequestLogger = Callable[[dict[str, Any]], Awaitable[None]]
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implement this to add support for any LLM backend.

    The framework never calls model APIs directly — always through
    this interface. This is what makes LLMs swappable.

    Implementations:
        OllamaProvider  — local models via Ollama
        AnthropicProvider — Claude (future)
        OpenAIProvider — GPT (future)
        MockProvider — for testing
    """

    def set_request_logger(self, request_logger: LLMRequestLogger | None) -> None:
        """Attach an optional async logger for outbound request payloads."""
        self._request_logger = request_logger

    async def _log_request(self, record: dict[str, Any]) -> None:
        """Emit a raw request record when debug logging is enabled."""
        request_logger = getattr(self, "_request_logger", None)
        if request_logger is not None:
            try:
                await request_logger(record)
            except Exception as e:
                logger.warning(f"Failed to emit raw LLM request log: {e}")

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """
        Generate a response from the LLM.

        Args:
            messages: Conversation history in Arc Message format
            tools: Available tools the LLM can call (optional)
            temperature: Randomness (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate (None = model default)
            stop_sequences: Stop generation when these strings appear

        Yields:
            LLMChunk objects with streaming text and/or tool calls.
            The LAST chunk will have stop_reason set.
            The LAST chunk will have input_tokens and output_tokens populated.

        Raises:
            LLMError: On API failures, rate limits, connection errors
        """
        ...

    @abstractmethod
    async def count_tokens(self, messages: list[Message]) -> int:
        """
        Estimate token count for a list of messages.

        Used by the context composer for budget management.
        Doesn't need to be exact — within 10-20% is fine.

        For providers without a tokenizer, use rough estimate:
            len(text) / 4 for English text
        """
        ...

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Return static model metadata.

        Called once at registration, result may be cached.
        Must return accurate context_window and pricing.
        """
        ...
