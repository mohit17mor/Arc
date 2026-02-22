"""
Ollama LLM Provider — connects to Ollama API.

Ollama runs LLMs locally or on a remote server.
API docs: https://github.com/ollama/ollama/blob/main/docs/api.md

This provider:
- Streams responses via Ollama's /api/chat endpoint
- Supports tool calling (Ollama 0.4+)
- Converts Arc Message format ↔ Ollama format
- Works with any Ollama-hosted model (llama3, mistral, codellama, etc.)
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from arc.core.errors import LLMError
from arc.core.types import (
    LLMChunk,
    Message,
    ModelInfo,
    StopReason,
    ToolCall,
    ToolSpec,
)
from arc.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    LLM provider for Ollama.

    Usage:
        provider = OllamaProvider(
            base_url="http://localhost:11434",
            model="llama3.1",
        )

        async for chunk in provider.generate([Message.user("Hello")]):
            print(chunk.text, end="")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        context_window: int = 128000,
        max_output_tokens: int = 8192,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._context_window = context_window
        self._max_output_tokens = max_output_tokens
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=120.0,   # LLMs can take a while
                    write=10.0,
                    pool=10.0,
                ),
            )
        return self._client

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Stream a response from Ollama."""
        client = await self._get_client()

        # Convert Arc messages to Ollama format
        ollama_messages = [self._convert_message(msg) for msg in messages]

        # Build request payload
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        # Add tools if provided
        if tools:
            payload["tools"] = [self._convert_tool_spec(t) for t in tools]

        try:
            async with client.stream(
                "POST",
                "/api/chat",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise LLMError(
                        f"Ollama API error ({response.status_code}): {error_body.decode()}",
                        provider="ollama",
                        model=self._model,
                        retryable=response.status_code >= 500,
                    )

                collected_text = ""
                input_tokens = 0
                output_tokens = 0

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Check for errors in stream
                    if "error" in data:
                        raise LLMError(
                            f"Ollama stream error: {data['error']}",
                            provider="ollama",
                            model=self._model,
                            retryable=True,
                        )

                    message_data = data.get("message", {})

                    # Handle tool calls
                    if message_data.get("tool_calls"):
                        tool_calls = []
                        for tc in message_data["tool_calls"]:
                            func = tc.get("function", {})
                            tool_calls.append(
                                ToolCall.new(
                                    name=func.get("name", ""),
                                    arguments=func.get("arguments", {}),
                                )
                            )

                        # Get token counts from final response
                        if data.get("done"):
                            input_tokens = data.get("prompt_eval_count", 0)
                            output_tokens = data.get("eval_count", 0)

                        yield LLMChunk(
                            tool_calls=tool_calls,
                            stop_reason=StopReason.TOOL_USE,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        )
                        return

                    # Handle text content
                    content = message_data.get("content", "")
                    if content:
                        collected_text += content
                        yield LLMChunk(text=content)

                    # Check if done
                    if data.get("done"):
                        input_tokens = data.get("prompt_eval_count", 0)
                        output_tokens = data.get("eval_count", 0)

                        yield LLMChunk(
                            stop_reason=StopReason.COMPLETE,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        )
                        return

        except httpx.ConnectError as e:
            raise LLMError(
                f"Cannot connect to Ollama at {self._base_url}. "
                f"Is Ollama running? Error: {e}",
                provider="ollama",
                model=self._model,
                retryable=True,
            ) from e
        except httpx.TimeoutException as e:
            raise LLMError(
                f"Ollama request timed out: {e}",
                provider="ollama",
                model=self._model,
                retryable=True,
            ) from e
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                f"Unexpected error communicating with Ollama: {e}",
                provider="ollama",
                model=self._model,
                retryable=False,
            ) from e

    async def count_tokens(self, messages: list[Message]) -> int:
        """
        Estimate token count.

        Ollama doesn't have a standalone tokenize endpoint in all versions,
        so we use a rough estimate: ~4 characters per token for English.
        """
        total_chars = 0
        for msg in messages:
            if msg.content:
                total_chars += len(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(json.dumps(tc.arguments))
        return max(total_chars // 4, 1)

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            provider="ollama",
            model=self._model,
            context_window=self._context_window,
            max_output_tokens=self._max_output_tokens,
            cost_per_input_token=0.0,   # Ollama is free
            cost_per_output_token=0.0,
            supports_tools=True,
            supports_vision=False,
            supports_streaming=True,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ━━━ Format Conversion ━━━

    @staticmethod
    def _convert_message(msg: Message) -> dict[str, Any]:
        """Convert Arc Message to Ollama message format."""
        result: dict[str, Any] = {"role": msg.role}

        if msg.role == "tool":
            # Ollama expects tool results in a specific format
            result["content"] = msg.content or ""
        elif msg.tool_calls:
            # Assistant message with tool calls
            result["content"] = msg.content or ""
            result["tool_calls"] = [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                }
                for tc in msg.tool_calls
            ]
        else:
            result["content"] = msg.content or ""

        return result

    @staticmethod
    def _convert_tool_spec(spec: ToolSpec) -> dict[str, Any]:
        """Convert Arc ToolSpec to Ollama tool format."""
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }