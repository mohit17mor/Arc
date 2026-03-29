"""
OpenAI-compatible LLM provider.

Works with any API that implements OpenAI's ``/v1/chat/completions``
endpoint — including OpenAI itself, OpenRouter (Claude, Gemini, etc.),
Groq, Together, vLLM, LM Studio, and more.

One provider to rule them all.
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


class OpenAICompatibleProvider(LLMProvider):
    """
    LLM provider for any OpenAI-compatible API.

    Covers:
        - OpenAI (GPT-4o, o1, o3-mini, etc.)
        - OpenRouter (Claude, Gemini, Llama, Mistral — any model)
        - Groq (ultra-fast inference)
        - Together (open-source models)
        - vLLM / LM Studio / text-generation-webui (local servers)

    Usage::

        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4o",
        )

        async for chunk in provider.generate([Message.user("Hello")]):
            print(chunk.text, end="")
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o",
        context_window: int = 128000,
        max_output_tokens: int = 4096,
        provider_name: str = "openai",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._context_window = context_window
        self._max_output_tokens = max_output_tokens
        self._provider_name = provider_name
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {
                "Content-Type": "application/json",
            }
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            # OpenRouter recommends these headers
            if "openrouter" in self._base_url:
                headers["HTTP-Referer"] = "https://github.com/ArcAI-xyz/Arc"
                headers["X-Title"] = "Arc AI Agent"

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=120.0,
                    write=10.0,
                    pool=10.0,
                ),
            )
        return self._client

    # ── generate ───────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        client = await self._get_client()

        payload = self._build_payload(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
        )
        await self._log_request(
            {
                "provider": self._provider_name,
                "model": self._model,
                "endpoint": "/chat/completions",
                "request_format": "chat_completions",
                "message_count": len(payload["messages"]),
                "tool_count": len(payload.get("tools", [])),
                "tool_names": [tool["function"]["name"] for tool in payload.get("tools", [])],
                "payload": payload,
            }
        )

        try:
            async with client.stream(
                "POST", "/chat/completions", json=payload,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    error_msg = body.decode(errors="replace")
                    # Try to extract error message from JSON
                    try:
                        err_data = json.loads(error_msg)
                        error_msg = (
                            err_data.get("error", {}).get("message", "")
                            or error_msg
                        )
                    except json.JSONDecodeError:
                        pass
                    raise LLMError(
                        f"API error ({response.status_code}): {error_msg}",
                        provider=self._provider_name,
                        model=self._model,
                        retryable=response.status_code in (429, 500, 502, 503, 504)
                                  or response.status_code >= 500,
                    )

                # Accumulate tool calls across chunks
                tool_call_accum: dict[int, dict[str, Any]] = {}
                input_tokens = 0
                output_tokens = 0
                cached_input_tokens = 0
                got_content = False

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Usage stats (may come in a separate chunk or in the last delta)
                    usage = data.get("usage")
                    if usage:
                        input_tokens = usage.get("prompt_tokens", input_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)
                        details = (
                            usage.get("prompt_tokens_details")
                            or usage.get("input_tokens_details")
                            or {}
                        )
                        cached_input_tokens = details.get("cached_tokens", cached_input_tokens)

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Text content
                    content = delta.get("content")
                    if content:
                        got_content = True
                        yield LLMChunk(text=content)

                    # Tool calls (streamed incrementally)
                    tc_deltas = delta.get("tool_calls", [])
                    for tc_delta in tc_deltas:
                        idx = tc_delta.get("index", 0)
                        if idx not in tool_call_accum:
                            tool_call_accum[idx] = {
                                "id": tc_delta.get("id", ""),
                                "name": "",
                                "arguments": "",
                            }

                        func = tc_delta.get("function", {})
                        if "name" in func:
                            tool_call_accum[idx]["name"] = func["name"]
                        if "arguments" in func:
                            tool_call_accum[idx]["arguments"] += func["arguments"]

                    # Stream finished
                    if finish_reason:
                        if finish_reason in ("tool_calls", "function_call"):
                            # Emit accumulated tool calls
                            calls = []
                            for idx in sorted(tool_call_accum.keys()):
                                tc = tool_call_accum[idx]
                                try:
                                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                                except json.JSONDecodeError:
                                    args = {"_raw": tc["arguments"]}
                                calls.append(
                                    ToolCall.new(
                                        name=tc["name"],
                                        arguments=args,
                                    )
                                )

                            yield LLMChunk(
                                tool_calls=calls,
                                stop_reason=StopReason.TOOL_USE,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                cached_input_tokens=cached_input_tokens,
                            )
                            return

                        # Normal completion
                        yield LLMChunk(
                            stop_reason=StopReason.COMPLETE,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cached_input_tokens=cached_input_tokens,
                        )
                        return

                # If no finish_reason was received, emit a final chunk
                if tool_call_accum:
                    calls = []
                    for idx in sorted(tool_call_accum.keys()):
                        tc = tool_call_accum[idx]
                        try:
                            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {"_raw": tc["arguments"]}
                        calls.append(ToolCall.new(name=tc["name"], arguments=args))
                    yield LLMChunk(
                        tool_calls=calls,
                        stop_reason=StopReason.TOOL_USE,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cached_input_tokens=cached_input_tokens,
                    )
                else:
                    yield LLMChunk(
                        stop_reason=StopReason.COMPLETE,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cached_input_tokens=cached_input_tokens,
                    )

        except httpx.ConnectError as e:
            raise LLMError(
                f"Cannot connect to {self._base_url}. Error: {e}",
                provider=self._provider_name,
                model=self._model,
                retryable=True,
            ) from e
        except httpx.TimeoutException as e:
            raise LLMError(
                f"Request timed out: {e}",
                provider=self._provider_name,
                model=self._model,
                retryable=True,
            ) from e
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                f"Unexpected error: {e}",
                provider=self._provider_name,
                model=self._model,
                retryable=False,
            ) from e

    def _build_payload(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        temperature: float,
        max_tokens: int | None,
        stop_sequences: list[str] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [self._convert_message(m) for m in messages],
            "stream": True,
            "temperature": temperature,
            # Request usage stats in stream (OpenAI extension, ignored by others)
            "stream_options": {"include_usage": True},
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if stop_sequences:
            payload["stop"] = stop_sequences

        if tools:
            payload["tools"] = [self._convert_tool_spec(t) for t in tools]
            payload["tool_choice"] = "auto"

        return payload

    # ── count_tokens ───────────────────────────────────────────────

    async def count_tokens(self, messages: list[Message]) -> int:
        total_chars = 0
        for msg in messages:
            if msg.content:
                total_chars += len(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(json.dumps(tc.arguments))
        return max(total_chars // 4, 1)

    # ── get_model_info ─────────────────────────────────────────────

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            provider=self._provider_name,
            model=self._model,
            context_window=self._context_window,
            max_output_tokens=self._max_output_tokens,
            cost_per_input_token=0.0,   # varies by model, not tracked here
            cost_per_output_token=0.0,
            supports_tools=True,
            supports_vision=False,
            supports_streaming=True,
        )

    # ── close ──────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── format conversion ──────────────────────────────────────────

    @staticmethod
    def _convert_message(msg: Message) -> dict[str, Any]:
        """Convert Arc Message → OpenAI message format."""
        result: dict[str, Any] = {"role": msg.role}

        if msg.role == "tool":
            result["content"] = msg.content or ""
            result["tool_call_id"] = msg.tool_call_id or ""
        elif msg.tool_calls:
            result["content"] = msg.content or ""
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        else:
            result["content"] = msg.content or ""

        return result

    @staticmethod
    def _convert_tool_spec(spec: ToolSpec) -> dict[str, Any]:
        """Convert Arc ToolSpec → OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
