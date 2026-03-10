"""
OpenAI Responses API provider.

Supports any endpoint that implements the ``/v1/responses`` wire format,
including Codex and newer OpenAI deployments.

The Responses API is OpenAI's successor to Chat Completions.  Key
differences this provider handles:

    - Endpoint:        POST /v1/responses  (not /v1/chat/completions)
    - System prompt:   top-level ``instructions`` field  (not a message)
    - Conversation:    ``input`` array  (not ``messages``)
    - Tool results:    ``function_call_output`` items  (not role=tool)
    - Streaming:       Named SSE events  (not generic data-only lines)

Everything else — tool definitions, text streaming, stop reasons —
maps 1:1 to Arc's existing types.
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


class ResponsesAPIProvider(LLMProvider):
    """
    LLM provider for the OpenAI Responses API (``/v1/responses``).

    Works with Codex and any endpoint that speaks this wire format.

    Usage::

        provider = ResponsesAPIProvider(
            base_url="https://your-corp-endpoint/v1",
            api_key="your-key",
            model="codex-model",
        )

        async for chunk in provider.generate([Message.user("Hello")]):
            print(chunk.text, end="")
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "",
        context_window: int = 128000,
        max_output_tokens: int = 4096,
        provider_name: str = "codex",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._context_window = context_window
        self._max_output_tokens = max_output_tokens
        self._provider_name = provider_name
        self._extra_headers = extra_headers or {}
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {
                "Content-Type": "application/json",
            }
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            headers.update(self._extra_headers)

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

    # ── generate ───────────────────────────────────────────────

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        client = await self._get_client()

        # Separate system prompt (→ instructions) from conversation (→ input)
        instructions, input_items = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "input": input_items,
            "stream": True,
            "temperature": temperature,
        }

        if instructions:
            payload["instructions"] = instructions

        if max_tokens:
            payload["max_output_tokens"] = max_tokens

        if tools:
            payload["tools"] = [self._convert_tool_spec(t) for t in tools]
            payload["tool_choice"] = "auto"

        try:
            async with client.stream(
                "POST", "/responses", json=payload,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    error_msg = body.decode(errors="replace")
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

                # Accumulate function calls by output_index
                func_calls: dict[int, dict[str, Any]] = {}
                input_tokens = 0
                output_tokens = 0

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue

                    # Parse SSE: lines are "event: <type>" or "data: <json>"
                    event_type, data = self._parse_sse_line(line)
                    if data is None:
                        continue

                    # ── Text delta ──
                    if event_type == "response.output_text.delta":
                        delta = data.get("delta", "")
                        if delta:
                            yield LLMChunk(text=delta)

                    # ── Function call start ──
                    elif event_type == "response.output_item.added":
                        item = data.get("item", {})
                        if item.get("type") == "function_call":
                            idx = data.get("output_index", 0)
                            func_calls[idx] = {
                                "call_id": item.get("call_id", ""),
                                "name": item.get("name", ""),
                                "arguments": "",
                            }

                    # ── Function call arguments streaming ──
                    elif event_type == "response.function_call_arguments.delta":
                        idx = data.get("output_index", 0)
                        if idx in func_calls:
                            func_calls[idx]["arguments"] += data.get("delta", "")

                    # ── Response completed ──
                    elif event_type == "response.completed":
                        response_obj = data.get("response", data)
                        usage = response_obj.get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)

                        if func_calls:
                            yield LLMChunk(
                                tool_calls=self._build_tool_calls(func_calls),
                                stop_reason=StopReason.TOOL_USE,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                            )
                        else:
                            yield LLMChunk(
                                stop_reason=StopReason.COMPLETE,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                            )
                        return

                    # ── Response failed ──
                    elif event_type == "response.failed":
                        response_obj = data.get("response", data)
                        err = response_obj.get("error", {})
                        raise LLMError(
                            f"Response failed: {err.get('message', 'unknown error')}",
                            provider=self._provider_name,
                            model=self._model,
                            retryable=True,
                        )

                # Stream ended without response.completed — emit what we have
                if func_calls:
                    yield LLMChunk(
                        tool_calls=self._build_tool_calls(func_calls),
                        stop_reason=StopReason.TOOL_USE,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                else:
                    yield LLMChunk(
                        stop_reason=StopReason.COMPLETE,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

        except httpx.ConnectError as e:
            raise LLMError(
                f"Cannot connect to {self._base_url}: {e}",
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

    # ── count_tokens ───────────────────────────────────────────

    async def count_tokens(self, messages: list[Message]) -> int:
        total_chars = 0
        for msg in messages:
            if msg.content:
                total_chars += len(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(json.dumps(tc.arguments))
        return max(total_chars // 4, 1)

    # ── get_model_info ─────────────────────────────────────────

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            provider=self._provider_name,
            model=self._model,
            context_window=self._context_window,
            max_output_tokens=self._max_output_tokens,
            supports_tools=True,
            supports_vision=False,
            supports_streaming=True,
        )

    # ── close ──────────────────────────────────────────────────

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Format Conversion ──────────────────────────────────────

    @staticmethod
    def _convert_messages(
        messages: list[Message],
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Convert Arc Messages to Responses API format.

        Returns (instructions, input_items):
            instructions: extracted system prompt (or "")
            input_items:  list of input items for the ``input`` field
        """
        instructions = ""
        items: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                # System messages become the top-level instructions field.
                # If multiple system messages, concatenate them.
                if instructions:
                    instructions += "\n\n"
                instructions += msg.content or ""

            elif msg.role == "user":
                items.append({
                    "role": "user",
                    "content": msg.content or "",
                })

            elif msg.role == "assistant":
                if msg.tool_calls:
                    # Assistant requested tool calls — emit as output items
                    # so the model sees them in its conversation history.
                    output_items: list[dict[str, Any]] = []
                    for tc in msg.tool_calls:
                        output_items.append({
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        })
                    if msg.content:
                        output_items.insert(0, {
                            "type": "message",
                            "role": "assistant",
                            "content": msg.content,
                        })
                    items.extend(output_items)
                else:
                    items.append({
                        "role": "assistant",
                        "content": msg.content or "",
                    })

            elif msg.role == "tool":
                # Tool results become function_call_output items
                items.append({
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id or "",
                    "output": msg.content or "",
                })

        return instructions, items

    @staticmethod
    def _convert_tool_spec(spec: ToolSpec) -> dict[str, Any]:
        """Convert Arc ToolSpec to Responses API tool format."""
        return {
            "type": "function",
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        }

    @staticmethod
    def _parse_sse_line(line: str) -> tuple[str, dict[str, Any] | None]:
        """
        Parse a single SSE line from the Responses API stream.

        The Responses API uses named events::

            event: response.output_text.delta
            data: {"delta": "Hello"}

        Some servers also send plain ``data:`` lines without an event prefix.
        We handle both — tracking the most recent ``event:`` line and pairing
        it with the next ``data:`` line.

        Returns (event_type, parsed_data) or ("", None) for non-data lines.
        """
        # This is called line-by-line; for proper SSE parsing we'd need
        # to track state across lines (event: then data:).  Since httpx
        # aiter_lines gives us one line at a time, we handle both formats:

        if line.startswith("event:"):
            # Store event type — will be used by next data line.
            # Return nothing yet; the data line follows.
            return line[len("event:"):].strip(), None

        if line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                return "", None
            try:
                data = json.loads(data_str)
                # The event type is embedded in the data for Responses API
                # via the "type" field when not using paired event/data lines.
                event_type = data.get("type", "")
                return event_type, data
            except json.JSONDecodeError:
                return "", None

        return "", None

    @staticmethod
    def _build_tool_calls(
        func_calls: dict[int, dict[str, Any]],
    ) -> list[ToolCall]:
        """Convert accumulated function call data into ToolCall objects."""
        calls: list[ToolCall] = []
        for idx in sorted(func_calls.keys()):
            fc = func_calls[idx]
            try:
                args = json.loads(fc["arguments"]) if fc["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_raw": fc["arguments"]}
            calls.append(ToolCall(
                id=fc.get("call_id", "") or ToolCall.new("", {}).id,
                name=fc["name"],
                arguments=args,
            ))
        return calls
