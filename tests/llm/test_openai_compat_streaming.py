from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from arc.core.errors import LLMError
from arc.core.types import Message, StopReason, ToolSpec
from arc.llm.openai_compat import OpenAICompatibleProvider


class _FakeResponse:
    def __init__(self, *, status_code=200, lines=None, body=b""):
        self.status_code = status_code
        self._lines = list(lines or [])
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aread(self):
        return self._body

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeClient:
    def __init__(self, response=None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls: list[tuple[str, str, dict]] = []
        self.is_closed = False
        self.close_calls = 0

    def stream(self, method, url, json):
        if self._error is not None:
            raise self._error
        self.calls.append((method, url, json))
        return self._response

    async def aclose(self):
        self.is_closed = True
        self.close_calls += 1


class TestOpenAICompatibleProviderStreaming:
    @pytest.mark.asyncio
    async def test_get_client_builds_openrouter_headers_and_reuses_open_client(self, monkeypatch):
        created: list[dict] = []
        fake_client = _FakeClient()

        def fake_async_client(**kwargs):
            created.append(kwargs)
            return fake_client

        monkeypatch.setattr("arc.llm.openai_compat.httpx.AsyncClient", fake_async_client)
        provider = OpenAICompatibleProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key="secret",
            model="gpt-4o",
        )

        client1 = await provider._get_client()
        client2 = await provider._get_client()

        assert client1 is fake_client
        assert client2 is fake_client
        assert created[0]["headers"]["Authorization"] == "Bearer secret"
        assert created[0]["headers"]["HTTP-Referer"] == "https://github.com/ArcAI-xyz/Arc"
        assert created[0]["headers"]["X-Title"] == "Arc AI Agent"

    @pytest.mark.asyncio
    async def test_generate_streams_text_and_final_completion_chunk(self):
        response = _FakeResponse(
            lines=[
                'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
                'data: {"usage":{"prompt_tokens":3,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":2}},"choices":[{"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
        client = _FakeClient(response=response)
        provider = OpenAICompatibleProvider(model="gpt-4o")
        provider._client = client

        chunks = [chunk async for chunk in provider.generate([Message.user("Hi")])]

        assert chunks[0].text == "Hello"
        assert chunks[1].stop_reason == StopReason.COMPLETE
        assert chunks[1].input_tokens == 3
        assert chunks[1].output_tokens == 5
        assert chunks[1].cached_input_tokens == 2
        assert client.calls[0][0:2] == ("POST", "/chat/completions")

    @pytest.mark.asyncio
    async def test_generate_includes_tools_stop_sequences_and_max_tokens(self):
        response = _FakeResponse(lines=['data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'])
        client = _FakeClient(response=response)
        provider = OpenAICompatibleProvider(model="gpt-4o")
        provider._client = client
        tool = ToolSpec(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        [chunk async for chunk in provider.generate(
            [Message.user("Hi")],
            tools=[tool],
            max_tokens=200,
            stop_sequences=["DONE"],
        )]

        payload = client.calls[0][2]
        assert payload["tools"][0]["function"]["name"] == "read_file"
        assert payload["tool_choice"] == "auto"
        assert payload["max_tokens"] == 200
        assert payload["stop"] == ["DONE"]

    @pytest.mark.asyncio
    async def test_generate_logs_raw_request_payload_when_logger_attached(self):
        response = _FakeResponse(lines=['data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'])
        client = _FakeClient(response=response)
        provider = OpenAICompatibleProvider(model="gpt-4o")
        provider._client = client
        request_logger = AsyncMock()
        provider.set_request_logger(request_logger)

        [chunk async for chunk in provider.generate(
            [Message.system("You are helpful."), Message.user("Hi")],
        )]

        request_logger.assert_awaited_once()
        record = request_logger.await_args.args[0]
        assert record["provider"] == "openai"
        assert record["model"] == "gpt-4o"
        assert record["payload"]["messages"][0]["role"] == "system"
        assert record["payload"]["messages"][1]["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_generate_emits_tool_calls_on_finish_reason(self):
        response = _FakeResponse(
            lines=[
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{\\"path\\""}}]},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":": \\"main.py\\"}"}}]},"finish_reason":"tool_calls"}]}',
            ]
        )
        client = _FakeClient(response=response)
        provider = OpenAICompatibleProvider(model="gpt-4o")
        provider._client = client

        chunks = [chunk async for chunk in provider.generate([Message.user("Read it")])]

        assert chunks[0].stop_reason == StopReason.TOOL_USE
        assert chunks[0].tool_calls[0].name == "read_file"
        assert chunks[0].tool_calls[0].arguments == {"path": "main.py"}

    @pytest.mark.asyncio
    async def test_generate_emits_fallback_tool_calls_without_finish_reason(self):
        response = _FakeResponse(
            lines=[
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"not-json"}}]},"finish_reason":null}]}',
                "data: [DONE]",
            ]
        )
        client = _FakeClient(response=response)
        provider = OpenAICompatibleProvider(model="gpt-4o")
        provider._client = client

        chunks = [chunk async for chunk in provider.generate([Message.user("Read it")])]

        assert chunks[0].stop_reason == StopReason.TOOL_USE
        assert chunks[0].tool_calls[0].arguments == {"_raw": "not-json"}

    @pytest.mark.asyncio
    async def test_generate_raises_structured_api_error(self):
        response = _FakeResponse(
            status_code=429,
            body=json.dumps({"error": {"message": "rate limited"}}).encode(),
        )
        client = _FakeClient(response=response)
        provider = OpenAICompatibleProvider(model="gpt-4o")
        provider._client = client

        with pytest.raises(LLMError) as exc:
            [chunk async for chunk in provider.generate([Message.user("Hi")])]

        assert "rate limited" in str(exc.value)
        assert exc.value.retryable is True

    @pytest.mark.asyncio
    async def test_generate_wraps_connect_timeout_and_unexpected_errors(self):
        provider = OpenAICompatibleProvider(base_url="https://api.example.test/v1", model="gpt-4o")

        provider._client = _FakeClient(error=httpx.ConnectError("boom"))
        with pytest.raises(LLMError, match="Cannot connect"):
            [chunk async for chunk in provider.generate([Message.user("Hi")])]

        provider._client = _FakeClient(error=httpx.TimeoutException("slow"))
        with pytest.raises(LLMError, match="Request timed out"):
            [chunk async for chunk in provider.generate([Message.user("Hi")])]

        provider._client = _FakeClient(error=RuntimeError("weird"))
        with pytest.raises(LLMError, match="Unexpected error"):
            [chunk async for chunk in provider.generate([Message.user("Hi")])]

    @pytest.mark.asyncio
    async def test_close_closes_live_http_client(self):
        provider = OpenAICompatibleProvider(model="gpt-4o")
        provider._client = _FakeClient()

        await provider.close()

        assert provider._client.is_closed is True
        assert provider._client.close_calls == 1
