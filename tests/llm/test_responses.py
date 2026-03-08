"""Tests for the Responses API (Codex) LLM provider."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.core.types import (
    Message,
    ModelInfo,
    StopReason,
    ToolCall,
    ToolSpec,
)
from arc.llm.responses import ResponsesAPIProvider


# ── Construction & model info ────────────────────────────────────


class TestConstruction:
    """Tests for provider construction and get_model_info."""

    def test_basic_construction(self):
        p = ResponsesAPIProvider(
            base_url="https://corp.example.com/v1",
            api_key="key-123",
            model="codex-model",
        )
        info = p.get_model_info()
        assert info.provider == "codex"
        assert info.model == "codex-model"
        assert info.context_window == 128000
        assert info.supports_tools is True
        assert info.supports_streaming is True

    def test_custom_provider_name(self):
        p = ResponsesAPIProvider(
            base_url="http://localhost:8080/v1",
            model="local-model",
            provider_name="my-responses",
        )
        assert p.get_model_info().provider == "my-responses"

    def test_custom_context_window(self):
        p = ResponsesAPIProvider(
            base_url="http://localhost/v1",
            model="m",
            context_window=200000,
        )
        assert p.get_model_info().context_window == 200000

    def test_extra_headers_stored(self):
        p = ResponsesAPIProvider(
            base_url="http://localhost/v1",
            model="m",
            extra_headers={"client": "codex-cli"},
        )
        assert p._extra_headers == {"client": "codex-cli"}


# ── Token counting ───────────────────────────────────────────────


class TestTokenCounting:

    @pytest.mark.asyncio
    async def test_count_tokens_basic(self):
        p = ResponsesAPIProvider(base_url="http://x/v1", model="m")
        count = await p.count_tokens([Message.user("hello world test")])
        assert count > 0
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_count_tokens_with_tool_calls(self):
        p = ResponsesAPIProvider(base_url="http://x/v1", model="m")
        msg = Message.assistant(
            tool_calls=[ToolCall(id="1", name="t", arguments={"key": "val"})]
        )
        count = await p.count_tokens([msg])
        assert count > 0

    @pytest.mark.asyncio
    async def test_count_tokens_empty_returns_one(self):
        p = ResponsesAPIProvider(base_url="http://x/v1", model="m")
        count = await p.count_tokens([Message.user("")])
        assert count == 1


# ── Message conversion ───────────────────────────────────────────


class TestMessageConversion:
    """Tests for Arc Message → Responses API format conversion."""

    def test_system_becomes_instructions(self):
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
        ]
        instructions, items = ResponsesAPIProvider._convert_messages(messages)
        assert instructions == "You are helpful."
        assert len(items) == 1
        assert items[0]["role"] == "user"
        assert items[0]["content"] == "Hello"

    def test_multiple_system_concatenated(self):
        messages = [
            Message.system("Rule 1."),
            Message.system("Rule 2."),
            Message.user("Hi"),
        ]
        instructions, items = ResponsesAPIProvider._convert_messages(messages)
        assert "Rule 1." in instructions
        assert "Rule 2." in instructions
        assert len(items) == 1

    def test_user_message(self):
        messages = [Message.user("Hello")]
        instructions, items = ResponsesAPIProvider._convert_messages(messages)
        assert instructions == ""
        assert items == [{"role": "user", "content": "Hello"}]

    def test_assistant_message(self):
        messages = [Message.assistant("Hi there")]
        _, items = ResponsesAPIProvider._convert_messages(messages)
        assert items == [{"role": "assistant", "content": "Hi there"}]

    def test_assistant_with_tool_calls(self):
        messages = [
            Message.assistant(
                tool_calls=[
                    ToolCall(id="call_1", name="read_file", arguments={"path": "x.py"}),
                ]
            ),
        ]
        _, items = ResponsesAPIProvider._convert_messages(messages)
        assert len(items) == 1
        assert items[0]["type"] == "function_call"
        assert items[0]["name"] == "read_file"
        assert items[0]["call_id"] == "call_1"
        assert json.loads(items[0]["arguments"]) == {"path": "x.py"}

    def test_assistant_with_text_and_tool_calls(self):
        messages = [
            Message.assistant(
                content="Let me check.",
                tool_calls=[
                    ToolCall(id="c1", name="ls", arguments={}),
                ],
            ),
        ]
        _, items = ResponsesAPIProvider._convert_messages(messages)
        # Text message + function call
        assert len(items) == 2
        assert items[0]["type"] == "message"
        assert items[0]["content"] == "Let me check."
        assert items[1]["type"] == "function_call"

    def test_tool_result(self):
        messages = [
            Message.tool_result(tool_call_id="call_1", content="file contents"),
        ]
        _, items = ResponsesAPIProvider._convert_messages(messages)
        assert items == [{
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "file contents",
        }]

    def test_full_conversation_roundtrip(self):
        """A full think→act→observe cycle converts correctly."""
        messages = [
            Message.system("You are an assistant."),
            Message.user("Read test.py"),
            Message.assistant(
                tool_calls=[
                    ToolCall(id="c1", name="read_file", arguments={"path": "test.py"}),
                ],
            ),
            Message.tool_result(tool_call_id="c1", content="print('hello')"),
            Message.user("Thanks, what does it do?"),
        ]
        instructions, items = ResponsesAPIProvider._convert_messages(messages)
        assert instructions == "You are an assistant."
        assert len(items) == 4
        assert items[0]["role"] == "user"
        assert items[1]["type"] == "function_call"
        assert items[2]["type"] == "function_call_output"
        assert items[3]["role"] == "user"


# ── Tool spec conversion ────────────────────────────────────────


class TestToolSpecConversion:

    def test_basic_tool_spec(self):
        spec = ToolSpec(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        converted = ResponsesAPIProvider._convert_tool_spec(spec)
        assert converted["type"] == "function"
        assert converted["name"] == "read_file"
        assert converted["description"] == "Read a file"
        assert "path" in converted["parameters"]["properties"]

    def test_tool_spec_no_wrapper(self):
        """Responses API tools have name at top level (no function wrapper)."""
        spec = ToolSpec(name="ls", description="List", parameters={})
        converted = ResponsesAPIProvider._convert_tool_spec(spec)
        assert "function" not in converted  # no nesting unlike chat completions
        assert converted["name"] == "ls"


# ── SSE parsing ──────────────────────────────────────────────────


class TestSSEParsing:

    def test_parse_text_delta_event(self):
        line = 'data: {"type": "response.output_text.delta", "delta": "Hello"}'
        event_type, data = ResponsesAPIProvider._parse_sse_line(line)
        assert event_type == "response.output_text.delta"
        assert data["delta"] == "Hello"

    def test_parse_function_call_added(self):
        payload = {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
            },
            "output_index": 0,
        }
        line = f"data: {json.dumps(payload)}"
        event_type, data = ResponsesAPIProvider._parse_sse_line(line)
        assert event_type == "response.output_item.added"
        assert data["item"]["name"] == "read_file"

    def test_parse_completed_event(self):
        payload = {
            "type": "response.completed",
            "response": {
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        }
        line = f"data: {json.dumps(payload)}"
        event_type, data = ResponsesAPIProvider._parse_sse_line(line)
        assert event_type == "response.completed"
        assert data["response"]["usage"]["input_tokens"] == 100

    def test_parse_done_sentinel(self):
        event_type, data = ResponsesAPIProvider._parse_sse_line("data: [DONE]")
        assert data is None

    def test_parse_empty_line(self):
        event_type, data = ResponsesAPIProvider._parse_sse_line("")
        assert data is None

    def test_parse_event_prefix_line(self):
        """An event: line alone returns type but no data."""
        event_type, data = ResponsesAPIProvider._parse_sse_line(
            "event: response.output_text.delta"
        )
        assert event_type == "response.output_text.delta"
        assert data is None

    def test_parse_invalid_json(self):
        event_type, data = ResponsesAPIProvider._parse_sse_line("data: {invalid")
        assert data is None


# ── Tool call building ───────────────────────────────────────────


class TestBuildToolCalls:

    def test_single_tool_call(self):
        func_calls = {
            0: {"call_id": "c1", "name": "read_file", "arguments": '{"path": "x.py"}'},
        }
        calls = ResponsesAPIProvider._build_tool_calls(func_calls)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "x.py"}
        assert calls[0].id == "c1"

    def test_multiple_tool_calls_ordered(self):
        func_calls = {
            1: {"call_id": "c2", "name": "write_file", "arguments": "{}"},
            0: {"call_id": "c1", "name": "read_file", "arguments": '{"path": "a"}'},
        }
        calls = ResponsesAPIProvider._build_tool_calls(func_calls)
        assert len(calls) == 2
        assert calls[0].name == "read_file"  # index 0 first
        assert calls[1].name == "write_file"

    def test_invalid_json_arguments(self):
        func_calls = {
            0: {"call_id": "c1", "name": "t", "arguments": "not-json"},
        }
        calls = ResponsesAPIProvider._build_tool_calls(func_calls)
        assert calls[0].arguments == {"_raw": "not-json"}

    def test_empty_arguments(self):
        func_calls = {
            0: {"call_id": "c1", "name": "t", "arguments": ""},
        }
        calls = ResponsesAPIProvider._build_tool_calls(func_calls)
        assert calls[0].arguments == {}


# ── Factory integration ──────────────────────────────────────────


class TestFactoryIntegration:

    def test_codex_preset_exists(self):
        from arc.llm.factory import get_preset
        p = get_preset("codex")
        assert p is not None
        assert p["class"] == "responses"
        assert p["needs_key"] is True

    def test_create_codex_provider(self):
        from arc.llm.factory import create_llm
        llm = create_llm(
            "codex",
            model="codex-model",
            base_url="http://localhost:8080/v1",
            api_key="key-123",
        )
        assert isinstance(llm, ResponsesAPIProvider)
        info = llm.get_model_info()
        assert info.provider == "codex"
        assert info.model == "codex-model"

    def test_codex_missing_base_url_raises(self):
        from arc.llm.factory import create_llm
        with pytest.raises(ValueError, match="base_url is required"):
            create_llm("codex", model="m")

    def test_codex_missing_model_raises(self):
        from arc.llm.factory import create_llm
        with pytest.raises(ValueError, match="model is required"):
            create_llm("codex", base_url="http://x/v1")

    def test_codex_in_provider_list(self):
        from arc.llm.factory import list_provider_names
        assert "codex" in list_provider_names()

    def test_extra_headers_passed_through(self):
        from arc.llm.factory import create_llm
        llm = create_llm(
            "codex",
            model="m",
            base_url="http://x/v1",
            api_key="k",
            extra_headers={"client": "codex-cli"},
        )
        assert llm._extra_headers == {"client": "codex-cli"}


# ── Streaming simulation ────────────────────────────────────────


def _make_sse_lines(events: list[dict]) -> list[str]:
    """Build SSE data lines from a list of event dicts."""
    return [f"data: {json.dumps(e)}" for e in events]


class TestStreaming:
    """Tests for generate() with simulated SSE streams."""

    def _make_provider(self) -> ResponsesAPIProvider:
        return ResponsesAPIProvider(
            base_url="http://test/v1",
            model="test-model",
            api_key="key",
        )

    @pytest.mark.asyncio
    async def test_text_streaming(self):
        """Text deltas are yielded as LLMChunks, completed closes the stream."""
        sse_events = [
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.output_text.delta", "delta": " world"},
            {
                "type": "response.completed",
                "response": {
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            },
        ]
        provider = self._make_provider()
        chunks = await self._simulate_stream(provider, sse_events)

        texts = [c.text for c in chunks if c.text]
        assert texts == ["Hello", " world"]

        final = chunks[-1]
        assert final.stop_reason == StopReason.COMPLETE
        assert final.input_tokens == 10
        assert final.output_tokens == 5

    @pytest.mark.asyncio
    async def test_tool_call_streaming(self):
        """Function calls are accumulated and emitted on completed."""
        sse_events = [
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "read_file",
                },
                "output_index": 0,
            },
            {
                "type": "response.function_call_arguments.delta",
                "delta": '{"path":',
                "output_index": 0,
            },
            {
                "type": "response.function_call_arguments.delta",
                "delta": '"test.py"}',
                "output_index": 0,
            },
            {
                "type": "response.completed",
                "response": {
                    "usage": {"input_tokens": 20, "output_tokens": 10},
                },
            },
        ]
        provider = self._make_provider()
        chunks = await self._simulate_stream(provider, sse_events)

        final = chunks[-1]
        assert final.stop_reason == StopReason.TOOL_USE
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0].name == "read_file"
        assert final.tool_calls[0].arguments == {"path": "test.py"}
        assert final.tool_calls[0].id == "call_abc"
        assert final.input_tokens == 20

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        """Multiple function calls in one response."""
        sse_events = [
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "read_file",
                },
                "output_index": 0,
            },
            {
                "type": "response.function_call_arguments.delta",
                "delta": '{"path": "a.py"}',
                "output_index": 0,
            },
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "c2",
                    "name": "write_file",
                },
                "output_index": 1,
            },
            {
                "type": "response.function_call_arguments.delta",
                "delta": '{"path": "b.py", "content": "hi"}',
                "output_index": 1,
            },
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 0, "output_tokens": 0}},
            },
        ]
        provider = self._make_provider()
        chunks = await self._simulate_stream(provider, sse_events)

        final = chunks[-1]
        assert final.stop_reason == StopReason.TOOL_USE
        assert len(final.tool_calls) == 2
        assert final.tool_calls[0].name == "read_file"
        assert final.tool_calls[1].name == "write_file"

    @pytest.mark.asyncio
    async def test_response_failed_raises(self):
        """response.failed event raises LLMError."""
        from arc.core.errors import LLMError

        sse_events = [
            {
                "type": "response.failed",
                "response": {
                    "error": {"message": "rate limited"},
                },
            },
        ]
        provider = self._make_provider()
        with pytest.raises(LLMError, match="rate limited"):
            await self._simulate_stream(provider, sse_events)

    @pytest.mark.asyncio
    async def test_stream_ends_without_completed(self):
        """If stream ends without response.completed, emit what we have."""
        sse_events = [
            {"type": "response.output_text.delta", "delta": "partial"},
        ]
        provider = self._make_provider()
        chunks = await self._simulate_stream(provider, sse_events)

        texts = [c.text for c in chunks if c.text]
        assert texts == ["partial"]
        # Should still get a final chunk with stop reason
        final = chunks[-1]
        assert final.stop_reason == StopReason.COMPLETE

    # ── Helpers ──────────────────────────────────────────────────

    async def _simulate_stream(
        self,
        provider: ResponsesAPIProvider,
        sse_events: list[dict],
    ) -> list:
        """
        Simulate an SSE stream by mocking httpx.

        Returns collected LLMChunks.
        """
        sse_lines = _make_sse_lines(sse_events)

        # Build a mock streaming response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = self._async_line_iter(sse_lines)

        # Make mock_response work as async context manager
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        provider._client = mock_client

        chunks = []
        async for chunk in provider.generate([Message.user("test")]):
            chunks.append(chunk)
        return chunks

    @staticmethod
    def _async_line_iter(lines: list[str]):
        """Create an async iterator factory from a list of strings."""
        async def _iter():
            for line in lines:
                yield line
        return _iter
