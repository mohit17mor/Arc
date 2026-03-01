"""Tests for OpenAI-compatible LLM provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.core.types import (
    Message,
    ModelInfo,
    StopReason,
    ToolCall,
    ToolSpec,
)
from arc.llm.openai_compat import OpenAICompatibleProvider


# ── Construction & model info ────────────────────────────────────


class TestConstruction:
    """Tests for provider construction and get_model_info."""

    def test_basic_construction(self):
        p = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
        )
        info = p.get_model_info()
        assert info.provider == "openai"
        assert info.model == "gpt-4o"
        assert info.context_window == 128000

    def test_custom_provider_name(self):
        p = OpenAICompatibleProvider(
            base_url="http://localhost:1234/v1",
            api_key="",
            model="local-model",
            provider_name="lmstudio",
        )
        info = p.get_model_info()
        assert info.provider == "lmstudio"
        assert info.model == "local-model"

    def test_custom_context_window(self):
        p = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
            context_window=200000,
        )
        assert p.get_model_info().context_window == 200000

    def test_custom_max_output(self):
        p = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
            max_output_tokens=8192,
        )
        # max_output is stored internally
        assert p._max_output_tokens == 8192


# ── Token counting ───────────────────────────────────────────────


class TestTokenCounting:
    """Tests for count_tokens (rough estimate)."""

    @pytest.mark.asyncio
    async def test_count_tokens_basic(self):
        p = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="",
            model="gpt-4o",
        )
        count = await p.count_tokens([Message.user("hello world, this is a test")])
        assert count > 0  # rough estimate
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_count_tokens_empty(self):
        p = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="",
            model="gpt-4o",
        )
        count = await p.count_tokens([Message.user("")])
        # Even empty message should return at least 1
        assert isinstance(count, int)


# ── Message conversion ──────────────────────────────────────────


class TestMessageConversion:
    """Tests for Arc Message → OpenAI format conversion."""

    def setup_method(self):
        self.provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="",
            model="gpt-4o",
        )

    def test_user_message(self):
        msg = Message.user("Hello")
        converted = self.provider._convert_message(msg)
        assert converted == {"role": "user", "content": "Hello"}

    def test_assistant_message(self):
        msg = Message.assistant("Hi there!")
        converted = self.provider._convert_message(msg)
        assert converted == {"role": "assistant", "content": "Hi there!"}

    def test_system_message(self):
        msg = Message.system("You are an assistant.")
        converted = self.provider._convert_message(msg)
        assert converted == {"role": "system", "content": "You are an assistant."}

    def test_tool_result_message(self):
        msg = Message(role="tool", content="result data", tool_call_id="call_123")
        converted = self.provider._convert_message(msg)
        assert converted["role"] == "tool"
        assert converted["content"] == "result data"
        assert converted["tool_call_id"] == "call_123"

    def test_assistant_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="call_1", name="read_file", arguments={"path": "x.py"}),
            ],
        )
        converted = self.provider._convert_message(msg)
        assert converted["role"] == "assistant"
        assert len(converted["tool_calls"]) == 1
        tc = converted["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "read_file"
        assert '"path"' in tc["function"]["arguments"]


# ── Tool spec conversion ────────────────────────────────────────


class TestToolSpecConversion:
    """Tests for Arc ToolSpec → OpenAI format conversion."""

    def setup_method(self):
        self.provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="",
            model="gpt-4o",
        )

    def test_basic_tool_spec(self):
        spec = ToolSpec(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        )
        converted = self.provider._convert_tool_spec(spec)
        assert converted["type"] == "function"
        assert converted["function"]["name"] == "read_file"
        assert converted["function"]["description"] == "Read a file"
        assert "path" in converted["function"]["parameters"]["properties"]

    def test_tool_spec_no_params(self):
        spec = ToolSpec(
            name="list_files",
            description="List files",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        converted = self.provider._convert_tool_spec(spec)
        assert converted["function"]["name"] == "list_files"


# ── Config integration ──────────────────────────────────────────


class TestLLMConfig:
    """Tests for LLMConfig worker model fields."""

    def test_default_no_worker_override(self):
        from arc.core.config import LLMConfig
        cfg = LLMConfig()
        assert not cfg.has_worker_override

    def test_worker_override_set(self):
        from arc.core.config import LLMConfig
        cfg = LLMConfig(
            worker_provider="groq",
            worker_model="llama-3.3-70b-versatile",
        )
        assert cfg.has_worker_override

    def test_worker_override_partial(self):
        """Provider without model is not a valid override."""
        from arc.core.config import LLMConfig
        cfg = LLMConfig(worker_provider="groq")
        assert not cfg.has_worker_override

    def test_worker_fields_in_config(self):
        from arc.core.config import ArcConfig
        cfg = ArcConfig(
            llm={
                "default_provider": "openai",
                "default_model": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-test",
                "worker_provider": "groq",
                "worker_model": "llama-3.3-70b-versatile",
                "worker_base_url": "https://api.groq.com/openai/v1",
                "worker_api_key": "gsk_test",
            }
        )
        assert cfg.llm.has_worker_override
        assert cfg.llm.worker_provider == "groq"
        assert cfg.llm.worker_model == "llama-3.3-70b-versatile"


# ── Worker skill wiring ─────────────────────────────────────────


class TestWorkerSkillWiring:
    """Tests for WorkerSkill accepting separate worker_llm."""

    def test_worker_skill_accepts_worker_llm(self):
        from arc.skills.builtin.worker import WorkerSkill
        from arc.llm.mock import MockLLMProvider

        main_llm = MockLLMProvider()
        worker_llm = MockLLMProvider()

        skill = WorkerSkill()
        skill.set_dependencies(
            llm=main_llm,
            worker_llm=worker_llm,
            skill_manager=None,
            escalation_bus=None,
            notification_router=None,
            agent_registry=None,
        )
        assert skill._worker_llm is worker_llm
        assert skill._llm is main_llm

    def test_worker_skill_fallback_to_main(self):
        from arc.skills.builtin.worker import WorkerSkill
        from arc.llm.mock import MockLLMProvider

        main_llm = MockLLMProvider()

        skill = WorkerSkill()
        skill.set_dependencies(
            llm=main_llm,
            skill_manager=None,
            escalation_bus=None,
            notification_router=None,
            agent_registry=None,
        )
        # When worker_llm not provided, falls back to main
        assert skill._worker_llm is main_llm
