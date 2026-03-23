from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from arc.core.types import Message, ToolCall
from arc.memory.compaction import (
    COMPACTION_THRESHOLD,
    CompactionState,
    find_safe_cut_index,
    summarise_messages,
)


class _FakeLLM:
    def __init__(self, chunks=None, error: Exception | None = None):
        self._chunks = list(chunks or [])
        self._error = error
        self.calls: list[dict] = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        for text in self._chunks:
            yield SimpleNamespace(text=text)


class TestFindSafeCutIndex:
    def test_returns_zero_when_not_enough_messages(self):
        messages = [Message.user("hello"), Message.assistant("hi")]

        assert find_safe_cut_index(messages, keep_recent=2) == 0

    def test_prefers_boundary_after_complete_assistant_turn(self):
        messages = [
            Message.user("first"),
            Message.assistant("done"),
            Message.user("second"),
            Message.assistant("later"),
            Message.user("recent"),
            Message.assistant("latest"),
        ]

        assert find_safe_cut_index(messages, keep_recent=2) == 4

    def test_can_fall_back_to_user_message_boundary(self):
        messages = [
            Message.user("first"),
            Message.assistant(tool_calls=[ToolCall.new("search", {})]),
            Message.user("second"),
            Message.assistant(tool_calls=[ToolCall.new("open", {})]),
            Message.user("recent"),
            Message.assistant("latest"),
        ]

        assert find_safe_cut_index(messages, keep_recent=2) == 2

    def test_returns_zero_when_no_safe_boundary_exists(self):
        messages = [
            Message.assistant(tool_calls=[ToolCall.new("search", {})]),
            Message.assistant(tool_calls=[ToolCall.new("open", {})]),
            Message.assistant(tool_calls=[ToolCall.new("click", {})]),
            Message.assistant(tool_calls=[ToolCall.new("fetch", {})]),
            Message.assistant(tool_calls=[ToolCall.new("parse", {})]),
            Message.assistant(tool_calls=[ToolCall.new("save", {})]),
            Message.assistant(tool_calls=[ToolCall.new("done", {})]),
        ]

        assert find_safe_cut_index(messages, keep_recent=2) == 0


@pytest.mark.asyncio
class TestSummariseMessages:
    async def test_builds_summary_prompt_from_messages_and_tool_calls(self):
        llm = _FakeLLM(chunks=["- decision", "\n- next step"])
        long_text = "x" * 1005
        messages = [
            Message.system("system"),
            Message.user("Need a deployment plan"),
            Message.assistant(tool_calls=[ToolCall.new("search", {"q": "deploy"})]),
            Message.assistant(long_text),
        ]

        summary = await summarise_messages(messages, llm)

        assert summary == "- decision\n- next step"
        prompt = llm.calls[0]["messages"][1].content
        assert "User: Need a deployment plan" in prompt
        assert "Assistant: [called tools: search]" in prompt
        assert ("Assistant: " + ("x" * 1000) + "...") in prompt

    async def test_returns_empty_when_only_system_messages_exist(self):
        llm = _FakeLLM(chunks=["unused"])

        summary = await summarise_messages([Message.system("system only")], llm)

        assert summary == ""
        assert llm.calls == []

    async def test_returns_empty_when_llm_fails(self):
        llm = _FakeLLM(error=RuntimeError("boom"))

        summary = await summarise_messages([Message.user("hello"), Message.assistant("world")], llm)

        assert summary == ""


@pytest.mark.asyncio
class TestCompactionState:
    async def test_check_and_start_background_skips_when_below_threshold(self):
        state = CompactionState()
        session = SimpleNamespace(messages=[Message.user("hello")] * 10)

        state.check_and_start_background(
            session,
            token_count=int(COMPACTION_THRESHOLD * 100) - 1,
            token_budget=100,
            llm=_FakeLLM(),
        )

        assert state._pending_task is None

    async def test_check_and_start_background_skips_when_result_or_task_is_pending(self):
        state = CompactionState()
        session = SimpleNamespace(messages=[Message.user("hello")] * 10)
        state._pending_summary = "ready"

        state.check_and_start_background(session, token_count=100, token_budget=100, llm=_FakeLLM())
        assert state._pending_task is None

        state._pending_summary = None
        state._pending_task = SimpleNamespace(done=lambda: False)
        state.check_and_start_background(session, token_count=100, token_budget=100, llm=_FakeLLM())
        assert state._pending_message_count == 0

    async def test_check_and_start_background_records_snapshot_and_cut_index(self, monkeypatch):
        state = CompactionState()
        session = SimpleNamespace(
            messages=[
                Message.user("start"),
                Message.assistant("done"),
                Message.user("recent"),
                Message.assistant("latest"),
            ]
        )
        created: list[asyncio.Task] = []
        real_create_task = asyncio.create_task

        monkeypatch.setattr("arc.memory.compaction.find_safe_cut_index", lambda messages: 2)

        def fake_create_task(coro):
            task = real_create_task(asyncio.sleep(0))
            created.append(task)
            coro.close()
            return task

        monkeypatch.setattr("arc.memory.compaction.asyncio.create_task", fake_create_task)

        state.check_and_start_background(session, token_count=100, token_budget=100, llm=_FakeLLM())

        assert state._pending_cut_index == 2
        assert state._pending_message_count == 4
        assert state._pending_task is created[0]
        await created[0]

    async def test_run_background_stores_non_empty_summary(self, monkeypatch):
        state = CompactionState()
        monkeypatch.setattr("arc.memory.compaction.summarise_messages", lambda messages, llm: asyncio.sleep(0, result="summary"))

        await state._run_background([Message.user("hello")], _FakeLLM())

        assert state._pending_summary == "summary"

    async def test_run_background_resets_cut_index_when_summary_is_empty(self, monkeypatch):
        state = CompactionState()
        state._pending_cut_index = 3
        state._pending_message_count = 5
        monkeypatch.setattr("arc.memory.compaction.summarise_messages", lambda messages, llm: asyncio.sleep(0, result=""))

        await state._run_background([Message.user("hello")], _FakeLLM())

        assert state._pending_summary is None
        assert state._pending_cut_index == 0
        assert state._pending_message_count == 0

    async def test_apply_if_ready_returns_false_without_pending_summary(self):
        state = CompactionState()

        assert state.apply_if_ready(SimpleNamespace(messages=[])) is False

    async def test_apply_if_ready_skips_when_session_is_shorter_than_cut_index(self):
        state = CompactionState()
        state._pending_summary = "summary"
        state._pending_cut_index = 3
        state._pending_message_count = 3
        session = SimpleNamespace(messages=[Message.user("only one")])

        assert state.apply_if_ready(session) is False
        assert state._pending_cut_index == 0
        assert state._pending_message_count == 0

    async def test_apply_if_ready_replaces_old_messages_with_summary_message(self):
        state = CompactionState()
        state._pending_summary = "- important context"
        state._pending_cut_index = 2
        session = SimpleNamespace(
            messages=[
                Message.user("old one"),
                Message.assistant("old two"),
                Message.user("recent"),
            ]
        )

        applied = state.apply_if_ready(session)

        assert applied is True
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert "earlier messages were compacted" in session.messages[0].content
        assert "- important context" in session.messages[0].content

    async def test_maybe_compact_sync_skips_below_threshold_or_without_safe_cut(self, monkeypatch):
        state = CompactionState()
        session = SimpleNamespace(messages=[Message.user("hello")] * 10)

        applied = await state.maybe_compact_sync(session, token_count=10, token_budget=100, llm=_FakeLLM())
        assert applied is False

        monkeypatch.setattr("arc.memory.compaction.find_safe_cut_index", lambda messages: 0)
        applied = await state.maybe_compact_sync(session, token_count=100, token_budget=100, llm=_FakeLLM())
        assert applied is False

    async def test_maybe_compact_sync_applies_summary_when_available(self, monkeypatch):
        state = CompactionState()
        session = SimpleNamespace(
            messages=[
                Message.user("old"),
                Message.assistant("older"),
                Message.user("recent"),
                Message.assistant("latest"),
            ]
        )
        monkeypatch.setattr("arc.memory.compaction.find_safe_cut_index", lambda messages: 2)
        monkeypatch.setattr(
            "arc.memory.compaction.summarise_messages",
            lambda messages, llm: asyncio.sleep(0, result="- summary"),
        )

        applied = await state.maybe_compact_sync(session, token_count=100, token_budget=100, llm=_FakeLLM())

        assert applied is True
        assert session.messages[0].content.endswith("- summary")
