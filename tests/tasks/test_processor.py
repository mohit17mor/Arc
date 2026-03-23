"""Tests for arc/tasks/processor.py — TaskProcessor daemon."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arc.tasks.processor import TaskProcessor, _ALWAYS_EXCLUDED
from arc.tasks.store import TaskStore
from arc.tasks.types import AgentDef, Task, TaskComment, TaskStatus, TaskStep


class _FakeTaskAgentLoop:
    def __init__(self, *args, run_control=None, agent_id="agent", **kwargs):
        self._run_control = run_control
        self._agent_id = agent_id
        self.current_run_id = None
        self.last_run_id = None

    async def run(self, user_input: str):
        if self._run_control is not None:
            handle = self._run_control.start_run(
                kind="agent",
                source=self._agent_id,
                metadata={"agent_id": self._agent_id},
            )
            self.current_run_id = handle.run_id
            self.last_run_id = handle.run_id
        else:
            handle = None

        try:
            for chunk in ["working", " ", "still-working"]:
                await asyncio.sleep(0.02)
                if handle is not None:
                    await handle.checkpoint()
                yield chunk
            if handle is not None:
                await handle.finish_completed()
        finally:
            self.current_run_id = None


class _FakeClassifierLLM:
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


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_store(tmp_path):
    return TaskStore(db_path=tmp_path / "test_tasks.db")


@pytest.fixture
def mock_kernel():
    k = MagicMock()
    k.emit = AsyncMock()
    return k


@pytest.fixture
def mock_notification_router():
    r = MagicMock()
    r.route = AsyncMock()
    return r


@pytest.fixture
def mock_skill_manager():
    sm = MagicMock()
    sm.manifests = {
        "browsing": MagicMock(),
        "filesystem": MagicMock(),
        "terminal": MagicMock(),
        "worker": MagicMock(),
        "scheduler": MagicMock(),
        "task_board": MagicMock(),
    }
    return sm


@pytest.fixture
def sample_agents():
    return {
        "researcher": AgentDef(name="researcher", role="Web research"),
        "writer": AgentDef(name="writer", role="Content creation"),
        "reviewer": AgentDef(name="reviewer", role="Code review"),
    }


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.close = AsyncMock()
    return llm


@pytest.fixture
def processor(tmp_store, sample_agents, mock_skill_manager, mock_llm,
              mock_notification_router, mock_kernel):
    return TaskProcessor(
        store=tmp_store,
        agents=sample_agents,
        skill_manager=mock_skill_manager,
        default_llm=mock_llm,
        notification_router=mock_notification_router,
        kernel=mock_kernel,
    )


# ── Unit tests ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestTaskProcessorUnit:

    async def test_start_and_stop(self, processor, tmp_store):
        await tmp_store.initialize()
        await processor.start()
        assert processor._running
        await processor.stop()
        assert not processor._running
        await tmp_store.close()

    async def test_reload_agents(self, processor):
        new_agents = {"new_agent": AgentDef(name="new_agent", role="test")}
        processor.reload_agents(new_agents)
        assert "new_agent" in processor._agents
        assert "researcher" not in processor._agents

    async def test_stop_closes_cached_agent_llms(self, processor):
        closable = SimpleNamespace(close=AsyncMock())
        processor._running = True
        processor._task = asyncio.create_task(asyncio.sleep(10))
        processor._agent_llms = {
            "writer": closable,
            "reviewer": SimpleNamespace(),
        }

        await processor.stop()

        closable.close.assert_awaited_once()
        assert processor._agent_llms == {}

    async def test_compute_excluded_default(self, processor, sample_agents):
        """Default exclusion includes worker, scheduler, task_board."""
        excluded = processor._compute_excluded(sample_agents["researcher"])
        assert "worker" in excluded
        assert "scheduler" in excluded
        assert "task_board" in excluded

    async def test_compute_excluded_with_blacklist(self, processor):
        agent = AgentDef(name="test", role="test", exclude_skills=["terminal"])
        excluded = processor._compute_excluded(agent)
        assert "terminal" in excluded
        assert "worker" in excluded  # always excluded

    async def test_compute_excluded_with_whitelist(self, processor):
        agent = AgentDef(name="test", role="test", skills=["browsing"])
        excluded = processor._compute_excluded(agent)
        # Only browsing should NOT be excluded (plus the always-excluded ones)
        assert "browsing" not in excluded
        assert "filesystem" in excluded  # not in whitelist
        assert "worker" in excluded  # always excluded

    async def test_build_context_empty_comments(self, processor):
        task = Task(title="T", instruction="i")
        result = processor._build_context(task, [])
        assert result == ""

    async def test_build_prompt_basic(self, processor):
        task = Task(title="Research AI", instruction="Find papers")
        prompt = processor._build_prompt(task, "researcher", "")
        assert "Research AI" in prompt
        assert "Find papers" in prompt

    async def test_build_prompt_with_revision(self, processor):
        task = Task(title="T", instruction="i", status=TaskStatus.REVISION_NEEDED)
        prompt = processor._build_prompt(task, "writer", "[reviewer] Fix the intro")
        assert "reviewer requested changes" in prompt.lower()

    async def test_build_prompt_multistep_includes_handoff_guidance(self, processor):
        task = Task(
            title="Research and write memo",
            instruction="Create a final recommendation memo.",
            steps=[
                TaskStep(step_index=0, agent_name="researcher"),
                TaskStep(step_index=1, agent_name="writer"),
                TaskStep(step_index=2, agent_name="reviewer"),
            ],
            current_step=1,
        )
        prompt = processor._build_prompt(task, "writer", "[researcher] Source notes")
        assert "step 2 of 3" in prompt.lower()
        assert "previous steps are complete" in prompt.lower()
        assert "do not repeat prior work" in prompt.lower()
        assert "complete your part based on your role" in prompt.lower()

    async def test_is_review_step_true(self, processor):
        steps = [TaskStep(step_index=0, agent_name="writer", review_by="reviewer")]
        task = Task(title="T", instruction="i", steps=steps, current_step=0)
        assert processor._is_review_step(task, "reviewer")

    async def test_is_review_step_false_wrong_agent(self, processor):
        steps = [TaskStep(step_index=0, agent_name="writer", review_by="reviewer")]
        task = Task(title="T", instruction="i", steps=steps, current_step=0)
        assert not processor._is_review_step(task, "writer")

    async def test_is_review_step_false_no_reviewer(self, processor):
        steps = [TaskStep(step_index=0, agent_name="writer")]
        task = Task(title="T", instruction="i", steps=steps, current_step=0)
        assert not processor._is_review_step(task, "writer")

    async def test_is_review_step_false_without_steps(self, processor):
        task = Task(title="T", instruction="i")
        assert not processor._is_review_step(task, "writer")

    async def test_agent_needs_human_input_placeholder_returns_false(self, processor):
        assert processor._agent_needs_human_input("Should I continue?") is False

    async def test_check_needs_human_input_skips_short_responses(self, processor, sample_agents):
        processor._get_agent_llm = AsyncMock()

        result = await processor._check_needs_human_input("too short", sample_agents["researcher"])

        assert result is False
        processor._get_agent_llm.assert_not_awaited()

    async def test_check_needs_human_input_returns_classifier_verdict(self, processor, sample_agents):
        llm = _FakeClassifierLLM(["yes"])
        processor._get_agent_llm = AsyncMock(return_value=llm)

        result = await processor._check_needs_human_input(
            "I cannot continue until you confirm which provider to use.",
            sample_agents["researcher"],
        )

        assert result is True
        assert llm.calls

    async def test_check_needs_human_input_falls_back_to_false_on_llm_error(self, processor, sample_agents):
        processor._get_agent_llm = AsyncMock(return_value=_FakeClassifierLLM(error=RuntimeError("boom")))

        result = await processor._check_needs_human_input(
            "Please choose between option A and option B before I proceed.",
            sample_agents["researcher"],
        )

        assert result is False

    async def test_get_agent_llm_uses_default_without_override(self, processor, sample_agents, mock_llm):
        assert await processor._get_agent_llm(sample_agents["researcher"]) is mock_llm

    async def test_get_agent_llm_uses_default_when_factory_is_missing(self, processor, mock_llm):
        agent = AgentDef(name="writer", role="Drafts", llm_provider="openai", llm_model="gpt-4.1")

        assert await processor._get_agent_llm(agent) is mock_llm

    async def test_get_agent_llm_caches_factory_instances(self, processor):
        created: list[tuple[str, dict[str, str | None]]] = []

        def factory(provider, **kwargs):
            llm = SimpleNamespace(provider=provider, kwargs=kwargs)
            created.append((provider, kwargs))
            return llm

        processor._llm_factory = factory
        agent = AgentDef(
            name="writer",
            role="Drafts",
            llm_provider="openai",
            llm_model="gpt-4.1",
            llm_base_url="https://example.test",
            llm_api_key="secret",
        )

        llm1 = await processor._get_agent_llm(agent)
        llm2 = await processor._get_agent_llm(agent)

        assert llm1 is llm2
        assert created == [
            (
                "openai",
                {
                    "model": "gpt-4.1",
                    "base_url": "https://example.test",
                    "api_key": "secret",
                },
            )
        ]

    async def test_build_context_summarizes_old_comments(self, processor):
        comments = [
            TaskComment(id=i, task_id="t-1", step_index=0, agent_name=f"agent{i}", content=f"line {i}")
            for i in range(6)
        ]

        text = processor._build_context(Task(title="T", instruction="i"), comments)

        assert "(... 2 earlier entries summarized ...)" in text
        assert "  [agent0] line 0..." in text
        assert "[agent5] line 5" in text

    async def test_build_context_truncates_long_history(self, processor):
        comments = [
            TaskComment(id=i, task_id="t-1", step_index=0, agent_name=f"agent{i}", content="x" * 80)
            for i in range(4)
        ]

        with patch("arc.tasks.processor._MAX_CONTEXT_CHARS", 60):
            text = processor._build_context(Task(title="T", instruction="i"), comments)

        assert text.startswith("(... truncated ...)\n")


@pytest.mark.asyncio
class TestTaskProcessorHandleHumanReply:

    async def test_reply_to_blocked_task(self, processor, tmp_store):
        await tmp_store.initialize()

        task = Task(title="Blocked", instruction="i", status=TaskStatus.BLOCKED,
                    steps=[TaskStep(step_index=0, agent_name="researcher")])
        await tmp_store.save(task)

        msg = await processor.handle_human_reply(task.id, "Use PostgreSQL", "approve")
        assert "resume" in msg.lower()

        found = await tmp_store.get_by_id(task.id)
        assert found is not None
        assert found.status == TaskStatus.QUEUED

        comments = await tmp_store.get_comments(task.id)
        assert any("PostgreSQL" in c.content for c in comments)
        await tmp_store.close()

    async def test_reply_to_awaiting_human_approve(self, processor, tmp_store):
        await tmp_store.initialize()

        task = Task(title="Review", instruction="i", status=TaskStatus.AWAITING_HUMAN,
                    steps=[TaskStep(step_index=0, agent_name="writer"),
                           TaskStep(step_index=1, agent_name="publisher")],
                    current_step=0)
        await tmp_store.save(task)

        # Add a fake agent result
        await tmp_store.add_comment(task.id, "writer", "Here is my content")

        msg = await processor.handle_human_reply(task.id, "Looks good", "approve")
        assert "approved" in msg.lower()
        await tmp_store.close()

    async def test_reply_to_awaiting_human_revise(self, processor, tmp_store):
        await tmp_store.initialize()

        task = Task(title="Review", instruction="i", status=TaskStatus.AWAITING_HUMAN,
                    steps=[TaskStep(step_index=0, agent_name="writer")],
                    current_step=0)
        await tmp_store.save(task)

        msg = await processor.handle_human_reply(task.id, "Make it shorter", "revise")
        assert "revision" in msg.lower()

        found = await tmp_store.get_by_id(task.id)
        assert found is not None
        assert found.status == TaskStatus.REVISION_NEEDED
        assert found.bounce_count == 1
        await tmp_store.close()

    async def test_reply_to_nonexistent_task(self, processor, tmp_store):
        await tmp_store.initialize()
        msg = await processor.handle_human_reply("t-ghost", "hello", "approve")
        assert "not waiting" in msg.lower()
        await tmp_store.close()

    async def test_reply_to_unexpected_state(self, processor):
        processor._store.get_blocked_task = AsyncMock(
            return_value=Task(title="Done", instruction="i", status=TaskStatus.DONE)
        )

        msg = await processor.handle_human_reply("t-done", "hello", "approve")

        assert "unexpected state" in msg.lower()

    async def test_reply_revise_at_max_bounces(self, processor, tmp_store):
        await tmp_store.initialize()

        task = Task(title="Stuck", instruction="i", status=TaskStatus.AWAITING_HUMAN,
                    steps=[TaskStep(step_index=0, agent_name="writer")],
                    current_step=0, bounce_count=3, max_bounces=3)
        await tmp_store.save(task)

        msg = await processor.handle_human_reply(task.id, "Try again", "revise")
        assert "bounce limit" in msg.lower()
        await tmp_store.close()


@pytest.mark.asyncio
class TestTaskProcessorReviewHandling:

    async def test_handle_review_result_approved(self, processor, tmp_store):
        await tmp_store.initialize()

        task = Task(title="T", instruction="i",
                    steps=[TaskStep(step_index=0, agent_name="writer", review_by="reviewer"),
                           TaskStep(step_index=1, agent_name="publisher")])
        await tmp_store.save(task)

        review_text = "The content is excellent.\n\nVERDICT: APPROVED"
        await processor._handle_review_result(task, "reviewer", review_text)

        # After approval, task should advance (either queued for next step or done)
        found = await tmp_store.get_by_id(task.id)
        assert found is not None
        # Should have advanced past step 0
        await tmp_store.close()

    async def test_handle_review_result_needs_revision(self, processor, tmp_store):
        await tmp_store.initialize()

        task = Task(title="T", instruction="i",
                    steps=[TaskStep(step_index=0, agent_name="writer", review_by="reviewer")])
        await tmp_store.save(task)

        review_text = "The intro is weak.\n\nVERDICT: NEEDS_REVISION"
        await processor._handle_review_result(task, "reviewer", review_text)

        found = await tmp_store.get_by_id(task.id)
        assert found is not None
        assert found.status == TaskStatus.REVISION_NEEDED
        assert found.bounce_count == 1
        await tmp_store.close()

    async def test_handle_review_max_bounces_completes(self, processor, tmp_store):
        await tmp_store.initialize()

        task = Task(title="T", instruction="i", bounce_count=3, max_bounces=3,
                    steps=[TaskStep(step_index=0, agent_name="writer", review_by="reviewer")])
        await tmp_store.save(task)

        review_text = "Still bad.\n\nVERDICT: NEEDS_REVISION"
        await processor._handle_review_result(task, "reviewer", review_text)

        found = await tmp_store.get_by_id(task.id)
        assert found is not None
        assert found.status == TaskStatus.DONE  # forced completion at limit
        await tmp_store.close()

    async def test_handle_failure(self, processor, tmp_store):
        await tmp_store.initialize()

        task = Task(title="Fail", instruction="i",
                    steps=[TaskStep(step_index=0, agent_name="researcher")])
        await tmp_store.save(task)

        await processor._handle_failure(task, "researcher", "LLM timeout")

        found = await tmp_store.get_by_id(task.id)
        assert found is not None
        assert found.status == TaskStatus.FAILED

        comments = await tmp_store.get_comments(task.id)
        assert any("LLM timeout" in c.content for c in comments)
        await tmp_store.close()

    async def test_handle_cancelled_marks_task_cancelled_when_store_is_not_already_cancelled(self, processor):
        processor._store.update_status_with_comment = AsyncMock()
        processor._is_task_cancelled = AsyncMock(return_value=False)
        task = Task(
            title="Cancelled",
            instruction="i",
            steps=[TaskStep(step_index=0, agent_name="researcher")],
        )

        with patch("arc.tasks.processor.time.time", return_value=123):
            await processor._handle_cancelled(task, "researcher", "cancel")

        processor._store.update_status_with_comment.assert_awaited_once_with(
            task.id,
            TaskStatus.CANCELLED,
            "system",
            "Task cancelled while agent 'researcher' was running (cancel).",
            step_index=task.current_step,
            extra_updates={"completed_at": 123},
        )

    async def test_advance_task_routes_to_agent_reviewer(self, processor):
        processor._store.update_status_with_comment = AsyncMock()
        processor._notify = AsyncMock()
        task = Task(
            title="Review me",
            instruction="i",
            steps=[TaskStep(step_index=0, agent_name="writer", review_by="reviewer")],
        )

        await processor._advance_task(task, "writer", "draft")

        processor._store.update_status_with_comment.assert_awaited_once_with(
            task.id,
            TaskStatus.IN_REVIEW,
            "system",
            "Submitted for review by 'reviewer'.",
            step_index=task.current_step,
        )
        processor._notify.assert_not_called()

    async def test_advance_task_routes_to_human_review(self, processor):
        processor._store.update_status_with_comment = AsyncMock()
        processor._notify = AsyncMock()
        task = Task(
            title="Review me",
            instruction="i",
            steps=[TaskStep(step_index=0, agent_name="writer", review_by="human")],
        )

        await processor._advance_task(task, "writer", "draft result")

        processor._store.update_status_with_comment.assert_awaited_once_with(
            task.id,
            TaskStatus.AWAITING_HUMAN,
            "system",
            "Awaiting human review.",
            step_index=task.current_step,
        )
        processor._notify.assert_awaited_once()

    async def test_advance_task_moves_directly_to_next_step_without_reviewer(self, processor):
        processor._move_to_next_step = AsyncMock()
        task = Task(
            title="Ship it",
            instruction="i",
            steps=[TaskStep(step_index=0, agent_name="writer")],
        )

        await processor._advance_task(task, "writer", "final")

        processor._move_to_next_step.assert_awaited_once_with(task, "final")

    async def test_move_to_next_step_queues_following_workflow_step(self, processor):
        processor._store.update_status_with_comment = AsyncMock()
        task = Task(
            title="Pipeline",
            instruction="i",
            steps=[
                TaskStep(step_index=0, agent_name="researcher"),
                TaskStep(step_index=1, agent_name="writer"),
            ],
            current_step=0,
        )

        await processor._move_to_next_step(task, "research complete")

        processor._store.update_status_with_comment.assert_awaited_once_with(
            task.id,
            TaskStatus.QUEUED,
            "system",
            "Step 1 complete. Moving to step 2.",
            step_index=task.current_step,
            extra_updates={"current_step": 1, "bounce_count": 0},
        )

    async def test_move_to_next_step_completes_task_and_notifies(self, processor):
        processor._store.update_status_with_comment = AsyncMock()
        processor._emit = AsyncMock()
        processor._notify = AsyncMock()
        task = Task(
            title="Pipeline",
            instruction="i",
            steps=[TaskStep(step_index=0, agent_name="researcher")],
            current_step=0,
        )

        with patch("arc.tasks.processor.time.time", return_value=456):
            await processor._move_to_next_step(task, "finished output")

        processor._store.update_status_with_comment.assert_awaited_once_with(
            task.id,
            TaskStatus.DONE,
            "system",
            "Task completed successfully.",
            step_index=task.current_step,
            extra_updates={"result": "finished output", "completed_at": 456},
        )
        processor._emit.assert_awaited_once_with(
            "task:complete",
            {"task_id": task.id, "task_title": task.title},
        )
        processor._notify.assert_awaited_once()

    async def test_tick_reviews_dispatches_available_reviewer(self, processor, sample_agents, monkeypatch):
        task = Task(
            title="Needs review",
            instruction="i",
            steps=[TaskStep(step_index=0, agent_name="writer", review_by="reviewer")],
            status=TaskStatus.IN_REVIEW,
        )
        processor._store.get_all = AsyncMock(return_value=[task])
        created: list[str] = []
        real_create_task = asyncio.create_task

        def fake_create_task(coro, name=None):
            created.append(name)
            coro.close()
            return real_create_task(asyncio.sleep(0))

        monkeypatch.setattr("arc.tasks.processor.asyncio.create_task", fake_create_task)

        await processor._tick_reviews()

        assert created == [f"review:{task.id}:reviewer"]
        assert processor._in_flight["reviewer"] == 1

    async def test_process_review_routes_successful_reviewer_output(self, processor):
        reviewer = AgentDef(name="reviewer", role="Reviews")
        task = Task(
            title="Needs review",
            instruction="i",
            steps=[TaskStep(step_index=0, agent_name="writer", review_by="reviewer")],
        )
        processor._in_flight["reviewer"] = 1
        processor._store.get_comments = AsyncMock(return_value=[])
        processor._store.add_comment = AsyncMock()
        processor._run_agent = AsyncMock(return_value=("VERDICT: APPROVED", None))
        processor._handle_review_result = AsyncMock()

        await processor._process_review(task, reviewer)

        processor._store.add_comment.assert_awaited_once_with(
            task.id,
            reviewer.name,
            "VERDICT: APPROVED",
            step_index=task.current_step,
        )
        processor._handle_review_result.assert_awaited_once_with(task, reviewer.name, "VERDICT: APPROVED")
        assert processor._in_flight["reviewer"] == 0

    async def test_process_review_handles_agent_errors(self, processor):
        reviewer = AgentDef(name="reviewer", role="Reviews")
        task = Task(
            title="Needs review",
            instruction="i",
            steps=[TaskStep(step_index=0, agent_name="writer", review_by="reviewer")],
        )
        processor._in_flight["reviewer"] = 1
        processor._store.get_comments = AsyncMock(return_value=[])
        processor._run_agent = AsyncMock(return_value=("", "review failed"))
        processor._handle_failure = AsyncMock()

        await processor._process_review(task, reviewer)

        processor._handle_failure.assert_awaited_once_with(task, reviewer.name, "review failed")
        assert processor._in_flight["reviewer"] == 0

@pytest.mark.asyncio
async def test_cancel_task_stops_inflight_run(tmp_store, sample_agents, mock_skill_manager,
                                              mock_llm, mock_notification_router,
                                              mock_kernel, monkeypatch):
    from arc.core.run_control import RunControlManager, RunStatus

    await tmp_store.initialize()
    run_control = RunControlManager()
    processor = TaskProcessor(
        store=tmp_store,
        agents=sample_agents,
        skill_manager=mock_skill_manager,
        default_llm=mock_llm,
        notification_router=mock_notification_router,
        kernel=mock_kernel,
        run_control=run_control,
    )

    task = Task(
        title="Long task",
        instruction="Do some long running work",
        steps=[TaskStep(step_index=0, agent_name="researcher")],
    )
    await tmp_store.save(task)

    monkeypatch.setattr("arc.agent.loop.AgentLoop", _FakeTaskAgentLoop)
    monkeypatch.setattr(processor, "_check_needs_human_input", AsyncMock(return_value=False))

    processing = asyncio.create_task(processor._process_task(task, sample_agents["researcher"]))

    deadline = time.monotonic() + 1.0
    active_run_id = None
    while time.monotonic() < deadline:
        runs = run_control.list_runs(active_only=True)
        if runs:
            active_run_id = runs[0].run_id
            break
        await asyncio.sleep(0.005)

    assert active_run_id is not None
    assert await processor.cancel_task(task.id)

    await processing

    found = await tmp_store.get_by_id(task.id)
    assert found is not None
    assert found.status == TaskStatus.CANCELLED

    snapshot = run_control.get_run(active_run_id)
    assert snapshot is not None
    assert snapshot.status == RunStatus.CANCELLED

    comments = await tmp_store.get_comments(task.id)
    researcher_comments = [c for c in comments if c.agent_name == "researcher"]
    assert researcher_comments == []

    await tmp_store.close()


@pytest.mark.asyncio
async def test_cancel_task_returns_false_when_store_rejects(processor):
    processor._store.cancel = AsyncMock(return_value=False)

    assert await processor.cancel_task("t-missing") is False
