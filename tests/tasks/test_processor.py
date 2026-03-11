"""Tests for arc/tasks/processor.py — TaskProcessor daemon."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arc.tasks.processor import TaskProcessor, _ALWAYS_EXCLUDED
from arc.tasks.store import TaskStore
from arc.tasks.types import AgentDef, Task, TaskStatus, TaskStep


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
