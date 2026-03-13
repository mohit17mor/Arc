"""Tests for arc/tasks/skill.py — TaskSkill LLM-facing tools."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from arc.tasks.skill import TaskSkill
from arc.tasks.store import TaskStore
from arc.tasks.types import AgentDef, Task, TaskStatus, TaskStep


@pytest.fixture
def tmp_store(tmp_path):
    return TaskStore(db_path=tmp_path / "test_tasks.db")


@pytest.fixture
def sample_agents():
    return {
        "researcher": AgentDef(name="researcher", role="Web research"),
        "writer": AgentDef(name="writer", role="Content creation"),
        "reviewer": AgentDef(name="reviewer", role="Quality review"),
    }


@pytest.fixture
def mock_processor():
    p = MagicMock()
    p.handle_human_reply = AsyncMock(return_value="Answer delivered.")
    return p


@pytest.fixture
async def skill(tmp_store, sample_agents, mock_processor):
    await tmp_store.initialize()
    s = TaskSkill()
    s.set_dependencies(store=tmp_store, agents=sample_agents, processor=mock_processor)
    return s


# ── Manifest ─────────────────────────────────────────────────────────────────


class TestManifest:
    def test_manifest_has_tools(self, sample_agents, mock_processor):
        skill = TaskSkill()
        skill.set_dependencies(
            store=MagicMock(), agents=sample_agents, processor=mock_processor,
        )
        manifest = skill.manifest()
        assert manifest.name == "task_board"
        tool_names = [t.name for t in manifest.tools]
        assert "queue_task" in tool_names
        assert "list_tasks" in tool_names
        assert "task_detail" in tool_names
        assert "cancel_task" in tool_names
        assert "clear_tasks" in tool_names
        assert "reply_to_task" in tool_names
        assert "list_agents" in tool_names

    def test_manifest_mentions_agents(self, sample_agents, mock_processor):
        skill = TaskSkill()
        skill.set_dependencies(
            store=MagicMock(), agents=sample_agents, processor=mock_processor,
        )
        manifest = skill.manifest()
        assert "researcher" in manifest.description
        assert "writer" in manifest.description


# ── queue_task ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestQueueTask:

    async def test_queue_simple_task(self, skill, tmp_store):
        result = await skill.execute_tool("queue_task", {
            "title": "Find AI papers",
            "instruction": "Search arxiv for recent AI papers",
            "assigned_agent": "researcher",
        })
        assert result.success
        assert "queued" in result.output.lower()
        assert "researcher" in result.output

        tasks = await tmp_store.get_all()
        assert len(tasks) == 1
        assert tasks[0].title == "Find AI papers"

    async def test_queue_task_unknown_agent(self, skill):
        result = await skill.execute_tool("queue_task", {
            "title": "T",
            "instruction": "I",
            "assigned_agent": "nonexistent",
        })
        assert not result.success
        assert "nonexistent" in result.error

    async def test_queue_task_no_agent_no_steps(self, skill):
        result = await skill.execute_tool("queue_task", {
            "title": "T",
            "instruction": "I",
        })
        assert not result.success
        assert "required" in result.error.lower()

    async def test_queue_multi_step_task(self, skill, tmp_store):
        result = await skill.execute_tool("queue_task", {
            "title": "Content pipeline",
            "instruction": "Create blog post about AI",
            "steps": [
                {"agent": "researcher"},
                {"agent": "writer", "review_by": "reviewer"},
            ],
        })
        assert result.success
        assert "Workflow" in result.output

        tasks = await tmp_store.get_all()
        assert len(tasks) == 1
        task = tasks[0]
        assert len(task.steps) == 2
        assert task.steps[0].agent_name == "researcher"
        assert task.steps[1].review_by == "reviewer"

    async def test_queue_task_with_human_reviewer(self, skill, tmp_store):
        result = await skill.execute_tool("queue_task", {
            "title": "Reviewed task",
            "instruction": "Write content",
            "steps": [
                {"agent": "writer", "review_by": "human"},
            ],
        })
        assert result.success

        tasks = await tmp_store.get_all()
        assert tasks[0].steps[0].review_by == "human"

    async def test_queue_task_unknown_reviewer(self, skill):
        result = await skill.execute_tool("queue_task", {
            "title": "T",
            "instruction": "I",
            "steps": [{"agent": "writer", "review_by": "ghost_reviewer"}],
        })
        assert not result.success
        assert "ghost_reviewer" in result.error

    async def test_queue_task_with_dependency(self, skill, tmp_store):
        # Create parent task first
        parent = Task(title="Parent", instruction="i")
        await tmp_store.save(parent)

        result = await skill.execute_tool("queue_task", {
            "title": "Child task",
            "instruction": "Depends on parent",
            "assigned_agent": "researcher",
            "depends_on": parent.id,
        })
        assert result.success
        assert parent.id in result.output

    async def test_queue_task_invalid_dependency(self, skill):
        result = await skill.execute_tool("queue_task", {
            "title": "T",
            "instruction": "I",
            "assigned_agent": "researcher",
            "depends_on": "t-nonexistent",
        })
        assert not result.success
        assert "not found" in result.error.lower()

    async def test_queue_task_with_priority(self, skill, tmp_store):
        result = await skill.execute_tool("queue_task", {
            "title": "Urgent",
            "instruction": "I",
            "assigned_agent": "researcher",
            "priority": 1,
        })
        assert result.success

        tasks = await tmp_store.get_all()
        assert tasks[0].priority == 1

    async def test_queue_task_blank_dependency_is_treated_as_none(self, skill, tmp_store):
        result = await skill.execute_tool("queue_task", {
            "title": "No real dependency",
            "instruction": "I",
            "assigned_agent": "researcher",
            "depends_on": "",
        })
        assert result.success

        tasks = await tmp_store.get_all()
        assert tasks[0].depends_on is None


# ── list_tasks ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestListTasks:

    async def test_list_empty(self, skill):
        result = await skill.execute_tool("list_tasks", {})
        assert result.success
        assert "no tasks" in result.output.lower()

    async def test_list_with_tasks(self, skill, tmp_store):
        await tmp_store.save(Task(title="Task A", instruction="i"))
        await tmp_store.save(Task(title="Task B", instruction="i"))

        result = await skill.execute_tool("list_tasks", {})
        assert result.success
        assert "2 task" in result.output
        assert "Task A" in result.output
        assert "Task B" in result.output

    async def test_list_with_status_filter(self, skill, tmp_store):
        await tmp_store.save(Task(title="Queued", instruction="i", status=TaskStatus.QUEUED))
        await tmp_store.save(Task(title="Done", instruction="i", status=TaskStatus.DONE))

        result = await skill.execute_tool("list_tasks", {"status": "done"})
        assert result.success
        assert "Done" in result.output
        assert "Queued" not in result.output


# ── task_detail ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestTaskDetail:

    async def test_detail_found(self, skill, tmp_store):
        task = Task(title="Detail me", instruction="Full instructions here")
        await tmp_store.save(task)
        await tmp_store.add_comment(task.id, "researcher", "Working on it")

        result = await skill.execute_tool("task_detail", {"task_id": task.id})
        assert result.success
        assert "Detail me" in result.output
        assert "Full instructions here" in result.output
        assert "researcher" in result.output

    async def test_detail_not_found(self, skill):
        result = await skill.execute_tool("task_detail", {"task_id": "t-ghost"})
        assert not result.success
        assert "not found" in result.error.lower()


# ── cancel_task ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestCancelTask:

    async def test_cancel_queued_task(self, skill, tmp_store):
        task = Task(title="Cancel me", instruction="i")
        await tmp_store.save(task)

        result = await skill.execute_tool("cancel_task", {"task_id": task.id})
        assert result.success
        assert "cancelled" in result.output.lower()

    async def test_cancel_nonexistent(self, skill):
        result = await skill.execute_tool("cancel_task", {"task_id": "t-nope"})
        assert not result.success


# ── clear_tasks ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestClearTasks:

    async def test_clear_terminal_task(self, skill, tmp_store):
        task = Task(title="Done", instruction="i", status=TaskStatus.DONE)
        await tmp_store.save(task)

        result = await skill.execute_tool("clear_tasks", {"task_ids": [task.id]})
        assert result.success
        assert "cleared 1 task" in result.output.lower()
        assert await tmp_store.get_by_id(task.id) is None

    async def test_clear_requires_terminal_task(self, skill, tmp_store):
        task = Task(title="Queued", instruction="i", status=TaskStatus.QUEUED)
        await tmp_store.save(task)

        result = await skill.execute_tool("clear_tasks", {"task_ids": [task.id]})
        assert not result.success
        assert "completed" in result.error.lower()


# ── reply_to_task ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestReplyToTask:

    async def test_reply_delegates_to_processor(self, skill, mock_processor):
        result = await skill.execute_tool("reply_to_task", {
            "task_id": "t-123",
            "reply": "Use PostgreSQL",
            "action": "approve",
        })
        assert result.success
        mock_processor.handle_human_reply.assert_called_once_with(
            "t-123", "Use PostgreSQL", "approve",
        )


# ── list_agents ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestListAgents:

    async def test_list_agents(self, skill):
        result = await skill.execute_tool("list_agents", {})
        assert result.success
        assert "researcher" in result.output
        assert "writer" in result.output
        assert "3 agent" in result.output

    async def test_list_agents_empty(self, tmp_store, mock_processor):
        await tmp_store.initialize()
        skill = TaskSkill()
        skill.set_dependencies(store=tmp_store, agents={}, processor=mock_processor)

        result = await skill.execute_tool("list_agents", {})
        assert result.success
        assert "no agents" in result.output.lower()


# ── unknown tool ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestUnknownTool:

    async def test_unknown_tool(self, skill):
        result = await skill.execute_tool("nonexistent_tool", {})
        assert not result.success
        assert "unknown" in result.error.lower()
