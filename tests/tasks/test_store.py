"""Tests for arc/tasks/store.py"""
from __future__ import annotations

import time

import pytest

from arc.tasks.store import TaskStore
from arc.tasks.types import Task, TaskStatus, TaskStep


@pytest.fixture
def tmp_store(tmp_path):
    return TaskStore(db_path=tmp_path / "test_tasks.db")


@pytest.mark.asyncio
class TestTaskStore:

    # ── Initialization ───────────────────────────────────────────────────────

    async def test_initialize_creates_tables(self, tmp_store):
        await tmp_store.initialize()
        tasks = await tmp_store.get_all()
        assert isinstance(tasks, list)
        assert len(tasks) == 0
        await tmp_store.close()

    # ── Save and retrieve ────────────────────────────────────────────────────

    async def test_save_and_get_by_id(self, tmp_store):
        await tmp_store.initialize()
        task = Task(title="Research AI", instruction="Find latest papers")
        await tmp_store.save(task)

        found = await tmp_store.get_by_id(task.id)
        assert found is not None
        assert found.title == "Research AI"
        assert found.instruction == "Find latest papers"
        assert found.status == TaskStatus.QUEUED
        await tmp_store.close()

    async def test_save_with_steps(self, tmp_store):
        await tmp_store.initialize()
        steps = [
            TaskStep(step_index=0, agent_name="designer", review_by="reviewer"),
            TaskStep(step_index=1, agent_name="implementer"),
        ]
        task = Task(title="Build", instruction="Build it", steps=steps)
        await tmp_store.save(task)

        found = await tmp_store.get_by_id(task.id)
        assert found is not None
        assert len(found.steps) == 2
        assert found.steps[0].agent_name == "designer"
        assert found.steps[0].review_by == "reviewer"
        assert found.steps[1].review_by is None
        await tmp_store.close()

    async def test_get_by_id_missing_returns_none(self, tmp_store):
        await tmp_store.initialize()
        assert await tmp_store.get_by_id("t-nonexistent") is None
        await tmp_store.close()

    # ── Actionable tasks ─────────────────────────────────────────────────────

    async def test_get_actionable_tasks_returns_queued(self, tmp_store):
        await tmp_store.initialize()
        task = Task(title="T1", instruction="Do stuff", assigned_agent="researcher",
                    steps=[TaskStep(step_index=0, agent_name="researcher")])
        await tmp_store.save(task)

        actionable = await tmp_store.get_actionable_tasks(["researcher"])
        assert len(actionable) == 1
        assert actionable[0].id == task.id
        await tmp_store.close()

    async def test_get_actionable_tasks_returns_revision_needed(self, tmp_store):
        await tmp_store.initialize()
        task = Task(title="T1", instruction="Do stuff", assigned_agent="writer",
                    status=TaskStatus.REVISION_NEEDED,
                    steps=[TaskStep(step_index=0, agent_name="writer")])
        await tmp_store.save(task)

        actionable = await tmp_store.get_actionable_tasks(["writer"])
        assert len(actionable) == 1
        await tmp_store.close()

    async def test_get_actionable_tasks_ignores_done(self, tmp_store):
        await tmp_store.initialize()
        task = Task(title="Done", instruction="Done task", assigned_agent="a",
                    status=TaskStatus.DONE,
                    steps=[TaskStep(step_index=0, agent_name="a")])
        await tmp_store.save(task)

        actionable = await tmp_store.get_actionable_tasks(["a"])
        assert len(actionable) == 0
        await tmp_store.close()

    async def test_get_actionable_tasks_filters_by_agent(self, tmp_store):
        await tmp_store.initialize()
        t1 = Task(title="For A", instruction="i", assigned_agent="a",
                  steps=[TaskStep(step_index=0, agent_name="a")])
        t2 = Task(title="For B", instruction="i", assigned_agent="b",
                  steps=[TaskStep(step_index=0, agent_name="b")])
        await tmp_store.save(t1)
        await tmp_store.save(t2)

        result = await tmp_store.get_actionable_tasks(["a"])
        assert len(result) == 1
        assert result[0].title == "For A"
        await tmp_store.close()

    async def test_get_actionable_respects_dependencies(self, tmp_store):
        await tmp_store.initialize()
        parent = Task(title="Parent", instruction="i", assigned_agent="a",
                      steps=[TaskStep(step_index=0, agent_name="a")])
        child = Task(title="Child", instruction="i", assigned_agent="a",
                     depends_on=parent.id,
                     steps=[TaskStep(step_index=0, agent_name="a")])
        await tmp_store.save(parent)
        await tmp_store.save(child)

        # Child should NOT appear because parent is still queued
        actionable = await tmp_store.get_actionable_tasks(["a"])
        titles = [t.title for t in actionable]
        assert "Parent" in titles
        assert "Child" not in titles

        # Complete parent
        await tmp_store.update_status_with_comment(
            parent.id, TaskStatus.DONE, "system", "Done",
        )

        # Now child should appear
        actionable = await tmp_store.get_actionable_tasks(["a"])
        titles = [t.title for t in actionable]
        assert "Child" in titles
        await tmp_store.close()

    # ── Status transitions with comments ─────────────────────────────────────

    async def test_update_status_with_comment(self, tmp_store):
        await tmp_store.initialize()
        task = Task(title="T", instruction="i")
        await tmp_store.save(task)

        await tmp_store.update_status_with_comment(
            task.id, TaskStatus.IN_PROGRESS, "researcher",
            "Picked up the task.", step_index=0,
            extra_updates={"started_at": int(time.time())},
        )

        found = await tmp_store.get_by_id(task.id)
        assert found is not None
        assert found.status == TaskStatus.IN_PROGRESS
        assert found.started_at > 0

        comments = await tmp_store.get_comments(task.id)
        assert len(comments) == 1
        assert comments[0].agent_name == "researcher"
        assert "Picked up" in comments[0].content
        await tmp_store.close()

    async def test_multiple_comments_ordered(self, tmp_store):
        await tmp_store.initialize()
        task = Task(title="T", instruction="i")
        await tmp_store.save(task)

        await tmp_store.add_comment(task.id, "agent_a", "First comment")
        await tmp_store.add_comment(task.id, "agent_b", "Second comment")
        await tmp_store.add_comment(task.id, "human", "Third comment")

        comments = await tmp_store.get_comments(task.id)
        assert len(comments) == 3
        assert comments[0].agent_name == "agent_a"
        assert comments[1].agent_name == "agent_b"
        assert comments[2].agent_name == "human"
        await tmp_store.close()

    # ── Blocked / awaiting human ─────────────────────────────────────────────

    async def test_get_blocked_task(self, tmp_store):
        await tmp_store.initialize()
        task = Task(title="Blocked", instruction="i", status=TaskStatus.BLOCKED)
        await tmp_store.save(task)

        found = await tmp_store.get_blocked_task(task.id)
        assert found is not None
        assert found.status == TaskStatus.BLOCKED
        await tmp_store.close()

    async def test_get_blocked_task_returns_none_for_non_blocked(self, tmp_store):
        await tmp_store.initialize()
        task = Task(title="Normal", instruction="i", status=TaskStatus.QUEUED)
        await tmp_store.save(task)

        found = await tmp_store.get_blocked_task(task.id)
        assert found is None
        await tmp_store.close()

    async def test_get_blocked_task_returns_awaiting_human(self, tmp_store):
        await tmp_store.initialize()
        task = Task(title="Awaiting", instruction="i", status=TaskStatus.AWAITING_HUMAN)
        await tmp_store.save(task)

        found = await tmp_store.get_blocked_task(task.id)
        assert found is not None
        await tmp_store.close()

    # ── Cancel ───────────────────────────────────────────────────────────────

    async def test_cancel_task(self, tmp_store):
        await tmp_store.initialize()
        task = Task(title="Cancel me", instruction="i")
        await tmp_store.save(task)

        ok = await tmp_store.cancel(task.id)
        assert ok is True

        found = await tmp_store.get_by_id(task.id)
        assert found is not None
        assert found.status == TaskStatus.CANCELLED
        assert found.completed_at > 0
        await tmp_store.close()

    async def test_cancel_done_task_fails(self, tmp_store):
        await tmp_store.initialize()
        task = Task(title="Done", instruction="i", status=TaskStatus.DONE)
        await tmp_store.save(task)

        ok = await tmp_store.cancel(task.id)
        assert ok is False
        await tmp_store.close()

    async def test_cancel_nonexistent_fails(self, tmp_store):
        await tmp_store.initialize()
        ok = await tmp_store.cancel("t-ghost")
        assert ok is False
        await tmp_store.close()

    # ── Get all with filters ─────────────────────────────────────────────────

    async def test_get_all_with_status_filter(self, tmp_store):
        await tmp_store.initialize()
        t1 = Task(title="Q", instruction="i", status=TaskStatus.QUEUED)
        t2 = Task(title="D", instruction="i", status=TaskStatus.DONE)
        await tmp_store.save(t1)
        await tmp_store.save(t2)

        queued = await tmp_store.get_all(status="queued")
        assert len(queued) == 1
        assert queued[0].title == "Q"

        done = await tmp_store.get_all(status="done")
        assert len(done) == 1
        assert done[0].title == "D"
        await tmp_store.close()

    async def test_get_all_respects_limit(self, tmp_store):
        await tmp_store.initialize()
        for i in range(10):
            await tmp_store.save(Task(title=f"Task {i}", instruction="i"))

        result = await tmp_store.get_all(limit=3)
        assert len(result) == 3
        await tmp_store.close()

    # ── Count in progress ────────────────────────────────────────────────────

    async def test_count_in_progress(self, tmp_store):
        await tmp_store.initialize()
        t1 = Task(title="Active", instruction="i", status=TaskStatus.IN_PROGRESS,
                  assigned_agent="a", steps=[TaskStep(step_index=0, agent_name="a")])
        t2 = Task(title="Queued", instruction="i", status=TaskStatus.QUEUED,
                  assigned_agent="a", steps=[TaskStep(step_index=0, agent_name="a")])
        await tmp_store.save(t1)
        await tmp_store.save(t2)

        count = await tmp_store.count_in_progress("a")
        assert count == 1
        await tmp_store.close()

    # ── Priority ordering ────────────────────────────────────────────────────

    async def test_actionable_ordered_by_priority(self, tmp_store):
        await tmp_store.initialize()
        low = Task(title="Low", instruction="i", priority=5, assigned_agent="a",
                   steps=[TaskStep(step_index=0, agent_name="a")])
        high = Task(title="High", instruction="i", priority=1, assigned_agent="a",
                    steps=[TaskStep(step_index=0, agent_name="a")])
        await tmp_store.save(low)
        await tmp_store.save(high)

        result = await tmp_store.get_actionable_tasks(["a"])
        assert result[0].title == "High"
        assert result[1].title == "Low"
        await tmp_store.close()
