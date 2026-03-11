"""Tests for arc/tasks/types.py"""
from __future__ import annotations

from arc.tasks.types import (
    AgentDef,
    Task,
    TaskComment,
    TaskStatus,
    TaskStep,
)


class TestTaskStatus:
    def test_terminal_states(self):
        done = Task(title="t", instruction="i", status=TaskStatus.DONE)
        assert done.is_terminal

        failed = Task(title="t", instruction="i", status=TaskStatus.FAILED)
        assert failed.is_terminal

        cancelled = Task(title="t", instruction="i", status=TaskStatus.CANCELLED)
        assert cancelled.is_terminal

    def test_non_terminal_states(self):
        for status in (TaskStatus.QUEUED, TaskStatus.IN_PROGRESS, TaskStatus.IN_REVIEW,
                       TaskStatus.BLOCKED, TaskStatus.AWAITING_HUMAN):
            t = Task(title="t", instruction="i", status=status)
            assert not t.is_terminal


class TestTask:
    def test_round_trip_dict(self):
        steps = [
            TaskStep(step_index=0, agent_name="designer", review_by="reviewer"),
            TaskStep(step_index=1, agent_name="implementer"),
        ]
        task = Task(
            title="Build feature",
            instruction="Build the caching feature",
            steps=steps,
            assigned_agent="designer",
            max_bounces=5,
            priority=2,
            depends_on="t-abc12345",
        )

        d = task.to_dict()
        restored = Task.from_dict(d)

        assert restored.title == task.title
        assert restored.instruction == task.instruction
        assert len(restored.steps) == 2
        assert restored.steps[0].review_by == "reviewer"
        assert restored.steps[1].review_by is None
        assert restored.assigned_agent == "designer"
        assert restored.max_bounces == 5
        assert restored.priority == 2
        assert restored.depends_on == "t-abc12345"
        assert restored.id == task.id

    def test_current_agent_with_steps(self):
        steps = [
            TaskStep(step_index=0, agent_name="agent_a"),
            TaskStep(step_index=1, agent_name="agent_b"),
        ]
        task = Task(title="t", instruction="i", steps=steps, current_step=0)
        assert task.current_agent == "agent_a"

        task.current_step = 1
        assert task.current_agent == "agent_b"

    def test_current_agent_no_steps(self):
        task = Task(title="t", instruction="i", assigned_agent="solo")
        assert task.current_agent == "solo"

    def test_current_reviewer(self):
        steps = [
            TaskStep(step_index=0, agent_name="writer", review_by="reviewer"),
            TaskStep(step_index=1, agent_name="publisher"),
        ]
        task = Task(title="t", instruction="i", steps=steps, current_step=0)
        assert task.current_reviewer == "reviewer"

        task.current_step = 1
        assert task.current_reviewer is None

    def test_id_auto_generated(self):
        t1 = Task(title="a", instruction="b")
        t2 = Task(title="a", instruction="b")
        assert t1.id != t2.id
        assert t1.id.startswith("t-")

    def test_default_values(self):
        task = Task(title="t", instruction="i")
        assert task.status == TaskStatus.QUEUED
        assert task.bounce_count == 0
        assert task.max_bounces == 3
        assert task.priority == 1
        assert task.result == ""
        assert task.depends_on is None


class TestTaskStep:
    def test_round_trip_dict(self):
        step = TaskStep(step_index=0, agent_name="coder", review_by="human")
        d = step.to_dict()
        restored = TaskStep.from_dict(d)
        assert restored.step_index == 0
        assert restored.agent_name == "coder"
        assert restored.review_by == "human"

    def test_no_reviewer(self):
        step = TaskStep(step_index=0, agent_name="fetcher")
        assert step.review_by is None
        d = step.to_dict()
        assert d["review_by"] is None


class TestAgentDef:
    def test_has_llm_override(self):
        agent = AgentDef(name="test", role="test", llm_provider="openai", llm_model="gpt-4o")
        assert agent.has_llm_override

    def test_no_llm_override(self):
        agent = AgentDef(name="test", role="test")
        assert not agent.has_llm_override

    def test_build_system_prompt(self):
        agent = AgentDef(name="researcher", role="Web research", personality="thorough")
        prompt = agent.build_system_prompt()
        assert "researcher" in prompt
        assert "Web research" in prompt
        assert "thorough" in prompt
        assert "task queue" in prompt.lower()
