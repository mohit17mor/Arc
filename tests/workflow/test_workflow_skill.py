from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from arc.workflow.skill import WorkflowSkill
from arc.workflow.types import Workflow, WorkflowStep


def _workflow(name: str, *, description: str = "", triggers: list[str] | None = None) -> Workflow:
    return Workflow(
        name=name,
        description=description,
        trigger_patterns=triggers or [],
        steps=[WorkflowStep(instruction=f"run {name}")],
    )


async def _yield_chunks(chunks):
    for chunk in chunks:
        yield chunk


class _FakeEngine:
    def __init__(self, chunks=None, error: Exception | None = None):
        self._chunks = list(chunks or [])
        self._error = error
        self.calls: list[tuple[Workflow, str]] = []
        self.is_waiting_for_input = False
        self.provided: list[str] = []

    async def run(self, workflow: Workflow, user_message: str = ""):
        self.calls.append((workflow, user_message))
        if self._error is not None:
            raise self._error
        async for chunk in _yield_chunks(self._chunks):
            yield chunk

    def provide_input(self, user_input: str) -> bool:
        self.provided.append(user_input)
        return True


class TestWorkflowSkill:
    def test_manifest_mentions_available_workflows(self):
        skill = WorkflowSkill()
        skill._workflows = [_workflow("jira-rca"), _workflow("deploy-check")]

        manifest = skill.manifest()

        assert manifest.name == "workflow"
        assert manifest.tools[0].name == "list_workflows"
        assert "jira-rca, deploy-check" in manifest.tools[0].description

    def test_set_dependencies_builds_engine(self):
        skill = WorkflowSkill()
        agent = object()
        kernel = object()

        with patch("arc.workflow.skill.WorkflowEngine", return_value="engine") as mock_engine:
            skill.set_dependencies(agent=agent, kernel=kernel)

        mock_engine.assert_called_once_with(agent=agent, kernel=kernel)
        assert skill._engine == "engine"

    @pytest.mark.asyncio
    async def test_activate_loads_workflows_once_used(self):
        skill = WorkflowSkill()

        with patch("arc.workflow.skill.load_workflows", return_value=[_workflow("jira-rca")]) as mock_load:
            await skill.activate()

        mock_load.assert_called_once_with()
        assert skill._activated is True
        assert skill.workflow_names == ["jira-rca"]

    @pytest.mark.asyncio
    async def test_execute_tool_activates_on_first_call_and_rejects_unknown_tool(self):
        skill = WorkflowSkill()
        skill.activate = AsyncMock()

        result = await skill.execute_tool("unknown_tool", {})

        skill.activate.assert_awaited_once()
        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_list_workflows_returns_help_when_none_are_available(self):
        skill = WorkflowSkill()

        result = await skill._list_workflows()

        assert result.success is True
        assert "No workflows found" in result.output
        assert "Example workflow file" in result.output

    @pytest.mark.asyncio
    async def test_list_workflows_formats_names_descriptions_and_triggers(self):
        skill = WorkflowSkill()
        skill._workflows = [
            _workflow("jira-rca", description="Investigate ticket", triggers=["jira", "rca"]),
            _workflow("cleanup"),
        ]

        result = await skill._list_workflows()

        assert result.success is True
        assert "Available workflows (2)" in result.output
        assert "jira-rca" in result.output
        assert "Investigate ticket" in result.output
        assert "Triggers: jira, rca" in result.output

    @pytest.mark.asyncio
    async def test_run_workflow_requires_initialized_engine(self):
        skill = WorkflowSkill()

        result = await skill._run_workflow({"name": "jira-rca"})

        assert result.success is False
        assert "engine not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_run_workflow_reports_unknown_name(self):
        skill = WorkflowSkill()
        skill._engine = _FakeEngine()
        skill._workflows = [_workflow("jira-rca"), _workflow("deploy-check")]

        result = await skill._run_workflow({"name": "missing"})

        assert result.success is False
        assert "Workflow 'missing' not found" in result.error
        assert "jira-rca, deploy-check" in result.error

    @pytest.mark.asyncio
    async def test_run_workflow_starts_background_task(self, monkeypatch):
        skill = WorkflowSkill()
        skill._engine = _FakeEngine()
        skill._workflows = [_workflow("jira-rca")]
        created: list[asyncio.Task] = []
        real_create_task = asyncio.create_task

        def fake_create_task(coro, name=None):
            task = real_create_task(asyncio.sleep(0))
            created.append(task)
            coro.close()
            return task

        monkeypatch.setattr(asyncio, "create_task", fake_create_task)

        result = await skill._run_workflow({"name": "jira-rca", "context": "PROJ-123"})

        assert result.success is True
        assert "Workflow 'jira-rca' has been started" in result.output
        await created[0]

    @pytest.mark.asyncio
    async def test_run_interactive_workflow_drains_engine_output(self):
        skill = WorkflowSkill()
        skill._engine = _FakeEngine(chunks=["step 1", "step 2"])
        workflow = _workflow("jira-rca")

        await skill._run_interactive_workflow(workflow, "ticket context")

        assert skill._engine.calls == [(workflow, "ticket context")]

    @pytest.mark.asyncio
    async def test_run_interactive_workflow_logs_failures(self):
        skill = WorkflowSkill()
        workflow = _workflow("jira-rca")
        skill._engine = _FakeEngine(error=RuntimeError("boom"))

        with patch("arc.workflow.skill.logger") as mock_logger:
            await skill._run_interactive_workflow(workflow, "ticket context")

        mock_logger.error.assert_called_once()

    def test_get_workflow_returns_matching_definition(self):
        skill = WorkflowSkill()
        workflow = _workflow("jira-rca")
        skill._workflows = [workflow]

        assert skill.get_workflow("jira-rca") is workflow
        assert skill.get_workflow("missing") is None

    @pytest.mark.asyncio
    async def test_stream_workflow_reports_missing_engine(self):
        skill = WorkflowSkill()
        skill._activated = True

        chunks = [chunk async for chunk in skill.stream_workflow("jira-rca")]

        assert chunks == ["Workflow engine not initialized\n"]

    @pytest.mark.asyncio
    async def test_stream_workflow_reports_missing_name(self):
        skill = WorkflowSkill()
        skill._activated = True
        skill._engine = _FakeEngine()
        skill._workflows = [_workflow("jira-rca"), _workflow("deploy-check")]

        chunks = [chunk async for chunk in skill.stream_workflow("missing")]

        assert chunks == ["Workflow 'missing' not found. Available: jira-rca, deploy-check\n"]

    @pytest.mark.asyncio
    async def test_stream_workflow_yields_engine_chunks(self):
        skill = WorkflowSkill()
        workflow = _workflow("jira-rca")
        skill._activated = True
        skill._engine = _FakeEngine(chunks=["one", "two"])
        skill._workflows = [workflow]

        chunks = [chunk async for chunk in skill.stream_workflow("jira-rca", "from cli")]

        assert chunks == ["one", "two"]
        assert skill._engine.calls == [(workflow, "from cli")]

    def test_provide_input_and_waiting_flags_delegate_to_engine(self):
        skill = WorkflowSkill()
        engine = _FakeEngine()
        engine.is_waiting_for_input = True
        skill._engine = engine

        assert skill.provide_input("continue") is True
        assert engine.provided == ["continue"]
        assert skill.is_waiting_for_input is True

    def test_provide_input_and_waiting_flags_return_false_without_engine(self):
        skill = WorkflowSkill()

        assert skill.provide_input("continue") is False
        assert skill.is_waiting_for_input is False
