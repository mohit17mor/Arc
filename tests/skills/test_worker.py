"""Tests for WorkerSkill."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.skills.builtin.worker import WorkerSkill, _ALWAYS_EXCLUDED
from arc.core.types import ToolResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def skill():
    """WorkerSkill with no dependencies injected (raw)."""
    return WorkerSkill()


@pytest.fixture
def mock_skill_manager():
    mgr = MagicMock()
    mgr.skill_names = ["filesystem", "terminal", "browsing", "worker", "scheduler"]
    return mgr


@pytest.fixture
def mock_notification_router():
    router = AsyncMock()
    router.route = AsyncMock()
    return router


@pytest.fixture
def mock_agent_registry():
    registry = MagicMock()
    registry.register_worker = MagicMock()
    return registry


@pytest.fixture
def wired_skill(mock_skill_manager, mock_notification_router, mock_agent_registry):
    """WorkerSkill with all dependencies injected."""
    s = WorkerSkill()
    s.set_dependencies(
        llm=MagicMock(),
        skill_manager=mock_skill_manager,
        escalation_bus=MagicMock(),
        notification_router=mock_notification_router,
        agent_registry=mock_agent_registry,
    )
    return s


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_name(skill):
    assert skill.manifest().name == "worker"


def test_manifest_has_delegate_task_tool(skill):
    tools = {t.name for t in skill.manifest().tools}
    assert "delegate_task" in tools


def test_manifest_has_list_workers_tool(skill):
    tools = {t.name for t in skill.manifest().tools}
    assert "list_workers" in tools


def test_manifest_delegate_task_requires_task_name_and_prompt(skill):
    tool = next(t for t in skill.manifest().tools if t.name == "delegate_task")
    required = tool.parameters.get("required", [])
    assert "task_name" in required
    assert "prompt" in required


def test_manifest_allowed_skills_is_optional(skill):
    tool = next(t for t in skill.manifest().tools if t.name == "delegate_task")
    required = tool.parameters.get("required", [])
    assert "allowed_skills" not in required


# ---------------------------------------------------------------------------
# _compute_excluded
# ---------------------------------------------------------------------------

def test_compute_excluded_defaults_to_always_excluded(wired_skill):
    excluded = wired_skill._compute_excluded(None)
    assert excluded == _ALWAYS_EXCLUDED


def test_compute_excluded_with_allowed_skills(wired_skill):
    """Only browsing allowed → everything else + always_excluded get excluded."""
    excluded = wired_skill._compute_excluded(["browsing"])
    # browsing should NOT be excluded
    assert "browsing" not in excluded
    # worker and scheduler always excluded
    assert "worker" in excluded
    assert "scheduler" in excluded
    # other skills excluded because not in allowed list
    assert "filesystem" in excluded
    assert "terminal" in excluded


def test_compute_excluded_allowed_cant_include_always_excluded(wired_skill):
    """Even if caller lists 'worker' in allowed, it stays excluded."""
    excluded = wired_skill._compute_excluded(["browsing", "worker", "scheduler"])
    assert "worker" in excluded
    assert "scheduler" in excluded


def test_compute_excluded_all_skills_allowed(wired_skill):
    """If all skills listed as allowed, only always_excluded are excluded."""
    excluded = wired_skill._compute_excluded(
        ["filesystem", "terminal", "browsing"]
    )
    assert excluded == _ALWAYS_EXCLUDED


# ---------------------------------------------------------------------------
# execute_tool — unknown tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_unknown_tool(wired_skill):
    result = await wired_skill.execute_tool("nonexistent", {})
    assert not result.success
    assert "Unknown tool" in (result.error or result.output)


# ---------------------------------------------------------------------------
# list_workers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_workers_no_registry(skill):
    result = await skill.execute_tool("list_workers", {})
    assert result.success  # graceful, not an error


@pytest.mark.asyncio
async def test_list_workers_none_running(wired_skill, mock_agent_registry):
    mock_agent_registry.list_worker_ids = MagicMock(return_value=[])
    result = await wired_skill.execute_tool("list_workers", {})
    assert result.success
    assert "no" in result.output.lower() or "0" in result.output


@pytest.mark.asyncio
async def test_list_workers_shows_active_ids(wired_skill, mock_agent_registry):
    mock_agent_registry.list_worker_ids = MagicMock(
        return_value=["research_a1b2c3d4", "analysis_e5f6a7b8"]
    )
    result = await wired_skill.execute_tool("list_workers", {})
    assert result.success
    assert "research_a1b2c3d4" in result.output
    assert "analysis_e5f6a7b8" in result.output


# ---------------------------------------------------------------------------
# execute_tool — dependencies not injected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delegate_task_without_dependencies_returns_error(skill):
    result = await skill.execute_tool(
        "delegate_task",
        {"task_name": "test", "prompt": "do something"},
    )
    assert not result.success
    assert "not initialised" in result.output.lower() or "not initialised" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# delegate_task — fire and forget: returns immediately
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delegate_task_returns_immediately(wired_skill):
    """delegate_task must return success immediately without waiting for worker."""
    with patch.object(
        wired_skill, "_run_and_notify", new=AsyncMock()
    ):
        result = await wired_skill.execute_tool(
            "delegate_task",
            {"task_name": "research", "prompt": "find AI news"},
        )

    # Returns immediately with success
    assert result.success
    assert "started" in result.output.lower()


@pytest.mark.asyncio
async def test_delegate_task_confirmation_contains_task_name(wired_skill):
    with patch.object(wired_skill, "_run_and_notify", new=AsyncMock()):
        result = await wired_skill.execute_tool(
            "delegate_task",
            {"task_name": "my_research_task", "prompt": "do work"},
        )
    assert "my_research_task" in result.output


@pytest.mark.asyncio
async def test_delegate_task_registers_with_agent_registry(wired_skill, mock_agent_registry):
    """Worker task must be registered for graceful shutdown."""
    with patch.object(wired_skill, "_run_and_notify", new=AsyncMock()):
        await wired_skill.execute_tool(
            "delegate_task",
            {"task_name": "reg_test", "prompt": "do work"},
        )
    # Give the create_task a chance to be registered
    await asyncio.sleep(0)
    mock_agent_registry.register_worker.assert_called_once()


# ---------------------------------------------------------------------------
# delegate_task — timeout / max_iterations forwarded and clamped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delegate_task_forwards_custom_timeout_and_iterations(wired_skill):
    captured = {}

    async def capture(task_id, task_name, prompt, excluded, **kwargs):
        captured.update(kwargs)

    with patch.object(wired_skill, "_run_and_notify", side_effect=capture):
        await wired_skill.execute_tool(
            "delegate_task",
            {"task_name": "long", "prompt": "deep work", "timeout_seconds": 600, "max_iterations": 40},
        )

    await asyncio.sleep(0)
    assert captured.get("timeout_seconds") == 600
    assert captured.get("max_iterations") == 40


@pytest.mark.asyncio
async def test_delegate_task_clamps_timeout_to_ceiling(wired_skill):
    captured = {}

    async def capture(task_id, task_name, prompt, excluded, **kwargs):
        captured.update(kwargs)

    with patch.object(wired_skill, "_run_and_notify", side_effect=capture):
        await wired_skill.execute_tool(
            "delegate_task",
            {"task_name": "greedy", "prompt": "run forever", "timeout_seconds": 99999},
        )

    await asyncio.sleep(0)
    assert captured.get("timeout_seconds") == WorkerSkill._MAX_TIMEOUT


@pytest.mark.asyncio
async def test_delegate_task_clamps_iterations_to_ceiling(wired_skill):
    captured = {}

    async def capture(task_id, task_name, prompt, excluded, **kwargs):
        captured.update(kwargs)

    with patch.object(wired_skill, "_run_and_notify", side_effect=capture):
        await wired_skill.execute_tool(
            "delegate_task",
            {"task_name": "greedy", "prompt": "run forever", "max_iterations": 999},
        )

    await asyncio.sleep(0)
    assert captured.get("max_iterations") == WorkerSkill._MAX_ITERATIONS


@pytest.mark.asyncio
async def test_delegate_task_confirmation_shows_timeout(wired_skill):
    with patch.object(wired_skill, "_run_and_notify", new=AsyncMock()):
        result = await wired_skill.execute_tool(
            "delegate_task",
            {"task_name": "timed", "prompt": "go", "timeout_seconds": 300},
        )
    assert "5m" in result.output  # 300s → "5m"


# ---------------------------------------------------------------------------
# _run_and_notify — result delivered via notification router
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_and_notify_routes_success_notification(wired_skill, mock_notification_router):
    with patch.object(
        wired_skill, "_run_worker", new=AsyncMock(return_value=("great result", None))
    ):
        await wired_skill._run_and_notify("t1", "my_task", "prompt", frozenset())

    mock_notification_router.route.assert_called_once()
    notification = mock_notification_router.route.call_args[0][0]
    assert "great result" in notification.content
    assert notification.job_name == "my_task"


@pytest.mark.asyncio
async def test_run_and_notify_routes_error_notification_on_double_failure(wired_skill, mock_notification_router):
    with patch.object(
        wired_skill, "_run_worker", new=AsyncMock(return_value=("", "network error"))
    ):
        await wired_skill._run_and_notify("t2", "bad_task", "prompt", frozenset())

    mock_notification_router.route.assert_called_once()
    notification = mock_notification_router.route.call_args[0][0]
    assert "failed" in notification.content.lower() or "❌" in notification.content


@pytest.mark.asyncio
async def test_run_and_notify_retries_once_on_failure(wired_skill, mock_notification_router):
    call_count = 0

    async def flaky(task_id, prompt, excluded, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ("", "timeout")
        return ("recovered", None)

    with patch.object(wired_skill, "_run_worker", side_effect=flaky):
        await wired_skill._run_and_notify("t3", "flaky_task", "prompt", frozenset())

    assert call_count == 2
    notification = mock_notification_router.route.call_args[0][0]
    assert "recovered" in notification.content


# ---------------------------------------------------------------------------
# delegate_task — allowed_skills scoping forwarded correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delegate_task_passes_excluded_to_run_worker(wired_skill):
    captured = {}

    async def capture(task_id, task_name, prompt, excluded, **kwargs):
        captured["excluded"] = excluded

    with patch.object(wired_skill, "_run_and_notify", side_effect=capture):
        await wired_skill.execute_tool(
            "delegate_task",
            {
                "task_name": "scoped",
                "prompt": "browse only",
                "allowed_skills": ["browsing"],
            },
        )

    await asyncio.sleep(0)  # let the background task fire
    assert "worker" in captured.get("excluded", frozenset())
    assert "scheduler" in captured.get("excluded", frozenset())
    assert "browsing" not in captured.get("excluded", frozenset())


# ---------------------------------------------------------------------------
# Integration — run with real VirtualPlatform + MockLLMProvider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delegate_task_integration_notification_delivered(kernel):
    """
    Full integration: WorkerSkill → VirtualPlatform → AgentLoop → MockLLM
    → NotificationRouter → pending_queue.
    """
    from arc.llm.mock import MockLLMProvider
    from arc.skills.manager import SkillManager
    from arc.notifications.router import NotificationRouter
    from arc.notifications.channels.cli import CLIChannel

    mock_llm = MockLLMProvider()
    mock_llm.set_response("Here is the AI news summary: nothing major today.")

    skill_mgr = SkillManager(kernel)
    from arc.skills.builtin.filesystem import FilesystemSkill
    await skill_mgr.register(FilesystemSkill())

    pending_queue: asyncio.Queue = asyncio.Queue()
    cli_channel = CLIChannel(pending_queue)
    cli_channel.set_active(True)
    router = NotificationRouter()
    router.register(cli_channel)

    registry_mock = MagicMock()
    registry_mock.register_worker = MagicMock()

    worker_skill = WorkerSkill()
    worker_skill.set_dependencies(
        llm=mock_llm,
        skill_manager=skill_mgr,
        escalation_bus=MagicMock(),
        notification_router=router,
        agent_registry=registry_mock,
    )
    await worker_skill.initialize(kernel, {})

    result = await worker_skill.execute_tool(
        "delegate_task",
        {"task_name": "news_summary", "prompt": "Summarise AI news"},
    )

    # Returns immediately
    assert result.success
    assert "news_summary" in result.output

    # Wait for background task to complete and deliver notification
    notification = await asyncio.wait_for(pending_queue.get(), timeout=10.0)
    assert notification.job_name == "news_summary"
    assert len(notification.content) > 0
