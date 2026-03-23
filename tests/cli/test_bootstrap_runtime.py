from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arc.cli.bootstrap import ArcRuntime
from arc.mcp.gateway import MCPGatewaySkill
from arc.mcp.manager import MCPManager
from arc.skills.builtin.worker import WorkerSkill


class _SkillManagerStub:
    def __init__(self, skills: dict[str, object] | None = None) -> None:
        self.skills = dict(skills or {})
        self.registered: list[str] = []
        self.unregistered: list[str] = []

    def get_skill(self, name: str) -> object | None:
        return self.skills.get(name)

    async def register(self, skill: object) -> None:
        name = skill.manifest().name
        self.skills[name] = skill
        self.registered.append(name)

    async def unregister(self, name: str) -> bool:
        self.unregistered.append(name)
        return self.skills.pop(name, None) is not None


class _AgentStub:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def set_system_prompt(self, prompt: str) -> None:
        self.prompts.append(prompt)


class _AsyncStub:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def shutdown_all(self) -> None:
        return None


class _WorkerLogStub:
    def close(self) -> None:
        return None


class _ConfigServiceStub:
    def __init__(self, *, valid: bool, text: str) -> None:
        self.valid = valid
        self.text = text
        self.marked: list[tuple[str, list[str]]] = []
        self.started = False

    def inspect(self) -> dict[str, object]:
        return {"valid": self.valid, "text": self.text}

    def mark_applied(self, text: str, active_server_names: list[str]) -> None:
        self.marked.append((text, list(active_server_names)))

    async def start(self) -> None:
        self.started = True


def _build_runtime(
    tmp_path: Path,
    *,
    initial_servers: dict[str, dict[str, object]] | None = None,
) -> tuple[ArcRuntime, _SkillManagerStub, _AgentStub, WorkerSkill]:
    manager = MCPManager(config_path=tmp_path / "mcp.json")
    manager.discover({"mcpServers": initial_servers or {}})

    prompt_state = {"names": manager.server_names}
    agent = _AgentStub()
    worker_skill = WorkerSkill()

    skills: dict[str, object] = {"worker": worker_skill}
    if manager.has_servers:
        skills["mcp_gateway"] = MCPGatewaySkill(manager)
    skill_manager = _SkillManagerStub(skills)

    runtime = ArcRuntime(
        config=SimpleNamespace(scheduler=SimpleNamespace(enabled=False)),
        identity={},
        kernel=_AsyncStub(),
        llm=_AsyncStub(),
        worker_llm=_AsyncStub(),
        agent=agent,
        skill_manager=skill_manager,
        skill_router=SimpleNamespace(),
        security=SimpleNamespace(),
        cost_tracker=SimpleNamespace(),
        memory_manager=None,
        sched_store=_AsyncStub(),
        scheduler_engine=None,
        notification_router=SimpleNamespace(),
        agent_registry=_AsyncStub(),
        escalation_bus=SimpleNamespace(),
        worker_log=_WorkerLogStub(),
        mcp_manager=manager,
        run_control=SimpleNamespace(),
        turn_controller=SimpleNamespace(),
        make_sub_agent=lambda: None,
        system_prompt="main:none",
        env_info="",
        task_store=None,
        task_processor=None,
        agent_defs={},
        mcp_config_service=None,
        build_main_system_prompt=lambda: f"main:{','.join(prompt_state['names']) or 'none'}",
        build_worker_system_prompt=lambda: f"worker:{','.join(prompt_state['names']) or 'none'}",
        mcp_prompt_state=prompt_state,
    )

    return runtime, skill_manager, agent, worker_skill


@pytest.mark.asyncio
async def test_apply_mcp_config_registers_gateway_and_refreshes_prompts(tmp_path):
    runtime, skill_manager, agent, worker_skill = _build_runtime(tmp_path)

    data = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            }
        }
    }

    await runtime.apply_mcp_config(json.dumps(data), data)

    assert runtime.mcp_manager.server_names == ["filesystem"]
    assert runtime.mcp_prompt_state["names"] == ["filesystem"]
    assert "mcp_gateway" in skill_manager.skills
    assert isinstance(skill_manager.skills["mcp_gateway"], MCPGatewaySkill)
    assert runtime.system_prompt == "main:filesystem"
    assert agent.prompts == ["main:filesystem"]
    assert worker_skill._worker_system_prompt == "worker:filesystem"


@pytest.mark.asyncio
async def test_apply_mcp_config_unregisters_gateway_when_servers_removed(tmp_path):
    runtime, skill_manager, agent, worker_skill = _build_runtime(
        tmp_path,
        initial_servers={"github": {"command": "echo"}},
    )

    data = {"mcpServers": {}}

    await runtime.apply_mcp_config(json.dumps(data), data)

    assert runtime.mcp_manager.server_names == []
    assert "mcp_gateway" not in skill_manager.skills
    assert skill_manager.unregistered == ["mcp_gateway"]
    assert runtime.system_prompt == "main:none"
    assert agent.prompts == ["main:none"]
    assert worker_skill._worker_system_prompt == "worker:none"


@pytest.mark.asyncio
async def test_runtime_start_does_not_mark_invalid_mcp_config_as_applied(tmp_path):
    runtime, _, _, _ = _build_runtime(tmp_path)
    runtime.mcp_config_service = _ConfigServiceStub(valid=False, text='{"broken":')

    await runtime.start()

    assert runtime.mcp_config_service.started is True
    assert runtime.mcp_config_service.marked == []
