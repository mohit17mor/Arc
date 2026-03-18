"""
Shared bootstrap — initialises all Arc subsystems once.

Used by `arc chat`, `arc gateway`, and `arc telegram` so they share
exact the same setup code.  Add a new skill or config option here and
every command picks it up automatically.
"""

from __future__ import annotations

import asyncio
import logging
import platform as plat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ArcRuntime:
    """Everything that gets created during bootstrap.

    Platform-specific runners pull what they need from here.
    """

    config: Any  # ArcConfig
    identity: dict
    kernel: Any  # Kernel
    llm: Any  # LLMProvider
    worker_llm: Any  # LLMProvider
    agent: Any  # AgentLoop
    skill_manager: Any  # SkillManager
    skill_router: Any  # SkillRouter
    security: Any  # SecurityEngine
    cost_tracker: Any  # CostTracker
    memory_manager: Any | None  # MemoryManager or None
    sched_store: Any  # SchedulerStore
    scheduler_engine: Any | None  # SchedulerEngine or None
    notification_router: Any  # NotificationRouter
    agent_registry: Any  # AgentRegistry
    escalation_bus: Any  # EscalationBus
    worker_log: Any  # WorkerActivityLog
    mcp_manager: Any  # MCPManager
    run_control: Any  # RunControlManager
    turn_controller: Any  # ForegroundTurnController
    make_sub_agent: Callable  # factory for sub-agents
    system_prompt: str
    env_info: str
    task_store: Any | None = None  # TaskStore
    task_processor: Any | None = None  # TaskProcessor
    agent_defs: dict = field(default_factory=dict)  # name → AgentDef

    async def start(self) -> None:
        """Start kernel + scheduler + task processor."""
        await self.kernel.start()
        if self.scheduler_engine:
            await self.scheduler_engine.start()
        if self.task_processor:
            await self.task_processor.start()

    async def shutdown(self) -> None:
        """Shut down everything in the correct order."""
        self.worker_log.close()
        if self.task_processor:
            await self.task_processor.stop()
        await self.agent_registry.shutdown_all()
        if self.scheduler_engine:
            await self.scheduler_engine.stop()
        if self.config.scheduler.enabled:
            await self.sched_store.close()
        if self.task_store:
            await self.task_store.close()
        await self.skill_manager.shutdown_all()
        await self.kernel.stop()
        await self.llm.close()
        if self.worker_llm is not self.llm:
            await self.worker_llm.close()
        if self.memory_manager is not None:
            await self.memory_manager.close()


async def bootstrap(
    *,
    log_level: int = logging.WARNING,
    model_override: str | None = None,
    platform_name: str = "cli",
    interactive_security: bool = True,
) -> ArcRuntime:
    """
    Bootstrap all Arc subsystems and return an ArcRuntime.

    Args:
        log_level: Logging level (DEBUG for verbose).
        model_override: Override the configured LLM model.
        platform_name: Injected into the system prompt so the agent knows
            which platform it's running on.
        interactive_security: If True, use interactive approval prompts.
            Set False for headless platforms (gateway, telegram).
    """
    from arc.core.kernel import Kernel
    from arc.core.config import ArcConfig
    from arc.core.events import EventType
    from arc.llm.factory import create_llm
    from arc.skills.manager import SkillManager
    from arc.skills.loader import discover_skills, discover_soft_skills
    from arc.security.engine import SecurityEngine
    from arc.agent.loop import AgentLoop, AgentConfig
    from arc.identity.soul import SoulManager
    from arc.middleware.cost import CostTracker
    from arc.middleware.logging import setup_logging, EventLogger
    from arc.memory.manager import MemoryManager
    from arc.notifications.router import NotificationRouter
    from arc.notifications.channels.file import FileChannel
    from arc.notifications.channels.telegram import TelegramChannel
    from arc.scheduler.store import SchedulerStore
    from arc.scheduler.engine import SchedulerEngine
    from arc.skills.builtin.scheduler import SchedulerSkill
    from arc.skills.builtin.worker import WorkerSkill
    from arc.agent.registry import AgentRegistry
    from arc.core.escalation import EscalationBus
    from arc.agent.worker_log import WorkerActivityLog
    from arc.skills.router import SkillRouter
    from arc.mcp.manager import MCPManager
    from arc.mcp.gateway import MCPGatewaySkill
    from arc.skills.builtin.browser_control import BrowserControlSkill
    from arc.core.run_control import RunControlManager
    from arc.core.foreground_turns import ForegroundTurnController

    arc_home = Path.home() / ".arc"
    config_path = arc_home / "config.toml"
    identity_path = arc_home / "identity.md"

    # ── Logging ──
    setup_logging(log_dir=arc_home / "logs", console_level=log_level)

    # ── Config ──
    config = ArcConfig.load()
    if model_override:
        config.llm.default_model = model_override

    # ── Identity ──
    soul = SoulManager(identity_path)
    identity = soul.load()

    # ── Kernel + middleware ──
    kernel = Kernel(config=config)
    event_logger = EventLogger(log_dir=arc_home / "logs")
    kernel.use(event_logger.middleware)
    cost_tracker = CostTracker()
    kernel.use(cost_tracker.middleware)

    # ── LLM ──
    llm = create_llm(
        config.llm.default_provider,
        model=config.llm.default_model,
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
    )
    cost_tracker.context_window = llm.get_model_info().context_window

    if config.llm.has_worker_override:
        worker_llm = create_llm(
            config.llm.worker_provider,
            model=config.llm.worker_model,
            base_url=config.llm.worker_base_url or config.llm.base_url,
            api_key=config.llm.worker_api_key or config.llm.api_key,
        )
    else:
        worker_llm = llm

    # ── Skills ──
    skill_manager = SkillManager(kernel)
    for skill in discover_skills():
        await skill_manager.register(skill)

    # ── MCP ──
    mcp_manager = MCPManager()
    mcp_manager.discover()
    if mcp_manager.has_servers:
        mcp_gw = MCPGatewaySkill(mcp_manager)
        await skill_manager.register(mcp_gw)

    # ── Scheduler store ──
    sched_store = SchedulerStore(db_path=Path(config.scheduler.db_path).expanduser())
    if config.scheduler.enabled:
        await sched_store.initialize()
        sched_skill = skill_manager.get_skill("scheduler")
        if sched_skill and isinstance(sched_skill, SchedulerSkill):
            sched_skill.set_store(sched_store)

    # ── Security ──
    if interactive_security:
        security = SecurityEngine(config.security, kernel)
    else:
        security = SecurityEngine.make_permissive(kernel)

    # ── Skill router ──
    skill_router = SkillRouter(skill_manager)

    # ── System prompt ──
    env_info = (
        f"\n\nEnvironment:\n"
        f"- OS: {plat.system()} {plat.release()}\n"
        f"- Working directory: {Path.cwd()}\n"
        f"- Platform: {platform_name}\n"
    )

    # Soft skills = bundled strategies (tool usage, research, browser, delegation)
    # + user custom .md files from ~/.arc/skills/
    # Main agent gets delegation strategy; sub-agents do not.
    soft_skill_text = discover_soft_skills(include_delegation=True)
    soft_skill_text_no_delegation = discover_soft_skills(include_delegation=False)

    from arc.agent.prompts import get_reliability_block

    system_prompt = (
        identity["system_prompt"]
        + env_info
        + soft_skill_text
        + get_reliability_block("main")
    )

    if mcp_manager.has_servers:
        system_prompt += (
            "\n\nMCP (Model Context Protocol) Servers:\n"
            "External tool servers are available via mcp_list_tools and mcp_call.\n"
            f"Configured servers: {', '.join(mcp_manager.server_names)}\n"
        )

    run_control = RunControlManager()

    # ── Memory ──
    mem_db_path = arc_home / "memory" / "memory.db"
    memory_manager: MemoryManager | None = MemoryManager(db_path=str(mem_db_path))
    try:
        await memory_manager.initialize()
    except Exception as e:
        logger.warning(f"Long-term memory init failed: {e}")
        memory_manager = None

    # ── Agent ──
    agent = AgentLoop(
        kernel=kernel,
        llm=llm,
        skill_manager=skill_manager,
        security=security,
        system_prompt=system_prompt,
        config=AgentConfig(
            max_iterations=config.agent.max_iterations,
            temperature=config.agent.temperature,
        ),
        memory_manager=memory_manager,
        router=skill_router,
        run_control=run_control,
    )

    turn_controller = ForegroundTurnController(
        agent=agent,
        run_control=run_control,
        kernel=kernel,
    )

    # ── Sub-agent factory ──
    sub_agent_system_prompt = (
        "You are a proactive background assistant completing a scheduled task. "
        "Use tools as needed to fulfil the task fully and accurately. "
        "Return a concise, well-structured answer — do not ask follow-up questions."
        + env_info
        + soft_skill_text_no_delegation        + get_reliability_block("scheduler")    )

    def make_sub_agent(agent_id: str = "scheduler") -> AgentLoop:
        return AgentLoop(
            kernel=kernel,
            llm=worker_llm,
            skill_manager=skill_manager,
            security=SecurityEngine.make_permissive(kernel),
            system_prompt=sub_agent_system_prompt,
            config=AgentConfig(
                max_iterations=config.agent.max_iterations,
                temperature=0.5,
                excluded_skills=frozenset({"scheduler"}),
            ),
            memory_manager=None,
            agent_id=agent_id,
            run_control=run_control,
        )

    # ── Multi-agent infra ──
    agent_registry = AgentRegistry()
    escalation_bus = EscalationBus(kernel)

    # ── Notification router ──
    notification_router = NotificationRouter()
    if config.telegram.configured:
        notification_router.register(
            TelegramChannel(config.telegram.token, config.telegram.chat_id)
        )
    notification_router.register(FileChannel(arc_home / "notifications.log"))

    # ── Scheduler engine ──
    scheduler_engine: SchedulerEngine | None = None
    if config.scheduler.enabled:
        scheduler_engine = SchedulerEngine(
            store=sched_store,
            llm=llm,
            agent_factory=make_sub_agent,
            router=notification_router,
            kernel=kernel,
            agent_registry=agent_registry,
        )

    # ── Inject skill dependencies ──
    worker_skill = skill_manager.get_skill("worker")
    if worker_skill and isinstance(worker_skill, WorkerSkill):
        worker_system_prompt = (
            "You are a focused background worker completing a specific sub-task. "
            "Do not ask clarifying questions — make your best effort with the "
            "information provided. Return a clear, structured result."
            + env_info
            + soft_skill_text_no_delegation            + get_reliability_block("worker")        )
        worker_skill.set_dependencies(
            llm=llm,
            worker_llm=worker_llm,
            skill_manager=skill_manager,
            escalation_bus=escalation_bus,
            notification_router=notification_router,
            agent_registry=agent_registry,
            system_prompt=worker_system_prompt,
        )

    browser_skill = skill_manager.get_skill("browser_control")
    if browser_skill and isinstance(browser_skill, BrowserControlSkill):
        browser_skill.set_dependencies(escalation_bus=escalation_bus)

    # ── Workflow engine ──
    from arc.workflow.skill import WorkflowSkill
    workflow_skill = WorkflowSkill()
    await skill_manager.register(workflow_skill)
    workflow_skill.set_dependencies(agent=agent, kernel=kernel)

    # ── Task Board ──
    from arc.tasks.store import TaskStore
    from arc.tasks.agents import load_agent_defs
    from arc.tasks.processor import TaskProcessor
    from arc.tasks.skill import TaskSkill

    agent_defs = load_agent_defs()
    task_store = TaskStore(db_path=arc_home / "tasks.db")
    await task_store.initialize()

    task_processor = TaskProcessor(
        store=task_store,
        agents=agent_defs,
        skill_manager=skill_manager,
        default_llm=worker_llm,
        notification_router=notification_router,
        kernel=kernel,
        llm_factory=create_llm,
        env_info=env_info,
        soft_skills=soft_skill_text_no_delegation,
        run_control=run_control,
    ) if agent_defs else None

    task_skill = TaskSkill()
    await skill_manager.register(task_skill)
    task_skill.set_dependencies(
        store=task_store,
        agents=agent_defs,
        processor=task_processor,
    )

    # ── Worker activity log ──
    worker_log = WorkerActivityLog(arc_home / "worker_activity.log")
    worker_log.open()

    # ── Wire worker log events ──
    kernel.on(EventType.AGENT_SPAWNED, worker_log.handle)
    kernel.on(EventType.AGENT_THINKING, worker_log.handle)
    kernel.on(EventType.SKILL_TOOL_CALL, worker_log.handle)
    kernel.on(EventType.SKILL_TOOL_RESULT, worker_log.handle)
    kernel.on(EventType.AGENT_TASK_COMPLETE, worker_log.handle)
    kernel.on(EventType.AGENT_ERROR, worker_log.handle)
    kernel.on(EventType.AGENT_PLAN_UPDATE, worker_log.handle)

    return ArcRuntime(
        config=config,
        identity=identity,
        kernel=kernel,
        llm=llm,
        worker_llm=worker_llm,
        agent=agent,
        skill_manager=skill_manager,
        skill_router=skill_router,
        security=security,
        cost_tracker=cost_tracker,
        memory_manager=memory_manager,
        sched_store=sched_store,
        scheduler_engine=scheduler_engine,
        notification_router=notification_router,
        agent_registry=agent_registry,
        escalation_bus=escalation_bus,
        worker_log=worker_log,
        mcp_manager=mcp_manager,
        run_control=run_control,
        turn_controller=turn_controller,
        make_sub_agent=make_sub_agent,
        system_prompt=system_prompt,
        env_info=env_info,
        task_store=task_store,
        task_processor=task_processor,
        agent_defs=agent_defs,
    )
