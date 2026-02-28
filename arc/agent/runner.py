"""
run_agent_on_virtual_platform — shared execution helper.

Both WorkerSkill and SchedulerEngine need the same pattern:
  1. Wrap an AgentLoop in a VirtualPlatform (silent — no terminal output)
  2. Send it a prompt and collect the full response
  3. Handle timeout and errors cleanly

Extracting this here means any future improvement (progress events,
better cancellation, memory injection, etc.) happens in one place.

Usage::

    from arc.agent.runner import run_agent_on_virtual_platform

    agent = AgentLoop(...)
    content, error = await run_agent_on_virtual_platform(
        agent=agent,
        prompt="Summarise AI news today",
        name="job:news",
        timeout_seconds=120.0,
    )
    if error:
        ...  # handle / retry
    else:
        ...  # use content
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arc.agent.loop import AgentLoop

logger = logging.getLogger(__name__)


async def run_agent_on_virtual_platform(
    agent: "AgentLoop",
    prompt: str,
    name: str = "virtual",
    timeout_seconds: float = 120.0,
) -> tuple[str, str | None]:
    """
    Run *agent* on a VirtualPlatform, send *prompt*, and return the result.

    The caller owns agent construction — this function only handles the
    VirtualPlatform lifecycle and timeout logic.

    Returns:
        (content, None)          on success
        ("",     error_message)  on timeout or unhandled exception
    """
    from arc.platforms.virtual.app import VirtualPlatform

    platform = VirtualPlatform(name=name)
    platform_task = asyncio.create_task(platform.run(agent.run), name=f"vp:{name}")
    try:
        content = await asyncio.wait_for(
            platform.send_message(prompt),
            timeout=float(timeout_seconds),
        )
        await platform.stop()
        # Give the platform loop a moment to drain cleanly
        await asyncio.wait_for(platform_task, timeout=5.0)
        return content, None

    except asyncio.TimeoutError:
        logger.warning(f"run_agent_on_virtual_platform '{name}' timed out after {timeout_seconds}s")
        platform_task.cancel()
        await asyncio.gather(platform_task, return_exceptions=True)
        return "", f"Timed out after {timeout_seconds:.0f}s"

    except Exception as exc:
        logger.error(f"run_agent_on_virtual_platform '{name}' failed: {exc}", exc_info=True)
        platform_task.cancel()
        await asyncio.gather(platform_task, return_exceptions=True)
        return "", str(exc)
