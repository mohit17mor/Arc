"""
Workflow Engine — deterministic step-by-step execution.

Integrates with Arc's core infrastructure:
  - **Kernel event bus** for lifecycle events (progress, completion, failure)
  - **AgentLoop** for executing each step's natural-language instruction
  - **10-minute hard timeout** per step to catch genuinely stuck calls

The engine emits events so any subscriber (CLI, Gateway, WebChat,
notifications, worker log) can react without the engine knowing about
any specific platform.

Events emitted:
  ``workflow:start``          — workflow begun
  ``workflow:step_start``     — step N starting
  ``workflow:step_complete``  — step N succeeded
  ``workflow:step_failed``    — step N failed (after all retries)
  ``workflow:complete``       — all steps done
  ``workflow:failed``         — workflow stopped due to failure
  ``workflow:paused``         — waiting for user help after failure
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from arc.agent.loop import AgentLoop
    from arc.core.kernel import Kernel

from arc.core.events import Event, EventType
from arc.workflow.types import (
    OnFail,
    StepResult,
    StepStatus,
    Workflow,
    WorkflowStep,
)

logger = logging.getLogger(__name__)


async def _stream_with_timeout(aiter, timeout: float):
    """Wrap an async iterator with a total timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    ait = aiter.__aiter__()
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise asyncio.TimeoutError()
        try:
            chunk = await asyncio.wait_for(ait.__anext__(), timeout=remaining)
            yield chunk
        except StopAsyncIteration:
            break


# Prompt prefix — keeps the agent focused on one step at a time.
_STEP_PREFIX = (
    "You are executing step {step_num} of {total_steps} in a workflow.\n"
    "INSTRUCTION: {instruction}\n\n"
    "RULES:\n"
    "- Do ONLY what this step asks. Nothing more.\n"
    "- If you don't have enough information to do this correctly, "
    "ask the user a clarifying question instead of guessing.\n"
    "- When the step is complete, summarize what you did and what you found.\n"
    "- Do NOT proceed to the next step — the workflow engine handles that.\n"
)


class WorkflowEngine:
    """
    Executes workflows step by step using an AgentLoop.

    Emits kernel events at every lifecycle point so CLI, Gateway,
    WebChat, notifications, and the worker log can all react
    independently.

    Usage::

        engine = WorkflowEngine(agent=agent, kernel=kernel)
        async for chunk in engine.run(workflow, user_message="check PAYMENTS-1234"):
            print(chunk, end="", flush=True)
    """

    def __init__(self, agent: "AgentLoop", kernel: "Kernel") -> None:
        self._agent = agent
        self._kernel = kernel

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit a workflow event through the kernel bus."""
        event = Event(type=event_type, source="workflow", data=data)
        await self._kernel.emit(event)

    # ━━━ Main execution ━━━

    async def run(
        self,
        workflow: Workflow,
        user_message: str = "",
    ) -> AsyncIterator[str]:
        """
        Execute a workflow and yield agent text output.

        The agent's streamed response for each step is yielded so
        platforms can display it in real-time.  All lifecycle progress
        (step started, completed, failed) is communicated via kernel
        events — not via yielded text.
        """
        total = len(workflow.steps)
        step_results: list[StepResult] = []
        context_parts: list[str] = []

        await self._emit(EventType.WORKFLOW_START, {
            "workflow": workflow.name,
            "total_steps": total,
            "description": workflow.description,
        })

        if user_message:
            context_parts.append(f"User request: {user_message}")

        for i, step in enumerate(workflow.steps):
            step_num = i + 1

            await self._emit(EventType.WORKFLOW_STEP_START, {
                "workflow": workflow.name,
                "step": step_num,
                "total_steps": total,
                "instruction": step.instruction,
            })

            # Execute the step — yields agent text, produces a StepResult
            result: StepResult | None = None
            async for agent_chunk, step_result in self._execute_step(
                step=step,
                step_num=step_num,
                total_steps=total,
                context_parts=context_parts,
                user_message=user_message,
            ):
                if agent_chunk is not None:
                    yield agent_chunk
                if step_result is not None:
                    result = step_result

            if result is None:
                result = StepResult(
                    step_index=step.index,
                    status=StepStatus.FAILED,
                    error="No result produced",
                )
            step_results.append(result)

            # ── Handle result ──

            if result.status == StepStatus.COMPLETED:
                context_parts.append(
                    f"Step {step_num} ({step.instruction}): {result.output}"
                )
                await self._emit(EventType.WORKFLOW_STEP_COMPLETE, {
                    "workflow": workflow.name,
                    "step": step_num,
                    "total_steps": total,
                    "instruction": step.instruction,
                    "attempts": result.attempts,
                })

            elif result.status == StepStatus.FAILED:
                await self._emit(EventType.WORKFLOW_STEP_FAILED, {
                    "workflow": workflow.name,
                    "step": step_num,
                    "total_steps": total,
                    "instruction": step.instruction,
                    "error": result.error,
                    "attempts": result.attempts,
                })

                if step.on_fail == OnFail.CONTINUE:
                    context_parts.append(
                        f"Step {step_num} ({step.instruction}): FAILED — {result.error}"
                    )
                else:
                    # Pause and ask user for help
                    completed = sum(
                        1 for r in step_results
                        if r.status == StepStatus.COMPLETED
                    )
                    remaining = [s.instruction for s in workflow.steps[i + 1:]]

                    await self._emit(EventType.WORKFLOW_PAUSED, {
                        "workflow": workflow.name,
                        "step": step_num,
                        "total_steps": total,
                        "instruction": step.instruction,
                        "error": result.error,
                        "completed_count": completed,
                        "remaining": remaining,
                    })
                    return

        # All steps done
        completed = sum(
            1 for r in step_results if r.status == StepStatus.COMPLETED
        )
        await self._emit(EventType.WORKFLOW_COMPLETE, {
            "workflow": workflow.name,
            "total_steps": total,
            "completed_steps": completed,
        })

    # ━━━ Step execution ━━━

    async def _execute_step(
        self,
        step: WorkflowStep,
        step_num: int,
        total_steps: int,
        context_parts: list[str],
        user_message: str,
    ) -> AsyncIterator[tuple[str | None, StepResult | None]]:
        """Execute a single step with retry, yielding agent output."""
        max_attempts = step.retry + 1
        step_timeout = 600  # 10-minute hard cap per attempt

        for attempt in range(max_attempts):
            step.attempts = attempt + 1

            try:
                prompt = self._build_step_prompt(
                    step, step_num, total_steps,
                    context_parts, user_message, attempt,
                )

                output = ""

                async def _stream():
                    nonlocal output
                    async for chunk in self._agent.run(prompt):
                        output += chunk
                        yield chunk

                async for chunk in _stream_with_timeout(_stream(), step_timeout):
                    yield (chunk, None)

                yield (None, StepResult(
                    step_index=step.index,
                    status=StepStatus.COMPLETED,
                    output=output[:2000],
                    attempts=attempt + 1,
                ))
                return

            except asyncio.TimeoutError:
                error = f"Step timed out after {step_timeout // 60} minutes"
                logger.warning(f"Step {step_num}: {error}")
                if attempt < max_attempts - 1:
                    continue
                yield (None, StepResult(
                    step_index=step.index,
                    status=StepStatus.FAILED,
                    error=error,
                    attempts=attempt + 1,
                ))
                return

            except Exception as e:
                logger.warning(
                    f"Step {step_num} attempt {attempt + 1} failed: {e}"
                )
                if attempt < max_attempts - 1:
                    continue
                yield (None, StepResult(
                    step_index=step.index,
                    status=StepStatus.FAILED,
                    error=str(e),
                    attempts=attempt + 1,
                ))
                return

    # ━━━ Prompt building ━━━

    def _build_step_prompt(
        self,
        step: WorkflowStep,
        step_num: int,
        total_steps: int,
        context_parts: list[str],
        user_message: str,
        attempt: int,
    ) -> str:
        prompt = _STEP_PREFIX.format(
            step_num=step_num,
            total_steps=total_steps,
            instruction=step.instruction,
        )

        if context_parts:
            prompt += "\nCONTEXT FROM PREVIOUS STEPS:\n"
            prompt += "\n".join(f"  - {c}" for c in context_parts)
            prompt += "\n"

        if step.ask_if_unclear:
            prompt += (
                "\nIMPORTANT: If you are unsure about any detail needed "
                "for this step, ASK the user rather than guessing.\n"
            )

        if attempt > 0:
            prompt += (
                f"\nNOTE: This is retry attempt {attempt + 1}. "
                "The previous attempt failed. Try a different approach.\n"
            )

        if step.shell:
            prompt += f"\nRun this exact shell command: {step.shell}\n"
        elif step.tool and step.args:
            prompt += (
                f"\nUse this exact tool call:\n"
                f"  Tool: {step.tool}\n"
                f"  Args: {step.args}\n"
            )

        return prompt
