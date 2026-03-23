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
from arc.core.run_control import RunCancelledError, RunControlManager, RunHandle
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

# Phrases that strongly indicate the agent is asking the user for input,
# not ending with a rhetorical question.
_QUESTION_INDICATORS = (
    "which ",
    "what ",
    "when ",
    "where ",
    "who ",
    "why ",
    "how ",
    "could you ",
    "would you ",
    "can you ",
    "please provide ",
    "before i proceed",
    "do you want ",
    "do you prefer ",
    "are you ",
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

    def __init__(self, agent: "AgentLoop", kernel: "Kernel", run_control: RunControlManager | None = None) -> None:
        self._agent = agent
        self._kernel = kernel
        self._run_control = run_control
        self._input_future: asyncio.Future[str] | None = None
        self._active_run_handle: RunHandle | None = None
        self._current_run_id: str | None = None
        self._last_run_id: str | None = None

    def provide_input(self, user_input: str) -> bool:
        if self._input_future is not None and not self._input_future.done():
            self._input_future.set_result(user_input)
            return True
        return False

    @property
    def is_waiting_for_input(self) -> bool:
        return self._input_future is not None and not self._input_future.done()

    @property
    def current_run_id(self) -> str | None:
        return self._current_run_id

    @property
    def last_run_id(self) -> str | None:
        return self._last_run_id

    async def _run_checkpoint(self) -> None:
        if self._active_run_handle is None:
            return
        await self._active_run_handle.checkpoint()

    async def _wait_for_user_input(
        self, workflow_name: str, step_num: int, total_steps: int, question: str
    ) -> str:
        loop = asyncio.get_running_loop()
        self._input_future = loop.create_future()

        await self._emit(EventType.WORKFLOW_WAITING_INPUT, {
            "workflow": workflow_name,
            "step": step_num,
            "total_steps": total_steps,
            "question": question,
        })

        logger.info(f"Workflow '{workflow_name}' waiting for user input at step {step_num}")

        try:
            while True:
                await self._run_checkpoint()
                try:
                    return await asyncio.wait_for(asyncio.shield(self._input_future), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
        finally:
            self._input_future = None

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        event = Event(type=event_type, source="workflow", data=data)
        await self._kernel.emit(event)

    async def run(self, workflow: Workflow, user_message: str = "") -> AsyncIterator[str]:
        run_handle: RunHandle | None = None
        if self._run_control is not None:
            run_handle = self._run_control.start_run(
                kind="workflow",
                source="workflow",
                metadata={"workflow": workflow.name},
            )
            self._active_run_handle = run_handle
            self._current_run_id = run_handle.run_id
            self._last_run_id = run_handle.run_id

        total = len(workflow.steps)
        step_results: list[StepResult] = []
        context_parts: list[str] = []

        await self._run_checkpoint()
        await self._emit(EventType.WORKFLOW_START, {
            "workflow": workflow.name,
            "total_steps": total,
            "description": workflow.description,
        })

        if user_message:
            context_parts.append(f"User request: {user_message}")

        try:
            for i, step in enumerate(workflow.steps):
                step_num = i + 1

                await self._run_checkpoint()
                await self._emit(EventType.WORKFLOW_STEP_START, {
                    "workflow": workflow.name,
                    "step": step_num,
                    "total_steps": total,
                    "instruction": step.instruction,
                })

                result: StepResult | None = None
                async for agent_chunk, step_result in self._execute_step(
                    step=step,
                    step_num=step_num,
                    total_steps=total,
                    context_parts=context_parts,
                    user_message=user_message,
                ):
                    if agent_chunk is not None:
                        await self._run_checkpoint()
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

                    needs_input = step.wait_for_input
                    if (
                        not needs_input
                        and step.ask_if_unclear
                        and i < total - 1
                        and _response_is_question(result.output)
                    ):
                        needs_input = True

                    if needs_input and i < total - 1:
                        question = result.output.strip()
                        if not question or not question.endswith("?"):
                            question = (
                                f"Workflow is waiting for your input.\n"
                                f"Step: {step.instruction}\n\n"
                                f"{result.output.strip()}" if result.output.strip()
                                else f"Workflow is waiting for your input.\n"
                                f"Step: {step.instruction}"
                            )

                        yield f"\n\n⏳ **Waiting for your input:**\n{question}\n"

                        user_answer = await self._wait_for_user_input(
                            workflow.name, step_num, total, question
                        )
                        context_parts.append(
                            f"User response to step {step_num}: {user_answer}"
                        )

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
                        completed = sum(1 for r in step_results if r.status == StepStatus.COMPLETED)
                        remaining = [s.instruction for s in workflow.steps[i + 1 :]]

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

            completed = sum(1 for r in step_results if r.status == StepStatus.COMPLETED)
            await self._emit(EventType.WORKFLOW_COMPLETE, {
                "workflow": workflow.name,
                "total_steps": total,
                "completed_steps": completed,
            })
            if run_handle is not None:
                await run_handle.finish_completed()

        except RunCancelledError as e:
            await self._emit(EventType.WORKFLOW_FAILED, {
                "workflow": workflow.name,
                "error": e.action.value,
            })
            return
        finally:
            self._active_run_handle = None
            self._current_run_id = None

    async def _execute_step(
        self,
        step: WorkflowStep,
        step_num: int,
        total_steps: int,
        context_parts: list[str],
        user_message: str,
    ) -> AsyncIterator[tuple[str | None, StepResult | None]]:
        max_attempts = step.retry + 1
        step_timeout = 600

        for attempt in range(max_attempts):
            step.attempts = attempt + 1

            try:
                prompt = self._build_step_prompt(
                    step, step_num, total_steps, context_parts, user_message, attempt
                )

                output = ""

                async def _stream():
                    nonlocal output
                    async for chunk in self._agent.run(prompt):
                        await self._run_checkpoint()
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

            except RunCancelledError:
                raise
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
                logger.warning(f"Step {step_num} attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    continue
                yield (None, StepResult(
                    step_index=step.index,
                    status=StepStatus.FAILED,
                    error=str(e),
                    attempts=attempt + 1,
                ))
                return

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

        if step.wait_for_input:
            prompt += (
                "\nIMPORTANT: This step requires user input. "
                "You MUST end your response with a CLEAR QUESTION "
                "asking the user for the specific information you need. "
                "Do NOT proceed without their answer. "
                "Phrase it as a direct question ending with a question mark.\n"
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

def _response_is_question(text: str) -> bool:
    """
    Detect if the agent's step output is asking the user a question.

    Looks for two signals:
    1. The response ends with a question mark (after stripping whitespace)
    2. The response contains question-indicator phrases

    Both must be true to avoid false positives on rhetorical questions
    embedded in longer summaries.  Only checks the last ~500 chars since
    the question is always at the end.
    """
    if not text:
        return False

    stripped = text.rstrip()
    if not stripped.endswith("?"):
        return False

    # Check the tail for question-indicator phrases
    tail = stripped[-500:].lower()
    return any(indicator in tail for indicator in _QUESTION_INDICATORS)
