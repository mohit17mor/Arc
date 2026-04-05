"""
Reliability prompt blocks — injected into every agent's system prompt.

These are the instructions that make agents plan, persist through failure,
verify their work, and communicate clearly when stuck.  They are terse by
design: every token here competes with user context, so we keep it tight.

Usage::

    from arc.agent.prompts import get_reliability_block

    system_prompt = base_prompt + get_reliability_block("main")
"""

from __future__ import annotations


# ── Planning instructions (shared across all agent types) ────────────────

_PLANNING_INSTRUCTIONS = """\

## Planning & Execution

For non-trivial tasks (more than 2 tool calls), create a plan FIRST:
1. Call update_plan with 3-7 short steps BEFORE doing anything else.
2. Set exactly ONE step to in_progress at a time.
3. After finishing a step, call update_plan to mark it completed and \
set the next step to in_progress.
4. If your approach changes, update the plan before continuing.
5. When all steps are done, ensure every step is marked completed."""


# ── Quality / error-handling instructions ────────────────────────────────

_QUALITY_INSTRUCTIONS = """\

## Quality & Error Handling

- If a tool call fails, READ the error message. Adjust your arguments \
or try a different tool. Do NOT repeat the exact same call.
- Before declaring done, verify: did you actually address the original request?
- Do not fabricate information. If you cannot find something, say so."""


# ── Agent-type-specific additions ────────────────────────────────────────

_MAIN_ADDITIONS = """\

## When You Are Stuck

If you genuinely cannot proceed because you need information only the \
user can provide, explain clearly:
1. What you tried
2. What is blocking you
3. What specific input you need from the user"""


_VOICE_INSTRUCTIONS = """\

## Spoken Summary

End every response with a [spoken] tag for TTS.

Make it sound like a friendly assistant speaking out loud: warm, natural, \
and conversational, not like a written summary or status report.

Rules:
- 1–2 short sentences, under 25 words
- use simple spoken English and natural contractions
- give the most useful takeaway first
- no code, URLs, markdown, or file paths
- avoid robotic words like "implemented", "completed", or "addressed"
- only mention "check chat" if the full answer is too detailed to say aloud

Example:
[spoken]I found the issue and fixed it. Voice mode should work properly now.[/spoken]"""


_BACKGROUND_ADDITIONS = """\

## Background Agent Rules

You are running unattended. Do not ask clarifying questions. \
If you cannot complete the task, explain what went wrong and \
what information was missing. Your output must be self-contained."""


_WORKER_ADDITIONS = """\

## Worker Efficiency

- Use the MINIMUM number of tool calls needed.
- Do NOT loop: search → read → search → read. One round is enough.
- If a tool call fails, try one alternative, then report with what you have."""


# ── Public API ───────────────────────────────────────────────────────────

def get_reliability_block(agent_type: str = "main", *, voice_mode: bool = False) -> str:
    """
    Return the reliability prompt block for the given agent type.

    Args:
        agent_type: One of "main", "worker", "scheduler", "task".
        voice_mode: If True, append spoken-summary instructions so the
            agent produces a ``[spoken]...[/spoken]`` tag for TTS.

    Returns:
        A string to append to the agent's system prompt.
    """
    parts = [_PLANNING_INSTRUCTIONS, _QUALITY_INSTRUCTIONS]

    if agent_type == "main":
        parts.append(_MAIN_ADDITIONS)
    elif agent_type == "worker":
        parts.append(_BACKGROUND_ADDITIONS)
        parts.append(_WORKER_ADDITIONS)
    elif agent_type == "scheduler":
        parts.append(_BACKGROUND_ADDITIONS)
    elif agent_type == "task":
        parts.append(_BACKGROUND_ADDITIONS)
    else:
        # Unknown type — give the safe background rules
        parts.append(_BACKGROUND_ADDITIONS)

    if voice_mode:
        parts.append(_VOICE_INSTRUCTIONS)

    return "\n".join(parts)
