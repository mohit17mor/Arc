# Multi-Step Chains and Reviews

This is where Arc moves from simple task queueing to structured multi-agent work.

## What A Chain Is

A chain is one task with multiple ordered steps.

Example:

- step 1: `researcher`
- step 2: `writer`
- step 3: `reviewer` or human review

## Current Model In Arc

A step currently stores:

- which agent runs the step
- which reviewer checks it, if any

The chain as a whole stores the main task instruction.

So today, the recommended pattern is:

- write one clear overall task description
- let each step's agent role determine how it contributes
- rely on previous task activity to carry the output forward

## Dashboard Support

The dashboard task form supports building multi-step chains.

You can:

- start with the main task description
- add steps with the `Add Step` control
- choose an agent per step
- choose a reviewer per step

That is the easiest UI for users who do not want to write chains only in chat or CLI.

## Example Chain

A research-to-writing flow might look like:

- step 1: `researcher`
- step 2: `writer` reviewed by `human`

The reviewer can also be another agent.

## How Handoff Works

When Arc advances to the next step, the next agent sees:

- the original task title and instruction
- a message that previous steps are complete
- previous activity on the task
- review feedback if the step is a revision pass

This is designed so the next agent continues the work instead of restarting it.

## Review Flow

If a step has `review_by` configured, the reviewer checks the output.

Possible outcomes:

- approved: move forward
- needs revision: bounce back to the same agent that produced the work

That bounce stays within the current review stage until approved or the bounce limit is reached.

## Bounce Limit Behavior

`max_bounces` applies to a review stage, not effectively to the whole chain.

Arc resets the counter when the task advances to the next step.

## Separate Tasks vs One Chain

Use separate tasks linked by `--after` when:

- each unit should stand on its own
- you want separate task IDs and history
- the work is better modeled as multiple queue items

Use one multi-step chain when:

- you want one task to flow through several agents
- you want a single shared task history
- you want reviews inside the same task
