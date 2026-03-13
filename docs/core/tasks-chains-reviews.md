# Tasks, Chains, and Reviews

Arc supports both simple queued tasks and multi-step chains.

## Single Task

A single task is one unit of work assigned to one task agent.

Example:

```bash
arc task add "Research AI coding tools" --assign researcher
```

This is the simplest background job model.

## Chain

A chain is one task that moves through multiple task agents in sequence.

Example mental model:

- step 1: `researcher`
- step 2: `writer`
- step 3: `reviewer` or human review

Arc stores the chain as one task with multiple steps.

## Dependency Between Separate Tasks

Arc also supports separate tasks that wait for earlier tasks.

```bash
arc task add "Research AI frameworks" --assign researcher
arc task add "Compare the top 3" --assign researcher --after t-001
arc task add "Write recommendation" --assign writer --after t-002
```

`--after` means the later task depends on the earlier task finishing first.

That is different from a multi-step chain:

- `--after` links **separate tasks**
- a chain keeps **multiple steps inside one task**

## How Handoff Works Inside A Chain

When one step finishes, Arc passes the earlier task activity forward as context.

The next agent sees:

- the main task title and instruction
- a note that earlier steps are complete
- previous task activity and comments
- any review feedback that needs to be addressed

The current step prompt explicitly tells the next agent to continue from previous work and not repeat earlier work unless necessary.

## Reviews

A step can optionally be reviewed by:

- another task agent
- a human

Possible outcomes:

- approve: move to the next step
- needs revision: send work back to the agent who just did that step

## Bounce Limits

Each task has `max_bounces`.

In a multi-step chain, the bounce counter is effectively used per review stage, not across the whole chain, because Arc resets it when moving to the next step.

## Important Current Limitation

A multi-step chain currently shares one overall task instruction.

The step definition itself stores:

- which agent should handle the step
- which reviewer, if any, should review it

It does **not** currently store a separate instruction per step.

So if you want strong step-by-step guidance today, put that guidance into the main task description in a clear structured way.
