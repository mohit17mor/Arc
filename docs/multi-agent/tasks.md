# Queue Tasks

Tasks are the units of background work that task agents pick up.

They are stored persistently, so they survive restarts.

## Simple Task

```bash
arc task add "Find top AI startups funded this year" --assign researcher
```

## With Priority

```bash
arc task add "Urgent: check incident summary" --assign researcher --priority 1
```

Lower numbers mean higher priority.

## Dependent Task

```bash
arc task add "Research AI frameworks" --assign researcher
arc task add "Compare the top 3" --assign researcher --after t-001
```

`--after` means this task waits for another task to finish.

## Where You Can Create Tasks

### CLI

Good for quick scripting and explicit control.

### Chat

You can ask the main agent to queue tasks for named task agents.

### Dashboard

The Task Board supports creating tasks directly from the UI.

## Task Statuses

Common statuses:

- `queued`
- `in_progress`
- `in_review`
- `awaiting_human`
- `blocked`
- `done`
- `failed`
- `cancelled`

## Replying To A Task

When a task is blocked or waiting for review:

```bash
arc task reply t-12345678 "Looks good" --action approve
arc task reply t-12345678 "Please make it more concise" --action revise
```

## Requirement To Actually Process Tasks

Queued tasks only get picked up when the background processor is running.

That means you need:

```bash
arc gateway
```

If `arc gateway` is not running, tasks can stay in `queued` state.
