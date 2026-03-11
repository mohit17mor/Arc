# Task Queue

Tasks are the units of work you assign to agents. They persist in SQLite, survive restarts, and are processed automatically by the TaskProcessor daemon.

## Creating Tasks

### Via CLI

```bash
# Simple task
arc task add "Find top AI startups funded in 2026" --assign researcher

# With priority (1 = highest)
arc task add "Urgent: check competitor launch" --assign researcher --priority 1

# With dependency (waits for another task)
arc task add "Write blog post from research" --assign writer --after t-a1b2c3d4
```

### Via Chat

Tell the main agent what you need:

> "Queue these for the researcher: 1) find quantum computing startups, 2) compare Rust vs Go, 3) summarize NIST standards"

The agent creates 3 tasks automatically.

### Via Dashboard

Open the **Task Board** tab, click **+ New Task**.

### Via REST API

```bash
curl -X POST http://localhost:18789/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"title": "Research AI trends", "assigned_agent": "researcher"}'
```

## Task Lifecycle

```
queued → in_progress → done
                     → failed

queued → in_progress → in_review → done          (with AI reviewer)
                                 → revision_needed → in_progress → ...

queued → in_progress → awaiting_human → done      (with human reviewer)
                                      → revision_needed → ...

queued → in_progress → blocked → queued → ...     (agent asks question)
```

### Status Reference

| Status | Meaning |
|--------|---------|
| `queued` | Waiting to be picked up |
| `in_progress` | Agent is actively working |
| `in_review` | Reviewer agent is checking work |
| `revision_needed` | Reviewer bounced it back |
| `awaiting_human` | Waiting for human review |
| `blocked` | Agent needs human input to continue |
| `done` | Completed successfully |
| `failed` | Failed after execution error |
| `cancelled` | Cancelled by user |

## Monitoring Tasks

```bash
arc task list                     # all tasks
arc task list --status in_progress  # filter by status
arc task show t-a1b2c3d4          # full detail + comments
```

Or use the **Task Board** tab in the dashboard — Kanban columns show tasks by status.

## Responding to Tasks

When a task is `blocked` (agent asked a question) or `awaiting_human`:

```bash
# Approve and advance
arc task reply t-a1b2c3d4 "Looks good" --action approve

# Send back with feedback
arc task reply t-a1b2c3d4 "Make the tone more casual" --action revise
```

Via Telegram: reply to the notification message.
Via Dashboard: click the task, use the approve/revise buttons.

## Task Dependencies

Create chains where one task waits for another:

```bash
arc task add "Research AI frameworks" --assign researcher    # → t-001
arc task add "Compare the top 3" --assign researcher --after t-001  # waits for t-001
arc task add "Write recommendation" --assign writer --after t-002   # waits for t-002
```

The second task only starts after the first completes. The completed task's result is injected as context.

## Priority

Tasks are processed in priority order (1 = highest). Same-priority tasks are processed in creation order.

```bash
arc task add "Normal task" --assign researcher                  # priority 1 (default)
arc task add "Low priority" --assign researcher --priority 5
```

## Cancellation

```bash
arc task cancel t-a1b2c3d4
```

Tasks in `done`, `failed`, or `cancelled` state cannot be cancelled.
