# Multi-Step Workflows

Chain multiple agents together with optional review steps. The output of one agent feeds into the next.

## How It Works

Define steps when creating a task. Each step specifies an agent and optionally a reviewer:

```
Step 1: researcher → gathers information
Step 2: writer → drafts content (reviewed by reviewer)
         ↕ reviewer bounces back if quality is low
Step 3: done → results delivered
```

## Creating Workflows

### Via Chat

> "Create a blog post about AI trends. Have the researcher gather info, the writer draft it, and the reviewer check quality."

### Via the `queue_task` Tool

The main agent creates a workflow by passing `steps`:

```json
{
  "title": "AI blog post",
  "instruction": "Create a comprehensive blog post about AI trends in 2026",
  "steps": [
    {"agent": "researcher"},
    {"agent": "writer", "review_by": "reviewer"}
  ]
}
```

## Review Loops

When a step has `review_by` set, the reviewer agent evaluates the work after that step completes.

The reviewer either:

- **Approves** → workflow advances to the next step
- **Requests revision** → work bounces back to the agent who did it, with the reviewer's feedback

The agent sees the feedback in the task comments and revises accordingly.

### Bounce Limits

Each task has a `max_bounces` setting (default: 3). After that many review cycles, the task completes with the latest output — preventing infinite loops.

### Human Review

Use `"review_by": "human"` for steps that need your sign-off:

```json
{
  "steps": [
    {"agent": "writer", "review_by": "human"},
    {"agent": "publisher"}
  ]
}
```

When the writer finishes, the task pauses at `awaiting_human`. You get a notification (Telegram, dashboard, CLI) and can:

- **Approve** → advances to publisher
- **Revise** → bounces back to writer with your feedback

## How Review Routing Works

The key design: **the agent who marked a step for review is always the one who gets the revision back.** No ambiguity about who fixes what.

```
researcher (no review) → writer (reviewed by reviewer)
                           ↕ bounces stay between writer ↔ reviewer
                         → publisher (no review) → done
```

The reviewer's feedback is saved as a comment on the task. When the writer picks it up again, it sees the full comment history including the feedback.

## Example: Content Pipeline

```bash
# Via chat:
# "Queue a content pipeline: researcher finds AI news,
#  writer creates IG captions, I review before posting"
```

This creates:

| Step | Agent | Reviewer |
|------|-------|----------|
| 1 | researcher | — |
| 2 | writer | human |

1. Researcher gathers information → output saved as comment
2. Writer reads researcher's output, creates captions → submitted for human review
3. You review → approve or send back with feedback
4. Done → results delivered

## Comments as Communication

All inter-agent communication happens through **task comments**. Each agent reads the full comment history when it picks up a task, so context flows naturally between agents. No information is lost between steps.
