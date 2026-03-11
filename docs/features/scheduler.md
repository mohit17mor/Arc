# Scheduler

Schedule recurring or one-time tasks for Arc to run proactively.

## Creating Jobs

Via chat:
```
"Remind me every morning at 9am to check my email"
"Fetch AI news every weekday at 8am"
"In 2 hours, search for the latest NVIDIA stock price"
```

The main agent creates scheduled jobs using the `schedule_job` tool.

## Trigger Types

| Type | Description | Example |
|------|-------------|---------|
| `cron` | Standard 5-field cron expression | `0 9 * * 1-5` (weekdays at 9am) |
| `interval` | Every N seconds | `1800` (every 30 minutes) |
| `oneshot` | Fire once at a specific time | `fire_after_seconds=3600` (in 1 hour) |

## With or Without Tools

Jobs can run in two modes:

- **Without tools** (default) — plain LLM text generation. Good for reminders, tips, summaries.
- **With tools** — full sub-agent with tool access. Good for tasks that need live data (web search, file reading).

## Managing Jobs

```bash
# CLI
arc task list                    # see scheduled jobs
```

```
# In chat
/jobs              # list all jobs
/jobs cancel <name>  # cancel a job
```

Or use the **Scheduler** tab in the dashboard.

## Results

When a job fires, results are delivered through the NotificationRouter:

1. Telegram (if configured)
2. CLI (if `arc chat` is open)
3. File log (`~/.arc/notifications.log`, always)

## Configuration

```toml
[scheduler]
enabled = true
db_path = "~/.arc/scheduler.db"
poll_interval = 30  # seconds between checks
```
