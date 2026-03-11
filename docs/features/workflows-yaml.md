# YAML Workflows

Define repeatable multi-step tasks in YAML. Unlike ad-hoc tool calls, workflows execute in a **fixed order** with retry, failure handling, human-in-the-loop pauses, and progress events.

Drop `.yaml` files in `~/.arc/workflows/` — discovered automatically.

## Quick Example

```yaml
# ~/.arc/workflows/jira-rca.yaml
name: jira-rca
description: Root cause analysis for a Jira ticket
trigger: "rca|root cause"
steps:
  - do: Fetch the Jira ticket details
    tool: mcp_call
    args: {server: jira, tool: get_issue, arguments: {key: "${ticket}"}}
  - do: Search the codebase for the relevant error
    retry: 2
  - do: Analyze the root cause and write a summary
    on_fail: continue
  - do: Post the RCA summary as a comment on the ticket
```

## Step Formats

```yaml
# Simple — plain English
steps:
  - search the web for NVIDIA news
  - summarize the results

# Extended — with control options
steps:
  - do: Search for relevant data
    retry: 2
    on_fail: continue        # continue|stop (default: stop)
    ask_if_unclear: true     # agent asks user instead of guessing
    wait_for_input: true     # pause and wait for user response

# Explicit tool call — bypass agent
steps:
  - do: Get the Jira ticket
    tool: mcp_call
    args: {server: jira, tool: get_issue, arguments: {key: "PROJ-123"}}

# Shell command
steps:
  - do: Check running pods
    shell: kubectl get pods -n payments
```

## Human-in-the-Loop

Workflows pause and wait for user input in two ways:

**Explicit** — mark a step with `wait_for_input: true`

**Implicit** — if the agent's response is a question, the workflow automatically pauses

In both cases, the workflow waits indefinitely. Your answer is injected as context for the next step. Works across CLI, WebChat, and voice.

## Usage

```
/workflow list              # see available workflows
/workflow jira-rca          # run a workflow
```

Or via chat: *"run the jira-rca workflow for PROJ-1234"*
