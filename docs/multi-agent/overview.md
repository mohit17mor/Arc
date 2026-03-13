# Multi-Agent Overview

Arc's multi-agent system is for background work handled by named task agents.

You do not need this to use Arc, but it becomes valuable when one chat agent is no longer enough.

## What Multi-Agent Means In Arc

It means all of the following together:

- named task agents
- a persistent task queue
- optional chains with multiple steps
- optional AI or human review
- the gateway processing work in the background

## When To Use It

Use multi-agent when you want:

- specialized workers like `researcher`, `writer`, or `reviewer`
- queued jobs that survive restarts
- work to continue while you are away
- different models for different roles
- structured chains and review loops

## The Core Pieces

### Task agents

These are named workers saved in `~/.arc/agents/*.toml`.

### Tasks

These are queued pieces of work stored persistently.

### Chains

One task can pass through multiple agents in order.

### Reviews

A step can be reviewed by another agent or by a human.

### Gateway

`arc gateway` runs the dashboard and the background processor that actually picks up queued tasks.

## The Basic Flow

1. create one or more task agents
2. queue a task or chain
3. run `arc gateway`
4. Arc processes the work and stores comments, results, and review state

## Where To Go Next

- [Create Agents](create-agents.md)
- [Queue Tasks](tasks.md)
- [Multi-Step Chains and Reviews](chains-and-reviews.md)
