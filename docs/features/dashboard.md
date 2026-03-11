# Web Dashboard

A single-page app served at `http://localhost:18789` by `arc gateway`.

## Tabs

### 📊 Dashboard
Overview of the system — task counts by status, agent list, uptime, total messages.

### 💬 Chat
Full WebChat with streaming responses, tool call events, worker notifications, and slash commands. Same functionality as `arc chat` but in the browser.

### 📋 Task Board
Kanban-style columns: Queued, In Progress, In Review, Needs Action, Done. Click any task to see full detail with comment history. Create tasks, approve/revise, cancel — all from the UI.

### 🤖 Agents
Create and manage named agents. The create form includes LLM provider/model picker and a system prompt editor. Delete agents with one click.

### ⏰ Scheduler
View all scheduled jobs with trigger info and next run time. Cancel jobs directly.

### 🧩 Skills & MCP
All loaded skills with their tools listed. MCP server status (connected/lazy). Useful for understanding what tools are available to agents.

### 📜 Logs
Real-time system event stream. Filter by source (main agent, task agents, workers, scheduler) and by event type (tool calls, errors, completions). Live mode auto-scrolls as events arrive.

## Architecture

The dashboard is a single HTML file using Alpine.js (~15KB) for reactivity. No npm, no build step, no node_modules.

- One WebSocket connection stays alive across all tabs
- Notifications appear on any tab (not just Chat)
- Data loaded via REST API endpoints
- Auto-refreshes every 10 seconds on Dashboard and Task Board

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/overview` | Dashboard stats |
| `GET` | `/api/tasks` | List tasks |
| `POST` | `/api/tasks` | Create task |
| `GET` | `/api/tasks/:id` | Task detail + comments |
| `POST` | `/api/tasks/:id/cancel` | Cancel task |
| `POST` | `/api/tasks/:id/reply` | Approve/revise/answer |
| `GET` | `/api/agents` | List agents |
| `POST` | `/api/agents` | Create agent |
| `DELETE` | `/api/agents/:name` | Delete agent |
| `GET` | `/api/scheduler` | List scheduled jobs |
| `GET` | `/api/skills` | List skills + tools |
| `GET` | `/api/mcp` | List MCP servers |
| `GET` | `/api/logs` | Recent system events |
