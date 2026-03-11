# Getting Started

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) (for local LLMs) or an API key for OpenAI/Groq/OpenRouter

## Installation

```bash
git clone https://github.com/ArcAI-xyz/Arc.git && cd Arc
python -m venv .venv
```

=== "Windows"

    ```bash
    .venv\Scripts\activate
    ```

=== "macOS / Linux"

    ```bash
    source .venv/bin/activate
    ```

```bash
pip install -e ".[dev]"
playwright install chromium
```

## Setup

```bash
arc init
```

The setup wizard walks you through:

1. Your name and agent personality
2. LLM provider selection (Ollama, OpenAI, Groq, OpenRouter, etc.)
3. Model selection
4. Optional: Telegram bot token, Tavily API key, ngrok token

## First Chat

```bash
arc chat
```

## Run the Daemon

For task processing, web dashboard, and Telegram:

```bash
arc gateway
```

Open [http://localhost:18789](http://localhost:18789) for the web dashboard.

## Create Your First Agent

```bash
arc agent create researcher --role "Web research" --model ollama/llama3.2
```

## Queue Your First Task

```bash
arc task add "Find the top 5 AI startups funded in 2026" --assign researcher
```

The agent picks it up automatically (requires `arc gateway` running).
