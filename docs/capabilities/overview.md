# Capabilities Overview

Capabilities are the tools and systems Arc can use beyond plain chat.

This section is about what Arc can do, when you would use it, and how to extend it.

## Built-In Capabilities

Arc already ships with a strong base set:

- **[Browser Automation](../features/browser.md)**: drive a real Chromium browser through structured page state
- **[Web Dashboard](../features/dashboard.md)**: browser UI for chat, tasks, agents, logs, and status
- **[Memory](../features/memory.md)**: persistent memory across sessions
- **[Code Intelligence](../features/code-intel.md)**: code-aware repo mapping and symbol search
- **[Voice Input](../features/voice.md)**: wake word plus speech-to-text flow
- **[Scheduler](../features/scheduler.md)**: recurring and one-shot jobs
- **[Liquid Web](../features/liquid-web.md)**: product search and comparison flow

## Extension Paths

Arc supports three main ways to add more capability.

### Skills

Skills are Arc-native extensions.

Use them when you want:

- local custom tools
- domain instructions
- reusable behavior that feels built into Arc

Read [Skills](skills.md).

### MCP

MCP lets Arc talk to external tool servers.

Use it when you already have an MCP-compatible server or want to connect to an external tool ecosystem without writing Arc-specific code.

Read [MCP](mcp.md).

### Build Your Own Skill

If you want your own local Arc extension, start with [Build Your Own Skill](build-your-own-skill.md).

## Which One Should You Use?

- Use a **built-in capability** when Arc already does what you need.
- Use a **skill** when you want a local Arc-native extension.
- Use **MCP** when the tool already exists as an MCP server or should stay outside Arc.
