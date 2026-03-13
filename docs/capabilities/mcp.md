# MCP

MCP stands for Model Context Protocol.

In Arc, MCP is the easiest way to connect external tool servers without writing Arc-specific integration code.

## What MCP Is For

Use MCP when you want Arc to access tools from:

- GitHub or other developer services
- external file or database servers
- internal automation servers
- any service that already exposes an MCP server

## How Arc Uses MCP

You configure servers in:

- `~/.arc/mcp.json`

Arc discovers those servers and exposes them through a gateway layer.

From the model's point of view, the main entry points are:

- `mcp_list_tools`
- `mcp_call`

That keeps the model surface small even when you connect many MCP servers.

## Example Config

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "ghp_xxx" }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
    }
  }
}
```

Remote SSE servers are also supported.

## How To Check MCP In Arc

Use:

```text
/mcp
```

That shows configured server status.

Use:

```text
/skills
```

to see the MCP gateway alongside the rest of Arc's capabilities.

## When To Prefer MCP Over A Skill

Prefer MCP when:

- the tool already exists as an MCP server
- multiple clients may share the same tool server
- you want a cleaner boundary between Arc and the external system
- the integration should stay provider-agnostic

Prefer a local Arc skill when the behavior is tiny, local, or very Arc-specific.
