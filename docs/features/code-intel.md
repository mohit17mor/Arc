# Code Intelligence

AST-aware code navigation using tree-sitter and grep-ast. Agents understand codebases structurally, not just as text files.

## Tools

| Tool | Description |
|------|-------------|
| `repo_map(path)` | Condensed project overview — all files with classes, functions, methods |
| `find_symbol(name, path)` | Locate any function/class definition and get the full source body |
| `search_code(pattern, path)` | Grep with AST context — shows matches within their enclosing function/class scope |

## Example: repo_map

```
Repository map: 9 files, 120 symbols

bus.py
  class EventBus:
    def __init__(self) -> None:
    def on(self, event_type: str, handler: EventHandler) -> None:
    async def emit(self, event: Event) -> Event:
    def emit_nowait(self, event: Event) -> None:

config.py
  class AgentConfig(BaseModel):
  class SecurityConfig(BaseModel):
  class LLMConfig(BaseModel):
```

One call → the agent has a mental map of the entire project. ~500-2000 tokens for a typical project.

## Example: find_symbol

```
Found 1 definition(s) of 'AgentLoop':

── loop.py (line 61) ──
  61 | class AgentLoop:
  62 |     """The agent's execution loop."""
  63 |     ...
  64 |     async def run(self, user_input: str) -> AsyncIterator[str]:
```

Returns the full function/class body with line numbers.

## Example: search_code

```
Found 3 match(es) across 2 file(s):

── core/bus.py (1 matches) ──
  28│class EventBus:
  ...
  98█    def emit_nowait(self, event: Event) -> None:
```

Shows the match highlighted (`█`) with its enclosing class/function scope for context.

## Supported Languages

Python, JavaScript, TypeScript, TSX, Rust, Go, Java, C, C++, C#, Ruby, PHP, Swift, Kotlin, YAML, TOML, JSON, Bash.

## Usage for Coding Tasks

When assigning coding tasks to agents, include the repo path in the instruction:

```bash
arc task add "Fix the auth bug in /home/user/projects/my-api" --assign developer
```

Or set the repo path in the agent's system prompt:

```toml
system_prompt = """
You are a backend developer working on the my-api project.
Project path: /home/user/projects/my-api

Always start with repo_map to understand the codebase before making changes.
"""
```

## Dependencies

```bash
pip install grep-ast  # included in Arc's default dependencies
```

Installed automatically with Arc. Uses tree-sitter-language-pack for multi-language grammar support.
