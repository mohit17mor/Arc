"""
Skill interface and @tool decorator.

A Skill is a collection of related tools with lifecycle management.
The @tool decorator is a shortcut for creating simple tools.
"""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, get_type_hints

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec


class Skill(ABC):
    """
    Abstract base class for skills.

    A skill is a package of related tools with:
    - Lifecycle (initialize, activate, deactivate, shutdown)
    - State (can persist between calls)
    - Tools (functions the LLM can call)

    Minimal implementation requires only manifest() and execute_tool().
    """

    @abstractmethod
    def manifest(self) -> SkillManifest:
        """Return skill metadata and tool specifications."""
        ...

    async def initialize(self, kernel: Any, config: dict[str, Any]) -> None:
        """Called when skill is registered. Store references for later."""
        self._kernel = kernel
        self._config = config

    async def activate(self) -> None:
        """Called on first use. Do heavy setup here (lazy initialization)."""
        pass

    @abstractmethod
    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool by name with given arguments."""
        ...

    async def get_state(self) -> dict[str, Any]:
        """Return serializable state snapshot for checkpointing."""
        return {}

    async def restore_state(self, state: dict[str, Any]) -> None:
        """Restore state from a previous snapshot."""
        pass

    async def deactivate(self) -> None:
        """Called when agent pauses. Release temporary resources."""
        pass

    async def shutdown(self) -> None:
        """Called when agent stops. Release ALL resources."""
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# @tool decorator for creating simple tools
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ToolDef:
    """A tool definition created by the @tool decorator."""

    name: str
    description: str
    func: Callable
    parameters: dict[str, Any]
    capabilities: frozenset[Capability] = frozenset()


def tool(
    name: str | None = None,
    description: str | None = None,
    capabilities: list[Capability] | None = None,
) -> Callable:
    """
    Decorator to define a tool function.

    Usage:
        @tool(name="read_file", description="Read a file's contents")
        async def read_file(path: str) -> str:
            '''
            Args:
                path: Path to the file to read
            '''
            with open(path) as f:
                return f.read()

    The decorator:
    - Extracts parameter schema from type hints
    - Extracts descriptions from docstring
    - Wraps the function to return ToolResult
    """

    def decorator(func: Callable) -> ToolDef:
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").split("\n")[0].strip()

        # Generate JSON schema from type hints
        params_schema = _generate_parameters_schema(func)

        return ToolDef(
            name=tool_name,
            description=tool_desc,
            func=func,
            parameters=params_schema,
            capabilities=frozenset(capabilities or []),
        )

    return decorator


def _generate_parameters_schema(func: Callable) -> dict[str, Any]:
    """Generate JSON Schema for function parameters from type hints."""
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    sig = inspect.signature(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = hints.get(param_name, str)
        json_type = _python_type_to_json(param_type)

        properties[param_name] = {
            "type": json_type,
            "description": f"The {param_name} parameter",
        }

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _python_type_to_json(py_type: Any) -> str:
    """Convert Python type to JSON Schema type."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    # Handle Optional, Union, etc.
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        # For Optional[X], Union[X, None], etc., just use the first arg
        args = getattr(py_type, "__args__", ())
        if args:
            return _python_type_to_json(args[0])

    return type_map.get(py_type, "string")


class FunctionSkill(Skill):
    """
    A skill built from @tool-decorated functions.

    Usage:
        @tool(name="greet")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        skill = FunctionSkill(
            name="greeter",
            description="A greeting skill",
            tools=[greet],
        )
    """

    def __init__(
        self,
        name: str,
        description: str,
        tools: list[ToolDef],
        version: str = "1.0.0",
    ) -> None:
        self._name = name
        self._description = description
        self._version = version
        self._tools = {t.name: t for t in tools}

    def manifest(self) -> SkillManifest:
        tool_specs = tuple(
            ToolSpec(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
                required_capabilities=t.capabilities,
            )
            for t in self._tools.values()
        )

        all_caps: set[Capability] = set()
        for t in self._tools.values():
            all_caps.update(t.capabilities)

        return SkillManifest(
            name=self._name,
            version=self._version,
            description=self._description,
            tools=tool_specs,
            capabilities=frozenset(all_caps),
        )

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        tool_def = self._tools.get(tool_name)
        if not tool_def:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
            )

        try:
            result = await tool_def.func(**arguments)
            return ToolResult(
                tool_call_id="",
                success=True,
                output=str(result),
            )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=str(e),
            )