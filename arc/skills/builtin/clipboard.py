"""Built-in clipboard access skill."""

from __future__ import annotations

from typing import Any

from arc.clipboard import ClipboardReadResult, SystemClipboardReader
from arc.core.types import SkillManifest, ToolResult, ToolSpec
from arc.skills.base import Skill


class ClipboardSkill(Skill):
    """Read the current system clipboard as plain text."""

    def __init__(self, reader: SystemClipboardReader | None = None) -> None:
        self._reader = reader or SystemClipboardReader()

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="clipboard",
            version="1.0.0",
            description=(
                "Read the current system clipboard as plain text when the user explicitly asks "
                "about copied text or highlighted text that they copied."
            ),
            always_available=True,
            tools=(
                ToolSpec(
                    name="get_clipboard_text",
                    description=(
                        "Read the current system clipboard as plain text. Use this only when the user "
                        "explicitly refers to copied text or says they highlighted text and copied it. "
                        "Do not call this tool for vague references like 'this', 'that', or general "
                        "questions that do not explicitly mention copied or highlighted text. "
                        "Fail if the clipboard is empty, unavailable, or does not currently contain text."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                    required_capabilities=frozenset(),
                ),
            ),
        )

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if tool_name != "get_clipboard_text":
            return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")

        result = self._reader.read()
        return self._to_tool_result(result)

    @staticmethod
    def _to_tool_result(result: ClipboardReadResult) -> ToolResult:
        if result.kind == "text" and result.text is not None:
            return ToolResult(
                success=True,
                output=f"Clipboard text from {result.source}:\n{result.text}",
            )

        if result.kind == "empty":
            return ToolResult(
                success=False,
                output="",
                error="The clipboard is empty right now.",
            )

        if result.kind == "non_text":
            return ToolResult(
                success=False,
                output="",
                error="The clipboard does not currently contain text.",
            )

        message = result.error or "Clipboard access failed."
        return ToolResult(
            success=False,
            output="",
            error=f"Could not read clipboard text: {message}",
        )
