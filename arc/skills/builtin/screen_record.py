"""Built-in general-purpose desktop screen recording skill."""

from __future__ import annotations

from typing import Any

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec
from arc.screen.recorder import FFmpegScreenRecorder
from arc.skills.base import Skill


class ScreenRecordSkill(Skill):
    """Start, stop, and inspect desktop screen recording."""

    def __init__(self, recorder: FFmpegScreenRecorder | None = None) -> None:
        self._recorder = recorder or FFmpegScreenRecorder()

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="screen_record",
            version="1.0.0",
            description="General-purpose desktop screen recording for evidence capture",
            capabilities=frozenset([Capability.SYSTEM_PROCESS, Capability.FILE_WRITE]),
            tools=(
                ToolSpec(
                    name="screen_record_start",
                    description=(
                        "Start recording the full desktop to a local MP4 file. "
                        "Use this when the user asks to record the screen while Arc works "
                        "or while the user performs actions manually."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "fps": {
                                "type": "integer",
                                "description": "Frames per second. Defaults to 30.",
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Optional custom path for the MP4 output file.",
                            },
                        },
                        "required": [],
                    },
                    required_capabilities=frozenset([Capability.SYSTEM_PROCESS, Capability.FILE_WRITE]),
                ),
                ToolSpec(
                    name="screen_record_stop",
                    description="Stop the active desktop recording and return the saved file path.",
                    parameters={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                    required_capabilities=frozenset([Capability.SYSTEM_PROCESS, Capability.FILE_WRITE]),
                ),
                ToolSpec(
                    name="screen_record_status",
                    description="Show whether desktop recording is currently active and where output is being saved.",
                    parameters={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                    required_capabilities=frozenset([Capability.SYSTEM_PROCESS]),
                ),
            ),
        )

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        handlers = {
            "screen_record_start": self._start,
            "screen_record_stop": self._stop,
            "screen_record_status": self._status,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")
        try:
            return await handler(arguments)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    async def _start(self, arguments: dict[str, Any]) -> ToolResult:
        raw_fps = arguments.get("fps")
        fps = 30 if raw_fps is None else int(raw_fps)
        if fps <= 0:
            return ToolResult(success=False, output="", error="fps must be greater than 0.")

        output_path = arguments.get("output_path")
        out = self._recorder.start(fps=fps, output_path=output_path)
        return ToolResult(
            success=True,
            output=f"Screen recording started at {fps} fps. Saving to {out}",
            artifacts=[str(out)],
        )

    async def _stop(self, arguments: dict[str, Any]) -> ToolResult:
        status = self._recorder.status()
        if not status.get("recording"):
            return ToolResult(
                success=False,
                output="",
                error="Screen recording is not recording right now.",
            )

        info = self._recorder.stop()
        output_path = str(info["output_path"])
        duration = info["duration_seconds"]
        fps = info["fps"]
        return ToolResult(
            success=True,
            output=(
                f"Screen recording stopped. Saved to {output_path} "
                f"({duration}s at {fps} fps)"
            ),
            artifacts=[output_path],
        )

    async def _status(self, arguments: dict[str, Any]) -> ToolResult:
        info = self._recorder.status()
        if not info["recording"]:
            return ToolResult(success=True, output="Screen recording is not recording right now.")
        return ToolResult(
            success=True,
            output=(
                f"Screen recording is active at {info['fps']} fps. "
                f"Saving to {info['output_path']}"
            ),
            artifacts=[str(info["output_path"])],
        )
