"""Built-in skill for updating the Gateway workspace surface."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from arc.core.events import Event, EventType
from arc.core.types import SkillManifest, ToolResult, ToolSpec
from arc.skills.base import Skill
from arc.workspace.models import (
    CardCollectionData,
    ChartBlockData,
    ComparisonTableData,
    DetailPanelData,
    MetricStripData,
    RecordTableData,
    SummaryHeaderData,
    WorkspaceUpdate,
    normalize_workspace_payload,
)


def _block_has_renderable_content(block) -> bool:
    data = block.data
    if isinstance(data, MetricStripData):
        return bool(data.items)
    if isinstance(data, CardCollectionData):
        return bool(data.items)
    if isinstance(data, RecordTableData):
        return bool(data.rows or data.columns)
    if isinstance(data, ComparisonTableData):
        return bool(data.rows or data.columns)
    if isinstance(data, SummaryHeaderData):
        return bool(data.summary or data.items)
    if isinstance(data, ChartBlockData):
        return bool(data.series)
    if isinstance(data, DetailPanelData):
        return bool(data.title or data.sections or data.fields or data.media)
    return False


def _has_renderable_block_content(payload: WorkspaceUpdate) -> bool:
    for block in payload.blocks:
        if _block_has_renderable_content(block):
            return True
    return False


class WorkspaceSkill(Skill):
    """Expose a validated structured rendering surface to the LLM."""

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="workspace",
            version="1.0.0",
            description=(
                "Render structured results in Arc's visual workspace when lists, tables, cards, "
                "metrics, or charts would communicate better than plain chat text."
            ),
            capabilities=frozenset(),
            always_available=True,
            tools=(
                ToolSpec(
                    name="update_workspace",
                    description=(
                        "Use this when structured visual presentation is better than prose. "
                        "Prefer it for news lists, products, flights, comparisons, and dashboards. "
                        "Never emit HTML, CSS, JavaScript, or JSX. Preserve canonical numbering from "
                        "tool output when it already exists. If the current tool output is optimized "
                        "for selection rather than presentation, call the source tool's richer read/list "
                        "view first, then render the workspace. For image galleries or local files, "
                        "prefer card_collection with item.image_url or item.media entries; a detail_panel "
                        "with media also works. Avoid using record_table for images unless you truly only "
                        "have tabular metadata. Use one canonical shape per block type. Put the canonical "
                        "payload under block.data. Use block.content only as a legacy alias when you are "
                        "repairing older output. Do not send empty shell blocks: every block must include "
                        "populated data. If a block cannot be populated, omit it instead of sending only a "
                        "title or summary. For summary_header include summary and/or items or badges; for "
                        "record_table include columns and rows; for card_collection include items; for "
                        "detail_panel include sections, fields, or media. For chart_block always send "
                        "data.chart_type, data.metrics, and data.series. data.series must be a list of row "
                        "objects such as [{\"label\":\"Jan\",\"net_revenue\":18918.93,\"profit\":7839.37}]. "
                        "Use supported chart_type values only: line, bar, column, pie, donut, histogram. "
                        "Do not send x_axis, x, labels, values, points, bar_line, or nested series arrays."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "workspace_id": {"type": "string", "description": "Workspace identity, usually 'main'."},
                            "revision": {"type": "integer", "description": "Monotonic update number."},
                            "mode": {"type": "string", "description": "replace or clear."},
                            "intent": {"type": "string", "description": "Semantic intent such as news_results."},
                            "title": {"type": "string", "description": "Workspace heading."},
                            "subtitle": {"type": "string", "description": "Optional context line."},
                            "layout": {"type": "string", "description": "Workspace layout hint."},
                            "blocks": {
                                "type": "array",
                                "description": "Ordered render blocks. Use generic block types like card_collection, record_table, metric_strip, comparison_table, chart_block, summary_header, or detail_panel.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "block_id": {"type": "string"},
                                        "type": {"type": "string"},
                                        "title": {"type": "string"},
                                        "layout_hint": {"type": "string"},
                                        "data": {
                                            "type": "object",
                                            "description": (
                                                "Canonical block payload. For chart_block this should contain "
                                                "chart_type, metrics, and series row objects."
                                            ),
                                        },
                                        "content": {
                                            "type": "object",
                                            "description": (
                                                "Legacy alias for data. Prefer data for all new payloads."
                                            ),
                                        },
                                        "summary": {"type": "string"},
                                        "meta": {"type": "object"},
                                    },
                                    "required": ["block_id", "type"],
                                },
                            },
                        },
                        "required": ["revision", "mode", "intent", "title", "layout", "blocks"],
                    },
                    required_capabilities=frozenset(),
                ),
            ),
        )

    async def initialize(self, kernel: Any, config: dict[str, Any]) -> None:
        self._kernel = kernel
        self._config = dict(config or {})

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if tool_name != "update_workspace":
            return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")

        try:
            payload = normalize_workspace_payload(arguments)
        except ValidationError as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Workspace validation failed: {exc}",
            )

        if payload.mode == "replace" and payload.blocks and not _has_renderable_block_content(payload):
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Workspace update rejected: every block is empty. "
                    "Send populated block data, not just titles and block shells."
                ),
            )

        if payload.mode == "replace":
            empty_blocks = [block.block_id for block in payload.blocks if not _block_has_renderable_content(block)]
            if empty_blocks:
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        "Workspace update rejected: empty render blocks detected for "
                        + ", ".join(empty_blocks)
                        + ". Send populated block data, not titles or summaries alone."
                    ),
                )

        dumped = payload.model_dump(mode="json", exclude_none=True)
        if getattr(self, "_kernel", None) is not None:
            await self._kernel.emit(
                Event(
                    type=EventType.WORKSPACE_UPDATE,
                    source=str(self._config.get("agent_id", "main")),
                    data={"payload": dumped},
                )
            )

        return ToolResult(
            success=True,
            output=(
                f"Workspace updated: {payload.title} "
                f"({len(payload.blocks)} block{'s' if len(payload.blocks) != 1 else ''}, "
                f"revision {payload.revision})."
            ),
        )
