"""Tests for the built-in workspace skill."""

from __future__ import annotations

import pytest

from arc.core.events import EventType
from arc.skills.builtin.workspace import WorkspaceSkill


class _KernelStub:
    def __init__(self) -> None:
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)


def test_workspace_tool_schema_declares_array_items():
    """Provider-facing tool schema must be a valid JSON schema for arrays."""
    skill = WorkspaceSkill()
    tool = skill.manifest().tools[0]
    blocks = tool.parameters["properties"]["blocks"]

    assert blocks["type"] == "array"
    assert "items" in blocks
    assert blocks["items"]["type"] == "object"

    def _assert_array_items(node):
        if isinstance(node, dict):
            if node.get("type") == "array":
                assert "items" in node
            for value in node.values():
                _assert_array_items(value)
        elif isinstance(node, list):
            for value in node:
                _assert_array_items(value)

    _assert_array_items(tool.parameters)


def test_workspace_tool_schema_allows_content_and_block_level_aliases():
    """Provider-facing schema should accept non-data aliases so sloppy model output reaches normalization."""
    skill = WorkspaceSkill()
    tool = skill.manifest().tools[0]
    block = tool.parameters["properties"]["blocks"]["items"]

    assert "content" in block["properties"]
    assert "data" not in block.get("required", [])


@pytest.mark.asyncio
async def test_workspace_skill_emits_validated_workspace_update():
    """update_workspace should normalize payloads and emit a workspace event."""
    skill = WorkspaceSkill()
    kernel = _KernelStub()
    await skill.initialize(kernel, {"agent_id": "main"})

    result = await skill.execute_tool(
        "update_workspace",
        {
            "workspace_id": "main",
            "revision": 7,
            "mode": "replace",
            "intent": "news_results",
            "title": "Latest AI News",
            "layout": "stack_with_metrics",
            "blocks": [
                {
                    "block_id": "news_1",
                    "type": "card_collection",
                    "data": {
                        "presentation": "article",
                        "items": [
                            {"id": "1", "title": "OpenAI update", "source": ""},
                        ],
                    },
                }
            ],
        },
    )

    assert result.success is True
    assert kernel.events
    event = kernel.events[-1]
    assert event.type == EventType.WORKSPACE_UPDATE
    assert event.data["payload"]["blocks"][0]["data"]["items"][0]["source"] == "Unknown source"


@pytest.mark.asyncio
async def test_workspace_skill_accepts_content_alias_from_model_output():
    """The tool should preserve populated block content even if the model uses content instead of data."""
    skill = WorkspaceSkill()
    kernel = _KernelStub()
    await skill.initialize(kernel, {"agent_id": "main"})

    result = await skill.execute_tool(
        "update_workspace",
        {
            "workspace_id": "main",
            "revision": 8,
            "mode": "replace",
            "intent": "news_results",
            "title": "Latest AI News",
            "layout": "stack_with_metrics",
            "blocks": [
                {
                    "block_id": "summary",
                    "type": "summary_header",
                    "content": {
                        "summary": "Canonical numbering preserved.",
                        "badges": [{"label": "Active Items", "value": "32"}],
                    },
                },
                {
                    "block_id": "top_news",
                    "type": "record_table",
                    "content": {
                        "columns": [{"key": "id", "label": "#"}],
                        "rows": [{"id": "1"}],
                    },
                },
            ],
        },
    )

    assert result.success is True
    event = kernel.events[-1]
    assert event.data["payload"]["blocks"][0]["data"]["items"][0]["value"] == "32"
    assert event.data["payload"]["blocks"][1]["data"]["rows"][0]["id"] == "1"


@pytest.mark.asyncio
async def test_workspace_skill_accepts_chart_layout_alias_and_nested_series():
    """Chart updates should tolerate model-friendly layout aliases and nested series arrays."""
    skill = WorkspaceSkill()
    kernel = _KernelStub()
    await skill.initialize(kernel, {"agent_id": "main"})

    result = await skill.execute_tool(
        "update_workspace",
        {
            "workspace_id": "main",
            "revision": 9,
            "mode": "replace",
            "intent": "chart_demo",
            "title": "Random Pie Chart Demo",
            "layout": "single",
            "blocks": [
                {
                    "block_id": "pie-chart",
                    "type": "chart_block",
                    "title": "Category Distribution",
                    "chart_type": "pie",
                    "series": [
                        {
                            "name": "Share",
                            "data": [
                                {"label": "Research", "value": 28},
                                {"label": "Product", "value": 22},
                            ],
                        }
                    ],
                }
            ],
        },
    )

    assert result.success is True
    event = kernel.events[-1]
    assert event.data["payload"]["layout"] == "stack"
    chart = event.data["payload"]["blocks"][0]["data"]
    assert chart["chart_type"] == "pie"
    assert chart["series"][0]["label"] == "Research"
    assert chart["series"][1]["value"] == 22


@pytest.mark.asyncio
async def test_workspace_skill_rejects_invalid_payload():
    """Workspace validation failures should return a tool error, not crash."""
    skill = WorkspaceSkill()
    kernel = _KernelStub()
    await skill.initialize(kernel, {"agent_id": "main"})

    result = await skill.execute_tool(
        "update_workspace",
        {
            "workspace_id": "main",
            "revision": 1,
            "mode": "replace",
            "intent": "news_results",
            "title": "Broken",
            "layout": "stack",
            "blocks": [
                {
                    "block_id": "bad_1",
                    "type": "nope",
                    "data": {},
                }
            ],
        },
    )

    assert result.success is False
    assert "validation" in (result.error or "").lower()
    assert kernel.events == []


@pytest.mark.asyncio
async def test_workspace_skill_rejects_empty_shell_blocks():
    """Shell-only workspace updates should fail so the model retries with real data."""
    skill = WorkspaceSkill()
    kernel = _KernelStub()
    await skill.initialize(kernel, {"agent_id": "main"})

    result = await skill.execute_tool(
        "update_workspace",
        {
            "workspace_id": "main",
            "revision": 2,
            "mode": "replace",
            "intent": "news_results",
            "title": "Latest AI News",
            "layout": "stack",
            "blocks": [
                {
                    "block_id": "summary",
                    "type": "summary_header",
                    "title": "Run Summary",
                },
                {
                    "block_id": "table",
                    "type": "record_table",
                    "title": "Top Articles",
                },
            ],
        },
    )

    assert result.success is False
    assert "empty" in (result.error or "").lower()
    assert kernel.events == []


@pytest.mark.asyncio
async def test_workspace_skill_rejects_partial_updates_with_empty_chart_shells():
    """Replace updates should fail if they include chart blocks with no renderable data."""
    skill = WorkspaceSkill()
    kernel = _KernelStub()
    await skill.initialize(kernel, {"agent_id": "main"})

    result = await skill.execute_tool(
        "update_workspace",
        {
            "workspace_id": "main",
            "revision": 4,
            "mode": "replace",
            "intent": "business_analysis",
            "title": "Q1 Analysis",
            "layout": "stack",
            "blocks": [
                {
                    "block_id": "summary",
                    "type": "summary_header",
                    "summary": "Topline metrics are available.",
                },
                {
                    "block_id": "monthly_trend",
                    "type": "chart_block",
                    "title": "Monthly Revenue and Profit Trend",
                    "summary": "Chart should exist here but has no points.",
                },
            ],
        },
    )

    assert result.success is False
    assert "monthly_trend" in (result.error or "")
    assert kernel.events == []


@pytest.mark.asyncio
async def test_workspace_skill_accepts_block_level_fields_as_data_alias():
    """Provider/model output may inline block body fields at block level."""
    skill = WorkspaceSkill()
    kernel = _KernelStub()
    await skill.initialize(kernel, {"agent_id": "main"})

    result = await skill.execute_tool(
        "update_workspace",
        {
            "workspace_id": "main",
            "revision": 3,
            "mode": "replace",
            "intent": "news_results",
            "title": "Latest AI News",
            "layout": "stack",
            "blocks": [
                {
                    "block_id": "run_summary",
                    "type": "summary_header",
                    "title": "Run Summary",
                    "summary": "Manual run completed successfully.",
                    "badges": [{"label": "Items", "value": "32"}],
                },
                {
                    "block_id": "latest_items",
                    "type": "record_table",
                    "title": "Top items",
                    "columns": [{"key": "id", "label": "#"}],
                    "rows": [{"id": "1"}],
                },
            ],
        },
    )

    assert result.success is True
    event = kernel.events[-1]
    assert event.data["payload"]["blocks"][0]["data"]["summary"] == "Manual run completed successfully."
    assert event.data["payload"]["blocks"][0]["data"]["items"][0]["value"] == "32"
    assert event.data["payload"]["blocks"][1]["data"]["rows"][0]["id"] == "1"
