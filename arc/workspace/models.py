"""Validated workspace payload models with normalization helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


WorkspaceMode = Literal["replace", "clear"]
WorkspaceLayout = Literal["stack", "stack_with_metrics", "grid_focus"]
BlockType = Literal[
    "summary_header",
    "card_collection",
    "record_table",
    "comparison_table",
    "metric_strip",
    "chart_block",
    "detail_panel",
]

_BLOCK_STRUCTURE_KEYS = {"block_id", "type", "title", "layout_hint", "data", "content", "meta"}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _chips(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        cleaned = _text(item)
        if cleaned:
            out.append(cleaned)
    return out


def _badge_text(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            label = _text(item.get("label"))
            badge_value = _text(item.get("value"))
            if label and badge_value:
                out.append(f"{label}: {badge_value}")
            elif label:
                out.append(label)
            elif badge_value:
                out.append(badge_value)
            continue
        cleaned = _text(item)
        if cleaned:
            out.append(cleaned)
    return out


def _slug_key(value: Any, fallback: str) -> str:
    text = (_text(value) or "").lower()
    chars: list[str] = []
    prev_sep = False
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
            prev_sep = False
            continue
        if not prev_sep:
            chars.append("_")
            prev_sep = True
    normalized = "".join(chars).strip("_")
    return normalized or fallback


def _extract_image_url(item: dict[str, Any]) -> str | None:
    image_url = _text(item.get("image_url"))
    if image_url:
        return image_url
    media = item.get("media")
    if not isinstance(media, (list, tuple)):
        return None
    for entry in media:
        if not isinstance(entry, dict):
            continue
        media_type = (_text(entry.get("type")) or "").lower()
        candidate = _text(entry.get("url")) or _text(entry.get("image_url")) or _text(entry.get("src"))
        if candidate and (not media_type or media_type == "image"):
            return candidate
    return None


class MetricItem(BaseModel):
    label: str
    value: str
    unit: str | None = None
    trend: str | None = None


class MetricStripData(BaseModel):
    items: list[MetricItem] = Field(default_factory=list)


class CardItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    source: str | None = None
    published_at: str | None = None
    image_url: str | None = None
    summary: str | None = None
    chips: list[str] = Field(default_factory=list)
    url: str | None = None
    price: str | None = None
    original_price: str | None = None
    rating: str | None = None
    merchant: str | None = None
    canonical_index: int | None = None


class CardCollectionData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    presentation: str = "generic"
    items: list[CardItem] = Field(default_factory=list)


class TableColumn(BaseModel):
    key: str
    label: str


class RecordTableData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    presentation: str = "generic"
    columns: list[TableColumn] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class ComparisonTableData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    columns: list[TableColumn] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    highlighted_rows: list[str] = Field(default_factory=list)


class SummaryHeaderData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str | None = None
    items: list[MetricItem] = Field(default_factory=list)


class ChartBlockData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chart_type: str
    series: list[dict[str, Any]] = Field(default_factory=list)


class DetailPanelData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    sections: list[dict[str, Any]] = Field(default_factory=list)
    fields: list[dict[str, Any]] = Field(default_factory=list)
    media: list[dict[str, Any]] = Field(default_factory=list)


class WorkspaceBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    block_id: str
    type: BlockType
    title: str | None = None
    layout_hint: str | None = None
    data: Any = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_data(self) -> "WorkspaceBlock":
        data = self.data or {}
        if self.type == "metric_strip":
            self.data = MetricStripData.model_validate(data)
        elif self.type == "card_collection":
            self.data = CardCollectionData.model_validate(data)
        elif self.type == "record_table":
            self.data = RecordTableData.model_validate(data)
        elif self.type == "comparison_table":
            self.data = ComparisonTableData.model_validate(data)
        elif self.type == "summary_header":
            self.data = SummaryHeaderData.model_validate(data)
        elif self.type == "chart_block":
            self.data = ChartBlockData.model_validate(data)
        elif self.type == "detail_panel":
            self.data = DetailPanelData.model_validate(data)
        return self


class WorkspaceUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    workspace_id: str
    revision: int
    mode: WorkspaceMode
    intent: str
    title: str
    subtitle: str | None = None
    layout: WorkspaceLayout
    blocks: list[WorkspaceBlock] = Field(default_factory=list)


def _normalize_metric_strip_data(data: Any) -> dict[str, Any]:
    raw = data or {}
    raw_items = raw.get("items")
    if not isinstance(raw_items, list):
        raw_items = raw.get("badges")
    if not isinstance(raw_items, list):
        raw_items = raw.get("metrics")

    items: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw_items or [], start=1):
        if not isinstance(entry, dict):
            cleaned = _text(entry)
            if cleaned:
                items.append({"label": f"Item {idx}", "value": cleaned, "unit": None, "trend": None})
            continue
        label = _text(entry.get("label"))
        value = _text(entry.get("value"))
        if not label or not value:
            continue
        items.append(
            {
                "label": label,
                "value": value,
                "unit": _text(entry.get("unit")),
                "trend": _text(entry.get("trend")),
            }
        )
    return {"items": items}


def _normalize_card_collection_data(block_id: str, data: Any) -> dict[str, Any]:
    raw = data or {}
    presentation = _text(raw.get("presentation")) or "generic"
    items: list[dict[str, Any]] = []
    for idx, item in enumerate(raw.get("items", []), start=1):
        if not isinstance(item, dict):
            continue
        title = _text(item.get("title")) or _text(item.get("name")) or "Untitled item"
        item_id = _text(item.get("id")) or f"{block_id}-item-{idx}"
        source = _text(item.get("source"))
        merchant = _text(item.get("merchant"))
        if presentation == "article" and not source:
            source = "Unknown source"
        summary = _text(item.get("summary")) or _text(item.get("description"))
        image_url = _extract_image_url(item)
        price = _text(item.get("price"))
        rating = _text(item.get("rating"))
        chips = _chips(item.get("chips"))
        subtitle = _text(item.get("subtitle"))
        if subtitle:
            chips.append(subtitle)
        for badge in _badge_text(item.get("badges")):
            lower_badge = badge.lower()
            if lower_badge.startswith("price:") and not price:
                price = _text(badge.split(":", 1)[1])
                continue
            if lower_badge.startswith("rating:") and not rating:
                rating = _text(badge.split(":", 1)[1])
                continue
            chips.append(badge)
        normalized: dict[str, Any] = {
            "id": item_id,
            "title": title,
            "source": source,
            "published_at": _text(item.get("published_at")),
            "image_url": image_url,
            "summary": summary,
            "chips": chips,
            "url": _text(item.get("url")),
            "price": price,
            "original_price": _text(item.get("original_price")),
            "rating": rating,
            "merchant": merchant,
        }
        canonical_index = item.get("canonical_index")
        if not isinstance(canonical_index, int):
            canonical_index = item.get("index")
        if isinstance(canonical_index, int) and canonical_index > 0:
            normalized["canonical_index"] = canonical_index
        items.append(normalized)
    return {"presentation": presentation, "items": items}


def _normalize_record_table_data(data: Any) -> dict[str, Any]:
    raw = data or {}
    columns: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    raw_columns = raw.get("columns", [])
    for idx, col in enumerate(raw_columns, start=1):
        if not isinstance(col, dict):
            label = _text(col)
            if not label:
                continue
            base_key = _slug_key(label, f"col_{idx}")
            key = base_key
            suffix = 2
            while key in seen_keys:
                key = f"{base_key}_{suffix}"
                suffix += 1
            seen_keys.add(key)
            columns.append({"key": key, "label": label})
            continue
        key = _text(col.get("key"))
        label = _text(col.get("label")) or key
        if not key or not label:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        columns.append({"key": key, "label": label})
    rows: list[dict[str, Any]] = []
    for row in raw.get("rows", []):
        if isinstance(row, dict):
            rows.append(row)
            continue
        if not isinstance(row, (list, tuple)):
            continue
        if not columns:
            for idx, _ in enumerate(row, start=1):
                key = f"col_{idx}"
                if any(col["key"] == key for col in columns):
                    continue
                columns.append({"key": key, "label": f"Column {idx}"})
        coerced: dict[str, Any] = {}
        for idx, value in enumerate(row):
            if idx >= len(columns):
                key = f"col_{idx+1}"
                columns.append({"key": key, "label": f"Column {idx+1}"})
            coerced[columns[idx]["key"]] = value
        rows.append(coerced)
    return {
        "presentation": _text(raw.get("presentation")) or "generic",
        "columns": columns,
        "rows": rows,
    }


def _normalize_comparison_table_data(data: Any) -> dict[str, Any]:
    raw = data or {}
    return {
        "columns": _normalize_record_table_data(raw).get("columns", []),
        "rows": [row for row in raw.get("rows", []) if isinstance(row, dict)],
        "highlighted_rows": [str(v) for v in raw.get("highlighted_rows", []) if _text(v)],
    }


def _normalize_summary_header_data(data: Any) -> dict[str, Any]:
    raw = data or {}
    return {
        "summary": _text(raw.get("summary")),
        "items": _normalize_metric_strip_data(raw).get("items", []),
    }


def _normalize_chart_block_data(data: Any) -> dict[str, Any]:
    raw = data or {}
    return {
        "chart_type": _text(raw.get("chart_type")) or "histogram",
        "series": [row for row in raw.get("series", []) if isinstance(row, dict)],
    }


def _normalize_detail_panel_data(data: Any) -> dict[str, Any]:
    raw = data or {}
    return {
        "title": _text(raw.get("title")),
        "sections": [row for row in raw.get("sections", []) if isinstance(row, dict)],
        "fields": [row for row in raw.get("fields", []) if isinstance(row, dict)],
        "media": [row for row in raw.get("media", []) if isinstance(row, dict)],
    }


def normalize_workspace_payload(payload: dict[str, Any]) -> WorkspaceUpdate:
    """Normalize sparse payloads, then validate the workspace structure."""
    normalized = deepcopy(payload)
    blocks: list[dict[str, Any]] = []
    for idx, raw_block in enumerate(normalized.get("blocks", []), start=1):
        if not isinstance(raw_block, dict):
            continue
        block_id = _text(raw_block.get("block_id")) or f"block-{idx}"
        block_type = _text(raw_block.get("type")) or ""
        raw_data = raw_block.get("data")
        if not isinstance(raw_data, dict):
            raw_data = raw_block.get("content")
        if not isinstance(raw_data, dict):
            raw_data = {
                key: value
                for key, value in raw_block.items()
                if key not in _BLOCK_STRUCTURE_KEYS
            }
        if not isinstance(raw_data, dict):
            raw_data = {}

        if block_type == "metric_strip":
            data = _normalize_metric_strip_data(raw_data)
        elif block_type == "card_collection":
            data = _normalize_card_collection_data(block_id, raw_data)
        elif block_type == "record_table":
            data = _normalize_record_table_data(raw_data)
        elif block_type == "comparison_table":
            data = _normalize_comparison_table_data(raw_data)
        elif block_type == "summary_header":
            data = _normalize_summary_header_data(raw_data)
        elif block_type == "chart_block":
            data = _normalize_chart_block_data(raw_data)
        elif block_type == "detail_panel":
            data = _normalize_detail_panel_data(raw_data)
        else:
            data = raw_data

        blocks.append(
            {
                "block_id": block_id,
                "type": block_type,
                "title": _text(raw_block.get("title")),
                "layout_hint": _text(raw_block.get("layout_hint")),
                "data": data,
                "meta": raw_block.get("meta") if isinstance(raw_block.get("meta"), dict) else {},
            }
        )

    normalized["workspace_id"] = _text(normalized.get("workspace_id")) or "main"
    normalized["intent"] = _text(normalized.get("intent")) or "generic_results"
    normalized["title"] = _text(normalized.get("title")) or "Workspace"
    normalized["subtitle"] = _text(normalized.get("subtitle"))
    normalized["layout"] = _text(normalized.get("layout")) or "stack"
    normalized["blocks"] = blocks

    try:
        return WorkspaceUpdate.model_validate(normalized)
    except ValidationError:
        raise
