"""Tests for workspace payload models and normalization."""

from __future__ import annotations

from arc.workspace.models import normalize_workspace_payload


def test_normalize_workspace_payload_fills_missing_article_fields():
    """Sparse article cards degrade safely instead of failing validation."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 3,
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
                            {
                                "id": "a1",
                                "name": "Model launch roundup",
                            },
                            {
                                "title": "Missing id should still survive",
                                "source": "",
                                "summary": "",
                            },
                        ],
                    },
                }
            ],
        }
    )

    block = payload.blocks[0]
    assert block.type == "card_collection"
    items = block.data.items

    assert items[0].title == "Model launch roundup"
    assert items[0].source == "Unknown source"
    assert items[0].summary is None
    assert items[0].image_url is None

    assert items[1].id.startswith("news_1-item-")
    assert items[1].title == "Missing id should still survive"
    assert items[1].source == "Unknown source"
    assert items[1].summary is None


def test_normalize_workspace_payload_uses_index_as_canonical_index():
    """Incoming selection indices should survive for workspace numbering."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 5,
            "mode": "replace",
            "intent": "news_results",
            "title": "Latest AI News",
            "layout": "stack",
            "blocks": [
                {
                    "block_id": "news_1",
                    "type": "card_collection",
                    "data": {
                        "presentation": "article",
                        "items": [
                            {"id": "a1", "title": "Article One", "index": 1},
                        ],
                    },
                }
            ],
        }
    )

    items = payload.blocks[0].data.items
    assert items[0].canonical_index == 1


def test_normalize_workspace_payload_maps_card_media_badges_and_description():
    """Card items should preserve image/media and common rich-card aliases."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 6,
            "mode": "replace",
            "intent": "product_results",
            "title": "Products",
            "layout": "stack",
            "blocks": [
                {
                    "block_id": "products",
                    "type": "card_collection",
                    "data": {
                        "items": [
                            {
                                "title": "Product One",
                                "subtitle": "Amazon's Choice",
                                "description": "Strong everyday white tee.",
                                "badges": [
                                    {"label": "Price", "value": "₹199"},
                                    {"label": "Rating", "value": "4.0/5"},
                                    {"label": "Deal", "value": "400+ bought in past month"},
                                ],
                                "media": [
                                    {"type": "image", "url": "file:///tmp/product-one.jpg"},
                                ],
                            }
                        ]
                    },
                }
            ],
        }
    )

    item = payload.blocks[0].data.items[0]
    assert item.image_url == "file:///tmp/product-one.jpg"
    assert item.summary == "Strong everyday white tee."
    assert item.price == "₹199"
    assert item.rating == "4.0/5"
    assert "Amazon's Choice" in item.chips
    assert "Deal: 400+ bought in past month" in item.chips


def test_normalize_workspace_payload_accepts_sparse_record_table_rows():
    """Missing row cells should remain renderable via generic table blocks."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 4,
            "mode": "replace",
            "intent": "flight_results",
            "title": "Flights",
            "layout": "stack",
            "blocks": [
                {
                    "block_id": "flights_1",
                    "type": "record_table",
                    "data": {
                        "presentation": "flight",
                        "columns": [
                            {"key": "airline", "label": "Airline"},
                            {"key": "price", "label": "Price"},
                        ],
                        "rows": [
                            {"id": "f1", "airline": "IndiGo"},
                        ],
                    },
                }
            ],
        }
    )

    block = payload.blocks[0]
    assert block.type == "record_table"
    assert block.data.columns[0].key == "airline"
    assert block.data.rows[0]["airline"] == "IndiGo"
    assert "price" not in block.data.rows[0]


def test_normalize_workspace_payload_coerces_string_summary_items_and_sequence_tables():
    """Loose model output should still survive when summaries use strings and tables use sequences."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 11,
            "mode": "replace",
            "intent": "image_gallery",
            "title": "Local Images",
            "layout": "grid_focus",
            "blocks": [
                {
                    "block_id": "downloads_summary",
                    "type": "summary_header",
                    "summary": "Found 3 images in Arc/Downloads",
                    "items": [
                        "instagram_current_slide_img_index_2.jpg",
                        "instagram_test_slide1.jpg",
                    ],
                },
                {
                    "block_id": "downloads_files",
                    "type": "record_table",
                    "columns": ["Filename", "Path", "Type"],
                    "rows": [
                        [
                            "instagram_current_slide_img_index_2.jpg",
                            "file:///Users/mmor/scratch/Arc/Downloads/instagram_current_slide_img_index_2.jpg",
                            "jpg",
                        ]
                    ],
                },
            ],
        }
    )

    summary = payload.blocks[0].data
    assert summary.summary == "Found 3 images in Arc/Downloads"
    assert summary.items[0].label == "Item 1"
    assert summary.items[0].value == "instagram_current_slide_img_index_2.jpg"

    table = payload.blocks[1].data
    assert table.columns[0].key == "filename"
    assert table.columns[1].key == "path"
    assert table.rows[0]["filename"] == "instagram_current_slide_img_index_2.jpg"
    assert table.rows[0]["path"] == "file:///Users/mmor/scratch/Arc/Downloads/instagram_current_slide_img_index_2.jpg"
    assert table.rows[0]["type"] == "jpg"


def test_normalize_workspace_payload_accepts_content_alias_and_summary_badges():
    """LLMs may send block payloads under content; normalization should preserve them."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 9,
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
                        "badges": [
                            {"label": "Run ID", "value": "20260331T081616Z"},
                            {"label": "Active Items", "value": "32"},
                        ],
                    },
                },
                {
                    "block_id": "news_table",
                    "type": "record_table",
                    "content": {
                        "columns": [
                            {"key": "id", "label": "#"},
                            {"key": "title", "label": "Title"},
                        ],
                        "rows": [
                            {"id": "1", "title": "The Silicon Valley congressional race is getting ugly"},
                        ],
                    },
                },
                {
                    "block_id": "themes",
                    "type": "detail_panel",
                    "content": {
                        "sections": [
                            {"heading": "Warnings", "items": ["VentureBeat AI feed returned HTTP 404"]},
                        ],
                    },
                },
            ],
        }
    )

    summary = payload.blocks[0].data
    assert summary.summary == "Canonical numbering preserved."
    assert len(summary.items) == 2
    assert summary.items[0].label == "Run ID"
    assert summary.items[0].value == "20260331T081616Z"

    table = payload.blocks[1].data
    assert table.columns[0].key == "id"
    assert table.rows[0]["id"] == "1"

    detail = payload.blocks[2].data
    assert detail.sections[0]["heading"] == "Warnings"


def test_normalize_workspace_payload_accepts_block_level_fields_as_data_alias():
    """If a model inlines block body fields at the block level, preserve them."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 10,
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
                    "badges": [
                        {"label": "Items", "value": "32"},
                        {"label": "New", "value": "0"},
                    ],
                },
                {
                    "block_id": "latest_items",
                    "type": "record_table",
                    "title": "Top items",
                    "columns": [
                        {"key": "id", "label": "#"},
                        {"key": "title", "label": "Title"},
                    ],
                    "rows": [
                        {"id": "1", "title": "The Silicon Valley congressional race is getting ugly"},
                    ],
                },
            ],
        }
    )

    summary = payload.blocks[0].data
    assert summary.summary == "Manual run completed successfully."
    assert summary.items[0].label == "Items"
    assert summary.items[0].value == "32"

    table = payload.blocks[1].data
    assert table.columns[1].key == "title"
    assert table.rows[0]["id"] == "1"
