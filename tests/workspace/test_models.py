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


def test_normalize_workspace_payload_preserves_chart_configuration():
    """Chart blocks should keep renderer-relevant configuration for multiple chart types."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 12,
            "mode": "replace",
            "intent": "analytics",
            "title": "Charts",
            "layout": "stack",
            "blocks": [
                {
                    "block_id": "traffic",
                    "type": "chart_block",
                    "content": {
                        "chart_type": "pie",
                        "label_key": "segment",
                        "value_key": "share",
                        "colors": ["#ff6600", "#0099ff"],
                        "series": [
                            {"segment": "Organic", "share": 62},
                            {"segment": "Paid", "share": 38},
                        ],
                    },
                }
            ],
        }
    )

    chart = payload.blocks[0].data
    assert chart.chart_type == "pie"
    assert chart.label_key == "segment"
    assert chart.value_key == "share"
    assert chart.colors == ["#ff6600", "#0099ff"]
    assert chart.series[0]["segment"] == "Organic"


def test_normalize_workspace_payload_accepts_chart_layout_alias_and_nested_series_data():
    """Chart payloads should tolerate common model aliases for layout and nested series arrays."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 13,
            "mode": "replace",
            "intent": "chart_demo",
            "title": "Pie Demo",
            "layout": "single",
            "blocks": [
                {
                    "block_id": "pie_chart",
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
        }
    )

    assert payload.layout == "stack"
    chart = payload.blocks[0].data
    assert chart.chart_type == "pie"
    assert chart.series == [
        {"label": "Research", "value": 28},
        {"label": "Product", "value": 22},
    ]


def test_normalize_workspace_payload_preserves_block_level_chart_data_arrays():
    """Chart blocks should keep row arrays when models place point data in block.data."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 14,
            "mode": "replace",
            "intent": "analytics",
            "title": "Revenue Charts",
            "layout": "stack",
            "blocks": [
                {
                    "block_id": "category_chart",
                    "type": "chart_block",
                    "title": "Revenue by Category",
                    "chart_type": "bar",
                    "x_key": "category",
                    "series": [
                        {"key": "net_revenue", "label": "Net Revenue"},
                    ],
                    "data": [
                        {"category": "Electronics", "net_revenue": 27236.97},
                        {"category": "Home", "net_revenue": 15078.72},
                    ],
                }
            ],
        }
    )

    chart = payload.blocks[0].data
    assert chart.chart_type == "bar"
    assert chart.x_key == "category"
    assert chart.y_key == "net_revenue"
    assert chart.series == [
        {"category": "Electronics", "net_revenue": 27236.97},
        {"category": "Home", "net_revenue": 15078.72},
    ]
    assert chart.metrics == [{"key": "net_revenue", "label": "Net Revenue"}]


def test_normalize_workspace_payload_accepts_axis_and_numeric_series_arrays():
    """Chart blocks should tolerate x_axis plus series[{name,data:[...]}] payloads."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 15,
            "mode": "replace",
            "intent": "analytics",
            "title": "Trend Charts",
            "layout": "stack",
            "blocks": [
                {
                    "block_id": "monthly_trend",
                    "type": "chart_block",
                    "title": "Monthly trend",
                    "chart_type": "line",
                    "x_axis": ["2026-01", "2026-02", "2026-03"],
                    "series": [
                        {"name": "Net revenue", "data": [18918.93, 13154.53, 27612.05]},
                        {"name": "Profit", "data": [7839.37, 5617.69, 9935.95]},
                    ],
                }
            ],
        }
    )

    chart = payload.blocks[0].data
    assert chart.chart_type == "line"
    assert chart.x_key == "label"
    assert chart.metrics == [
        {"key": "net_revenue", "label": "Net revenue"},
        {"key": "profit", "label": "Profit"},
    ]
    assert chart.series == [
        {"label": "2026-01", "net_revenue": 18918.93, "profit": 7839.37},
        {"label": "2026-02", "net_revenue": 13154.53, "profit": 5617.69},
        {"label": "2026-03", "net_revenue": 27612.05, "profit": 9935.95},
    ]


def test_normalize_workspace_payload_accepts_pie_labels_and_values_arrays():
    """Pie chart blocks should tolerate labels[] plus values[] payloads."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 16,
            "mode": "replace",
            "intent": "analytics",
            "title": "Channel Mix",
            "layout": "stack",
            "blocks": [
                {
                    "block_id": "channel_mix",
                    "type": "chart_block",
                    "title": "Channel mix",
                    "chart_type": "pie",
                    "labels": ["Mobile", "Marketplace", "Web", "Retail"],
                    "values": [22381.72, 15264.49, 14919.79, 7119.51],
                }
            ],
        }
    )

    chart = payload.blocks[0].data
    assert chart.chart_type == "pie"
    assert chart.label_key == "label"
    assert chart.value_key == "value"
    assert chart.series == [
        {"label": "Mobile", "value": 22381.72},
        {"label": "Marketplace", "value": 15264.49},
        {"label": "Web", "value": 14919.79},
        {"label": "Retail", "value": 7119.51},
    ]


def test_normalize_workspace_payload_accepts_object_x_axis_and_bar_line_alias():
    """Chart blocks should tolerate x:[{value}] plus combo chart aliases like bar_line."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 17,
            "mode": "replace",
            "intent": "analytics",
            "title": "Combo Chart",
            "layout": "vertical",
            "blocks": [
                {
                    "block_id": "monthly_trend",
                    "type": "chart_block",
                    "title": "Monthly Revenue and Profit Trend",
                    "chart_type": "bar_line",
                    "x": [{"value": "2026-01"}, {"value": "2026-02"}, {"value": "2026-03"}],
                    "series": [
                        {"name": "Net Revenue", "type": "bar", "data": [18918.93, 13154.53, 27612.05]},
                        {"name": "Profit", "type": "line", "data": [7839.37, 5617.69, 9935.95]},
                    ],
                }
            ],
        }
    )

    assert payload.layout == "stack"
    chart = payload.blocks[0].data
    assert chart.chart_type == "line"
    assert chart.x_key == "label"
    assert chart.metrics == [
        {"key": "net_revenue", "label": "Net Revenue"},
        {"key": "profit", "label": "Profit"},
    ]
    assert chart.series[0] == {"label": "2026-01", "net_revenue": 18918.93, "profit": 7839.37}


def test_normalize_workspace_payload_uses_explicit_metrics_for_row_based_charts():
    """Row-based chart series should honor data.metrics instead of inventing metrics from row labels."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 18,
            "mode": "replace",
            "intent": "analytics",
            "title": "Charts",
            "layout": "stack",
            "blocks": [
                {
                    "block_id": "monthly_trend",
                    "type": "chart_block",
                    "title": "Monthly Revenue vs Profit",
                    "data": {
                        "chart_type": "column",
                        "metrics": ["net_revenue", "profit"],
                        "series": [
                            {"label": "Jan", "net_revenue": 18918.93, "profit": 7839.37},
                            {"label": "Feb", "net_revenue": 13154.53, "profit": 5617.69},
                            {"label": "Mar", "net_revenue": 27612.05, "profit": 9935.95},
                        ],
                    },
                }
            ],
        }
    )

    chart = payload.blocks[0].data
    assert chart.chart_type == "column"
    assert chart.x_key == "label"
    assert [metric["key"] for metric in chart.metrics] == ["net_revenue", "profit"]
    assert chart.metrics[0]["label"] == "Net Revenue"
    assert chart.metrics[1]["label"] == "Profit"


def test_normalize_workspace_payload_uses_explicit_metric_for_row_based_pie_chart():
    """Pie charts with row objects should preserve the metric field as value_key."""
    payload = normalize_workspace_payload(
        {
            "workspace_id": "main",
            "revision": 19,
            "mode": "replace",
            "intent": "analytics",
            "title": "Charts",
            "layout": "stack",
            "blocks": [
                {
                    "block_id": "channel_chart",
                    "type": "chart_block",
                    "title": "Channel Mix",
                    "data": {
                        "chart_type": "donut",
                        "metrics": ["net_revenue"],
                        "series": [
                            {"label": "Mobile", "net_revenue": 22381.72},
                            {"label": "Marketplace", "net_revenue": 15264.49},
                            {"label": "Web", "net_revenue": 14919.79},
                        ],
                    },
                }
            ],
        }
    )

    chart = payload.blocks[0].data
    assert chart.chart_type == "donut"
    assert chart.label_key == "label"
    assert chart.value_key == "net_revenue"
    assert [metric["key"] for metric in chart.metrics] == ["net_revenue"]


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
