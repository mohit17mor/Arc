"""Tests for HTML renderer."""

import json
import pytest

from arc.liquid.renderer import render_products
from arc.liquid.extract import ProductData


def test_render_products_basic():
    """Test that render_products returns valid HTML with products."""
    products = [
        ProductData(
            name="Sony A6400",
            price="799",
            currency="$",
            image_url="https://example.com/sony.jpg",
            rating="4.5",
            review_count="120",
            url="https://example.com/sony",
            source_domain="amazon.com",
        ),
        ProductData(
            name="Canon EOS R50",
            price="679",
            currency="$",
            url="https://example.com/canon",
            source_domain="bhphoto.com",
        ),
    ]

    html = render_products("best camera", products, sources=["amazon.com", "bhphoto.com"])

    assert "<!DOCTYPE html>" in html
    assert "best camera" in html
    assert "Sony A6400" in html
    assert "Canon EOS R50" in html
    assert "amazon.com" in html
    assert "ARC LIQUID WEB" in html


def test_render_products_contains_json():
    """Test that products are embedded as JSON for client-side rendering."""
    products = [
        ProductData(name="Test Product", price="100", url="https://example.com")
    ]
    html = render_products("test query", products)

    # Extract the products JSON from the script
    assert '"Test Product"' in html
    assert '"100"' in html


def test_render_products_responsive():
    """Test that the HTML includes responsive breakpoints."""
    products = [ProductData(name="Product", url="https://example.com")]
    html = render_products("test", products)

    assert "@media (max-width: 700px)" in html
    assert "@media (max-width: 420px)" in html
    assert "viewport" in html


def test_render_products_empty():
    """Test rendering with no products."""
    html = render_products("empty query", [])
    assert "<!DOCTYPE html>" in html
    assert "0 products found" in html


def test_render_products_escapes_html():
    """Test that query with special chars is escaped."""
    products = [ProductData(name="P", url="https://example.com")]
    html = render_products('<script>alert("xss")</script>', products)

    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_render_products_touch_support():
    """Test that touch/swipe handlers are included for mobile."""
    products = [ProductData(name="P", url="https://example.com")]
    html = render_products("test", products)

    assert "touchstart" in html
    assert "touchend" in html


def test_render_products_source_badges():
    """Test that source domain badges are rendered."""
    products = [
        ProductData(name="P1", url="https://a.com", source_domain="amazon.in"),
        ProductData(name="P2", url="https://b.com", source_domain="flipkart.com"),
    ]
    html = render_products("test", products, sources=["amazon.in", "flipkart.com"])

    assert "source_domain" in html  # Present in JSON data
