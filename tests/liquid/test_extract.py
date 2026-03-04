"""Tests for product data extractors."""

import pytest

from arc.liquid.extract import (
    ProductData,
    _extract_jsonld,
    _extract_opengraph,
    _products_from_jsonld,
    _product_from_opengraph,
    _product_quality_score,
    filter_quality_products,
)


# ── JSON-LD extraction ────────────────────────────────────────


def test_extract_jsonld_single_product():
    """Test extraction of a single Product JSON-LD block."""
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@type": "Product", "name": "Sony A6400", "offers": {"price": "799", "priceCurrency": "USD"}}
    </script>
    </head></html>
    """
    items = _extract_jsonld(html)
    assert len(items) == 1
    assert items[0]["name"] == "Sony A6400"


def test_extract_jsonld_with_graph():
    """Test extraction from @graph wrapper."""
    html = """
    <script type="application/ld+json">
    {"@graph": [
        {"@type": "Product", "name": "Camera A"},
        {"@type": "Product", "name": "Camera B"}
    ]}
    </script>
    """
    items = _extract_jsonld(html)
    assert len(items) == 2


def test_extract_jsonld_invalid_json():
    """Test that invalid JSON is skipped gracefully."""
    html = '<script type="application/ld+json">not valid json{</script>'
    items = _extract_jsonld(html)
    assert items == []


def test_products_from_jsonld():
    """Test building ProductData from JSON-LD."""
    html = """
    <script type="application/ld+json">
    {"@type": "Product", "name": "Canon EOS R50",
     "brand": {"@type": "Brand", "name": "Canon"},
     "image": "https://example.com/camera.jpg",
     "offers": {"price": "679", "priceCurrency": "USD"},
     "aggregateRating": {"ratingValue": "4.5", "reviewCount": "120"}}
    </script>
    """
    products = _products_from_jsonld(html, "https://example.com/product", "example.com")
    assert len(products) == 1
    p = products[0]
    assert p.name == "Canon EOS R50"
    assert p.brand == "Canon"
    assert p.price == "679"
    assert p.currency == "USD"
    assert p.rating == "4.5"
    assert p.review_count == "120"
    assert p.image_url == "https://example.com/camera.jpg"


def test_products_from_jsonld_itemlist():
    """Test extraction from ItemList JSON-LD."""
    html = """
    <script type="application/ld+json">
    {"@type": "ItemList", "itemListElement": [
        {"@type": "ListItem", "item": {"@type": "Product", "name": "Product A"}},
        {"@type": "ListItem", "item": {"@type": "Product", "name": "Product B"}}
    ]}
    </script>
    """
    products = _products_from_jsonld(html, "https://example.com", "example.com")
    assert len(products) == 2
    assert products[0].name == "Product A"
    assert products[1].name == "Product B"


# ── OpenGraph extraction ──────────────────────────────────────


def test_extract_opengraph():
    """Test OpenGraph meta tag extraction."""
    html = """
    <meta property="og:title" content="Best Camera 2024">
    <meta property="og:image" content="https://example.com/img.jpg">
    <meta property="og:description" content="Top cameras reviewed">
    <meta content="29.99" property="og:price:amount">
    """
    og = _extract_opengraph(html)
    assert og["title"] == "Best Camera 2024"
    assert og["image"] == "https://example.com/img.jpg"
    assert og["price:amount"] == "29.99"


def test_product_from_opengraph():
    """Test building ProductData from OpenGraph tags."""
    html = """
    <meta property="og:title" content="Nikon Z50">
    <meta property="og:image" content="https://example.com/nikon.jpg">
    <meta property="og:price:amount" content="899">
    <meta property="og:price:currency" content="USD">
    """
    product = _product_from_opengraph(html, "https://example.com/nikon", "example.com")
    assert product is not None
    assert product.name == "Nikon Z50"
    assert product.price == "899"
    assert product.currency == "USD"


def test_product_from_opengraph_no_title():
    """Test that OpenGraph with no title returns None."""
    html = '<meta property="og:image" content="https://example.com/img.jpg">'
    product = _product_from_opengraph(html, "https://example.com", "example.com")
    assert product is None


# ── ProductData ───────────────────────────────────────────────


def test_product_data_to_dict_omits_empty():
    """Test that to_dict only includes non-empty fields."""
    p = ProductData(name="Test", price="100", currency="USD")
    d = p.to_dict()
    assert "name" in d
    assert "price" in d
    assert "image_url" not in d
    assert "rating" not in d


# ── Quality Scoring & Filtering ──────────────────────────────


def test_quality_score_full_product():
    """Product with all fields scores highest."""
    p = ProductData(name="Camera", price="799", image_url="https://img.com/a.jpg", rating="4.5")
    assert _product_quality_score(p) == 5


def test_quality_score_name_only():
    """Product with only a name (blog title) scores lowest."""
    p = ProductData(name="10 Best Cameras in India 2026")
    assert _product_quality_score(p) == 1


def test_quality_score_name_and_price():
    """Product with name + price scores 3."""
    p = ProductData(name="Camera", price="799")
    assert _product_quality_score(p) == 3


def test_filter_removes_blog_titles():
    """filter_quality_products keeps real products, drops blog page titles."""
    products = [
        ProductData(name="10 Best Selfie Sticks - Review", source_domain="blog.com"),
        ProductData(name="Top Sticks 2026", image_url="https://blog.com/hero.jpg", source_domain="blog2.com"),
        ProductData(name="Mi Stick", price="599", image_url="https://amazon.in/img.jpg", source_domain="amazon.in"),
        ProductData(name="DJI Osmo", price="8999", image_url="https://amazon.in/dji.jpg", rating="4.5", source_domain="amazon.in"),
    ]
    filtered = filter_quality_products(products)
    # Should keep real products (have price or image+name) and drop name-only blog titles
    assert len(filtered) >= 2
    names = [p.name for p in filtered]
    assert "Mi Stick" in names
    assert "DJI Osmo" in names
    # Blog title with only name and no price/image should be dropped
    assert "10 Best Selfie Sticks - Review" not in names


def test_filter_sorts_by_quality():
    """Better products should come first."""
    products = [
        ProductData(name="Basic", price="100", source_domain="x.com"),
        ProductData(name="Full", price="200", image_url="https://img.com/a.jpg", rating="4.0", source_domain="y.com"),
    ]
    filtered = filter_quality_products(products)
    assert filtered[0].name == "Full"  # Higher quality first


def test_filter_empty_list():
    """Empty input returns empty output."""
    assert filter_quality_products([]) == []


def test_filter_all_low_quality_returns_all():
    """If ALL products are low quality (no price/image), return them anyway."""
    products = [
        ProductData(name="Blog Post Title A", source_domain="blog.com"),
        ProductData(name="Blog Post Title B", source_domain="blog2.com"),
    ]
    filtered = filter_quality_products(products)
    assert len(filtered) == 2  # Returns them since nothing better is available
