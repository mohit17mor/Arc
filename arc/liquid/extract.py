"""
Product data extraction from web pages.

Generic-first approach: JSON-LD → OpenGraph → DOM heuristics.
Optional site-specific extractors for higher quality on known domains.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Callable, Awaitable
from urllib.parse import urlparse

if TYPE_CHECKING:
    from playwright.async_api import Page

logger = logging.getLogger(__name__)


@dataclass
class ProductData:
    """Extracted product data — flexible for any product type."""

    name: str = ""
    price: str = ""
    currency: str = ""
    image_url: str = ""
    rating: str = ""
    review_count: str = ""
    description: str = ""
    brand: str = ""
    url: str = ""
    original_price: str = ""
    source_domain: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


# Type for site-specific extractors
SiteExtractor = Callable[["Page", str], Awaitable[list[ProductData]]]

# Registry of site-specific extractors keyed by domain
EXTRACTORS: dict[str, SiteExtractor] = {}


def register_extractor(domain: str):
    """Decorator to register a site-specific extractor."""
    def decorator(func: SiteExtractor) -> SiteExtractor:
        EXTRACTORS[domain] = func
        return func
    return decorator


# ── Generic extraction (works on any site) ────────────────────


def _extract_jsonld(html: str) -> list[dict]:
    """Extract all JSON-LD blocks from HTML."""
    pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
    results = []
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict):
                if "@graph" in data:
                    results.extend(data["@graph"])
                else:
                    results.append(data)
        except json.JSONDecodeError:
            continue
    return results


def _extract_opengraph(html: str) -> dict:
    """Extract OpenGraph meta tags."""
    og: dict[str, str] = {}
    patterns = [
        r'<meta\s+property=["\']og:(\w+(?::\w+)*)["\']\s+content=["\']([^"\']*)["\']',
        r'<meta\s+content=["\']([^"\']*)["\']\s+property=["\']og:(\w+(?::\w+)*)["\']',
    ]
    for i, pattern in enumerate(patterns):
        for match in re.finditer(pattern, html, re.IGNORECASE):
            groups = match.groups()
            key, value = (groups[0], groups[1]) if i == 0 else (groups[1], groups[0])
            og[key] = value
    return og


def _get_nested(d: dict, *keys: str):
    """Safely get nested dict values."""
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _product_from_jsonld_item(item: dict, url: str, domain: str) -> ProductData | None:
    """Build ProductData from a single JSON-LD item."""
    item_type = item.get("@type", "")
    types = [t.lower() for t in item_type] if isinstance(item_type, list) else [item_type.lower()]

    if "product" not in types:
        return None

    p = ProductData(source_domain=domain, url=item.get("url", url))
    p.name = item.get("name", "")
    p.description = (item.get("description", "") or "")[:300]
    p.brand = _get_nested(item, "brand", "name") or item.get("brand", "")
    if isinstance(p.brand, dict):
        p.brand = p.brand.get("name", "")

    # Image
    img = item.get("image", "")
    if isinstance(img, list):
        p.image_url = img[0] if img else ""
        if isinstance(p.image_url, dict):
            p.image_url = p.image_url.get("url", "")
    elif isinstance(img, dict):
        p.image_url = img.get("url", "")
    else:
        p.image_url = str(img)

    # Price from offers
    offers = item.get("offers", {})
    if isinstance(offers, list):
        offers = offers[0] if offers else {}
    if isinstance(offers, dict):
        p.price = str(offers.get("price", offers.get("lowPrice", "")))
        p.currency = offers.get("priceCurrency", "")

    # Rating
    rating = item.get("aggregateRating", {})
    if isinstance(rating, dict):
        p.rating = str(rating.get("ratingValue", ""))
        p.review_count = str(rating.get("reviewCount", rating.get("ratingCount", "")))

    return p if p.name else None


def _products_from_jsonld(html: str, url: str, domain: str) -> list[ProductData]:
    """Extract products from JSON-LD blocks."""
    products = []
    items = _extract_jsonld(html)

    for item in items:
        item_type = item.get("@type", "")
        types = [t.lower() for t in item_type] if isinstance(item_type, list) else [item_type.lower()]

        if "itemlist" in types:
            for elem in item.get("itemListElement", []):
                if isinstance(elem, dict):
                    inner = elem.get("item", elem)
                    p = _product_from_jsonld_item(inner, url, domain)
                    if p:
                        products.append(p)
        else:
            p = _product_from_jsonld_item(item, url, domain)
            if p:
                products.append(p)

    return products


def _product_from_opengraph(html: str, url: str, domain: str) -> ProductData | None:
    """Build a ProductData from OpenGraph tags."""
    og = _extract_opengraph(html)
    if not og.get("title"):
        return None

    p = ProductData(source_domain=domain, url=url)
    p.name = og.get("title", "")
    p.description = (og.get("description", "") or "")[:300]
    p.image_url = og.get("image", "")
    p.price = og.get("price:amount", og.get("product:price:amount", ""))
    p.currency = og.get("price:currency", og.get("product:price:currency", ""))
    return p if p.name else None


def _product_quality_score(p: ProductData) -> int:
    """Score a product's data quality (higher = better).

    Products from blog titles with no price/image score 0-1.
    Real products with price+image+rating score 3-5.
    """
    score = 0
    if p.name:
        score += 1
    if p.price:
        score += 2  # price is the strongest signal of a real product
    if p.image_url:
        score += 1
    if p.rating:
        score += 1
    return score


def filter_quality_products(products: list[ProductData]) -> list[ProductData]:
    """Filter out low-quality extractions (e.g., blog page titles with no product data)."""
    if not products:
        return products

    # Separate real products (have price or image) from page-level entries
    real = [p for p in products if p.price or (p.image_url and p.name)]

    if real:
        # Sort by quality score, best first
        real.sort(key=_product_quality_score, reverse=True)
        return real

    # If nothing has price/image, return what we have (better than nothing)
    return products


async def extract_generic(page: Page, url: str) -> list[ProductData]:
    """Generic extractor — works on any site via JSON-LD, OpenGraph, and DOM."""
    domain = urlparse(url).netloc.removeprefix("www.")
    html = await page.content()

    # Try JSON-LD first (richest structured data)
    products = _products_from_jsonld(html, url, domain)
    if products:
        return products

    # Try OpenGraph — but only return immediately if it looks like a real product
    # (has price). Otherwise, continue to DOM extraction which may find actual
    # product cards inside review/listicle pages.
    og_product = _product_from_opengraph(html, url, domain)
    og_has_product_data = og_product and og_product.price

    # DOM heuristic: find product-like cards via JS
    products = []
    try:
        dom_products = await page.evaluate(r"""() => {
            const results = [];

            // Strategy 1: Amazon search result cards
            const amazonCards = document.querySelectorAll(
                '[data-component-type="s-search-result"]'
            );
            if (amazonCards.length > 0) {
                for (const card of [...amazonCards].slice(0, 50)) {
                    const p = {};
                    const img = card.querySelector('.s-image, img[data-image-latency]');
                    if (img) {
                        p.image_url = img.src || '';
                        p.name = img.alt || '';
                    }
                    if (!p.name) {
                        const titleEl = card.querySelector('h2 a span, .a-text-normal');
                        if (titleEl) p.name = titleEl.textContent.trim();
                    }
                    const link = card.querySelector('h2 a, a.a-link-normal[href*="/dp/"]');
                    if (link) {
                        p.url = link.href;
                    }
                    const priceWhole = card.querySelector('.a-price:not([data-a-strike]) .a-price-whole');
                    const priceFrac = card.querySelector('.a-price:not([data-a-strike]) .a-price-fraction');
                    if (priceWhole) {
                        let price = priceWhole.textContent.replace(/[,.]/g, '').trim();
                        if (priceFrac) price += '.' + priceFrac.textContent.trim();
                        p.price = price;
                    }
                    if (p.name && p.name.length > 5) results.push(p);
                }
                return results;
            }

            // Strategy 2: Generic product/item cards
            const selectors = [
                '[class*="product"]', '[class*="Product"]',
                '[class*="item-card"]', '[class*="listing"]',
                '[class*="search-result"]', '[class*="SearchResult"]',
                '[class*="card"]', 'article',
                '[data-product]', '[data-item]',
            ];
            const allCards = document.querySelectorAll(selectors.join(', '));
            const seen = new Set();
            for (const card of [...allCards].slice(0, 50)) {
                const p = {};
                // Get image
                const img = card.querySelector('img[src*="http"]');
                if (img) {
                    p.image_url = img.src;
                    p.name = img.alt || '';
                }
                // Get link
                const link = card.querySelector('a[href]');
                if (link) {
                    p.url = link.href;
                    if (!p.name) {
                        // Try heading or link text
                        const heading = card.querySelector('h1, h2, h3, h4, [class*="title"], [class*="name"]');
                        p.name = heading
                            ? heading.textContent.trim().substring(0, 200)
                            : link.textContent.trim().substring(0, 200);
                    }
                }
                // Price heuristic — look for currency symbols
                const priceSelectors = [
                    '[class*="price"]', '[class*="Price"]',
                    '[data-price]', '[class*="cost"]',
                    '[class*="amount"]', '[class*="offer"]',
                ];
                const priceEl = card.querySelector(priceSelectors.join(', '));
                if (priceEl) {
                    const text = priceEl.textContent.trim();
                    const match = text.match(/[\$\£\€\₹]?\s*[\d,]+\.?\d*/);
                    if (match) p.price = match[0];
                }
                // Rating
                const ratingEl = card.querySelector('[class*="rating"], [class*="stars"]');
                if (ratingEl) {
                    const match = ratingEl.textContent.match(/(\d+\.?\d*)\s*(?:\/\s*5|out of|stars)/i);
                    if (match) p.rating = match[1];
                }
                // Deduplicate by name
                if (p.name && p.name.length > 10 && !seen.has(p.name)) {
                    seen.add(p.name);
                    results.push(p);
                }
            }
            return results;
        }""")
        for item in dom_products:
            products.append(ProductData(
                name=item.get("name", ""),
                price=item.get("price", ""),
                image_url=item.get("image_url", ""),
                url=item.get("url", url),
                rating=item.get("rating", ""),
                source_domain=domain,
            ))
    except Exception as e:
        logger.debug("DOM extraction failed for %s: %s", url, e)

    # If DOM extraction found real products, prefer those over OG page-level entry
    if products:
        return products

    # Fall back to OpenGraph (even if it's just a page title — better than nothing)
    if og_product:
        return [og_product]

    return products


# ── Amazon-specific extractor ─────────────────────────────────


@register_extractor("amazon.in")
@register_extractor("amazon.com")
@register_extractor("amazon.co.uk")
@register_extractor("amazon.de")
@register_extractor("amazon.co.jp")
async def extract_amazon(page: Page, url: str) -> list[ProductData]:
    """Optimized extractor for Amazon search/product pages."""
    domain = urlparse(url).netloc.removeprefix("www.")

    raw_products = await page.evaluate(r"""() => {
        const cards = document.querySelectorAll('[data-component-type="s-search-result"]');
        const results = [];
        for (const card of [...cards].slice(0, 50)) {
            const p = {};
            p.asin = card.getAttribute('data-asin') || '';

            const img = card.querySelector('.s-image, img[data-image-latency]');
            if (img) {
                p.image_url = img.src || '';
                p.name = img.alt || '';
            }
            if (!p.name) {
                const titleEl = card.querySelector('h2 a span, .a-text-normal');
                if (titleEl) p.name = titleEl.textContent.trim();
            }

            const link = card.querySelector('h2 a, a.a-link-normal[href*="/dp/"]');
            if (link) {
                const href = link.href;
                const dpMatch = href.match(/\/dp\/([A-Z0-9]+)/);
                p.url = dpMatch
                    ? 'https://' + location.hostname + '/dp/' + dpMatch[1]
                    : href.split('?')[0];
            }

            const priceWhole = card.querySelector('.a-price:not([data-a-strike]) .a-price-whole');
            const priceFrac = card.querySelector('.a-price:not([data-a-strike]) .a-price-fraction');
            const priceSym = card.querySelector('.a-price:not([data-a-strike]) .a-price-symbol');
            if (priceWhole) {
                let price = priceWhole.textContent.replace(/[,.]/g, '').trim();
                if (priceFrac) price += '.' + priceFrac.textContent.trim();
                p.price = price;
            }
            if (priceSym) p.currency = priceSym.textContent.trim();

            const origPrice = card.querySelector('.a-price[data-a-strike] .a-offscreen');
            if (origPrice) {
                const match = origPrice.textContent.match(/[\d,]+\.?\d*/);
                if (match) p.original_price = match[0].replace(/,/g, '');
            }

            const ratingEl = card.querySelector('.a-icon-star-small .a-icon-alt, .a-icon-star .a-icon-alt');
            if (ratingEl) {
                const match = ratingEl.textContent.match(/(\d+\.?\d*)/);
                if (match) p.rating = match[1];
            }

            const reviewEl = card.querySelector('a[href*="#customerReviews"] span');
            if (reviewEl) {
                const text = reviewEl.textContent.replace(/,/g, '').trim();
                const match = text.match(/(\d+)/);
                if (match) p.review_count = match[1];
            }

            if (p.name && p.url) results.push(p);
        }

        // Single product page fallback
        if (results.length === 0) {
            const title = document.getElementById('productTitle');
            if (title) {
                const p = {};
                p.name = title.textContent.trim();
                p.url = location.href.split('?')[0];
                const img = document.getElementById('landingImage') || document.querySelector('#imgTagWrapperId img');
                if (img) p.image_url = img.src;
                const priceEl = document.querySelector('.a-price .a-offscreen');
                if (priceEl) p.price = priceEl.textContent.trim();
                results.push(p);
            }
        }

        return results;
    }""")

    products = []
    for item in raw_products:
        products.append(ProductData(
            name=item.get("name", ""),
            price=item.get("price", ""),
            currency=item.get("currency", ""),
            image_url=item.get("image_url", ""),
            rating=item.get("rating", ""),
            review_count=item.get("review_count", ""),
            url=item.get("url", url),
            original_price=item.get("original_price", ""),
            source_domain=domain,
        ))
    return products


@register_extractor("flipkart.com")
async def extract_flipkart(page: Page, url: str) -> list[ProductData]:
    """Optimized extractor for Flipkart search/product pages."""
    domain = "flipkart.com"

    raw_products = await page.evaluate(r"""() => {
        const results = [];
        // Flipkart uses _1AtVbE or similar dynamic classes — use structural selectors
        const cards = document.querySelectorAll('[data-id], a[href*="/p/"]');
        const seen = new Set();

        for (const card of [...cards].slice(0, 50)) {
            const link = card.tagName === 'A' ? card : card.querySelector('a[href*="/p/"]');
            if (!link) continue;
            const href = link.href;
            if (seen.has(href)) continue;
            seen.add(href);

            const p = {url: href};
            const img = card.querySelector('img[src*="http"]');
            if (img) {
                p.image_url = img.src;
                p.name = img.alt || '';
            }
            if (!p.name) {
                const titleEl = card.querySelector('a[title], div[class*="title"]');
                if (titleEl) p.name = (titleEl.getAttribute('title') || titleEl.textContent || '').trim();
            }

            // Price — Flipkart uses ₹ symbol in text
            const allText = card.textContent || '';
            const priceMatch = allText.match(/₹\s*([\d,]+)/);
            if (priceMatch) {
                p.price = priceMatch[1].replace(/,/g, '');
                p.currency = '₹';
            }

            const ratingEl = card.querySelector('div[class*="rating"] span, span[id*="rating"]');
            if (ratingEl) {
                const match = ratingEl.textContent.match(/(\d+\.?\d*)/);
                if (match) p.rating = match[1];
            }

            if (p.name && p.name.length > 5) results.push(p);
        }
        return results.slice(0, 50);
    }""")

    products = []
    for item in raw_products:
        products.append(ProductData(
            name=item.get("name", ""),
            price=item.get("price", ""),
            currency=item.get("currency", "₹"),
            image_url=item.get("image_url", ""),
            rating=item.get("rating", ""),
            url=item.get("url", url),
            source_domain=domain,
        ))
    return products


# ── Dispatcher ────────────────────────────────────────────────


async def extract_products(page: Page, url: str) -> list[ProductData]:
    """
    Extract products from a page. Uses site-specific extractor if available,
    falls back to generic extraction.
    """
    domain = urlparse(url).netloc.removeprefix("www.")

    # Check for site-specific extractor
    extractor = EXTRACTORS.get(domain)
    if extractor:
        try:
            products = await extractor(page, url)
            if products:
                logger.info("Site extractor for %s returned %d products", domain, len(products))
                return products
        except Exception as e:
            logger.warning("Site extractor for %s failed, falling back to generic: %s", domain, e)

    # Generic fallback
    products = await extract_generic(page, url)
    logger.info("Generic extractor for %s returned %d products", domain, len(products))
    return products
