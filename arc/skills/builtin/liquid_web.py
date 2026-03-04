"""
Liquid Web skill — parallel scrape → aggregate → render pipeline.

The agent calls `liquid_search` with a query, and this skill:
1. Searches via Tavily to find relevant product URLs
2. Scrapes them in parallel with headless Chromium
3. Deduplicates and renders a beautiful comparison UI
4. Serves it via ngrok and returns the public URL
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from arc.core.types import (
    Capability,
    SkillManifest,
    ToolResult,
    ToolSpec,
)
from arc.skills.base import Skill

logger = logging.getLogger(__name__)


class LiquidWebSkill(Skill):
    """Skill that provides Liquid Web — dynamic product comparison UI."""

    def __init__(self):
        self._config = None
        self._server = None

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="liquid_web",
            version="0.1.0",
            description=(
                "Search, compare, and browse products across the web. "
                "Renders a visual comparison page (3D carousel) served via a public URL. "
                "Use this for ANY request involving finding, comparing, shopping, or browsing products, "
                "gadgets, electronics, clothes, accessories, books, or any purchasable items."
            ),
            tools=(
                ToolSpec(
                    name="liquid_search",
                    description=(
                        "Search for products across the web and create a beautiful visual comparison page. "
                        "Use this tool whenever the user asks to find, search, compare, shop for, browse, or look up "
                        "ANY kind of product — electronics, gadgets, cameras, headphones, phones, laptops, appliances, "
                        "clothing, shoes, accessories, books, toys, furniture, or any purchasable item. "
                        "Also use this when the user says things like 'best X under Y price', 'find me a ...', "
                        "'compare ...', 'search for ...', 'show me options for ...', or 'what are good ...'. "
                        "The tool scrapes multiple shopping sites in parallel (Amazon, Flipkart, etc.), "
                        "extracts product data (name, price, rating, image), deduplicates, and renders "
                        "an interactive 3D carousel UI accessible via a public URL (works on mobile, Telegram, any browser). "
                        "Do NOT use browser_go or web_search for product queries — use this tool instead."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Product search query (e.g., 'best mirrorless camera under $800')",
                            },
                            "country": {
                                "type": "string",
                                "description": "Full country name to boost localized results (e.g., 'india', 'united states', 'united kingdom', 'germany'). Use lowercase. Infer from user's context.",
                            },
                            "include_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Optional list of domains to search within "
                                    "(e.g., ['amazon.in', 'flipkart.com']). "
                                    "If omitted, searches all relevant sites."
                                ),
                            },
                            "exclude_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of domains to exclude from results.",
                            },
                            "search_depth": {
                                "type": "string",
                                "enum": ["basic", "advanced"],
                                "description": "Search depth — 'basic' (faster, 1 credit) or 'advanced' (better relevance, 2 credits). Default 'basic'.",
                            },
                        },
                        "required": ["query"],
                    },
                    required_capabilities=frozenset([Capability.NETWORK_HTTP, Capability.BROWSER]),
                ),
            ),
            capabilities=(Capability.NETWORK_HTTP, Capability.BROWSER),
        )

    async def initialize(self, kernel: Any = None, config: Any = None) -> None:
        # kernel.config is the ArcConfig with tavily/ngrok sections
        if kernel and hasattr(kernel, "config"):
            self._config = kernel.config
        else:
            self._config = config

    async def activate(self) -> None:
        pass

    async def execute_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        if tool_name != "liquid_search":
            return ToolResult(
                tool_call_id="",
                success=False,
                output=f"Unknown tool: {tool_name}",
            )

        query = arguments.get("query", "")
        if not query:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="Missing required parameter: query",
            )

        country = arguments.get("country")
        include_domains = arguments.get("include_domains")
        exclude_domains = arguments.get("exclude_domains")
        search_depth = arguments.get("search_depth", "basic")

        try:
            result = await self._run_pipeline(
                query, country, include_domains, exclude_domains, search_depth,
            )
            return ToolResult(tool_call_id="", success=True, output=result)
        except Exception as e:
            logger.exception("Liquid Web pipeline failed")
            return ToolResult(
                tool_call_id="",
                success=False,
                output=f"Liquid Web failed: {e}",
            )

    async def _run_pipeline(
        self,
        query: str,
        country: str | None,
        include_domains: list[str] | None,
        exclude_domains: list[str] | None,
        search_depth: str,
    ) -> str:
        """Execute the full Liquid Web pipeline."""
        from arc.liquid.tavily import tavily_search
        from arc.liquid.pool import BrowserPool
        from arc.liquid.extract import filter_quality_products
        from arc.liquid.renderer import render_products
        from arc.liquid.server import LiquidServer

        # 1. Get API keys from config
        tavily_key = ""
        ngrok_token = ""
        if self._config:
            tavily_key = getattr(getattr(self._config, "tavily", None), "api_key", "")
            ngrok_token = getattr(getattr(self._config, "ngrok", None), "auth_token", "")

        if not tavily_key:
            return (
                "Tavily API key not configured. "
                "Set ARC_TAVILY_API_KEY environment variable or add [tavily] api_key to config.toml"
            )

        # 2. Search via Tavily
        # Append "buy" to the query to bias results toward e-commerce pages
        # (blog/review pages rarely have extractable structured product data)
        search_query = query
        buy_keywords = {"buy", "price", "shop", "purchase", "order", "deal"}
        query_lower = query.lower()
        if not any(kw in query_lower for kw in buy_keywords):
            search_query = f"{query} buy price"

        logger.info("Liquid Web: searching for '%s' (original: '%s')", search_query, query)
        search_results = await tavily_search(
            query=search_query,
            api_key=tavily_key,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            country=country,
            search_depth=search_depth,
            max_results=10,
        )

        if not search_results:
            return f"No search results found for: {query}"

        urls = [r.url for r in search_results]
        logger.info("Liquid Web: extracting from %d URLs: %s", len(urls), urls)

        # 3. Extract products in parallel
        pool = BrowserPool(max_concurrent=5, timeout_per_url=30.0)
        try:
            extraction_results = await pool.extract_all(urls, overall_timeout=90.0)
        finally:
            await pool.close()

        # 4. Collect and deduplicate products
        all_products = []
        source_domains = set()
        for result in extraction_results:
            if result.error:
                logger.warning(
                    "Liquid Web: extraction error for %s: %s",
                    result.url, result.error,
                )
            for product in result.products:
                all_products.append(product)
                if product.source_domain:
                    source_domains.add(product.source_domain)

        logger.info(
            "Liquid Web: extraction summary — %d/%d URLs yielded products, %d total products",
            sum(1 for r in extraction_results if r.products),
            len(extraction_results),
            len(all_products),
        )

        if not all_products:
            # Provide diagnostic info about what went wrong
            errors = [r.error for r in extraction_results if r.error]
            error_detail = "; ".join(errors[:3]) if errors else "pages had no extractable product data"
            return (
                f"Found {len(search_results)} pages but could not extract any product data. "
                f"Details: {error_detail}"
            )

        # Filter out low-quality entries (blog page titles with no price/image)
        quality_products = filter_quality_products(all_products)
        logger.info(
            "Liquid Web: quality filter — %d → %d products",
            len(all_products), len(quality_products),
        )

        # Deduplicate by name similarity
        unique_products = _deduplicate(quality_products)
        logger.info(
            "Liquid Web: %d products from %d sources (after dedup: %d)",
            len(all_products), len(source_domains), len(unique_products),
        )

        # 5. Render HTML
        html = render_products(
            query=query,
            products=unique_products[:20],  # Cap at 20 for UI performance
            sources=sorted(source_domains),
        )

        # 6. Serve and get public URL
        # Close previous server if any
        if self._server:
            await self._server.stop()

        self._server = LiquidServer(
            ngrok_auth_token=ngrok_token,
            auto_open=True,
            shutdown_timeout=600.0,
        )
        public_url = await self._server.start(html)

        return (
            f"Found {len(unique_products)} products from {len(source_domains)} sites. "
            f"Comparison page is live at: {public_url}\n\n"
            "IMPORTANT: The page has already been opened in the user's browser automatically. "
            "Do NOT try to browse, open, navigate to, or search for this URL. "
            "Your task is complete — just tell the user about the results and the link."
        )

    async def deactivate(self) -> None:
        if self._server:
            await self._server.stop()
            self._server = None

    async def shutdown(self) -> None:
        await self.deactivate()


def _deduplicate(products: list) -> list:
    """Remove near-duplicate products based on name similarity."""
    seen_names: list[str] = []
    unique = []

    for p in products:
        name = p.name.lower().strip()
        if not name:
            continue

        is_dup = False
        for seen in seen_names:
            # Simple overlap check: if >60% of words match, it's a duplicate
            words_a = set(name.split())
            words_b = set(seen.split())
            if not words_a or not words_b:
                continue
            overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
            if overlap > 0.6:
                is_dup = True
                break

        if not is_dup:
            seen_names.append(name)
            unique.append(p)

    return unique
