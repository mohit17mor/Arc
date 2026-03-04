"""
Tavily search integration for Liquid Web.

Searches the web via Tavily API and returns relevant product URLs
for parallel extraction.

API Reference: https://docs.tavily.com/sdk/python/reference
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from Tavily."""

    url: str
    title: str
    snippet: str
    domain: str
    score: float = 0.0


async def tavily_search(
    query: str,
    api_key: str,
    *,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    country: str | None = None,
    topic: str = "general",
    search_depth: str = "basic",
    max_results: int = 10,
    time_range: str | None = None,
) -> list[SearchResult]:
    """
    Search via Tavily and return structured results.

    Args:
        query: Search query (e.g., "best mirrorless camera under $800").
        api_key: Tavily API key.
        include_domains: Domains to restrict results to (max 300).
        exclude_domains: Domains to exclude from results (max 150).
        country: Full country name to boost results from (e.g., "india",
                 "united states", "united kingdom"). Only works when topic
                 is "general".
        topic: Search category — "general", "news", or "finance".
        search_depth: "basic" (1 credit) or "advanced" (2 credits, better relevance).
        max_results: Maximum results to return (1-20).
        time_range: Filter by recency — "day", "week", "month", or "year".

    Returns:
        List of SearchResult with URLs suitable for product extraction.
    """
    from tavily import AsyncTavilyClient

    client = AsyncTavilyClient(api_key=api_key)

    kwargs: dict = {
        "query": query,
        "max_results": min(max_results, 20),
        "search_depth": search_depth,
        "topic": topic,
    }
    if include_domains:
        kwargs["include_domains"] = include_domains
    if exclude_domains:
        kwargs["exclude_domains"] = exclude_domains
    if country and topic == "general":
        kwargs["country"] = country
    if time_range:
        kwargs["time_range"] = time_range

    try:
        response = await client.search(**kwargs)
    except Exception as e:
        logger.error("Tavily search failed: %s", e)
        return []

    results: list[SearchResult] = []
    for item in response.get("results", []):
        url = item.get("url", "")
        if not url:
            continue

        domain = ""
        try:
            domain = urlparse(url).netloc.removeprefix("www.")
        except Exception:
            pass

        results.append(
            SearchResult(
                url=url,
                title=item.get("title", ""),
                snippet=item.get("content", "")[:300],
                domain=domain,
                score=item.get("score", 0.0),
            )
        )

    logger.info("Tavily returned %d results for: %s", len(results), query)
    return results
