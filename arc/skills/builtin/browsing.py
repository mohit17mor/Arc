"""
Browsing Skill — web search and page reading without a browser.

Tools:
    web_search(query, num_results)  → structured search results
    web_read(url, max_length)       → clean page content via Jina Reader

Design notes:
    - Entirely stateless between calls (safe for multi-agent use)
    - No browser, no vision, no screenshots
    - httpx client created at activate(), closed at shutdown()
    - Output format is structured markdown so downstream agents/LLMs
      can parse it reliably (important for future pipeline orchestration)
    - max_content_length is configurable so a routing agent can cap
      how much text flows between hops
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
from ddgs import DDGS

from arc.core.types import Capability, SkillManifest, ToolResult, ToolSpec
from arc.skills.base import Skill

logger = logging.getLogger(__name__)

# Jina Reader — converts any URL to clean markdown for free
JINA_BASE = "https://r.jina.ai/"

# Default limits
DEFAULT_NUM_RESULTS = 5
DEFAULT_MAX_CONTENT = 8000   # chars — keeps token cost predictable
MAX_NUM_RESULTS = 10


class BrowsingSkill(Skill):
    """
    Skill for searching the web and reading pages.

    Designed to be stateless so multiple agents can share the same
    instance, or separate instances can run in parallel without
    interference. All state lives in the httpx client (just a
    connection pool) — no per-call state is stored on self.
    """

    def __init__(
        self,
        max_content_length: int = DEFAULT_MAX_CONTENT,
        timeout: float = 20.0,
    ) -> None:
        self._max_content_length = max_content_length
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._ddgs: DDGS | None = None  # one persistent session — avoids per-query rate limiting

    # ━━━ Lifecycle ━━━

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="browsing",
            version="1.0.0",
            description="Search the web and read pages",
            capabilities=frozenset([Capability.NETWORK_HTTP]),
            tools=(
                ToolSpec(
                    name="web_search",
                    description=(
                        "Search the web and return a list of relevant results. "
                        "Use this to find pages, articles, prices, comparisons, etc. "
                        "Returns title, URL, and a short snippet for each result. "
                        "IMPORTANT: Use this at most 1-2 times per user request. "
                        "Once you have relevant URLs from the snippets, switch to "
                        "web_read or http_get — do not keep searching."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                            "num_results": {
                                "type": "integer",
                                "description": (
                                    f"Number of results to return "
                                    f"(default {DEFAULT_NUM_RESULTS}, max {MAX_NUM_RESULTS})"
                                ),
                                "default": DEFAULT_NUM_RESULTS,
                            },
                        },
                        "required": ["query"],
                    },
                    required_capabilities=frozenset([Capability.NETWORK_HTTP]),
                ),
                ToolSpec(
                    name="web_read",
                    description=(
                        "Fetch and read the content of a web page as clean text. "
                        "Use this after web_search to get full content from a URL. "
                        "Works on articles, product pages, docs, news, etc. "
                        "Does NOT work on pages that require login. "
                        "IMPORTANT: Read at most 2-3 pages per user request, then "
                        "synthesize what you found and answer — do not keep reading more pages."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The full URL to read",
                            },
                            "max_length": {
                                "type": "integer",
                                "description": (
                                    f"Max characters to return (default {DEFAULT_MAX_CONTENT}). "
                                    f"Use smaller values when you only need a summary."
                                ),
                                "default": DEFAULT_MAX_CONTENT,
                            },
                        },
                        "required": ["url"],
                    },
                    required_capabilities=frozenset([Capability.NETWORK_HTTP]),
                ),
                ToolSpec(
                    name="http_get",
                    description=(
                        "Make a raw HTTP GET request to a URL and return the response body as-is. "
                        "Use this for JSON APIs, data endpoints, RSS feeds, or any URL that "
                        "returns structured data rather than a human-readable page. "
                        "Examples: crypto prices, weather APIs, exchange rates, stock data. "
                        "Do NOT use this for regular web pages — use web_read instead."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The full URL to request",
                            },
                            "headers": {
                                "type": "object",
                                "description": (
                                    "Optional extra HTTP headers as key-value pairs. "
                                    "Use for APIs that require Accept: application/json etc."
                                ),
                            },
                            "max_length": {
                                "type": "integer",
                                "description": (
                                    f"Max characters to return (default {DEFAULT_MAX_CONTENT}). "
                                    f"Useful for large API responses where you only need part."
                                ),
                                "default": DEFAULT_MAX_CONTENT,
                            },
                        },
                        "required": ["url"],
                    },
                    required_capabilities=frozenset([Capability.NETWORK_HTTP]),
                ),
            ),
        )

    async def activate(self) -> None:
        """Create the shared httpx client and DDG session (lazy init)."""
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers={
                # Look like a real browser to avoid 403s
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
            follow_redirects=True,
        )
        # One persistent DDGS session reused across all queries.
        # Creating a new DDGS() per query causes immediate rate-limiting
        # because DDG sees repeated session handshakes from the same IP.
        self._ddgs = DDGS()
        logger.debug("BrowsingSkill activated")

    async def shutdown(self) -> None:
        """Close the httpx client and DDG session cleanly."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._ddgs = None
        logger.debug("BrowsingSkill shut down")

    # ━━━ Tool routing ━━━

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        # Safety: ensure client exists (in case activate() wasn't called)
        if self._client is None:
            await self.activate()

        try:
            if tool_name == "web_search":
                return await self._web_search(
                    query=arguments["query"],
                    num_results=min(
                        int(arguments.get("num_results", DEFAULT_NUM_RESULTS)),
                        MAX_NUM_RESULTS,
                    ),
                )
            elif tool_name == "web_read":
                return await self._web_read(
                    url=arguments["url"],
                    max_length=int(
                        arguments.get("max_length", self._max_content_length)
                    ),
                )
            elif tool_name == "http_get":
                return await self._http_get(
                    url=arguments["url"],
                    headers=arguments.get("headers"),
                    max_length=int(
                        arguments.get("max_length", self._max_content_length)
                    ),
                )
            else:
                return ToolResult(
                    tool_call_id="",
                    success=False,
                    output="",
                    error=f"Unknown tool: {tool_name}",
                )
        except Exception as e:
            logger.exception(f"BrowsingSkill.{tool_name} failed")
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=str(e),
            )

    # ━━━ Implementations ━━━

    async def _web_search(self, query: str, num_results: int) -> ToolResult:
        """
        Search DuckDuckGo via the duckduckgo-search library.

        DDGS is synchronous (uses primp under the hood) so we run it in
        a thread-pool executor to avoid blocking the event loop.
        No API key, no cost. Future: swap backend by replacing this method
        without changing the tool interface or callers.
        """
        try:
            ddgs = self._ddgs
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: list(ddgs.text(query, max_results=num_results)),
            )
        except Exception as e:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Search request failed: {e}",
            )

        if not raw:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error="No results found. Try rephrasing the query.",
            )

        # Normalise to our internal format (library uses 'href'/'body')
        results = [
            {"title": r["title"], "url": r["href"], "snippet": r.get("body", "")}
            for r in raw
        ]

        # Format as structured text — easy for LLM to parse, easy for
        # a future orchestrator to extract URLs from for parallel reads
        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            if r["snippet"]:
                lines.append(f"   {r['snippet']}")
            lines.append("")

        return ToolResult(
            tool_call_id="",
            success=True,
            output="\n".join(lines).strip(),
            # Store raw structured data in artifacts for future pipeline use
            # (orchestrators can extract URLs without re-parsing text)
            artifacts=[json.dumps(results)],
        )

    async def _web_read(self, url: str, max_length: int) -> ToolResult:
        """
        Fetch page content via Jina Reader (r.jina.ai).

        Jina strips ads, nav, and boilerplate — returns just the main
        content as clean markdown. No JS execution required for most pages.

        Falls back with a descriptive error if Jina fails (e.g. paywalled,
        login required) so the LLM can explain the issue to the user.
        """
        jina_url = f"{JINA_BASE}{url}"

        try:
            response = await self._client.get(
                jina_url,
                headers={
                    # Ask Jina for plain text (faster, smaller)
                    "Accept": "text/plain",
                    "X-Return-Format": "markdown",
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                return ToolResult(
                    tool_call_id="",
                    success=False,
                    output="",
                    error=(
                        f"Cannot read this page (likely requires login or is paywalled): {url}"
                    ),
                )
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Failed to fetch {url}: HTTP {e.response.status_code}",
            )
        except httpx.HTTPError as e:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Network error reading {url}: {e}",
            )

        content = response.text.strip()

        if not content:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Page returned empty content: {url}",
            )

        # Truncate if needed — always truncate cleanly at a line boundary
        if len(content) > max_length:
            content = _truncate_at_line(content, max_length)
            content += (
                f"\n\n[Content truncated at {max_length} chars. "
                f"Use a larger max_length or read a specific section.]"
            )

        return ToolResult(
            tool_call_id="",
            success=True,
            output=content,
            artifacts=[url],  # source URL for citation
        )

    async def _http_get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        max_length: int = DEFAULT_MAX_CONTENT,
    ) -> ToolResult:
        """
        Raw HTTP GET — no Jina, no processing, response body returned as-is.

        Ideal for JSON APIs where Jina's HTML-to-markdown conversion would
        corrupt the data. The LLM receives the raw response and can parse
        JSON, XML, CSV, or whatever the API returns directly.
        """
        request_headers: dict[str, str] = {}
        if headers:
            request_headers.update(headers)

        try:
            response = await self._client.get(url, headers=request_headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"HTTP {e.response.status_code} from {url}",
            )
        except httpx.HTTPError as e:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Network error fetching {url}: {e}",
            )

        body = response.text.strip()

        if not body:
            return ToolResult(
                tool_call_id="",
                success=False,
                output="",
                error=f"Empty response from {url}",
            )

        if len(body) > max_length:
            body = _truncate_at_line(body, max_length)
            body += f"\n\n[Response truncated at {max_length} chars.]"

        return ToolResult(
            tool_call_id="",
            success=True,
            output=body,
            artifacts=[url],
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers (module-level so they're testable without instantiating the skill)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _truncate_at_line(text: str, max_length: int) -> str:
    """Truncate text at the last complete line before max_length."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_newline = truncated.rfind("\n")
    # Only cut at newline if it's not too far back (> 80% of max_length)
    if last_newline > max_length * 0.8:
        return truncated[:last_newline]
    return truncated
