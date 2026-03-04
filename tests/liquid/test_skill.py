"""Tests for LiquidWebSkill."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from arc.skills.builtin.liquid_web import LiquidWebSkill
from arc.liquid.extract import ProductData
from arc.liquid.tavily import SearchResult
from arc.liquid.pool import ExtractionResult


@pytest.fixture
def skill(config):
    """Create a LiquidWebSkill with config."""
    config.tavily.api_key = "test-tavily-key"
    config.ngrok.auth_token = ""  # No ngrok in tests
    s = LiquidWebSkill()
    s._config = config
    return s


async def test_skill_manifest():
    """Test that the skill manifest is correctly defined."""
    skill = LiquidWebSkill()
    manifest = skill.manifest()

    assert manifest.name == "liquid_web"
    assert len(manifest.tools) == 1
    assert manifest.tools[0].name == "liquid_search"

    params = manifest.tools[0].parameters
    assert "query" in params["properties"]
    assert "country" in params["properties"]
    assert "include_domains" in params["properties"]
    assert "query" in params["required"]


async def test_skill_missing_query(skill):
    """Test error when query is missing."""
    result = await skill.execute_tool("liquid_search", {})
    assert not result.success
    assert "Missing" in result.output


async def test_skill_unknown_tool(skill):
    """Test error for unknown tool name."""
    result = await skill.execute_tool("unknown_tool", {"query": "test"})
    assert not result.success
    assert "Unknown" in result.output


async def test_skill_missing_api_key(config):
    """Test error when Tavily API key is not configured."""
    config.tavily.api_key = ""
    skill = LiquidWebSkill()
    skill._config = config

    result = await skill.execute_tool("liquid_search", {"query": "camera"})
    assert result.success  # Tool call succeeds, but output explains the issue
    assert "Tavily API key" in result.output


async def test_skill_full_pipeline(skill):
    """Test the full pipeline with mocked dependencies."""
    mock_search_results = [
        SearchResult(url="https://amazon.in/dp/B123", title="Camera", snippet="", domain="amazon.in"),
        SearchResult(url="https://flipkart.com/cam/p/1", title="Camera", snippet="", domain="flipkart.com"),
    ]

    mock_extraction_results = [
        ExtractionResult(
            url="https://amazon.in/dp/B123",
            products=[
                ProductData(name="Sony A6400", price="799", source_domain="amazon.in"),
            ],
        ),
        ExtractionResult(
            url="https://flipkart.com/cam/p/1",
            products=[
                ProductData(name="Canon R50", price="679", source_domain="flipkart.com"),
            ],
        ),
    ]

    with patch("arc.liquid.tavily.tavily_search", new_callable=AsyncMock) as mock_tavily, \
         patch("arc.liquid.pool.BrowserPool") as MockPool, \
         patch("arc.liquid.server.LiquidServer") as MockServer:

        mock_tavily.return_value = mock_search_results

        mock_pool_instance = AsyncMock()
        mock_pool_instance.extract_all = AsyncMock(return_value=mock_extraction_results)
        MockPool.return_value = mock_pool_instance

        mock_server_instance = AsyncMock()
        mock_server_instance.start = AsyncMock(return_value="https://abc123.ngrok-free.app")
        MockServer.return_value = mock_server_instance

        result = await skill.execute_tool("liquid_search", {
            "query": "best camera under $800",
            "country": "india",
        })

    assert result.success
    assert "2 products" in result.output
    assert "2 sites" in result.output
    assert "https://abc123.ngrok-free.app" in result.output

    # Verify Tavily was called with right params
    mock_tavily.assert_called_once()
    call_kwargs = mock_tavily.call_args[1]
    assert call_kwargs["country"] == "india"


async def test_skill_no_search_results(skill):
    """Test handling when Tavily returns no results."""
    with patch("arc.liquid.tavily.tavily_search", new_callable=AsyncMock) as mock_tavily:
        mock_tavily.return_value = []

        result = await skill.execute_tool("liquid_search", {"query": "nonexistent"})

    assert result.success  # Not a tool error, just no results
    assert "No search results" in result.output


async def test_skill_no_products_extracted(skill):
    """Test handling when extraction yields no products."""
    with patch("arc.liquid.tavily.tavily_search", new_callable=AsyncMock) as mock_tavily, \
         patch("arc.liquid.pool.BrowserPool") as MockPool:

        mock_tavily.return_value = [
            SearchResult(url="https://example.com", title="Blog", snippet="", domain="example.com"),
        ]
        mock_pool = AsyncMock()
        mock_pool.extract_all = AsyncMock(return_value=[
            ExtractionResult(url="https://example.com", products=[]),
        ])
        MockPool.return_value = mock_pool

        result = await skill.execute_tool("liquid_search", {"query": "camera"})

    assert "could not extract" in result.output


async def test_skill_shutdown_stops_server(skill):
    """Test that shutdown cleans up the server."""
    mock_server = AsyncMock()
    skill._server = mock_server

    await skill.shutdown()

    mock_server.stop.assert_called_once()
    assert skill._server is None
