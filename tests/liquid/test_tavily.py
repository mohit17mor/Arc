"""Tests for Tavily search integration."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from arc.liquid.tavily import tavily_search, SearchResult


@pytest.fixture
def mock_tavily_response():
    """Sample Tavily API response."""
    return {
        "results": [
            {
                "url": "https://www.amazon.in/dp/B07MVCPZ1Q",
                "title": "Sony Alpha a6400 Camera",
                "content": "Great mirrorless camera for beginners",
                "score": 0.95,
            },
            {
                "url": "https://www.flipkart.com/sony-camera/p/itm123",
                "title": "Sony Camera - Flipkart",
                "content": "Buy Sony cameras online",
                "score": 0.87,
            },
            {
                "url": "",  # Empty URL — should be filtered
                "title": "Empty result",
                "content": "Should be skipped",
                "score": 0.1,
            },
        ]
    }


async def test_tavily_search_returns_results(mock_tavily_response):
    """Test that tavily_search parses response correctly."""
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value=mock_tavily_response)

    with patch("tavily.AsyncTavilyClient", return_value=mock_client):
        results = await tavily_search("best camera", api_key="test-key")

    assert len(results) == 2  # Empty URL filtered out
    assert results[0].url == "https://www.amazon.in/dp/B07MVCPZ1Q"
    assert results[0].domain == "amazon.in"
    assert results[0].title == "Sony Alpha a6400 Camera"
    assert results[1].domain == "flipkart.com"


async def test_tavily_search_passes_filters(mock_tavily_response):
    """Test that domain and country filters are passed to Tavily."""
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value=mock_tavily_response)

    with patch("tavily.AsyncTavilyClient", return_value=mock_client):
        await tavily_search(
            "camera",
            api_key="test-key",
            include_domains=["amazon.in"],
            country="india",
        )

    call_kwargs = mock_client.search.call_args[1]
    assert call_kwargs["include_domains"] == ["amazon.in"]
    assert call_kwargs["country"] == "india"
    assert call_kwargs["search_depth"] == "basic"
    assert call_kwargs["topic"] == "general"


async def test_tavily_search_handles_api_error():
    """Test graceful handling of Tavily API errors."""
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(side_effect=Exception("API rate limited"))

    with patch("tavily.AsyncTavilyClient", return_value=mock_client):
        results = await tavily_search("camera", api_key="bad-key")

    assert results == []


async def test_tavily_search_empty_results():
    """Test handling of empty results."""
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value={"results": []})

    with patch("tavily.AsyncTavilyClient", return_value=mock_client):
        results = await tavily_search("nonexistent product", api_key="test-key")

    assert results == []
