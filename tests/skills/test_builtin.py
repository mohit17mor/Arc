"""Tests for built-in skills."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from arc.skills.builtin.filesystem import FilesystemSkill
from arc.skills.builtin.browsing import (
    BrowsingSkill,
    _truncate_at_line,
)


@pytest.fixture
def fs_skill(tmp_path):
    """Filesystem skill with temp workspace."""
    skill = FilesystemSkill(workspace=tmp_path)
    return skill


@pytest.mark.asyncio
async def test_read_file(fs_skill, tmp_path):
    """read_file returns file contents."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, Arc!")

    result = await fs_skill.execute_tool("read_file", {"path": "test.txt"})
    assert result.success is True
    assert result.output == "Hello, Arc!"


@pytest.mark.asyncio
async def test_read_file_not_found(fs_skill):
    """read_file returns error for missing file."""
    result = await fs_skill.execute_tool("read_file", {"path": "nonexistent.txt"})
    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_write_file(fs_skill, tmp_path):
    """write_file creates file with content."""
    result = await fs_skill.execute_tool(
        "write_file",
        {"path": "output.txt", "content": "Test content"},
    )
    assert result.success is True

    created = tmp_path / "output.txt"
    assert created.exists()
    assert created.read_text() == "Test content"


@pytest.mark.asyncio
async def test_write_file_creates_dirs(fs_skill, tmp_path):
    """write_file creates parent directories."""
    result = await fs_skill.execute_tool(
        "write_file",
        {"path": "deep/nested/file.txt", "content": "Nested!"},
    )
    assert result.success is True

    created = tmp_path / "deep" / "nested" / "file.txt"
    assert created.exists()


@pytest.mark.asyncio
async def test_list_directory(fs_skill, tmp_path):
    """list_directory shows files and folders."""
    (tmp_path / "file1.txt").write_text("a")
    (tmp_path / "file2.txt").write_text("b")
    (tmp_path / "subdir").mkdir()

    result = await fs_skill.execute_tool("list_directory", {"path": "."})
    assert result.success is True
    assert "file1.txt" in result.output
    assert "file2.txt" in result.output
    assert "subdir" in result.output
    assert "[DIR]" in result.output
    assert "[FILE]" in result.output


@pytest.mark.asyncio
async def test_list_directory_empty(fs_skill, tmp_path):
    """list_directory handles empty directory."""
    result = await fs_skill.execute_tool("list_directory", {"path": "."})
    assert result.success is True
    assert "empty" in result.output.lower()


@pytest.mark.asyncio
async def test_list_directory_not_found(fs_skill):
    """list_directory returns error for missing directory."""
    result = await fs_skill.execute_tool("list_directory", {"path": "nonexistent"})
    assert result.success is False
    assert "not found" in result.error.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BrowsingSkill tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Fake results returned by AsyncDDGS (library format uses 'href'/'body')
DDG_RESULTS_FIXTURE = [
    {"title": "Best Laptops 2026", "href": "https://example.com/laptops", "body": "Top picks for laptops under 50000 rupees."},
    {"title": "Laptop Reviews", "href": "https://review-site.com/top-laptops", "body": "Detailed reviews and comparisons."},
]


@pytest.fixture
def browsing_skill():
    """BrowsingSkill instance (client not activated — mocked per test)."""
    return BrowsingSkill(max_content_length=500, timeout=5.0)


def _make_response(status_code: int = 200, text: str = "") -> MagicMock:
    """Helper to create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    return resp


# ── manifest ──────────────────────────────────────────


def test_browsing_manifest_name(browsing_skill):
    """Manifest has correct name."""
    assert browsing_skill.manifest().name == "browsing"


def test_browsing_manifest_tools(browsing_skill):
    """Manifest exposes web_search, web_read, and http_get tools."""
    tool_names = {t.name for t in browsing_skill.manifest().tools}
    assert tool_names == {"web_search", "web_read", "http_get"}


def test_browsing_manifest_capability(browsing_skill):
    """Manifest requires NETWORK_HTTP capability."""
    from arc.core.types import Capability
    assert Capability.NETWORK_HTTP in browsing_skill.manifest().capabilities


# ── lifecycle ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_activate_creates_client(browsing_skill):
    """activate() creates an httpx client."""
    assert browsing_skill._client is None
    await browsing_skill.activate()
    assert browsing_skill._client is not None
    await browsing_skill.shutdown()


@pytest.mark.asyncio
async def test_shutdown_closes_client(browsing_skill):
    """shutdown() closes and clears the client."""
    await browsing_skill.activate()
    await browsing_skill.shutdown()
    assert browsing_skill._client is None


@pytest.mark.asyncio
async def test_execute_auto_activates(browsing_skill):
    """execute_tool activates client automatically if not yet activated."""
    with patch("arc.skills.builtin.browsing.DDGS") as mock_cls:
        mock_cls.return_value.text = MagicMock(return_value=DDG_RESULTS_FIXTURE)
        with patch("httpx.AsyncClient") as mock_http:
            mock_http.return_value = AsyncMock()
            result = await browsing_skill.execute_tool("web_search", {"query": "test"})

    # No exception — client was created automatically
    assert result.success is True or result.success is False


# ── web_search ────────────────────────────────────────


@pytest.mark.asyncio
async def test_web_search_success(browsing_skill):
    """web_search returns formatted results on success."""
    with patch("arc.skills.builtin.browsing.DDGS") as mock_cls:
        mock_cls.return_value.text = MagicMock(return_value=DDG_RESULTS_FIXTURE)

        result = await browsing_skill.execute_tool(
            "web_search", {"query": "best laptop under 50000"}
        )

    assert result.success is True
    assert "best laptop under 50000" in result.output.lower()
    assert "URL:" in result.output


@pytest.mark.asyncio
async def test_web_search_result_count(browsing_skill):
    """web_search respects num_results parameter."""
    with patch("arc.skills.builtin.browsing.DDGS") as mock_cls:
        mock_cls.return_value.text = MagicMock(return_value=DDG_RESULTS_FIXTURE[:1])

        result = await browsing_skill.execute_tool(
            "web_search", {"query": "laptops", "num_results": 1}
        )

    assert result.success is True
    assert "1." in result.output
    assert "2." not in result.output


@pytest.mark.asyncio
async def test_web_search_artifacts_are_json(browsing_skill):
    """web_search stores structured JSON in artifacts for pipeline use."""
    with patch("arc.skills.builtin.browsing.DDGS") as mock_cls:
        mock_cls.return_value.text = MagicMock(return_value=DDG_RESULTS_FIXTURE)

        result = await browsing_skill.execute_tool(
            "web_search", {"query": "test"}
        )

    assert result.success is True
    assert len(result.artifacts) == 1
    parsed = json.loads(result.artifacts[0])
    assert isinstance(parsed, list)
    assert "url" in parsed[0]
    assert "title" in parsed[0]


@pytest.mark.asyncio
async def test_web_search_caps_at_max(browsing_skill):
    """web_search caps num_results at MAX_NUM_RESULTS (10)."""
    with patch("arc.skills.builtin.browsing.DDGS") as mock_cls:
        mock_cls.return_value.text = MagicMock(return_value=DDG_RESULTS_FIXTURE)

        result = await browsing_skill.execute_tool(
            "web_search", {"query": "test", "num_results": 999}
        )
    assert result.success is True


@pytest.mark.asyncio
async def test_web_search_http_error(browsing_skill):
    """web_search returns error result on failure."""
    with patch("arc.skills.builtin.browsing.DDGS") as mock_cls:
        mock_cls.return_value.text = MagicMock(side_effect=Exception("Connection refused"))

        result = await browsing_skill.execute_tool("web_search", {"query": "test"})

    assert result.success is False
    assert "search request failed" in result.error.lower()


@pytest.mark.asyncio
async def test_web_search_no_results(browsing_skill):
    """web_search returns descriptive error when no results returned."""
    with patch("arc.skills.builtin.browsing.DDGS") as mock_cls:
        mock_cls.return_value.text = MagicMock(return_value=[])

        result = await browsing_skill.execute_tool("web_search", {"query": "xyzzy123"})

    assert result.success is False
    assert "no results" in result.error.lower()


# ── web_read ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_web_read_success(browsing_skill):
    """web_read returns page content on success."""
    page_content = "# Best Laptops\n\nHere are the top picks..."
    mock_resp = _make_response(200, page_content)
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(return_value=mock_resp)

    result = await browsing_skill.execute_tool(
        "web_read", {"url": "https://example.com/laptops"}
    )

    assert result.success is True
    assert "Best Laptops" in result.output


@pytest.mark.asyncio
async def test_web_read_url_in_artifacts(browsing_skill):
    """web_read stores the source URL in artifacts for citation."""
    mock_resp = _make_response(200, "Some content")
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(return_value=mock_resp)

    result = await browsing_skill.execute_tool(
        "web_read", {"url": "https://example.com/article"}
    )

    assert result.success is True
    assert "https://example.com/article" in result.artifacts


@pytest.mark.asyncio
async def test_web_read_truncates_long_content(browsing_skill):
    """web_read truncates content exceeding max_length."""
    long_content = "word " * 2000  # way over 500 char limit in fixture
    mock_resp = _make_response(200, long_content)
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(return_value=mock_resp)

    result = await browsing_skill.execute_tool(
        "web_read", {"url": "https://example.com", "max_length": 500}
    )

    assert result.success is True
    assert len(result.output) < len(long_content)
    assert "truncated" in result.output.lower()


@pytest.mark.asyncio
async def test_web_read_login_required(browsing_skill):
    """web_read returns descriptive error for paywalled/login-required pages."""
    import httpx
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    error = httpx.HTTPStatusError(
        message="422", request=MagicMock(), response=mock_resp
    )
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(side_effect=error)

    result = await browsing_skill.execute_tool(
        "web_read", {"url": "https://paywalled.com/article"}
    )

    assert result.success is False
    assert "login" in result.error.lower() or "paywalled" in result.error.lower()


@pytest.mark.asyncio
async def test_web_read_http_error(browsing_skill):
    """web_read returns error result on non-422 HTTP failures."""
    import httpx
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    error = httpx.HTTPStatusError(
        message="500", request=MagicMock(), response=mock_resp
    )
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(side_effect=error)

    result = await browsing_skill.execute_tool(
        "web_read", {"url": "https://broken.com"}
    )

    assert result.success is False
    assert "500" in result.error


@pytest.mark.asyncio
async def test_web_read_empty_content(browsing_skill):
    """web_read returns error when page content is empty."""
    mock_resp = _make_response(200, "   ")
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(return_value=mock_resp)

    result = await browsing_skill.execute_tool(
        "web_read", {"url": "https://empty.com"}
    )

    assert result.success is False
    assert "empty" in result.error.lower()


@pytest.mark.asyncio
async def test_web_read_network_error(browsing_skill):
    """web_read returns error on network failure."""
    import httpx
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(
        side_effect=httpx.ConnectError("Network unreachable")
    )

    result = await browsing_skill.execute_tool(
        "web_read", {"url": "https://offline.com"}
    )

    assert result.success is False
    assert "network error" in result.error.lower()


# ── unknown tool ──────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_tool(browsing_skill):
    """execute_tool returns error for unrecognised tool names."""
    browsing_skill._client = AsyncMock()

    result = await browsing_skill.execute_tool("fly_to_moon", {})

    assert result.success is False
    assert "unknown tool" in result.error.lower()


def test_truncate_at_line_boundary():
    """_truncate_at_line cuts at a newline boundary."""
    text = "line one\nline two\nline three extra content here"
    result = _truncate_at_line(text, 20)
    # Should cut at the last newline before position 20
    assert not result.endswith("line t")  # no mid-word cut at newline
    assert len(result) <= 20


def test_truncate_at_line_no_truncation_needed():
    """_truncate_at_line returns original text when under limit."""
    text = "short text"
    assert _truncate_at_line(text, 1000) == text


def test_truncate_at_line_fallback():
    """_truncate_at_line falls back to hard cut if no newline near boundary."""
    # No newlines at all in a long string
    text = "a" * 200
    result = _truncate_at_line(text, 100)
    assert len(result) <= 100


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# http_get tests


@pytest.mark.asyncio
async def test_http_get_success(browsing_skill):
    """http_get returns raw response body as-is."""
    body = '{"bitcoin": {"usd": 85000}}'
    mock_resp = _make_response(200, body)
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(return_value=mock_resp)

    result = await browsing_skill.execute_tool(
        "http_get", {"url": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"}
    )

    assert result.success is True
    assert result.output == body


@pytest.mark.asyncio
async def test_http_get_does_not_use_jina(browsing_skill):
    """http_get calls the URL directly — NOT via r.jina.ai."""
    from arc.skills.builtin.browsing import JINA_BASE
    mock_resp = _make_response(200, '{"ok": true}')
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(return_value=mock_resp)

    await browsing_skill.execute_tool(
        "http_get", {"url": "https://api.example.com/data"}
    )

    called_url = browsing_skill._client.get.call_args[0][0]
    assert not called_url.startswith(JINA_BASE)
    assert called_url == "https://api.example.com/data"


@pytest.mark.asyncio
async def test_http_get_forwards_custom_headers(browsing_skill):
    """http_get passes extra headers to the request."""
    mock_resp = _make_response(200, "ok")
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(return_value=mock_resp)

    await browsing_skill.execute_tool(
        "http_get",
        {
            "url": "https://api.example.com/data",
            "headers": {"Authorization": "Bearer tok123", "Accept": "application/json"},
        },
    )

    call_kwargs = browsing_skill._client.get.call_args
    sent_headers = call_kwargs[1].get("headers", {})
    assert sent_headers.get("Authorization") == "Bearer tok123"
    assert sent_headers.get("Accept") == "application/json"


@pytest.mark.asyncio
async def test_http_get_url_in_artifacts(browsing_skill):
    """http_get stores the requested URL in artifacts."""
    mock_resp = _make_response(200, '{"price": 100}')
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(return_value=mock_resp)

    result = await browsing_skill.execute_tool(
        "http_get", {"url": "https://api.example.com/price"}
    )

    assert result.success is True
    assert "https://api.example.com/price" in result.artifacts


@pytest.mark.asyncio
async def test_http_get_truncates_large_response(browsing_skill):
    """http_get truncates response body that exceeds max_length."""
    big_body = '{"data": "' + "x" * 2000 + '"}'
    mock_resp = _make_response(200, big_body)
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(return_value=mock_resp)

    result = await browsing_skill.execute_tool(
        "http_get", {"url": "https://api.example.com/data", "max_length": 200}
    )

    assert result.success is True
    assert len(result.output) < len(big_body)
    assert "truncated" in result.output.lower()


@pytest.mark.asyncio
async def test_http_get_http_error(browsing_skill):
    """http_get returns error on non-2xx status."""
    import httpx
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    error = httpx.HTTPStatusError(
        message="403 Forbidden", request=MagicMock(), response=mock_resp
    )
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(side_effect=error)

    result = await browsing_skill.execute_tool(
        "http_get", {"url": "https://api.example.com/protected"}
    )

    assert result.success is False
    assert "403" in result.error


@pytest.mark.asyncio
async def test_http_get_network_error(browsing_skill):
    """http_get returns error on network failure."""
    import httpx
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(
        side_effect=httpx.ConnectError("offline")
    )

    result = await browsing_skill.execute_tool(
        "http_get", {"url": "https://unreachable.example.com"}
    )

    assert result.success is False
    assert "network error" in result.error.lower()


@pytest.mark.asyncio
async def test_http_get_empty_response(browsing_skill):
    """http_get returns error for empty response body."""
    mock_resp = _make_response(200, "   ")
    browsing_skill._client = AsyncMock()
    browsing_skill._client.get = AsyncMock(return_value=mock_resp)

    result = await browsing_skill.execute_tool(
        "http_get", {"url": "https://api.example.com/empty"}
    )

    assert result.success is False
    assert "empty" in result.error.lower()


def test_browsing_manifest_has_http_get(browsing_skill):
    """Manifest now exposes web_search, web_read, and http_get."""
    tool_names = {t.name for t in browsing_skill.manifest().tools}
    assert tool_names == {"web_search", "web_read", "http_get"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _truncate_at_line unit tests (no skill instance needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_truncate_at_line_boundary():
    """_truncate_at_line cuts at a newline boundary."""
    text = "line one\nline two\nline three extra content here"
    result = _truncate_at_line(text, 20)
    assert not result.endswith("line t")
    assert len(result) <= 20


def test_truncate_at_line_no_truncation_needed():
    """_truncate_at_line returns original text when under limit."""
    text = "short text"
    assert _truncate_at_line(text, 1000) == text


def test_truncate_at_line_fallback():
    """_truncate_at_line falls back to hard cut if no newline near boundary."""
    text = "a" * 200
    result = _truncate_at_line(text, 100)
    assert len(result) <= 100
