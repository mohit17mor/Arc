"""
Tests for arc.browser.actions — ActionExecutor and helpers.

Mocks Playwright's Page and Locator to test action logic
without a real browser.
"""

from __future__ import annotations

import platform
import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from arc.browser.actions import (
    ActionExecutor,
    ActionResult,
    ActionsResult,
    _select_all_shortcut,
    _score_suggestion,
    _normalize_date,
)
from arc.browser.snapshot import InteractiveElement, PageAnalyzer, PageSnapshot


# ━━━ Suggestion Scoring ━━━


class TestScoreSuggestion:
    def test_exact_match(self):
        assert _score_suggestion("email", "email") == 1.0

    def test_substring_match(self):
        assert _score_suggestion("email", "email address") >= 0.8

    def test_no_match(self):
        assert _score_suggestion("email", "telephone") < 0.5

    def test_empty_strings(self):
        assert _score_suggestion("", "something") == 0.0
        assert _score_suggestion("something", "") == 0.0

    def test_word_overlap(self):
        score = _score_suggestion("first name", "name first")
        assert score > 0.5

    def test_prefix_no_false_positive(self):
        """Partial prefix 'ema' should NOT match 'email' — avoids false positives."""
        score = _score_suggestion("email", "ema")
        assert score < 0.5  # Not a word-boundary match

    def test_word_boundary_paris(self):
        """Paris should match 'Paris, France' much better than 'Parish'."""
        good = _score_suggestion("paris", "Paris, France")
        bad = _score_suggestion("paris", "Portland Parish")
        assert good > bad
        assert good >= 0.80
        assert bad <= 0.40

    def test_word_boundary_delhi(self):
        """Delhi should match 'Delhi, India' well."""
        score = _score_suggestion("delhi", "Delhi, India")
        assert score >= 0.90


# ━━━ Date Normalization ━━━


class TestNormalizeDate:
    def test_iso_format_passthrough(self):
        assert _normalize_date("2026-03-15") == "2026-03-15"

    def test_month_name_format(self):
        assert _normalize_date("March 15, 2026") == "2026-03-15"

    def test_month_name_no_comma(self):
        assert _normalize_date("March 15 2026") == "2026-03-15"

    def test_day_month_year(self):
        assert _normalize_date("15 March 2026") == "2026-03-15"

    def test_slash_format(self):
        # 15/03/2026 — day > 12 so parsed as DD/MM/YYYY
        assert _normalize_date("15/03/2026") == "2026-03-15"

    def test_abbreviated_month(self):
        assert _normalize_date("Jan 1, 2025") == "2025-01-01"

    def test_unparseable_passthrough(self):
        assert _normalize_date("next tuesday") == "next tuesday"


# ━━━ Platform Helpers ━━━


class TestPlatformShortcuts:
    def test_select_all_shortcut_uses_meta_on_macos(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert _select_all_shortcut() == "Meta+a"

    def test_select_all_shortcut_uses_control_elsewhere(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert _select_all_shortcut() == "Control+a"


# ━━━ ActionResult ━━━


class TestActionsResult:
    def test_all_succeeded_true(self):
        result = ActionsResult(results=[
            ActionResult(success=True, action_type="click"),
            ActionResult(success=True, action_type="fill"),
        ])
        assert result.all_succeeded is True

    def test_all_succeeded_false(self):
        result = ActionsResult(results=[
            ActionResult(success=True, action_type="click"),
            ActionResult(success=False, action_type="fill", error="not found"),
        ])
        assert result.all_succeeded is False

    def test_summary_format(self):
        result = ActionsResult(results=[
            ActionResult(success=True, action_type="click", target="Login"),
            ActionResult(success=False, action_type="fill", target="Email", error="not found"),
        ])
        summary = result.summary
        assert "✓" in summary
        assert "✗" in summary
        assert "Login" in summary
        assert "not found" in summary


# ━━━ ActionExecutor ━━━


class TestActionExecutor:
    @pytest.fixture
    def analyzer(self):
        analyzer = PageAnalyzer()
        # Mock the analyze method to return a simple snapshot
        analyzer.analyze = AsyncMock(return_value=PageSnapshot(
            url="https://example.com",
            title="Example",
            page_type="other",
            elements=[],
            obstacles=[],
            text_content="",
            forms_count=0,
            links_count=0,
        ))
        return analyzer

    @pytest.fixture
    def executor(self, analyzer):
        return ActionExecutor(analyzer)

    @pytest.fixture
    def mock_page(self):
        """
        Create a mock Playwright Page with basic locator support.

        In Playwright, page.locator(), page.get_by_text(), page.get_by_role(),
        page.get_by_label(), page.get_by_placeholder() are all SYNCHRONOUS
        and return a Locator. Only the Locator's methods (.count(), .click(),
        .fill(), etc.) are async. Mocks must reflect this.
        """
        page = AsyncMock()
        page.url = "https://example.com"

        # make page.wait_for_load_state not fail
        page.wait_for_load_state = AsyncMock()
        page.wait_for_timeout = AsyncMock()

        # Default: all locator-returning methods return an empty locator
        empty_locator = MagicMock()
        empty_locator.count = AsyncMock(return_value=0)
        empty_locator.first = empty_locator

        page.get_by_text = MagicMock(return_value=empty_locator)
        page.get_by_role = MagicMock(return_value=empty_locator)
        page.get_by_label = MagicMock(return_value=empty_locator)
        page.get_by_placeholder = MagicMock(return_value=empty_locator)
        page.locator = MagicMock(return_value=empty_locator)

        return page

    def _make_locator(self, visible=True, tag="input", input_type="text"):
        """Helper to create a mock Playwright Locator."""
        locator = AsyncMock()
        locator.count = AsyncMock(return_value=1)
        locator.first = locator
        locator.is_visible = AsyncMock(return_value=visible)
        locator.evaluate = AsyncMock(return_value=tag if tag != "input" else input_type)
        locator.get_attribute = AsyncMock(return_value=input_type)
        locator.click = AsyncMock()
        locator.fill = AsyncMock()
        locator.press_sequentially = AsyncMock()
        locator.set_checked = AsyncMock()
        locator.select_option = AsyncMock()
        locator.nth = MagicMock(return_value=locator)
        locator.inner_text = AsyncMock(return_value="Option A")
        return locator

    @pytest.mark.asyncio
    async def test_unknown_action_type(self, executor, mock_page):
        result = await executor.execute(mock_page, [{"type": "dance"}])
        assert len(result.results) == 1
        assert result.results[0].success is False
        assert "Unknown action type" in result.results[0].error

    @pytest.mark.asyncio
    async def test_click_action(self, executor, mock_page):
        locator = self._make_locator()
        # get_by_text is SYNC in Playwright, returns a Locator
        mock_page.get_by_text = MagicMock(return_value=locator)

        result = await executor.execute(mock_page, [
            {"type": "click", "target": "Login"},
        ])

        assert len(result.results) == 1
        assert result.results[0].success is True
        assert result.results[0].action_type == "click"

    @pytest.mark.asyncio
    async def test_scroll_action(self, executor, mock_page):
        mock_page.evaluate = AsyncMock()

        result = await executor.execute(mock_page, [
            {"type": "scroll", "direction": "down"},
        ])

        assert result.results[0].success is True
        assert "down" in result.results[0].detail

    @pytest.mark.asyncio
    async def test_back_action(self, executor, mock_page):
        mock_page.go_back = AsyncMock()

        result = await executor.execute(mock_page, [
            {"type": "back"},
        ])

        assert result.results[0].success is True

    @pytest.mark.asyncio
    async def test_stops_on_failure(self, executor, mock_page):
        """Actions should stop executing after first failure."""
        # All locator-returning methods return empty (count=0)
        # mock_page already set up with empty locators by fixture

        result = await executor.execute(mock_page, [
            {"type": "click", "target": "Nonexistent"},
            {"type": "scroll", "direction": "down"},  # should NOT execute
        ])

        assert len(result.results) == 1  # Only first action attempted
        assert result.results[0].success is False

    @pytest.mark.asyncio
    async def test_fill_form_batch(self, executor, mock_page):
        """fill_form should fill multiple fields."""
        locator = self._make_locator()
        # get_by_label is SYNC in Playwright
        mock_page.get_by_text = MagicMock(return_value=locator)
        mock_page.get_by_label = MagicMock(return_value=locator)

        # Mock evaluate to return "text" type
        locator.evaluate = AsyncMock(side_effect=lambda js, *a, **kw: "text" if "tagName" in js else False)

        result = await executor.execute(mock_page, [
            {
                "type": "fill_form",
                "fields": {
                    "First Name": "John",
                    "Last Name": "Doe",
                    "Email": "john@example.com",
                },
            },
        ])

        assert len(result.results) == 1
        r = result.results[0]
        assert r.action_type == "fill_form"
        # If fill succeeded, detail should mention filled fields
        assert "filled" in r.detail

    @pytest.mark.asyncio
    async def test_submit_action(self, executor, mock_page):
        """submit should find and click a submit button."""
        locator = self._make_locator(tag="button", input_type="submit")
        locator.is_visible = AsyncMock(return_value=True)
        # page.locator is SYNC in Playwright
        mock_page.locator = MagicMock(return_value=locator)

        result = await executor.execute(mock_page, [
            {"type": "submit"},
        ])

        assert result.results[0].success is True

    @pytest.mark.asyncio
    async def test_js_action(self, executor, mock_page):
        mock_page.evaluate = AsyncMock(return_value=42)

        result = await executor.execute(mock_page, [
            {"type": "js", "code": "1 + 1"},
        ])

        assert result.results[0].success is True

    @pytest.mark.asyncio
    async def test_fill_autocomplete_uses_meta_on_macos(self, executor, mock_page, monkeypatch):
        locator = self._make_locator()
        mock_page.keyboard = AsyncMock()
        executor._pick_suggestion = AsyncMock(return_value=None)
        monkeypatch.setattr(platform, "system", lambda: "Darwin")

        result = await executor._fill_autocomplete(mock_page, locator, "Mumbai", "From")

        assert result.success is True
        mock_page.keyboard.press.assert_any_call("Meta+a")
        mock_page.keyboard.press.assert_any_call("Backspace")
        mock_page.keyboard.type.assert_awaited_once_with("Mumbai", delay=60)


# ━━━ Select Product ━━━


class TestSelectProduct:
    """Tests for the select_product action type."""

    @pytest.fixture
    def analyzer(self):
        analyzer = PageAnalyzer()
        analyzer.analyze = AsyncMock(return_value=PageSnapshot(
            url="https://amazon.in/dp/B09ABC",
            title="Product",
            page_type="other",
            elements=[],
            obstacles=[],
            text_content="",
            forms_count=0,
            links_count=0,
        ))
        return analyzer

    @pytest.fixture
    def executor(self, analyzer):
        return ActionExecutor(analyzer)

    @pytest.fixture
    def mock_page(self):
        page = AsyncMock()
        page.url = "https://amazon.in/s?k=camera"
        page.wait_for_load_state = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.goto = AsyncMock()
        empty = MagicMock()
        empty.count = AsyncMock(return_value=0)
        empty.first = empty
        page.get_by_text = MagicMock(return_value=empty)
        page.get_by_role = MagicMock(return_value=empty)
        page.get_by_label = MagicMock(return_value=empty)
        page.get_by_placeholder = MagicMock(return_value=empty)
        page.locator = MagicMock(return_value=empty)
        return page

    @pytest.fixture
    def products(self):
        from arc.liquid.extract import ProductData
        return [
            ProductData(name="Sony Alpha A6100", price="73159", url="https://amazon.in/dp/B09A"),
            ProductData(name="Canon EOS M50", price="52999", url="https://amazon.in/dp/B09B"),
            ProductData(name="Nikon Z50", price="86999", url="https://amazon.in/dp/B09C"),
        ]

    @pytest.mark.asyncio
    async def test_select_product_navigates(self, executor, mock_page, products):
        result = await executor.execute(
            mock_page,
            [{"type": "select_product", "index": 2}],
            products=products,
        )
        assert result.results[0].success is True
        assert result.results[0].action_type == "select_product"
        assert "Canon" in result.results[0].target
        mock_page.goto.assert_called_once_with(
            "https://amazon.in/dp/B09B", timeout=10000,
        )

    @pytest.mark.asyncio
    async def test_select_product_index_out_of_range(self, executor, mock_page, products):
        result = await executor.execute(
            mock_page,
            [{"type": "select_product", "index": 5}],
            products=products,
        )
        assert result.results[0].success is False
        assert "out of range" in result.results[0].error

    @pytest.mark.asyncio
    async def test_select_product_missing_index(self, executor, mock_page, products):
        result = await executor.execute(
            mock_page,
            [{"type": "select_product"}],
            products=products,
        )
        assert result.results[0].success is False
        assert "index" in result.results[0].error.lower()

    @pytest.mark.asyncio
    async def test_select_product_no_products(self, executor, mock_page):
        result = await executor.execute(
            mock_page,
            [{"type": "select_product", "index": 1}],
            products=[],
        )
        assert result.results[0].success is False
        assert "No products" in result.results[0].error

    @pytest.mark.asyncio
    async def test_select_product_no_url(self, executor, mock_page):
        from arc.liquid.extract import ProductData
        products = [
            ProductData(name="Mystery Product", price="999"),
            ProductData(name="Another One", price="888"),
        ]
        result = await executor.execute(
            mock_page,
            [{"type": "select_product", "index": 1}],
            products=products,
        )
        assert result.results[0].success is False
        assert "no URL" in result.results[0].error

    @pytest.mark.asyncio
    async def test_select_product_first_item(self, executor, mock_page, products):
        result = await executor.execute(
            mock_page,
            [{"type": "select_product", "index": 1}],
            products=products,
        )
        assert result.results[0].success is True
        assert "Sony" in result.results[0].target

    @pytest.mark.asyncio
    async def test_select_product_invalid_index_type(self, executor, mock_page, products):
        result = await executor.execute(
            mock_page,
            [{"type": "select_product", "index": "abc"}],
            products=products,
        )
        assert result.results[0].success is False
        assert "Invalid index" in result.results[0].error
