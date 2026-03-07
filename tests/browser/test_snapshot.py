"""
Tests for arc.browser.snapshot — PageAnalyzer, PageSnapshot, InteractiveElement.

These tests mock Playwright's Page object so they run without a real browser.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from arc.browser.snapshot import (
    InteractiveElement,
    Obstacle,
    PageAnalyzer,
    PageSnapshot,
)
from arc.browser.accessibility import AXElement, AXTreeExtractor


# ━━━ InteractiveElement ━━━


class TestInteractiveElement:
    def test_snapshot_line_text_input(self):
        el = InteractiveElement(
            id=1,
            tag="input",
            role="textbox",
            name="Email",
            type="email",
            value="",
            placeholder="you@example.com",
            selector="input[name='email']",
        )
        line = el.to_snapshot_line()
        assert "[1]" in line
        assert "Email" in line
        assert "textbox" in line
        assert "you@example.com" in line

    def test_snapshot_line_button(self):
        el = InteractiveElement(
            id=2,
            tag="button",
            role="button",
            name="Submit",
            type="submit",
            selector="button[type='submit']",
        )
        line = el.to_snapshot_line()
        assert "[2]" in line
        assert "Submit" in line
        assert "button" in line

    def test_snapshot_line_select_with_options(self):
        el = InteractiveElement(
            id=3,
            tag="select",
            role="combobox",
            name="Country",
            type="select",
            options=["US", "UK", "India"],
            selector="select[name='country']",
        )
        line = el.to_snapshot_line()
        assert "[3]" in line
        assert "Country" in line
        assert "US" in line

    def test_snapshot_line_checkbox(self):
        el = InteractiveElement(
            id=4,
            tag="input",
            role="checkbox",
            name="Accept terms",
            type="checkbox",
            checked=False,
            selector="#accept",
        )
        line = el.to_snapshot_line()
        assert "[4]" in line
        assert "Accept terms" in line
        assert "☐" in line or "checkbox" in line.lower()

    def test_snapshot_line_checked_checkbox(self):
        el = InteractiveElement(
            id=5,
            tag="input",
            role="checkbox",
            name="Newsletter",
            type="checkbox",
            checked=True,
            selector="#newsletter",
        )
        line = el.to_snapshot_line()
        assert "[5]" in line
        assert "☑" in line or "checked" in line.lower()


# ━━━ PageSnapshot ━━━


class TestPageSnapshot:
    def test_to_text_basic(self):
        snapshot = PageSnapshot(
            url="https://example.com",
            title="Example",
            page_type="other",
            elements=[
                InteractiveElement(
                    id=1, tag="input", role="textbox", name="Name",
                    type="text", selector="#name",
                ),
                InteractiveElement(
                    id=2, tag="button", role="button", name="Submit",
                    type="submit", selector="button",
                ),
            ],
            obstacles=[],
            text_content="Welcome to Example.com",
            forms_count=1,
            links_count=0,
        )
        text = snapshot.to_text()
        assert "Example" in text
        assert "example.com" in text
        assert "[1]" in text
        assert "[2]" in text
        assert "Name" in text
        assert "Submit" in text

    def test_to_text_with_obstacles(self):
        snapshot = PageSnapshot(
            url="https://example.com",
            title="Example",
            page_type="login",
            elements=[],
            obstacles=[
                Obstacle(type="captcha", description="reCAPTCHA detected"),
            ],
            text_content="",
            forms_count=0,
            links_count=0,
        )
        text = snapshot.to_text()
        assert "captcha" in text.lower() or "CAPTCHA" in text

    def test_to_text_empty_page(self):
        snapshot = PageSnapshot(
            url="about:blank",
            title="",
            page_type="other",
            elements=[],
            obstacles=[],
            text_content="",
            forms_count=0,
            links_count=0,
        )
        text = snapshot.to_text()
        assert "about:blank" in text

    def test_to_text_with_products(self):
        """Products section should appear when 2+ products are present."""
        from arc.liquid.extract import ProductData

        products = [
            ProductData(name="Sony Alpha A6100", price="73159", currency="₹", rating="4.5", brand="Sony"),
            ProductData(name="Canon EOS M50", price="52999", currency="₹"),
            ProductData(name="Nikon Z50", price="86999", currency="₹", rating="4.3"),
        ]
        snapshot = PageSnapshot(
            url="https://amazon.in/s?k=camera",
            title="Camera - Amazon.in",
            page_type="search_results",
            elements=[],
            obstacles=[],
            text_content="",
            forms_count=0,
            links_count=0,
            products=products,
        )
        text = snapshot.to_text()
        assert "[Products]" in text
        assert "3 found" in text
        assert "Sony Alpha A6100" in text
        assert "Canon EOS M50" in text
        assert "Nikon Z50" in text
        assert "select_product" in text
        assert "₹73159" in text

    def test_to_text_single_product_no_section(self):
        """A single product should NOT render the [Products] section."""
        from arc.liquid.extract import ProductData

        snapshot = PageSnapshot(
            url="https://amazon.in/dp/B09A",
            title="Sony Camera",
            page_type="listing",
            elements=[],
            obstacles=[],
            text_content="",
            forms_count=0,
            links_count=0,
            products=[ProductData(name="Sony Alpha", price="73159")],
        )
        text = snapshot.to_text()
        assert "[Products]" not in text

    def test_to_text_no_products_no_section(self):
        """No products = no [Products] section."""
        snapshot = PageSnapshot(
            url="https://google.com/travel/flights",
            title="Google Flights",
            page_type="form",
            elements=[],
            obstacles=[],
            text_content="",
            forms_count=0,
            links_count=0,
        )
        text = snapshot.to_text()
        assert "[Products]" not in text


# ━━━ PageAnalyzer ━━━


class TestPageAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return PageAnalyzer()

    @pytest.fixture
    def mock_page(self):
        """
        Create a mock Playwright page.

        Important: In Playwright, page.locator(), page.get_by_text(), etc.
        are SYNCHRONOUS and return a Locator. Only Locator methods like
        .count(), .click(), .fill() are async. Mocks must reflect this.
        """
        page = AsyncMock()
        page.url = "https://example.com/login"
        page.title = AsyncMock(return_value="Login - Example")

        # Default evaluate — first call returns interactive elements,
        # second call returns text content
        interactive_elements = [
            {
                "tag": "input",
                "type": "email",
                "name": "email",
                "id": "email-field",
                "role": "textbox",
                "ariaLabel": "Email address",
                "placeholder": "you@example.com",
                "value": "",
                "options": [],
                "checked": False,
                "disabled": False,
                "required": True,
                "label": "Email address",
                "selector": "#email-field",
            },
            {
                "tag": "input",
                "type": "password",
                "name": "password",
                "id": "password-field",
                "role": "textbox",
                "ariaLabel": "Password",
                "placeholder": "",
                "value": "",
                "options": [],
                "checked": False,
                "disabled": False,
                "required": True,
                "label": "Password",
                "selector": "#password-field",
            },
            {
                "tag": "button",
                "type": "submit",
                "name": "",
                "id": "login-btn",
                "role": "button",
                "ariaLabel": "Sign in",
                "placeholder": "",
                "value": "",
                "options": [],
                "checked": False,
                "disabled": False,
                "required": False,
                "label": "Sign in",
                "selector": "#login-btn",
            },
        ]

        # evaluate is called multiple times: once for elements, once for text
        eval_call_count = {"n": 0}
        async def evaluate_side_effect(js, *args, **kwargs):
            eval_call_count["n"] += 1
            if eval_call_count["n"] == 1:
                return interactive_elements
            return "Login to your account"

        page.evaluate = AsyncMock(side_effect=evaluate_side_effect)

        # page.locator() is SYNC in Playwright — returns a locator object
        # The locator's .count() and .inner_text() are ASYNC
        def make_locator(count=0, inner_text=""):
            loc = MagicMock()
            loc.count = AsyncMock(return_value=count)
            loc.inner_text = AsyncMock(return_value=inner_text)
            return loc

        # Default: no forms, no obstacles
        page.locator = MagicMock(side_effect=lambda sel: make_locator(0, ""))

        return page

    @pytest.mark.asyncio
    async def test_analyze_returns_snapshot(self, analyzer, mock_page):
        snapshot = await analyzer.analyze(mock_page)

        assert isinstance(snapshot, PageSnapshot)
        assert snapshot.url == "https://example.com/login"
        assert snapshot.title == "Login - Example"
        assert len(snapshot.elements) == 3

    @pytest.mark.asyncio
    async def test_analyze_elements_have_ids(self, analyzer, mock_page):
        snapshot = await analyzer.analyze(mock_page)

        # Elements should have sequential IDs
        ids = [el.id for el in snapshot.elements]
        assert ids == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_analyze_classifies_login_page(self, analyzer, mock_page):
        snapshot = await analyzer.analyze(mock_page)

        # Should classify as login page — has password field
        assert snapshot.page_type == "login"

    @pytest.mark.asyncio
    async def test_analyze_handles_evaluate_failure(self, analyzer, mock_page):
        # All evaluate calls fail
        mock_page.evaluate = AsyncMock(side_effect=Exception("JS error"))

        # Also need locator to work for the forms_count fallback
        def make_locator(count=0, inner_text=""):
            loc = MagicMock()
            loc.count = AsyncMock(return_value=count)
            loc.inner_text = AsyncMock(return_value=inner_text)
            return loc
        mock_page.locator = MagicMock(side_effect=lambda sel: make_locator(0, ""))

        snapshot = await analyzer.analyze(mock_page)

        # Should still return a valid snapshot with no elements
        assert isinstance(snapshot, PageSnapshot)
        assert len(snapshot.elements) == 0

    @pytest.mark.asyncio
    async def test_analyze_obstacle_detection(self, analyzer, mock_page):
        # Make locator return count=1 for captcha selectors
        def locator_side_effect(sel):
            loc = MagicMock()
            if "captcha" in sel.lower():
                loc.count = AsyncMock(return_value=1)
            else:
                loc.count = AsyncMock(return_value=0)
            loc.inner_text = AsyncMock(return_value="")
            return loc

        mock_page.locator = MagicMock(side_effect=locator_side_effect)

        snapshot = await analyzer.analyze(mock_page)

        # Should detect the CAPTCHA obstacle
        assert any(obs.type == "captcha" for obs in snapshot.obstacles)


# ━━━ AXTreeExtractor gap-fill ━━━

class TestAXTreeGapFill:
    """Tests for _build_gap_elements and _match_ax_to_dom improvements."""

    def test_build_gap_elements_catches_submit_input(self):
        """<input type="submit" value="Add to Cart"> missing from AX tree is gap-filled."""
        ext = AXTreeExtractor()

        unmatched_dom = [
            {"tag": "input", "type": "submit", "name": "Add to Cart",
             "value": "Add to Cart", "selector": "#add-to-cart-button", "disabled": False},
        ]
        ax_elements = [
            AXElement(role="button", name="Buy Now"),
        ]

        gaps = ext._build_gap_elements(unmatched_dom, ax_elements)
        assert len(gaps) == 1
        assert gaps[0].name == "Add to Cart"
        assert gaps[0].tag == "input"
        assert gaps[0].input_type == "submit"
        assert gaps[0].selector == "#add-to-cart-button"

    def test_build_gap_elements_skips_non_button_types(self):
        """Non-button DOM elements (links, text inputs) are NOT gap-filled."""
        ext = AXTreeExtractor()

        unmatched_dom = [
            {"tag": "a", "type": "", "name": "Home", "value": "",
             "selector": "#nav-home", "disabled": False},
            {"tag": "input", "type": "text", "name": "Search",
             "value": "", "selector": "#q", "disabled": False},
        ]
        gaps = ext._build_gap_elements(unmatched_dom, [])
        assert len(gaps) == 0

    def test_build_gap_elements_dedup_by_name(self):
        """If AX tree already has a button with matching name, skip the gap."""
        ext = AXTreeExtractor()

        unmatched_dom = [
            {"tag": "input", "type": "submit", "name": "Add to Cart",
             "value": "Add to Cart", "selector": "#add-btn", "disabled": False},
        ]
        ax_elements = [
            AXElement(role="button", name="Add to Cart"),
        ]

        gaps = ext._build_gap_elements(unmatched_dom, ax_elements)
        assert len(gaps) == 0

    def test_build_gap_elements_dedup_by_selector(self):
        """If AX tree already has an element with matching selector, skip."""
        ext = AXTreeExtractor()

        unmatched_dom = [
            {"tag": "button", "type": "", "name": "Submit",
             "value": "", "selector": "#submit-btn", "disabled": False},
        ]
        ax_elements = [
            AXElement(role="button", name="Submit Order", selector="#submit-btn"),
        ]

        gaps = ext._build_gap_elements(unmatched_dom, ax_elements)
        assert len(gaps) == 0

    def test_build_gap_elements_dedup_by_substring_name(self):
        """AX name 'Add to cart, shift, alt, K' should match DOM 'Add to cart'."""
        ext = AXTreeExtractor()

        unmatched_dom = [
            {"tag": "input", "type": "submit", "name": "Add to cart",
             "value": "Add to cart", "selector": "#atc", "disabled": False},
        ]
        ax_elements = [
            AXElement(role="button", name="Add to cart, shift, alt, K"),
        ]

        gaps = ext._build_gap_elements(unmatched_dom, ax_elements)
        assert len(gaps) == 0

    def test_build_gap_elements_skips_generic_button(self):
        """Generic <button> elements are NOT gap-filled (AX tree rarely misses them)."""
        ext = AXTreeExtractor()

        unmatched_dom = [
            {"tag": "button", "type": "button", "name": "Checkout",
             "value": "", "selector": "#checkout-btn", "disabled": False},
        ]

        gaps = ext._build_gap_elements(unmatched_dom, [])
        assert len(gaps) == 0

    def test_build_gap_elements_skips_unidentifiable(self):
        """Elements with no name, value, or selector are skipped."""
        ext = AXTreeExtractor()

        unmatched_dom = [
            {"tag": "input", "type": "submit", "name": "",
             "value": "", "selector": "", "disabled": False},
        ]

        gaps = ext._build_gap_elements(unmatched_dom, [])
        assert len(gaps) == 0

    def test_match_ax_to_dom_uses_input_value_as_name(self):
        """_match_ax_to_dom should match AX elements by DOM input value."""
        ext = AXTreeExtractor()

        ax_elements = [
            AXElement(role="button", name="Add to Cart"),
        ]
        dom_elements = [
            {"tag": "input", "type": "submit", "name": "",
             "value": "Add to Cart", "selector": "#atc"},
        ]

        unmatched = ext._match_ax_to_dom(ax_elements, dom_elements)
        # Should have matched — input value "Add to Cart" matches AX name
        assert ax_elements[0].selector == "#atc"
        assert ax_elements[0].tag == "input"
        assert len(unmatched) == 0
