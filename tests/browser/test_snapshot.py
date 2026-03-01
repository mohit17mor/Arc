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
