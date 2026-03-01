"""
PageAnalyzer — converts a live Playwright page into a structured text snapshot.

The snapshot is what the LLM reads to understand the page. It includes:
  - Page metadata (title, URL)
  - Interactive elements with numeric IDs (inputs, buttons, links, selects)
  - Form field states (current values, placeholder text)
  - Page type classification (form, product listing, article, login, error)
  - Detected obstacles (CAPTCHAs, cookie banners, popups)
  - Truncated visible text content

Design principle: the snapshot should be small enough (~500-2000 tokens)
for the LLM to read in one pass, yet complete enough to make all decisions
for the current page without needing another look.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page

logger = logging.getLogger(__name__)

# Max text content to include in snapshot (chars)
MAX_TEXT_CONTENT = 4000

# Known obstacle selectors
_CAPTCHA_SELECTORS = [
    "iframe[src*='captcha']:visible",
    "iframe[src*='recaptcha']:visible",
    "iframe[src*='hcaptcha']:visible",
    "iframe[src*='turnstile']:visible",
    "[class*='recaptcha' i]:visible",
    "#captcha-container:visible",
    # NOTE: We intentionally omit broad selectors like [class*='captcha' i]
    # and [id*='captcha' i] because they cause false positives on sites
    # like Shopify stores that embed analytics/scripts with 'captcha' in class names.
]

_COOKIE_BANNER_SELECTORS = [
    "[class*='cookie' i][class*='banner' i]",
    "[class*='cookie' i][class*='consent' i]",
    "[id*='cookie' i][id*='consent' i]",
    "[class*='gdpr' i]",
    "[id*='cookie-banner' i]",
    "[class*='consent-banner' i]",
    "[aria-label*='cookie' i]",
    "#onetrust-banner-sdk",
    ".cc-banner",
    "#CybotCookiebotDialog",
]

_BOT_WALL_PATTERNS = [
    "verify you are human",
    "checking your browser",
    "please wait while we verify",
    "are you a robot",
    "access denied",
    "cf-challenge",
    # NOTE: "just a moment" and "blocked" removed — too generic.
    # Many normal pages contain these phrases (loading states, content words).
]


@dataclass
class Obstacle:
    """An obstacle detected on the page that needs human intervention."""

    type: str  # "captcha" | "bot_wall" | "cookie_banner" | "login_wall" | "popup"
    description: str
    selector: str | None = None  # CSS selector if found


@dataclass
class InteractiveElement:
    """A single interactive element extracted from the page."""

    id: int  # numeric ID for LLM reference
    tag: str  # input, button, a, select, textarea
    role: str = ""  # textbox, button, link, combobox, checkbox, etc.
    name: str = ""  # accessible name / label
    type: str = ""  # input type: text, password, email, date, etc.
    value: str = ""  # current value
    placeholder: str = ""  # placeholder text
    options: list[str] = field(default_factory=list)  # for select/combobox
    checked: bool | None = None  # for checkbox/radio
    disabled: bool = False
    required: bool = False
    selector: str = ""  # CSS selector for targeting

    def to_snapshot_line(self) -> str:
        """Format this element as a single line for the snapshot."""
        parts = [f"[{self.id}]"]

        # Role / type
        if self.role:
            parts.append(self.role)
        elif self.tag:
            parts.append(self.tag)

        # Name / label
        if self.name:
            parts.append(f'"{self.name}"')

        # Current value — skip for buttons/links (name already contains the text)
        show_value = self.role not in ("button", "link", "tab", "menuitem")
        if show_value and self.value and self.value != self.name:
            parts.append(f'value="{self.value}"')
        elif self.placeholder:
            parts.append(f'placeholder="{self.placeholder}"')

        # Options for dropdowns
        if self.options:
            display_options = self.options[:6]
            opts_str = ", ".join(f'"{o}"' for o in display_options)
            if len(self.options) > 6:
                opts_str += f", ... (+{len(self.options) - 6} more)"
            parts.append(f"options=[{opts_str}]")

        # Checkbox state
        if self.checked is not None:
            parts.append(f"checked={self.checked}")

        # Flags
        if self.disabled:
            parts.append("(disabled)")
        if self.required:
            parts.append("(required)")

        return " ".join(parts)


@dataclass
class PageSnapshot:
    """Complete structured snapshot of a page."""

    url: str
    title: str
    page_type: str  # "form", "listing", "article", "login", "search_results", "error", "other"
    elements: list[InteractiveElement] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    text_content: str = ""
    forms_count: int = 0
    links_count: int = 0

    def to_text(self) -> str:
        """Render the snapshot as structured text for the LLM."""
        lines = [
            f"Page: {self.title}",
            f"URL: {self.url}",
            f"Type: {self.page_type}",
        ]

        # Obstacles first — LLM needs to know about these immediately
        if self.obstacles:
            lines.append("")
            lines.append("[Obstacles Detected]")
            for obs in self.obstacles:
                lines.append(f"  ⚠ {obs.type}: {obs.description}")

        # Interactive elements grouped by type
        forms_unsorted = [
            e for e in self.elements
            if e.tag in ("input", "textarea", "select")
            or e.role in ("combobox", "textbox", "searchbox", "spinbutton", "listbox")
        ]
        # Sort: dropdown/select-style comboboxes first (trip type, class),
        # then input comboboxes, then other form fields
        def _form_sort_key(e: InteractiveElement) -> tuple:
            if e.role == "combobox" and e.tag != "input":
                return (0, e.id)
            return (1, e.id)
        forms = sorted(forms_unsorted, key=_form_sort_key)
        buttons = [
            e for e in self.elements
            if (e.tag == "button" or e.role == "button") and e not in forms
        ]
        links = [e for e in self.elements if e.tag == "a" or e.role == "link"]

        if forms:
            lines.append("")
            lines.append("[Form Fields]")
            for el in forms:
                lines.append(f"  {el.to_snapshot_line()}")

        if buttons:
            lines.append("")
            lines.append("[Buttons]")
            for el in buttons:
                lines.append(f"  {el.to_snapshot_line()}")

        if links:
            lines.append("")
            lines.append(f"[Links] ({len(links)} total, showing first 15)")
            for el in links[:15]:
                lines.append(f"  {el.to_snapshot_line()}")
            if len(links) > 15:
                lines.append(f"  ... and {len(links) - 15} more links")

        # Text content — truncated
        if self.text_content:
            lines.append("")
            lines.append("[Page Content]")
            content = self.text_content
            if len(content) > MAX_TEXT_CONTENT:
                content = content[:MAX_TEXT_CONTENT] + "\n... (content truncated)"
            lines.append(content)

        return "\n".join(lines)


class PageAnalyzer:
    """
    Analyzes a Playwright page and produces a structured PageSnapshot.

    Uses a combination of accessibility tree inspection and targeted DOM
    queries to extract interactive elements, detect page type, and identify
    obstacles — all without vision or screenshots.
    """

    async def analyze(self, page: "Page") -> PageSnapshot:
        """Produce a complete PageSnapshot for the current page state."""
        title = await page.title()
        url = page.url

        # Gather data in parallel
        elements = await self._extract_interactive_elements(page)
        obstacles = await self._detect_obstacles(page)
        text_content = await self._extract_text_content(page)
        forms_count = await page.locator("form").count()

        # Classify page type
        page_type = self._classify_page(
            url=url,
            title=title,
            elements=elements,
            obstacles=obstacles,
            forms_count=forms_count,
            text_content=text_content,
        )

        return PageSnapshot(
            url=url,
            title=title,
            page_type=page_type,
            elements=elements,
            obstacles=obstacles,
            text_content=text_content,
            forms_count=forms_count,
            links_count=len([e for e in elements if e.tag == "a"]),
        )

    async def _extract_interactive_elements(self, page: "Page") -> list[InteractiveElement]:
        """Extract all interactive elements from the page."""
        elements: list[InteractiveElement] = []
        element_id = 1

        # Query all interactive elements in one shot using JS for speed
        try:
            raw_elements = await page.evaluate("""() => {
            const results = [];
            const seen = new Set();
            
            // Query interactive elements
            const selectors = [
                'input:not([type="hidden"])',
                'textarea',
                'select',
                'button',
                'a[href]',
                '[role="button"]',
                '[role="link"]',
                '[role="textbox"]',
                '[role="combobox"]',
                '[role="searchbox"]',
                '[role="checkbox"]',
                '[role="radio"]',
                '[role="menuitem"]',
                '[role="tab"]',
                '[contenteditable="true"]',
            ];
            
            for (const selector of selectors) {
                for (const el of document.querySelectorAll(selector)) {
                    // Skip invisible elements
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 && rect.height === 0) continue;
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') continue;
                    
                    // Dedupe
                    if (seen.has(el)) continue;
                    seen.add(el);
                    
                    // Build unique CSS selector
                    let css = '';
                    if (el.id) {
                        css = '#' + CSS.escape(el.id);
                    } else if (el.getAttribute('aria-label')) {
                        // aria-label is the most reliable selector for SPAs
                        // (Google Flights, Airbnb, etc.) where inputs lack id/name
                        const label = el.getAttribute('aria-label');
                        css = el.tagName.toLowerCase() + '[aria-label="' + label.replace(/"/g, '\\"') + '"]';
                    } else if (el.name) {
                        css = el.tagName.toLowerCase() + '[name="' + el.name + '"]';
                    } else {
                        // nth-child fallback
                        const parent = el.parentElement;
                        if (parent) {
                            const siblings = Array.from(parent.children).filter(s => s.tagName === el.tagName);
                            const idx = siblings.indexOf(el) + 1;
                            css = el.tagName.toLowerCase() + ':nth-of-type(' + idx + ')';
                            // Walk up to get a more specific selector
                            let ancestor = parent;
                            let depth = 0;
                            while (ancestor && ancestor !== document.body && depth < 3) {
                                let prefix = ancestor.tagName.toLowerCase();
                                if (ancestor.id) { prefix = '#' + CSS.escape(ancestor.id); css = prefix + ' > ' + css; break; }
                                if (ancestor.className && typeof ancestor.className === 'string') {
                                    const cls = ancestor.className.trim().split(/\\s+/)[0];
                                    if (cls) prefix += '.' + CSS.escape(cls);
                                }
                                css = prefix + ' > ' + css;
                                ancestor = ancestor.parentElement;
                                depth++;
                            }
                        }
                    }
                    
                    // Get label
                    let name = '';
                    if (el.labels && el.labels.length > 0) name = el.labels[0].textContent.trim();
                    if (!name) name = el.getAttribute('aria-label') || '';
                    if (!name) name = el.getAttribute('placeholder') || '';
                    if (!name) name = el.textContent?.trim().substring(0, 80) || '';
                    
                    // Get options for select / div combobox
                    const options = [];
                    if (el.tagName === 'SELECT') {
                        for (const opt of el.options) {
                            if (opt.value) options.push(opt.textContent.trim());
                        }
                    }
                    
                    // For div-based comboboxes, use textContent as value
                    let elValue = el.value || '';
                    if (!elValue && el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA' && el.tagName !== 'SELECT') {
                        elValue = (el.textContent || '').trim();
                    }
                    
                    results.push({
                        tag: el.tagName.toLowerCase(),
                        type: el.type || '',
                        role: el.getAttribute('role') || '',
                        name: name.substring(0, 100),
                        value: elValue.substring(0, 200),
                        placeholder: el.getAttribute('placeholder') || '',
                        options: options.slice(0, 20),
                        checked: el.type === 'checkbox' || el.type === 'radio' ? el.checked : null,
                        disabled: el.disabled || false,
                        required: el.required || el.getAttribute('aria-required') === 'true',
                        selector: css,
                        hasAutocomplete: el.getAttribute('autocomplete') !== 'off' && (
                            el.getAttribute('aria-autocomplete') === 'list' ||
                            el.getAttribute('aria-autocomplete') === 'both' ||
                            el.getAttribute('list') !== null ||
                            el.dataset.autocomplete !== undefined
                        ),
                    });
                }
            }
            return results;
        }""")
        except Exception as e:
            logger.warning(f"Failed to extract interactive elements: {e}")
            return elements

        for raw in raw_elements:
            # Determine role
            role = raw.get("role", "")
            if not role:
                tag = raw["tag"]
                input_type = raw.get("type", "text")
                role_map = {
                    "input": {
                        "text": "textbox", "email": "textbox", "password": "textbox",
                        "search": "searchbox", "url": "textbox", "tel": "textbox",
                        "number": "spinbutton", "date": "datepicker",
                        "datetime-local": "datepicker", "time": "timepicker",
                        "checkbox": "checkbox", "radio": "radio",
                        "file": "file", "submit": "button", "button": "button",
                    },
                    "textarea": "textbox",
                    "select": "combobox",
                    "button": "button",
                    "a": "link",
                }
                if tag == "input":
                    role = role_map.get("input", {}).get(input_type, "textbox")
                else:
                    r = role_map.get(tag, "")
                    role = r if isinstance(r, str) else ""

            el = InteractiveElement(
                id=element_id,
                tag=raw["tag"],
                role=role,
                name=raw.get("name", ""),
                type=raw.get("type", ""),
                value=raw.get("value", ""),
                placeholder=raw.get("placeholder", ""),
                options=raw.get("options", []),
                checked=raw.get("checked"),
                disabled=raw.get("disabled", False),
                required=raw.get("required", False),
                selector=raw.get("selector", ""),
            )
            elements.append(el)
            element_id += 1

        return elements

    async def _detect_obstacles(self, page: "Page") -> list[Obstacle]:
        """Detect CAPTCHAs, cookie banners, bot walls, login walls, popups."""
        obstacles: list[Obstacle] = []

        # Check CAPTCHAs
        for selector in _CAPTCHA_SELECTORS:
            try:
                count = await page.locator(selector).count()
                if count > 0:
                    obstacles.append(Obstacle(
                        type="captcha",
                        description="CAPTCHA detected — needs human to solve",
                        selector=selector,
                    ))
                    break  # One captcha obstacle is enough
            except Exception:
                continue

        # Check cookie banners
        for selector in _COOKIE_BANNER_SELECTORS:
            try:
                count = await page.locator(selector).count()
                if count > 0:
                    obstacles.append(Obstacle(
                        type="cookie_banner",
                        description="Cookie consent banner detected — needs human to dismiss",
                        selector=selector,
                    ))
                    break
            except Exception:
                continue

        # Check bot walls — look at page text
        try:
            body_text = await page.locator("body").inner_text()
            body_lower = body_text.lower()[:2000]  # Only check first 2000 chars
            for pattern in _BOT_WALL_PATTERNS:
                if pattern in body_lower:
                    # Require a Cloudflare challenge element for text-only matches
                    # to avoid false positives from sites that say "just a moment"
                    # or "blocked" in their normal content.
                    cf_challenge = await page.locator(
                        "#cf-challenge-running, .cf-challenge, "
                        "#challenge-running, .challenge-form"
                    ).count()
                    if cf_challenge > 0:
                        obstacles.append(Obstacle(
                            type="bot_wall",
                            description=f"Bot detection wall — matched '{pattern}'",
                        ))
                        break
                    # For the most specific patterns, trust text alone
                    elif pattern in ("verify you are human", "are you a robot",
                                     "checking your browser"):
                        # But only if the page is very short (challenge pages have minimal content)
                        if len(body_lower.strip()) < 500:
                            obstacles.append(Obstacle(
                                type="bot_wall",
                                description=f"Bot detection wall — matched '{pattern}'",
                            ))
                            break
        except Exception:
            pass

        # Check login walls — detect login forms on unexpected pages
        try:
            password_fields = await page.locator("input[type='password']").count()
            if password_fields > 0:
                url_lower = page.url.lower()
                login_keywords = ("login", "signin", "sign-in", "auth", "sso")
                if any(kw in url_lower for kw in login_keywords):
                    obstacles.append(Obstacle(
                        type="login_wall",
                        description="Login page detected — needs human to authenticate",
                    ))
        except Exception:
            pass

        return obstacles

    async def _extract_text_content(self, page: "Page") -> str:
        """Extract the main text content of the page."""
        try:
            # Try to get main content area first, fall back to body
            text = await page.evaluate("""() => {
                // Try semantic content selectors
                const contentSelectors = [
                    'main', 'article', '[role="main"]',
                    '#content', '#main-content', '.content',
                    '.product-detail', '.search-results',
                ];
                for (const sel of contentSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.textContent.trim().length > 100) {
                        return el.textContent.trim();
                    }
                }
                // Fall back to body, minus nav/header/footer
                const body = document.body.cloneNode(true);
                for (const el of body.querySelectorAll('nav, header, footer, script, style, noscript')) {
                    el.remove();
                }
                return body.textContent.trim();
            }""")

            if text:
                # Clean up whitespace
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r"[ \t]{2,}", " ", text)
                text = text.strip()

            return text[:MAX_TEXT_CONTENT] if text else ""
        except Exception as e:
            logger.warning(f"Failed to extract text content: {e}")
            return ""

    def _classify_page(
        self,
        url: str,
        title: str,
        elements: list[InteractiveElement],
        obstacles: list[Obstacle],
        forms_count: int,
        text_content: str,
    ) -> str:
        """Classify the page type based on its structure."""
        url_lower = url.lower()
        title_lower = title.lower()

        # Login page
        has_password = any(e.type == "password" for e in elements)
        if has_password and any(kw in url_lower for kw in ("login", "signin", "sign-in", "auth")):
            return "login"

        # Search results
        if any(kw in url_lower for kw in ("search", "results", "q=", "query=")):
            return "search_results"

        # Error page
        if any(kw in title_lower for kw in ("404", "not found", "error", "forbidden", "denied")):
            return "error"

        # Form page — has multiple input fields
        input_fields = [e for e in elements if e.tag in ("input", "textarea", "select") and not e.disabled]
        if len(input_fields) >= 3 or forms_count >= 1:
            return "form"

        # Product / listing — heuristic
        if any(kw in url_lower for kw in ("product", "item", "shop", "store")):
            return "listing"

        # Article — lots of text, few inputs
        if len(text_content) > 1000 and len(input_fields) <= 1:
            return "article"

        return "other"
