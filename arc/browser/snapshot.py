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

from arc.browser.accessibility import AXTreeExtractor, AXElement, _INTERACTIVE_ROLES
from arc.liquid.extract import ProductData, extract_generic, filter_quality_products, EXTRACTORS
from urllib.parse import urlparse

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

    # ── AX tree enrichment (set when PageAnalyzer uses CDP path) ──
    locator_strategy: str = ""    # "role" | "label" | "selector" | ""
    locator_value: str = ""       # e.g. "textbox::Where from?" or CSS
    expanded: bool | None = None  # combobox / menu open state
    selected: bool | None = None  # tab / option selected
    focused: bool = False         # currently focused element
    level: int | None = None      # heading level
    autocomplete: str = ""        # ARIA autocomplete hint
    haspopup: str = ""            # ARIA haspopup ("menu", "listbox", ...)
    description: str = ""         # accessible description
    invalid: str = ""             # "true" | "grammar" | "spelling" | ""
    context: str = ""             # nearest ancestor heading for disambiguation
    backend_dom_node_id: int | None = None  # CDP DOM node id

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

        # ARIA state annotations (from AX tree)
        if self.expanded is not None:
            parts.append(f"expanded={self.expanded}")
        if self.selected is True:
            parts.append("(selected)")
        if self.haspopup:
            parts.append(f"popup={self.haspopup}")
        if self.invalid and self.invalid != "false":
            parts.append("(invalid)")

        # Flags
        if self.disabled:
            parts.append("(disabled)")
        if self.required:
            parts.append("(required)")
        if self.focused:
            parts.append("(focused)")

        # Section context — disambiguates duplicate buttons like "Add to Cart"
        if self.context and self.role in ("button", "link"):
            parts.append(f"in: {self.context}")

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

    # ── Structured product data (for search results / listing pages) ──
    products: list[ProductData] = field(default_factory=list)

    # ── AX tree enrichment ──
    landmarks: list[str] = field(default_factory=list)    # e.g. ["navigation", "main", "search"]
    alerts: list[str] = field(default_factory=list)       # live region text
    focused_element_id: int | None = None                 # ID of the focused element
    ax_source: bool = False                               # True if snapshot used AX tree

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

        # Structured product data — when available, show a clean numbered list
        # so the LLM can use select_product instead of guessing element IDs
        if len(self.products) >= 2:
            lines.append("")
            lines.append(f"[Products] ({len(self.products)} found — use select_product action with the product number)")
            for i, p in enumerate(self.products, 1):
                parts = [f"  {i}. {p.name}"]
                if p.price:
                    currency = p.currency or ""
                    parts.append(f"  {currency}{p.price}")
                if p.rating:
                    parts.append(f"  ★{p.rating}")
                if p.brand:
                    parts.append(f"  by {p.brand}")
                lines.append(" —".join(parts))
            lines.append("")
            lines.append("  → To select a product: {type: 'select_product', index: 1}")

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

        # Landmarks — structural context for the LLM
        if self.landmarks:
            lines.append("")
            lines.append("[Page Structure]")
            for lm in self.landmarks:
                lines.append(f"  {lm}")

        # Alerts / live regions — validation errors, status messages
        if self.alerts:
            lines.append("")
            lines.append("[Alerts]")
            for alert in self.alerts:
                lines.append(f"  ⚠ {alert}")

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

    Two extraction paths:
    - **AX tree (preferred)**: CDP Accessibility.getFullAXTree → fast, accurate
      accessible names, ARIA states, shadow-DOM-aware. Augmented with targeted
      DOM queries for tag/input-type/options/href.
    - **DOM fallback**: original JS page.evaluate() with 15 CSS selectors.
      Used when CDP is unavailable (e.g. Firefox, headless shell).
    """

    def __init__(self, use_ax_tree: bool = True):
        self._use_ax_tree = use_ax_tree
        self._ax_extractor = AXTreeExtractor() if use_ax_tree else None

    async def analyze(self, page: "Page") -> PageSnapshot:
        """Produce a complete PageSnapshot for the current page state."""
        title = await page.title()
        url = page.url

        # Try AX tree extraction first, fall back to DOM
        elements: list[InteractiveElement] = []
        landmarks: list[str] = []
        alerts: list[str] = []
        ax_source = False

        if self._use_ax_tree and self._ax_extractor:
            try:
                elements, landmarks, alerts = await self._extract_interactive_elements_ax(page)
                ax_source = len(elements) > 0
            except Exception as e:
                logger.warning(f"AX tree extraction failed, falling back to DOM: {e}")

        if not elements:
            elements = await self._extract_interactive_elements_dom(page)

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

        # Find focused element
        focused_id = None
        for el in elements:
            if el.focused:
                focused_id = el.id
                break

        # Extract structured product data on listing / search / form pages
        # (many e-commerce search results have filter forms → classify as "form")
        products: list[ProductData] = []
        if page_type in ("search_results", "listing", "form"):
            products = await self._extract_products(page, url)

        return PageSnapshot(
            url=url,
            title=title,
            page_type=page_type,
            elements=elements,
            obstacles=obstacles,
            text_content=text_content,
            forms_count=forms_count,
            links_count=len([e for e in elements if e.tag == "a" or e.role == "link"]),
            products=products,
            landmarks=landmarks,
            alerts=alerts,
            focused_element_id=focused_id,
            ax_source=ax_source,
        )

    async def _extract_interactive_elements_ax(
        self,
        page: "Page",
    ) -> tuple[list[InteractiveElement], list[str], list[str]]:
        """
        Extract elements via CDP Accessibility Tree + DOM augmentation.

        Returns (elements, landmarks, alerts).
        """
        import time

        t0 = time.perf_counter()
        ax_result = await self._ax_extractor.extract(page)
        t_ax = time.perf_counter()
        logger.debug(f"AX tree: {len(ax_result.elements)} elements in {t_ax - t0:.2f}s")

        if not ax_result.elements:
            return [], ax_result.landmarks, ax_result.alerts

        # Augment with DOM data (tag, input_type, options, href, selector)
        # Also returns elements the AX tree missed (DOM gap-fill)
        dom_gap_elements = await self._ax_extractor.augment_from_dom(page, ax_result.elements)
        t_dom = time.perf_counter()
        logger.debug(f"DOM augment: {t_dom - t_ax:.2f}s, {len(dom_gap_elements)} gap elements")

        # Resolve locator strategies (sync — no browser round-trips)
        locators = await self._ax_extractor.resolve_locators(page, ax_result.elements)
        t_loc = time.perf_counter()
        logger.debug(f"Locators: {t_loc - t_dom:.2f}s")

        # Convert AXElement → InteractiveElement
        elements: list[InteractiveElement] = []
        element_id = 1

        for i, ax_el in enumerate(ax_result.elements):
            # Skip headings and images with no interactivity (keep for context)
            if ax_el.role in ("heading", "image"):
                # Include headings for page structure context
                el = InteractiveElement(
                    id=element_id,
                    tag=ax_el.tag or ("h" + str(ax_el.level or 2) if ax_el.role == "heading" else "img"),
                    role=ax_el.role,
                    name=ax_el.name,
                    level=ax_el.level,
                )
                elements.append(el)
                element_id += 1
                continue

            # Get locator info
            strategy, value = locators.get(i, ("", ""))

            el = InteractiveElement(
                id=element_id,
                tag=ax_el.tag or _role_to_tag(ax_el.role),
                role=ax_el.role,
                name=ax_el.name,
                type=ax_el.input_type,
                value=ax_el.value,
                placeholder=ax_el.placeholder,
                options=ax_el.options,
                checked=_parse_checked(ax_el.checked),
                disabled=ax_el.disabled,
                required=ax_el.required,
                selector=ax_el.selector,
                # AX enrichment
                locator_strategy=strategy,
                locator_value=value,
                expanded=ax_el.expanded,
                selected=ax_el.selected,
                focused=ax_el.focused,
                level=ax_el.level,
                autocomplete=ax_el.autocomplete,
                haspopup=ax_el.haspopup,
                description=ax_el.description,
                invalid=ax_el.invalid,
                context=ax_el.context,
                backend_dom_node_id=ax_el.backend_dom_node_id,
            )
            elements.append(el)
            element_id += 1

        # Add DOM-only elements that AX tree missed (gap-fill)
        for gap_el in dom_gap_elements:
            el = InteractiveElement(
                id=element_id,
                tag=gap_el.tag or "button",
                role=gap_el.role or "button",
                name=gap_el.name,
                type=gap_el.input_type,
                selector=gap_el.selector,
                disabled=gap_el.disabled,
                locator_strategy="selector" if gap_el.selector else "name",
                locator_value=gap_el.selector if gap_el.selector else gap_el.name,
            )
            elements.append(el)
            element_id += 1

        return elements, ax_result.landmarks, ax_result.alerts

    async def _extract_interactive_elements_dom(self, page: "Page") -> list[InteractiveElement]:
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

    async def _extract_products(self, page: "Page", url: str) -> list["ProductData"]:
        """
        Extract structured product data from a listing/search results page.

        Uses site-specific extractors when available, falls back to generic
        JSON-LD → OpenGraph → DOM heuristic extraction from Liquid Web.
        Only returns products if ≥ 2 are found (single product = detail page).
        """
        try:
            domain = urlparse(url).netloc.removeprefix("www.")
            extractor = EXTRACTORS.get(domain)
            if extractor:
                products = await extractor(page, url)
            else:
                products = await extract_generic(page, url)
            products = filter_quality_products(products)
            if len(products) < 2:
                return []
            return products[:15]  # cap to avoid snapshot bloat
        except Exception as e:
            logger.debug(f"Product extraction failed: {e}")
            return []

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
        if any(kw in url_lower for kw in ("search", "results", "q=", "query=", "/s?")):
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


# ━━━ Helpers ━━━

def _role_to_tag(role: str) -> str:
    """Map ARIA role to most likely HTML tag."""
    return {
        "textbox": "input",
        "searchbox": "input",
        "combobox": "select",
        "checkbox": "input",
        "radio": "input",
        "button": "button",
        "link": "a",
        "spinbutton": "input",
        "slider": "input",
        "switch": "input",
        "tab": "button",
        "menuitem": "li",
        "treeitem": "li",
        "option": "option",
        "heading": "h2",
        "image": "img",
    }.get(role, "div")


def _parse_checked(checked: str | None) -> bool | None:
    """Parse AX tree checked state to bool."""
    if checked is None:
        return None
    if checked == "true":
        return True
    if checked == "false":
        return False
    if checked == "mixed":
        return False  # "mixed" → treat as unchecked for display
    return None
