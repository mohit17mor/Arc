"""
ActionExecutor — smart strategies for interacting with page elements.

Handles the mechanical work of clicking, typing, filling forms, and
navigating — so the LLM doesn't need a round-trip per field.

Key strategies:
  - Plain text inputs: direct Playwright fill
  - Autocomplete fields: type → wait for dropdown → select best match
  - Date pickers: JS value injection → calendar navigation fallback
  - Dropdowns/selects: match by value or label text
  - Checkboxes/radios: set_checked with boolean
  - Form batch fill: fuzzy-match labels to fields, fill all in one pass
"""

from __future__ import annotations

import logging
import platform
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page, ElementHandle, Locator

from arc.browser.snapshot import InteractiveElement, PageAnalyzer, PageSnapshot
from arc.liquid.extract import ProductData

logger = logging.getLogger(__name__)

# Timeouts for interactive waits (ms)
AUTOCOMPLETE_WAIT_MS = 300
NAVIGATION_WAIT_MS = 1000
ACTION_TIMEOUT_MS = 10000


def _select_all_shortcut() -> str:
    """Return the OS-appropriate select-all shortcut for browser inputs."""
    return "Meta+a" if platform.system().lower() == "darwin" else "Control+a"


@dataclass
class ActionResult:
    """Result of executing a single action."""

    success: bool
    action_type: str
    target: str = ""
    detail: str = ""
    error: str | None = None


@dataclass
class ActionsResult:
    """Result of executing a batch of actions."""

    results: list[ActionResult] = field(default_factory=list)
    snapshot: PageSnapshot | None = None  # page state after all actions

    @property
    def all_succeeded(self) -> bool:
        return all(r.success for r in self.results)

    @property
    def summary(self) -> str:
        lines = []
        for r in self.results:
            status = "✓" if r.success else "✗"
            line = f"{status} {r.action_type}"
            if r.target:
                line += f" → {r.target}"
            if r.detail:
                line += f" ({r.detail})"
            if r.error:
                line += f" — ERROR: {r.error}"
            lines.append(line)
        return "\n".join(lines)


class ActionExecutor:
    """
    Executes browser actions on a Playwright page.

    Each action type has a dedicated strategy that handles the
    mechanical details — autocomplete dropdowns, date pickers,
    select elements, etc. — without needing the LLM.
    """

    def __init__(self, analyzer: PageAnalyzer) -> None:
        self._analyzer = analyzer

    async def execute(
        self,
        page: "Page",
        actions: list[dict[str, Any]],
        elements: list[InteractiveElement] | None = None,
        products: list[ProductData] | None = None,
    ) -> ActionsResult:
        """
        Execute a list of actions sequentially on the page.

        If any action fails, execution stops and the result includes
        the failure + current page snapshot so the LLM can course-correct.

        Args:
            page: The Playwright page to act on
            actions: List of action dicts (type, target, value, etc.)
            elements: Pre-extracted elements from a previous snapshot (optional)
            products: Structured product data from the snapshot (optional)

        Returns:
            ActionsResult with per-action results + final page snapshot
        """
        results: list[ActionResult] = []

        for action in actions:
            action_type = action.get("type", "")
            try:
                result = await self._dispatch_action(page, action, elements, products)
                results.append(result)

                if not result.success:
                    logger.warning(
                        f"Action failed: {action_type} — {result.error}"
                    )
                    break  # Stop on failure — LLM needs to see the state

            except Exception as e:
                logger.exception(f"Action {action_type} raised exception")
                results.append(ActionResult(
                    success=False,
                    action_type=action_type,
                    target=action.get("target", ""),
                    error=str(e),
                ))
                break

        # Take a fresh snapshot after all actions
        try:
            snapshot = await self._analyzer.analyze(page)
        except Exception as e:
            logger.warning(f"Failed to snapshot after actions: {e}")
            snapshot = None

        return ActionsResult(results=results, snapshot=snapshot)

    async def _dispatch_action(
        self,
        page: "Page",
        action: dict[str, Any],
        elements: list[InteractiveElement] | None,
        products: list[ProductData] | None = None,
    ) -> ActionResult:
        """Route an action to the right handler."""
        action_type = action.get("type", "")

        # select_product is special — needs product data, not elements
        if action_type == "select_product":
            return await self._do_select_product(page, action, products)

        handlers = {
            "click": self._do_click,
            "fill": self._do_fill,
            "fill_form": self._do_fill_form,
            "select": self._do_select,
            "check": self._do_check,
            "scroll": self._do_scroll,
            "submit": self._do_submit,
            "wait": self._do_wait,
            "back": self._do_back,
            "forward": self._do_forward,
            "js": self._do_js,
        }

        handler = handlers.get(action_type)
        if not handler:
            return ActionResult(
                success=False,
                action_type=action_type,
                error=f"Unknown action type: {action_type}",
            )

        return await handler(page, action, elements)

    # ━━━ Element Finding ━━━

    async def _locate_by_element_props(
        self,
        page: "Page",
        el: InteractiveElement,
    ) -> "Locator | None":
        """
        Try all locator strategies for a known snapshot element.

        Used when we know WHICH element we want (by snapshot ID or
        fuzzy match) but need to find it on the live page.
        """
        # 1. AX role+name locator
        if el.locator_strategy == "role" and el.locator_value:
            try:
                parts = el.locator_value.split("::", 1)
                if len(parts) == 2:
                    role, name = parts
                    locator = page.get_by_role(role, name=name)
                    if await locator.count() > 0:
                        return locator.first
            except Exception:
                pass

        # 2. Label locator
        if el.locator_strategy == "label" and el.locator_value:
            try:
                locator = page.get_by_label(el.locator_value)
                if await locator.count() > 0:
                    return locator.first
            except Exception:
                pass

        # 3. CSS selector
        if el.selector:
            try:
                locator = page.locator(el.selector)
                if await locator.count() > 0:
                    return locator.first
            except Exception:
                pass

        # 4. Text content match (element name as visible text)
        if el.name:
            try:
                locator = page.get_by_text(el.name, exact=True)
                if await locator.count() > 0:
                    return locator.first
            except Exception:
                pass

        # 5. CDP backend node ID — targets the exact DOM node
        if el.backend_dom_node_id:
            return await self._resolve_by_cdp_node(
                page, el.backend_dom_node_id, el.id,
            )

        return None

    async def _resolve_by_cdp_node(
        self,
        page: "Page",
        backend_node_id: int,
        element_id: int,
    ) -> "Locator | None":
        """
        Locate a DOM node by its CDP backend node ID (last-resort fallback).

        Injects a temporary data-attribute so Playwright can locate the node.
        """
        try:
            cdp = await page.context.new_cdp_session(page)
            try:
                result = await cdp.send("DOM.resolveNode", {
                    "backendNodeId": backend_node_id,
                })
                object_id = result["object"]["objectId"]
                await cdp.send("Runtime.callFunctionOn", {
                    "objectId": object_id,
                    "functionDeclaration":
                        f'function() {{ this.setAttribute("data-arc-target", "{element_id}"); }}',
                })
            finally:
                try:
                    await cdp.detach()
                except Exception:
                    pass
            locator = page.locator(f'[data-arc-target="{element_id}"]')
            if await locator.count() > 0:
                return locator.first
        except Exception as e:
            logger.debug(f"CDP node resolution failed for node {backend_node_id}: {e}")
        return None

    async def _find_element(
        self,
        page: "Page",
        target: str,
        elements: list[InteractiveElement] | None,
    ) -> "Locator | None":
        """
        Find an element on the page using a cascade of strategies.

        0. Snapshot ID — [3] matches element with id=3, uses all locator strategies
        1. CSS selector — #id, .class, [attr] passed through directly
        2. Playwright role locators — textbox, combobox, searchbox by name
        3. Label / placeholder association
        4. Playwright text locator — exact then partial match
        5. Role locators — button, link, checkbox, etc. by name
        6. Best-effort fuzzy match against snapshot elements
        """
        target = target.strip()

        # Strategy 0: Snapshot ID → use element's own locator strategies.
        # If matched, we either find the element or return None (don't
        # fall through to text strategies — "[3]" as text never helps).
        id_match = re.match(r"^\[?(\d+)\]?$", target)
        if id_match and elements:
            element_id = int(id_match.group(1))
            for el in elements:
                if el.id == element_id:
                    locator = await self._locate_by_element_props(page, el)
                    if locator:
                        return locator
                    break
            # ID target couldn't be located on page
            return None

        # Strategy 2: CSS selector pass-through
        if target.startswith(("#", ".", "[")) or "::" in target:
            try:
                locator = page.locator(target)
                if await locator.count() > 0:
                    return locator.first
            except Exception:
                pass

        # Strategy 3: Role-based locators (BEFORE text — prefer actual
        # interactive elements over decorative overlay divs)
        for role in ("textbox", "combobox", "searchbox", "spinbutton"):
            try:
                locator = page.get_by_role(role, name=target)
                if await locator.count() > 0:
                    return locator.first
            except Exception:
                continue

        # Strategy 4: Label association (finds the *input*, not the label div)
        try:
            locator = page.get_by_label(target)
            if await locator.count() > 0:
                return locator.first
        except Exception:
            pass

        # Strategy 5: Placeholder match
        try:
            locator = page.get_by_placeholder(target)
            if await locator.count() > 0:
                return locator.first
        except Exception:
            pass

        # Strategy 6: Playwright text match — exact first, then partial.
        # This CAN resolve to decorative overlay divs on SPAs, which is fine
        # because _smart_fill handles click-to-activate.
        try:
            locator = page.get_by_text(target, exact=True)
            if await locator.count() > 0:
                return locator.first
        except Exception:
            pass

        try:
            locator = page.get_by_text(target, exact=False)
            if await locator.count() > 0:
                return locator.first
        except Exception:
            pass

        # Strategy 7: Remaining role locators (buttons, links, etc.)
        for role in ("button", "link", "checkbox", "radio", "menuitem", "tab"):
            try:
                locator = page.get_by_role(role, name=target)
                if await locator.count() > 0:
                    return locator.first
            except Exception:
                continue

        # Strategy 8: Fuzzy match against snapshot elements
        if elements:
            target_lower = target.lower().strip()
            best_match = None
            best_score = 0.0

            for el in elements:
                # Compare against name, placeholder, value
                candidates = [el.name.lower(), el.placeholder.lower()]
                for candidate in candidates:
                    if not candidate:
                        continue
                    score = _score_suggestion(target_lower, candidate)
                    if score > best_score and score > 0.5:
                        best_score = score
                        best_match = el

            if best_match:
                locator = await self._locate_by_element_props(page, best_match)
                if locator:
                    return locator

        return None

    async def _find_element_or_fail(
        self,
        page: "Page",
        target: str,
        elements: list[InteractiveElement] | None,
        action_type: str,
    ) -> tuple["Locator | None", ActionResult | None]:
        """Find an element, returning an error ActionResult if not found."""
        locator = await self._find_element(page, target, elements)
        if locator is None:
            return None, ActionResult(
                success=False,
                action_type=action_type,
                target=target,
                error=f"Element not found: '{target}'",
            )
        return locator, None

    # ━━━ Action Handlers ━━━

    async def _do_click(
        self, page: "Page", action: dict, elements: list[InteractiveElement] | None,
    ) -> ActionResult:
        target = action.get("target", "")
        locator, err = await self._find_element_or_fail(page, target, elements, "click")
        if err:
            return err

        clicked = await _smart_click(page, locator, timeout=ACTION_TIMEOUT_MS)
        if not clicked:
            return ActionResult(
                success=False, action_type="click", target=target,
                error="All click strategies failed (normal, force, JS, mouse)",
            )
        # Smart wait — only full wait if navigation was triggered
        await _wait_after_click(page)

        return ActionResult(success=True, action_type="click", target=target)

    async def _do_fill(
        self, page: "Page", action: dict, elements: list[InteractiveElement] | None,
    ) -> ActionResult:
        """Fill a single field with smart input-type detection."""
        target = action.get("target", "")
        value = action.get("value", "")
        locator, err = await self._find_element_or_fail(page, target, elements, "fill")
        if err:
            return err

        return await self._smart_fill(page, locator, value, target)

    async def _do_fill_form(
        self, page: "Page", action: dict, elements: list[InteractiveElement] | None,
    ) -> ActionResult:
        """
        Batch-fill multiple form fields in one action.

        The LLM provides {"fields": {"Label": "value", ...}}.
        The engine fuzzy-matches each label to a form element and fills them all.
        """
        fields: dict[str, str] = action.get("fields", {})
        if not fields:
            return ActionResult(
                success=False, action_type="fill_form",
                error="No fields provided",
            )

        filled = []
        failed = []

        for label, value in fields.items():
            locator = await self._find_element(page, label, elements)
            if locator is None:
                failed.append(f"'{label}' (not found)")
                continue

            try:
                result = await self._smart_fill(page, locator, value, label)
                if result.success:
                    filled.append(label)
                else:
                    failed.append(f"'{label}' ({result.error})")
            except Exception as e:
                failed.append(f"'{label}' ({e})")

        detail = f"filled {len(filled)}/{len(fields)} fields"
        if failed:
            detail += f" — failed: {', '.join(failed)}"

        return ActionResult(
            success=len(failed) == 0,
            action_type="fill_form",
            detail=detail,
            error=f"Could not fill: {', '.join(failed)}" if failed else None,
        )

    async def _do_select(
        self, page: "Page", action: dict, elements: list[InteractiveElement] | None,
    ) -> ActionResult:
        target = action.get("target", "")
        value = action.get("value", "")
        locator, err = await self._find_element_or_fail(page, target, elements, "select")
        if err:
            return err

        # Div-based comboboxes (role=combobox but NOT <select>) need
        # click-to-open + pick-option, not native select_option().
        is_combobox = await _is_combobox(locator)
        if is_combobox:
            tag = await locator.evaluate("el => el.tagName.toLowerCase()")
            if tag != "select":
                return await self._fill_select_combobox(page, locator, value, target)

        try:
            await self._select_option(locator, value)
            return ActionResult(
                success=True, action_type="select", target=target,
                detail=f'selected "{value}"',
            )
        except Exception as e:
            return ActionResult(
                success=False, action_type="select", target=target,
                error=str(e),
            )

    async def _do_check(
        self, page: "Page", action: dict, elements: list[InteractiveElement] | None,
    ) -> ActionResult:
        target = action.get("target", "")
        checked = action.get("checked", True)
        locator, err = await self._find_element_or_fail(page, target, elements, "check")
        if err:
            return err

        await locator.set_checked(checked, timeout=ACTION_TIMEOUT_MS)
        return ActionResult(
            success=True, action_type="check", target=target,
            detail=f"checked={checked}",
        )

    async def _do_scroll(
        self, page: "Page", action: dict, elements: list[InteractiveElement] | None,
    ) -> ActionResult:
        direction = action.get("direction", "down")
        amount = 600  # pixels

        if direction == "down":
            await page.evaluate(f"window.scrollBy(0, {amount})")
        elif direction == "up":
            await page.evaluate(f"window.scrollBy(0, -{amount})")
        else:
            return ActionResult(
                success=False, action_type="scroll",
                error=f"Unknown direction: {direction}",
            )

        await page.wait_for_timeout(300)  # Let content load

        return ActionResult(
            success=True, action_type="scroll",
            detail=f"scrolled {direction}",
        )

    async def _do_submit(
        self, page: "Page", action: dict, elements: list[InteractiveElement] | None,
    ) -> ActionResult:
        """Find and click the submit button on the current page."""
        # Try common submit patterns
        submit_selectors = [
            "button[type='submit']",
            "input[type='submit']",
            "button:has-text('Submit')",
            "button:has-text('Search')",
            "button:has-text('Sign in')",
            "button:has-text('Log in')",
            "button:has-text('Continue')",
            "button:has-text('Next')",
            "button:has-text('Send')",
            "button:has-text('Go')",
            "button:has-text('Apply')",
            "button:has-text('Book')",
            "button:has-text('Confirm')",
        ]

        for selector in submit_selectors:
            try:
                locator = page.locator(selector)
                count = await locator.count()
                if count > 0:
                    # Find the first visible one
                    for i in range(count):
                        el = locator.nth(i)
                        if await el.is_visible():
                            await el.click(timeout=ACTION_TIMEOUT_MS)
                            await _wait_for_stable(page)
                            return ActionResult(
                                success=True, action_type="submit",
                                detail=f"clicked {selector}",
                            )
            except Exception:
                continue

        # Fallback: press Enter in the last focused input
        try:
            await page.keyboard.press("Enter")
            await _wait_for_stable(page)
            return ActionResult(
                success=True, action_type="submit",
                detail="pressed Enter (no submit button found)",
            )
        except Exception as e:
            return ActionResult(
                success=False, action_type="submit",
                error=f"No submit button found and Enter key failed: {e}",
            )

    async def _do_wait(
        self, page: "Page", action: dict, elements: list[InteractiveElement] | None,
    ) -> ActionResult:
        wait_for = action.get("for", "")
        timeout = action.get("timeout", NAVIGATION_WAIT_MS)

        try:
            if wait_for.startswith("#") or wait_for.startswith(".") or wait_for.startswith("["):
                # CSS selector — wait for element to appear
                await page.wait_for_selector(wait_for, timeout=timeout)
            else:
                # Text content — wait for text to appear on page
                await page.get_by_text(wait_for, exact=False).wait_for(timeout=timeout)

            return ActionResult(
                success=True, action_type="wait",
                detail=f'waited for "{wait_for}"',
            )
        except Exception as e:
            return ActionResult(
                success=False, action_type="wait",
                target=wait_for,
                error=f"Timed out waiting for '{wait_for}': {e}",
            )

    async def _do_back(
        self, page: "Page", action: dict, elements: list[InteractiveElement] | None,
    ) -> ActionResult:
        await page.go_back(timeout=NAVIGATION_WAIT_MS)
        await _wait_for_stable(page)
        return ActionResult(success=True, action_type="back")

    async def _do_forward(
        self, page: "Page", action: dict, elements: list[InteractiveElement] | None,
    ) -> ActionResult:
        await page.go_forward(timeout=NAVIGATION_WAIT_MS)
        await _wait_for_stable(page)
        return ActionResult(success=True, action_type="forward")

    async def _do_js(
        self, page: "Page", action: dict, elements: list[InteractiveElement] | None,
    ) -> ActionResult:
        code = action.get("code", "")
        if not code:
            return ActionResult(
                success=False, action_type="js",
                error="No JavaScript code provided",
            )
        try:
            result = await page.evaluate(code)
            return ActionResult(
                success=True, action_type="js",
                detail=str(result)[:500] if result else "executed",
            )
        except Exception as e:
            return ActionResult(
                success=False, action_type="js",
                error=str(e),
            )

    # ━━━ Product Selection ━━━

    async def _do_select_product(
        self, page: "Page", action: dict, products: list[ProductData] | None,
    ) -> ActionResult:
        """
        Navigate to a product by its index from the [Products] list.

        The LLM sees a numbered product list in the snapshot and
        says {type: 'select_product', index: 3}. We look up the
        corresponding ProductData and navigate to its URL.
        """
        index = action.get("index")
        if index is None:
            return ActionResult(
                success=False, action_type="select_product",
                error="Missing 'index' — specify the product number from the [Products] list",
            )

        try:
            index = int(index)
        except (TypeError, ValueError):
            return ActionResult(
                success=False, action_type="select_product",
                error=f"Invalid index: {index} — must be a number",
            )

        if not products:
            return ActionResult(
                success=False, action_type="select_product",
                error="No products available on this page. Use click with element [id] instead.",
            )

        if index < 1 or index > len(products):
            return ActionResult(
                success=False, action_type="select_product",
                error=f"Product index {index} out of range (1-{len(products)})",
            )

        product = products[index - 1]  # 1-indexed
        if not product.url:
            return ActionResult(
                success=False, action_type="select_product",
                error=f"Product '{product.name}' has no URL — use click with element [id] instead.",
            )

        try:
            await page.goto(product.url, timeout=NAVIGATION_WAIT_MS * 10)
        except Exception:
            pass  # Page may still be usable

        await _wait_for_stable(page)

        return ActionResult(
            success=True,
            action_type="select_product",
            target=product.name[:80],
            detail=f"Navigated to product page: {product.url}",
        )

    # ━━━ Smart Fill (core) ━━━

    async def _smart_fill(
        self, page: "Page", locator: "Locator", value: str, target: str,
    ) -> ActionResult:
        """
        Fill an element intelligently, handling modern SPA patterns.

        Modern sites (Google Flights, Airbnb, etc.) often use overlay divs as
        placeholders.  ``get_by_text("Where to?")`` resolves to the decorative
        ``<div>``, *not* the real ``<input>`` which only appears after a click.

        Strategy cascade
        ────────────────
        1. Detect input type → special-case checkbox / radio / select / file.
        2. Detect autocomplete attributes → ``_fill_autocomplete``.
        3. Detect date input → ``_fill_date``.
        4. **Try direct** ``locator.fill(value)`` with a *short* timeout.
        5. **Click-to-activate fallback** — click the element, wait for a real
           input to appear (via focus or DOM mutation), then fill *that*.
        6. **Last resort** — ``page.keyboard.type(value)`` into whatever is
           focused.
        """
        input_type = await _get_input_type(locator)

        # ── Special input types ──────────────────────────────────────────
        if input_type in ("checkbox", "radio"):
            checked = value.lower() in ("true", "yes", "on", "1", "checked")
            try:
                await locator.set_checked(checked, timeout=ACTION_TIMEOUT_MS)
                return ActionResult(
                    success=True, action_type="fill", target=target,
                    detail=f"checked={checked}",
                )
            except Exception as e:
                return ActionResult(
                    success=False, action_type="fill", target=target,
                    error=f"Check failed: {e}",
                )

        if input_type == "select":
            try:
                await self._select_option(locator, value)
                return ActionResult(
                    success=True, action_type="fill", target=target,
                    detail=f'selected "{value}"',
                )
            except Exception as e:
                return ActionResult(
                    success=False, action_type="fill", target=target,
                    error=f"Select failed: {e}",
                )

        if input_type == "file":
            try:
                await locator.set_input_files(value, timeout=ACTION_TIMEOUT_MS)
                return ActionResult(
                    success=True, action_type="fill", target=target,
                    detail=f'file: "{value}"',
                )
            except Exception as e:
                return ActionResult(
                    success=False, action_type="fill", target=target,
                    error=f"File input failed: {e}",
                )

        # ── Combobox / autocomplete? ─────────────────────────────────────
        # Combobox inputs MUST be typed character-by-character so the
        # site's JS fires autocomplete / typeahead handlers.
        # Using .fill() injects text silently — no dropdown ever appears.
        is_combobox = await _is_combobox(locator)
        if is_combobox:
            # Two kinds of combobox:
            #   1. Input/textarea combobox — type + pick suggestion
            #   2. Select combobox (<div role=combobox>) — click + pick option
            tag = await locator.evaluate("el => el.tagName.toLowerCase()")
            if tag in ("input", "textarea"):
                return await self._fill_autocomplete(page, locator, value, target)
            else:
                return await self._fill_select_combobox(page, locator, value, target)

        if await _has_autocomplete(locator):
            return await self._fill_autocomplete(page, locator, value, target)

        # ── Date input? ──────────────────────────────────────────────────
        # Detect by type="date" OR by field name (departure, return, date,
        # check-in, check-out).  SPAs use type="text" for date fields.
        is_date_field = input_type in ("date", "datetime-local")
        if not is_date_field:
            # Check if the field name/label/placeholder/id suggests a date
            _date_keywords = ("depart", "return", "date", "check-in", "check-out",
                              "checkin", "checkout", "arrival", "travel")
            field_hint = (target + " " + await locator.evaluate(
                "el => (el.getAttribute('aria-label') || '') + ' ' + "
                "      (el.getAttribute('placeholder') || '') + ' ' + "
                "      (el.getAttribute('name') || '') + ' ' + "
                "      (el.id || '')"
            )).lower()
            is_date_field = any(kw in field_hint for kw in _date_keywords)
        if is_date_field:
            return await self._fill_date(page, locator, value, target, input_type)

        # ── Standard fill (fast-path) ────────────────────────────────────
        # Only for plain inputs — NOT comboboxes (handled above).
        try:
            await locator.fill(value, timeout=3000)
            return ActionResult(
                success=True, action_type="fill", target=target,
                detail=f'filled "{value}"',
            )
        except Exception:
            pass  # Element is probably not editable — fall through

        # ── Click-to-activate fallback ───────────────────────────────────
        # The locator likely points at a decorative overlay div.  Click it
        # to reveal/activate the real input underneath, then fill *that*.
        return await self._click_and_fill(page, locator, value, target)

    async def _click_and_fill(
        self, page: "Page", locator: "Locator", value: str, target: str,
    ) -> ActionResult:
        """
        Click an element to reveal the real input, then type + pick suggestion.

        Modern SPAs (Google Flights, Airbnb, booking sites) use overlay divs.
        After clicking, a real input appears.  We MUST type character-by-
        character (not ``.fill()``) so the site's JS fires autocomplete /
        typeahead handlers.  Then we pick the best suggestion from the
        dropdown — that's the only way the field is truly "filled".
        """
        # ── Step 1: Click to activate ────────────────────────────────────
        try:
            await locator.click(timeout=ACTION_TIMEOUT_MS)
            await page.wait_for_timeout(400)
        except Exception:
            # Overlay label/span intercepting pointer events — force click
            try:
                await locator.click(timeout=ACTION_TIMEOUT_MS, force=True)
                await page.wait_for_timeout(400)
            except Exception as e:
                return ActionResult(
                    success=False, action_type="fill", target=target,
                    error=f"Could not click element to activate it: {e}",
                )

        # ── Step 2: Find the real input ──────────────────────────────────
        real_input = await self._find_activated_input(page, locator)

        # ── Step 3: Type + pick suggestion ───────────────────────────────
        if real_input is not None:
            return await self._type_and_pick(page, real_input, value, target)

        # ── Fallback: type into whatever has focus via keyboard ──────────
        return await self._type_and_pick_blind(page, value, target)

    async def _find_activated_input(
        self, page: "Page", clicked_locator: "Locator",
    ) -> "Locator | None":
        """After clicking an overlay, find the real editable input."""
        # A: Currently focused element
        try:
            focused = page.locator("*:focus")
            if await focused.count() > 0:
                tag = await focused.first.evaluate(
                    "el => el.tagName.toLowerCase()"
                )
                if tag in ("input", "textarea"):
                    return focused.first
        except Exception:
            pass

        # B: Visible input inside the clicked container
        try:
            inner = clicked_locator.locator(
                "input:visible, textarea:visible"
            )
            if await inner.count() > 0:
                return inner.first
        except Exception:
            pass

        # C: Any focused visible input on the page
        try:
            active = page.locator(
                "input:visible:focus, textarea:visible:focus"
            )
            if await active.count() > 0:
                return active.first
        except Exception:
            pass

        return None

    async def _type_and_pick(
        self, page: "Page", input_locator: "Locator", value: str, target: str,
    ) -> ActionResult:
        """
        Type into a located input character-by-character, then pick the best
        autocomplete suggestion if one appears.
        """
        try:
            # Clear any existing value.
            # .fill("") doesn't trigger JS events on SPAs (Google Flights, etc.)
            # so we also do the OS-native select-all chord → Backspace.
            try:
                await input_locator.fill("", timeout=2000)
            except Exception:
                pass  # might fail if contenteditable — that's fine
            await input_locator.click(timeout=2000)
            await page.keyboard.press(_select_all_shortcut())
            await page.wait_for_timeout(50)
            await page.keyboard.press("Backspace")
            await page.wait_for_timeout(150)
        except Exception:
            pass  # best-effort clear

        try:
            await input_locator.press_sequentially(value, delay=50)
        except Exception:
            # contenteditable or weird widget — fall back to keyboard API
            try:
                await input_locator.click(timeout=2000)
                await page.keyboard.type(value, delay=50)
            except Exception as e:
                return ActionResult(
                    success=False, action_type="fill", target=target,
                    error=f"Could not type into activated input: {e}",
                )

        # Wait for autocomplete / typeahead dropdown
        picked = await self._pick_suggestion(page, value, target)
        if picked is not None:
            return picked

        # No dropdown — value might be accepted as-is (e.g. plain text field)
        return ActionResult(
            success=True, action_type="fill", target=target,
            detail=f'typed "{value}" (no autocomplete dropdown)',
        )

    async def _type_and_pick_blind(
        self, page: "Page", value: str, target: str,
    ) -> ActionResult:
        """
        Last resort: type via keyboard into whatever has focus, then try to
        pick a suggestion.
        """
        # Clear any existing value in the focused field first
        try:
            await page.keyboard.press(_select_all_shortcut())
            await page.wait_for_timeout(50)
            await page.keyboard.press("Backspace")
            await page.wait_for_timeout(150)
        except Exception:
            pass  # best-effort clear

        try:
            await page.keyboard.type(value, delay=50)
        except Exception as e:
            return ActionResult(
                success=False, action_type="fill", target=target,
                error=f"All fill strategies exhausted for '{target}': {e}",
            )

        picked = await self._pick_suggestion(page, value, target)
        if picked is not None:
            return picked

        return ActionResult(
            success=True, action_type="fill", target=target,
            detail=f'keyboard-typed "{value}" (blind, no suggestion picked)',
        )

    async def _pick_suggestion(
        self, page: "Page", value: str, target: str,
    ) -> ActionResult | None:
        """
        Wait for an autocomplete / suggestion dropdown, find the best match,
        and click it.  Returns ``None`` if no dropdown appeared.
        """
        await page.wait_for_timeout(800)  # give JS time to render

        # Broad set of selectors covering most frameworks
        suggestion = page.locator(
            "[role='option']:visible, "
            "[role='listbox']:visible li:visible, "
            "[role='listbox']:visible [role='option']:visible, "
            "ul[role='listbox']:visible > *:visible, "
            "[class*='suggestion']:visible, "
            "[class*='Suggestion']:visible, "
            "[class*='autocomplete']:visible li:visible, "
            "[class*='Autocomplete']:visible li:visible, "
            "[class*='dropdown']:visible li:visible, "
            "[class*='Dropdown']:visible li:visible, "
            "[class*='typeahead']:visible li:visible, "
            "[class*='result'] li:visible, "
            "[class*='menu-item']:visible, "
            "[class*='MenuItem']:visible, "
            "datalist option"
        )

        try:
            count = await suggestion.count()
        except Exception:
            return None

        if count == 0:
            return None

        # If a matched item's text is very long with newlines, it's likely
        # a parent container — drill down to children
        if count <= 3:
            try:
                first_text = await suggestion.first.inner_text()
                if first_text.count("\n") > 3 and len(first_text) > 200:
                    # Container element matched — look for children
                    children = suggestion.first.locator(
                        "li, [role='option'], div > span, div > div"
                    )
                    child_count = await children.count()
                    if child_count > count:
                        # Use finer-grained children instead
                        suggestion = children
                        count = child_count
            except Exception:
                pass

        # Score each suggestion
        scored: list[tuple[float, int, str]] = []
        val_lower = value.lower()

        for i in range(min(count, 15)):
            try:
                text = await suggestion.nth(i).inner_text()
                text = text.strip()
                if not text:
                    continue
                # Skip containers (parent elements with tons of text)
                if text.count('\n') > 8 and len(text) > 400:
                    continue
                score = _score_suggestion(val_lower, text)
                scored.append((score, i, text))
            except Exception:
                continue

        if not scored:
            return None

        # Selection strategy: trust the site's ranking.
        # Sites sort suggestions by relevance.  When top items score well,
        # prefer the site's ordering.  Only override if a later item is
        # SIGNIFICANTLY better (>0.20 gap).
        max_score = max(s for s, _, _ in scored)
        threshold = max(max_score - 0.20, 0.50)

        # Among items scoring >= threshold, pick the first one (site's preferred)
        best_idx = -1
        best_score = 0.0
        best_text = ""

        for score, idx, text in scored:
            if score >= threshold:
                best_idx = idx
                best_score = score
                best_text = text
                break  # First item above threshold wins

        # If nothing passed threshold, fall back to absolute best
        if best_idx < 0:
            for score, idx, text in sorted(scored, key=lambda x: -x[0]):
                best_idx = idx
                best_score = score
                best_text = text
                break

        # Click the best match
        if best_score > 0.15 or count == 1:
            try:
                await suggestion.nth(best_idx).click(timeout=ACTION_TIMEOUT_MS)
                await page.wait_for_timeout(300)
                return ActionResult(
                    success=True, action_type="fill", target=target,
                    detail=f'typed "{value}" → selected "{best_text}"',
                )
            except Exception:
                # Suggestion might have vanished — try first visible one
                try:
                    await suggestion.first.click(timeout=ACTION_TIMEOUT_MS)
                    await page.wait_for_timeout(300)
                    first_text = await suggestion.first.inner_text()
                    return ActionResult(
                        success=True, action_type="fill", target=target,
                        detail=f'typed "{value}" → selected "{first_text.strip()}"',
                    )
                except Exception:
                    pass

        return None

    # ━━━ Smart Fill Strategies ━━━

    async def _fill_select_combobox(
        self, page: "Page", locator: "Locator", value: str, target: str,
    ) -> ActionResult:
        """
        Fill a select-style combobox (e.g. trip type: Round trip / One way).

        These are <div role="combobox"> elements that show a listbox of options
        when clicked.  No typing needed — just click to open, then click the
        matching option.
        """
        try:
            # Step 1: Click to open the dropdown
            try:
                await locator.click(timeout=ACTION_TIMEOUT_MS)
            except Exception:
                await locator.click(timeout=ACTION_TIMEOUT_MS, force=True)
            await page.wait_for_timeout(400)

            # Step 2: Find the matching option.
            # Try multiple selector patterns — Material Design, ARIA, plain lists.
            # Some frameworks don't set `:visible` properly, so we try with
            # and without the pseudo-class.
            _OPTION_SELECTORS = [
                "[role='option']:visible",
                "[role='option']",
                "[role='listbox'] li:visible",
                "[role='listbox'] li",
                "[role='menuitem']:visible",
                "[role='menuitemradio']:visible",
                "[role='menuitem']",
                "[role='menuitemradio']",
                "ul:visible li:visible",
            ]
            suggestion = page.locator(_OPTION_SELECTORS[0])
            count = 0

            for sel in _OPTION_SELECTORS:
                suggestion = page.locator(sel)
                count = await suggestion.count()
                if count > 0:
                    break

            if count == 0:
                return ActionResult(
                    success=False, action_type="fill", target=target,
                    error=f"No dropdown options appeared for '{target}'",
                )

            # Find best match by text.
            # Normalize hyphens/dashes so "One-way" matches "One way".
            best_idx = 0
            best_score = 0.0
            best_text = ""
            val_norm = re.sub(r'[-–—]', ' ', value.lower()).strip()

            for i in range(min(count, 20)):
                try:
                    text = await suggestion.nth(i).inner_text()
                    text = text.strip()
                    if not text:
                        continue
                    text_norm = re.sub(r'[-–—]', ' ', text.lower()).strip()
                    score = max(
                        _score_suggestion(val_norm, text_norm),
                        _score_suggestion(value.lower(), text),
                    )
                    if score > best_score:
                        best_score = score
                        best_idx = i
                        best_text = text
                except Exception:
                    continue

            if best_score > 0.15 or count == 1:
                try:
                    await suggestion.nth(best_idx).click(timeout=ACTION_TIMEOUT_MS)
                except Exception:
                    # Option may be in a Material Design overlay that Playwright
                    # considers "not visible" — force-click as fallback
                    await suggestion.nth(best_idx).click(timeout=ACTION_TIMEOUT_MS, force=True)
                await page.wait_for_timeout(300)
                return ActionResult(
                    success=True, action_type="fill", target=target,
                    detail=f'selected "{best_text}"',
                )

            return ActionResult(
                success=False, action_type="fill", target=target,
                error=f"No matching option for '{value}' in dropdown (best: '{best_text}' score={best_score:.2f})",
            )

        except Exception as e:
            return ActionResult(
                success=False, action_type="fill", target=target,
                error=f"Select combobox fill failed: {e}",
            )

    async def _fill_autocomplete(
        self, page: "Page", locator: "Locator", value: str, target: str,
    ) -> ActionResult:
        """
        Fill an autocomplete / combobox field like a human:
        click → clear → type char-by-char → pick suggestion.
        """
        try:
            # Step 1: Click to focus / activate the field
            try:
                await locator.click(timeout=ACTION_TIMEOUT_MS)
            except Exception:
                # Overlay intercepting pointer events — force click
                await locator.click(timeout=ACTION_TIMEOUT_MS, force=True)
            await page.wait_for_timeout(300)

            # Step 2: Clear any pre-filled value (e.g. "Bengaluru")
            #   .fill("") doesn't trigger JS events. The OS-native
            #   select-all chord → Backspace works on native inputs and
            #   contenteditable fields.
            await page.keyboard.press(_select_all_shortcut())
            await page.wait_for_timeout(50)
            await page.keyboard.press("Backspace")
            await page.wait_for_timeout(200)

            # Step 3: Type character-by-character to trigger autocomplete JS
            await page.keyboard.type(value, delay=60)

            # Step 4: Pick the best suggestion from the dropdown
            picked = await self._pick_suggestion(page, value, target)
            if picked is not None:
                return picked

            # No dropdown appeared — press Enter to accept typed value
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(200)

            return ActionResult(
                success=True, action_type="fill", target=target,
                detail=f'typed "{value}" (no autocomplete dropdown detected)',
            )

        except Exception as e:
            return ActionResult(
                success=False, action_type="fill", target=target,
                error=f"Autocomplete fill failed: {e}",
            )

    async def _fill_date(
        self, page: "Page", locator: "Locator", value: str, target: str,
        input_type: str = "",
    ) -> ActionResult:
        """
        Fill a date field.  Handles both native <input type="date"> and
        custom calendar date pickers (click to open, pick day from grid).

        Strategy order depends on the input type:
        - Native ``<input type="date">`` → JS injection / .fill() first
        - SPA date fields (type=text)   → calendar-first (click + pick day
          from grid + Done button)
        """
        date_value = _normalize_date(value)
        day, month_name, year = _parse_date_parts(date_value)
        is_native = input_type in ("date", "datetime-local")

        # ── Native date inputs: JS injection / fill ──────────────────────
        if is_native:
            try:
                selector = await _get_selector(locator)
                if selector:
                    injected = await page.evaluate("""(args) => {
                        const input = document.querySelector(args.selector);
                        if (!input) return false;
                        const nativeSetter = Object.getOwnPropertyDescriptor(
                            window.HTMLInputElement.prototype, 'value').set;
                        nativeSetter.call(input, args.value);
                        input.dispatchEvent(new Event('input', {bubbles: true}));
                        input.dispatchEvent(new Event('change', {bubbles: true}));
                        return input.value === args.value;
                    }""", {"selector": selector, "value": date_value})
                    if injected:
                        return ActionResult(
                            success=True, action_type="fill", target=target,
                            detail=f'date set to "{date_value}" via JS injection',
                        )
            except Exception:
                pass

            try:
                await locator.fill(date_value, timeout=2000)
                return ActionResult(
                    success=True, action_type="fill", target=target,
                    detail=f'date set to "{date_value}"',
                )
            except Exception:
                pass

        # ── SPA calendar approach (primary for non-native inputs) ────────
        # Click the field to open the calendar, then pick the day from the
        # grid.  This is the only reliable way for Google Flights, Airbnb,
        # Booking.com, etc.
        try:
            try:
                await locator.click(timeout=ACTION_TIMEOUT_MS)
            except Exception:
                # Overlay label/span intercepting pointer — force click
                await locator.click(timeout=ACTION_TIMEOUT_MS, force=True)
            await page.wait_for_timeout(800)

            grid = page.locator(
                "[role='grid']:visible, "
                "[class*='calendar' i]:visible, "
                "[class*='DayPicker']:visible, "
                "[class*='datepicker' i]:visible, "
                "[class*='date-picker']:visible, "
                "table[class*='month' i]:visible"
            )
            if await grid.count() > 0:
                result = await self._pick_calendar_date(
                    page, day, month_name, year, target,
                )
                # Ensure calendar is closed
                await self._close_calendar(page)
                return result
        except Exception:
            pass

        # ── Fallback: type the date string ───────────────────────────────
        try:
            try:
                await locator.click(timeout=ACTION_TIMEOUT_MS)
            except Exception:
                await locator.click(timeout=ACTION_TIMEOUT_MS, force=True)
            await page.keyboard.press(_select_all_shortcut())
            await page.wait_for_timeout(50)
            await page.keyboard.press("Backspace")
            await page.wait_for_timeout(100)
            await page.keyboard.type(value, delay=50)
            await self._close_calendar(page)
            return ActionResult(
                success=True, action_type="fill", target=target,
                detail=f'typed date "{value}"',
            )
        except Exception as e:
            await self._close_calendar(page)
            return ActionResult(
                success=False, action_type="fill", target=target,
                error=f"Date fill failed: {e}",
            )

    async def _close_calendar(self, page: "Page") -> None:
        """Ensure any open calendar/date-picker overlay is closed."""
        try:
            # Click Done/Apply/OK/Select/Confirm if present
            done_btn = page.locator(
                "button:has-text('Done'):visible, "
                "button:has-text('Apply'):visible, "
                "button:has-text('OK'):visible, "
                "button:has-text('Select'):visible, "
                "button:has-text('Confirm'):visible, "
                "button:has-text('Save'):visible"
            )
            if await done_btn.count() > 0:
                await done_btn.first.click(timeout=2000)
                await page.wait_for_timeout(300)
                return
        except Exception:
            pass

        try:
            # If calendar grid/overlay is still visible, press Escape
            overlay = page.locator(
                "[role='grid']:visible, "
                "[role='dialog']:visible, "
                "[class*='calendar' i]:visible, "
                "[class*='DayPicker']:visible, "
                "[class*='datepicker' i]:visible, "
                "[class*='date-picker']:visible"
            )
            if await overlay.count() > 0:
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(300)
        except Exception:
            pass

    async def _pick_calendar_date(
        self, page: "Page", day: int, month_name: str, year: int, target: str,
    ) -> ActionResult:
        """
        Pick a date from an open calendar widget.

        Modern calendars (Google Flights, Airbnb, etc.) have this DOM pattern:
          <div role="gridcell">
            <div role="button">
              <div aria-label="Sunday, March 15, 2026">15</div>
            </div>
          </div>

        The gridcell is a layout wrapper — the REAL clickable target is the
        inner element whose aria-label contains the full date text.  Clicking
        the outer gridcell often doesn't register.

        Strategy:
        1. Find element by aria-label containing "March 15, 2026" (most precise)
        2. Find element by aria-label containing "March 15" (no year)
        3. Fallback: gridcell text match + scroll navigation
        """
        day_str = str(day)

        # ── Strategy 0: data-iso (most precise, e.g. Google Flights) ─────
        # Google Flights uses data-iso="2026-04-10" on calendar cells.
        iso = f"{year}-{str(_month_index(month_name) + 1).zfill(2)}-{day_str.zfill(2)}"
        el = page.locator(f"[data-iso='{iso}']:visible")
        if await el.count() > 0:
            # Prefer inner button/clickable
            inner = el.first.locator("[role='button'], button, a")
            click_target = inner.first if await inner.count() > 0 else el.first
            try:
                await click_target.click(timeout=ACTION_TIMEOUT_MS)
                await page.wait_for_timeout(500)
                return ActionResult(
                    success=True, action_type="fill", target=target,
                    detail=f'picked {month_name} {day}, {year} via data-iso',
                )
            except Exception:
                try:
                    await click_target.click(timeout=ACTION_TIMEOUT_MS, force=True)
                    await page.wait_for_timeout(500)
                    return ActionResult(
                        success=True, action_type="fill", target=target,
                        detail=f'picked {month_name} {day}, {year} via data-iso (force)',
                    )
                except Exception:
                    pass

        # ── Strategy 1: aria-label with full date ────────────────────────
        # e.g., aria-label="Sunday, March 15, 2026"
        for label_pattern in [
            f"{month_name} {day}, {year}",       # "March 15, 2026"
            f"{month_name} {day_str}, {year}",   # same but explicit str
            f"{day} {month_name} {year}",         # "15 March 2026"
        ]:
            sel = f"[aria-label*='{label_pattern}']:visible"
            el = page.locator(sel)
            if await el.count() > 0:
                await el.first.click(timeout=ACTION_TIMEOUT_MS)
                await page.wait_for_timeout(500)
                return ActionResult(
                    success=True, action_type="fill", target=target,
                    detail=f'picked {month_name} {day}, {year} from calendar',
                )

        # ── Strategy 2: aria-label without year ──────────────────────────
        # e.g., aria-label="March 15"
        sel = f"[aria-label*='{month_name} {day_str}']:visible"
        el = page.locator(sel)
        if await el.count() > 0:
            # Pick the one whose aria-label also contains the year (if any)
            for i in range(await el.count()):
                label = await el.nth(i).evaluate(
                    "el => el.getAttribute('aria-label') || ''"
                )
                if str(year) in label:
                    await el.nth(i).click(timeout=ACTION_TIMEOUT_MS)
                    await page.wait_for_timeout(500)
                    return ActionResult(
                        success=True, action_type="fill", target=target,
                        detail=f'picked {month_name} {day}, {year} from calendar',
                    )
            # No year match — click first (most likely correct)
            await el.first.click(timeout=ACTION_TIMEOUT_MS)
            await page.wait_for_timeout(500)
            return ActionResult(
                success=True, action_type="fill", target=target,
                detail=f'picked {month_name} {day} from calendar',
            )

        # ── Strategy 3: Navigate months + click gridcell button ──────────
        # Some calendars show one month at a time with next/prev buttons.
        max_nav = 18
        for _ in range(max_nav):
            current_month = await page.evaluate("""() => {
                const headers = document.querySelectorAll(
                    '[role="heading"], [class*="month"], [class*="Month"],'
                    + '[class*="header"], [class*="Header"], [class*="title"],'
                    + '[class*="Title"]'
                );
                for (const h of headers) {
                    if (!h.offsetParent && !h.offsetWidth) continue;
                    const text = h.textContent?.trim() || '';
                    const match = text.match(/(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{4})/i);
                    if (match) return { month: match[1], year: parseInt(match[2]) };
                }
                return null;
            }""")

            if current_month:
                target_month_idx = _month_index(month_name)
                current_month_idx = _month_index(current_month["month"])
                target_total = year * 12 + target_month_idx
                current_total = current_month["year"] * 12 + current_month_idx

                if current_total == target_total:
                    break
                elif current_total < target_total:
                    next_btn = page.locator(
                        "button[aria-label*='Next']:visible, "
                        "button[aria-label*='next']:visible, "
                        "button[aria-label*='Forward']:visible, "
                        "[class*='next']:visible:not(li), "
                        "[class*='Next']:visible:not(li)"
                    )
                    if await next_btn.count() > 0:
                        await next_btn.first.click(timeout=3000)
                        await page.wait_for_timeout(300)
                    else:
                        break
                else:
                    prev_btn = page.locator(
                        "button[aria-label*='Prev']:visible, "
                        "button[aria-label*='prev']:visible, "
                        "button[aria-label*='Back']:visible, "
                        "[class*='prev']:visible:not(li), "
                        "[class*='Prev']:visible:not(li)"
                    )
                    if await prev_btn.count() > 0:
                        await prev_btn.first.click(timeout=3000)
                        await page.wait_for_timeout(300)
                    else:
                        break
            else:
                break

        # Now try to click a button/div inside a gridcell matching our day
        # Look for inner buttons with the right day text
        buttons_in_grid = page.locator(
            f"[role='gridcell'] [role='button']:has-text('{day_str}'):visible, "
            f"[role='gridcell'] button:has-text('{day_str}'):visible"
        )
        count = await buttons_in_grid.count()
        for i in range(count):
            text = (await buttons_in_grid.nth(i).inner_text()).strip()
            if text == day_str or text.startswith(day_str + "\n"):
                await buttons_in_grid.nth(i).click(timeout=ACTION_TIMEOUT_MS)
                await page.wait_for_timeout(500)
                return ActionResult(
                    success=True, action_type="fill", target=target,
                    detail=f'picked day {day_str} from calendar grid',
                )

        # Final fallback: try gridcells directly with exact text
        day_cells = page.locator(f"[role='gridcell']:has-text('{day_str}'):visible")
        count = await day_cells.count()
        for i in range(count):
            text = (await day_cells.nth(i).inner_text()).strip()
            lines = text.split("\n")
            if lines[0].strip() == day_str:
                await day_cells.nth(i).click(timeout=ACTION_TIMEOUT_MS)
                await page.wait_for_timeout(500)
                return ActionResult(
                    success=True, action_type="fill", target=target,
                    detail=f'picked day {day_str} from calendar (gridcell click)',
                )

        # ── Strategy 4: Class-based calendar cells (no ARIA roles) ───────
        # Many sites (MakeMyTrip, etc.) use <td>, <div class="day">, etc.
        class_cells = page.locator(
            f"td:has-text('{day_str}'):visible, "
            f"[class*='day' i]:has-text('{day_str}'):visible, "
            f"[data-day='{day_str}']:visible, "
            f"[data-date]:has-text('{day_str}'):visible"
        )
        count = await class_cells.count()
        for i in range(count):
            try:
                text = (await class_cells.nth(i).inner_text()).strip()
                first_line = text.split("\n")[0].strip()
                if first_line == day_str:
                    await class_cells.nth(i).click(timeout=ACTION_TIMEOUT_MS)
                    await page.wait_for_timeout(500)
                    return ActionResult(
                        success=True, action_type="fill", target=target,
                        detail=f'picked day {day_str} from calendar (class-based)',
                    )
            except Exception:
                continue

        return ActionResult(
            success=False, action_type="fill", target=target,
            error=f"Could not find {month_name} {day}, {year} in calendar",
        )

    async def _select_option(self, locator: "Locator", value: str) -> None:
        """Select an option from a <select> element by value or label."""
        try:
            # Try by value first
            await locator.select_option(value=value, timeout=ACTION_TIMEOUT_MS)
            return
        except Exception:
            pass

        try:
            # Try by label
            await locator.select_option(label=value, timeout=ACTION_TIMEOUT_MS)
            return
        except Exception:
            pass

        # Try partial label match
        options = await locator.locator("option").all_inner_texts()
        value_lower = value.lower()
        for opt_text in options:
            if value_lower in opt_text.lower():
                await locator.select_option(label=opt_text, timeout=ACTION_TIMEOUT_MS)
                return

        raise ValueError(f"Option '{value}' not found in select element")


# ━━━ Helper Functions ━━━


async def _smart_click(page: "Page", locator: "Locator", timeout: int = 5000) -> bool:
    """
    Click with escalating fallbacks for overlays / interception.

    Strategy cascade:
    1. Normal click
    2. Force click (bypasses overlap checks)
    3. Scroll into view + JS click
    4. Mouse click at element coordinates
    """
    # Strategy 1: Normal click
    try:
        await locator.click(timeout=timeout)
        return True
    except Exception:
        pass

    # Strategy 2: Force click
    try:
        await locator.click(timeout=timeout, force=True)
        return True
    except Exception:
        pass

    # Strategy 3: Scroll into view + JS click
    try:
        await locator.scroll_into_view_if_needed(timeout=2000)
        await locator.evaluate("el => el.click()")
        return True
    except Exception:
        pass

    # Strategy 4: Mouse coordinates
    try:
        box = await locator.bounding_box()
        if box:
            await page.mouse.click(
                box['x'] + box['width'] / 2,
                box['y'] + box['height'] / 2,
            )
            return True
    except Exception:
        pass

    # Strategy 5: Click nearest clickable ancestor
    # Handles cases where the locator points to an inner <span> inside
    # an <a> or <button>, and the inner element can't receive clicks.
    try:
        clicked = await locator.evaluate("""el => {
            const ancestor = el.closest('a, button, [role="link"], [role="button"]');
            if (ancestor && ancestor !== el) {
                ancestor.click();
                return true;
            }
            return false;
        }""")
        if clicked:
            return True
    except Exception:
        pass

    return False


async def _wait_for_stable(page: "Page", timeout_ms: int = NAVIGATION_WAIT_MS) -> None:
    """Wait for the page to be stable after a navigation-triggering action."""
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass
    try:
        # Short wait for any AJAX / dynamic rendering
        await page.wait_for_load_state("networkidle", timeout=1000)
    except Exception:
        pass  # networkidle can timeout on pages with persistent connections


async def _wait_after_click(page: "Page") -> None:
    """
    Smart post-click wait — only does a full stable wait if the click
    triggered a navigation.  For non-navigating clicks (dropdown toggle,
    checkbox, menu expand) just a brief settle is enough.
    """
    url_before = page.url
    # Brief pause to let any navigation start
    await page.wait_for_timeout(150)
    if page.url != url_before:
        # URL changed — real navigation, wait for it to finish
        await _wait_for_stable(page)
    else:
        # No navigation — just let dynamic content settle briefly
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=300)
        except Exception:
            pass


async def _get_input_type(locator: "Locator") -> str:
    """Get the input type of an element."""
    try:
        tag = await locator.evaluate("el => el.tagName.toLowerCase()")
        if tag == "select":
            return "select"
        elif tag == "textarea":
            return "textarea"
        return await locator.get_attribute("type") or "text"
    except Exception:
        return "text"


async def _is_combobox(locator: "Locator") -> bool:
    """Check if an element is a combobox (must be typed, not .fill()'d)."""
    try:
        return await locator.evaluate("""el => {
            return el.getAttribute('role') === 'combobox' ||
                   el.getAttribute('aria-haspopup') === 'listbox' ||
                   el.getAttribute('aria-haspopup') === 'true';
        }""")
    except Exception:
        return False


async def _has_autocomplete(locator: "Locator") -> bool:
    """Check if an element has autocomplete behavior."""
    try:
        return await locator.evaluate("""el => {
            return el.getAttribute('autocomplete') !== 'off' && (
                el.getAttribute('aria-autocomplete') === 'list' ||
                el.getAttribute('aria-autocomplete') === 'both' ||
                el.getAttribute('list') !== null ||
                el.dataset.autocomplete !== undefined ||
                el.classList.toString().includes('autocomplete') ||
                el.classList.toString().includes('typeahead')
            );
        }""")
    except Exception:
        return False


async def _get_selector(locator: "Locator") -> str | None:
    """Get a CSS selector string for a locator's element."""
    try:
        return await locator.evaluate("""el => {
            if (el.id) return '#' + CSS.escape(el.id);
            if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';
            return null;
        }""")
    except Exception:
        return None


def _score_suggestion(query: str, candidate_text: str) -> float:
    """
    Score how well a suggestion matches the query.

    Key improvement over the old _fuzzy_score: WORD BOUNDARY awareness.
    "Paris" matches "Paris, France" (word boundary) much better than
    "Portland Parish" (substring within word "Parish").
    """
    if not query or not candidate_text:
        return 0.0

    q = query.lower().strip()
    # Use first line only — multi-line items have metadata like
    # "Mumbai (BOM)\nIndia\n..."
    full = candidate_text.lower().strip()
    first_line = full.split('\n')[0].strip()

    # Exact match on first line
    if q == first_line:
        return 1.0

    # First line starts with query (e.g., "delhi" → "delhi, india")
    if first_line.startswith(q):
        return 0.97

    # Query matches first word/phrase before comma/paren
    # "mumbai" → "mumbai (bom)" or "mumbai, india"
    first_chunk = re.split(r'[,()\[\]|]', first_line)[0].strip()
    if first_chunk == q:
        return 0.96
    if first_chunk.startswith(q) and len(q) >= 3:
        return 0.93

    # Query at a WORD BOUNDARY in first line
    # "paris" in "paris, france" → word boundary → high score
    # "paris" in "portland parish" → inside word → low score
    try:
        boundary_match = re.search(r'\b' + re.escape(q) + r'\b', first_line)
        if boundary_match:
            pos = boundary_match.start()
            if pos == 0:
                return 0.95  # At the very start
            return 0.80     # Word boundary but not at start
    except re.error:
        pass

    # Multi-word query: check each word independently at word boundaries
    q_words = q.split()
    if len(q_words) > 1:
        matched = 0
        for w in q_words:
            try:
                if re.search(r'\b' + re.escape(w) + r'\b', first_line):
                    matched += 1
            except re.error:
                if w in first_line:
                    matched += 1
        ratio = matched / len(q_words)
        if ratio >= 1.0:
            return 0.88  # All query words found at word boundaries
        elif ratio >= 0.5:
            return 0.65  # At least half the words found

    # Query is a substring but NOT at word boundary
    # "paris" in "parish" — this is a FALSE positive, score low
    if q in first_line:
        return 0.35  # Significantly lower than word-boundary match

    # Single-word overlap as last resort
    q_set = set(q_words) if len(q_words) > 1 else set(q.split())
    c_set = set(re.split(r'[\s,()\[\]|]+', first_line))
    overlap = len(q_set & c_set)
    if overlap > 0:
        ratio = overlap / len(q_set)
        return 0.3 + ratio * 0.3

    return 0.0


def _normalize_date(value: str) -> str:
    """
    Try to normalize a date string to YYYY-MM-DD format for HTML date inputs.

    Handles common formats like "March 15, 2026", "15/03/2026", "2026-03-15".
    Falls back to returning the original string if parsing fails.
    """
    import re

    # Already in ISO format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value

    # Month name formats: "March 15, 2026" or "15 March 2026"
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "jun": "06", "jul": "07", "aug": "08", "sep": "09",
        "oct": "10", "nov": "11", "dec": "12",
    }

    # "March 15, 2026" or "March 15 2026"
    m = re.match(r"(\w+)\s+(\d{1,2}),?\s*(\d{4})", value, re.IGNORECASE)
    if m:
        month_name = m.group(1).lower()
        if month_name in months:
            return f"{m.group(3)}-{months[month_name]}-{int(m.group(2)):02d}"

    # "15 March 2026"
    m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", value, re.IGNORECASE)
    if m:
        month_name = m.group(2).lower()
        if month_name in months:
            return f"{m.group(3)}-{months[month_name]}-{int(m.group(1)):02d}"

    # "DD/MM/YYYY" or "MM/DD/YYYY" — ambiguous, assume DD/MM/YYYY
    m = re.match(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", value)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if d > 12:  # must be DD/MM/YYYY
            return f"{y}-{mo:02d}-{d:02d}"
        elif mo > 12:  # must be MM/DD/YYYY
            return f"{y}-{d:02d}-{mo:02d}"
        else:
            # Ambiguous — assume DD/MM/YYYY
            return f"{y}-{mo:02d}-{d:02d}"

    return value  # Return as-is if we can't parse


def _parse_date_parts(iso_date: str) -> tuple[int, str, int]:
    """
    Parse an ISO date (YYYY-MM-DD) or freeform date into (day, month_name, year).
    Falls back to (15, "March", 2026) if parsing fails.
    """
    import re

    _month_names = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", iso_date)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        month_name = _month_names[month] if 1 <= month <= 12 else "January"
        return day, month_name, year

    # Try to extract from freeform
    months_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
        "nov": 11, "dec": 12,
    }
    for word in iso_date.lower().split():
        if word.rstrip(",") in months_map:
            month_idx = months_map[word.rstrip(",")]
            month_name = _month_names[month_idx]
            # Extract numbers
            nums = re.findall(r"\d+", iso_date)
            if len(nums) >= 2:
                day = int(nums[0]) if int(nums[0]) <= 31 else int(nums[1])
                year = int(nums[-1]) if int(nums[-1]) > 31 else 2026
                return day, month_name, year

    return 15, "March", 2026  # fallback


def _month_index(month_name: str) -> int:
    """Convert month name to 0-based index."""
    months = {
        "january": 0, "february": 1, "march": 2, "april": 3,
        "may": 4, "june": 5, "july": 6, "august": 7,
        "september": 8, "october": 9, "november": 10, "december": 11,
    }
    return months.get(month_name.lower(), 0)
