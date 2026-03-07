"""
CDP Accessibility Tree Extractor — uses Chrome DevTools Protocol to
extract the browser's native accessibility tree.

The AX tree sees through shadow DOM, computes accessible names per
the W3C Accname spec, and exposes ARIA states (expanded, selected,
checked, pressed) — all things the DOM-query approach can miss.

Each extracted element gets a Playwright locator strategy derived
from its AX role and name, with a `backendDOMNodeId` fallback for
guaranteed targeting via CDP DOM resolution.

Usage:
    extractor = AXTreeExtractor()
    result = await extractor.extract(page)
    # result.elements  — flat list of AXElement
    # result.landmarks — ["navigation", "main", "search"]
    # result.alerts    — live region text ["Error: email required"]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page

logger = logging.getLogger(__name__)

# ARIA roles we consider "interesting" for LLM snapshots
_INTERACTIVE_ROLES = frozenset({
    "button", "checkbox", "combobox", "link", "listbox",
    "menuitem", "option", "radio", "searchbox", "slider",
    "spinbutton", "switch", "tab", "textbox", "treeitem",
    "menuitemcheckbox", "menuitemradio",
})

# Structural / landmark roles — extracted as context, not numbered
_LANDMARK_ROLES = frozenset({
    "banner", "complementary", "contentinfo", "form",
    "main", "navigation", "region", "search",
})

# Live region roles — capture text content for alerts
_LIVE_REGION_ROLES = frozenset({
    "alert", "alertdialog", "log", "marquee", "status", "timer",
})

# Roles to skip entirely (structural noise)
_SKIP_ROLES = frozenset({
    "none", "presentation", "generic", "group", "list",
    "listitem", "paragraph", "Section", "separator",
    "LayoutTable", "LayoutTableRow", "LayoutTableCell",
    "LineBreak", "InlineTextBox", "StaticText",
    "RootWebArea", "WebArea",
})

# Playwright get_by_role() accepts these exact role names
_PLAYWRIGHT_ROLES = frozenset({
    "button", "checkbox", "combobox", "heading", "link",
    "listbox", "menuitem", "option", "radio", "searchbox",
    "slider", "spinbutton", "switch", "tab", "tabpanel",
    "textbox", "treeitem",
})


@dataclass
class AXElement:
    """A single element extracted from the CDP accessibility tree."""

    role: str = ""
    name: str = ""
    value: str = ""
    description: str = ""
    disabled: bool = False
    required: bool = False
    expanded: bool | None = None
    selected: bool | None = None
    checked: str | None = None  # "true" | "false" | "mixed" | None
    focused: bool = False
    level: int | None = None
    autocomplete: str = ""
    haspopup: str = ""
    readonly: bool = False
    multiline: bool = False
    invalid: str = ""
    backend_dom_node_id: int | None = None
    ax_node_id: str = ""
    children_ids: list[str] = field(default_factory=list)
    context: str = ""  # nearest ancestor heading/label for disambiguation

    # Derived by DOM augmentation
    tag: str = ""
    input_type: str = ""
    placeholder: str = ""
    options: list[str] = field(default_factory=list)
    href: str = ""
    selector: str = ""  # CSS selector fallback


@dataclass
class AXTreeResult:
    """Result of AX tree extraction."""

    elements: list[AXElement] = field(default_factory=list)
    landmarks: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    total_ax_nodes: int = 0


class AXTreeExtractor:
    """
    Extracts interactive elements from the page using the CDP
    Accessibility.getFullAXTree protocol.

    Why CDP over page.accessibility.snapshot()?
    - CDP gives `backendDOMNodeId` on every node → guaranteed DOM targeting
    - Flat list is easier to process than nested tree
    - Full control over filtering logic
    """

    async def extract(self, page: "Page") -> AXTreeResult:
        """
        Extract the accessibility tree from the page.

        Returns an AXTreeResult with interactive elements, landmarks,
        and live region alerts.
        """
        cdp = None
        try:
            cdp = await page.context.new_cdp_session(page)
            raw = await cdp.send("Accessibility.getFullAXTree")
            nodes = raw.get("nodes", [])
        except Exception as e:
            logger.warning(f"CDP AX tree extraction failed: {e}")
            return AXTreeResult()
        finally:
            if cdp:
                try:
                    await cdp.detach()
                except Exception:
                    pass

        return self._process_nodes(nodes)

    def _process_nodes(self, nodes: list[dict[str, Any]]) -> AXTreeResult:
        """Process raw CDP AX nodes into structured result."""
        elements: list[AXElement] = []
        landmarks: list[str] = []
        alerts: list[str] = []

        # Build parent map and heading index from raw nodes for context resolution
        parent_map: dict[str, str] = {}   # nodeId → parentId
        heading_map: dict[str, str] = {}  # nodeId → heading name (for headings only)
        for node in nodes:
            nid = node.get("nodeId", "")
            pid = node.get("parentId", "")
            if nid and pid:
                parent_map[nid] = pid
            role = _get_str(node, "role")
            name = _get_str(node, "name")
            if role == "heading" and name:
                heading_map[nid] = name[:80]

        for node in nodes:
            if node.get("ignored", False):
                continue

            role = _get_str(node, "role")
            name = _get_str(node, "name")
            value = _get_str(node, "value")

            # Extract properties
            props = self._extract_properties(node)

            # Skip noise
            if role in _SKIP_ROLES:
                continue
            # Skip text-only nodes (StaticText, etc.)
            if role.startswith("Static") or role == "InlineTextBox":
                continue

            # Landmarks — capture for context
            if role in _LANDMARK_ROLES:
                label = f"{role}"
                if name:
                    label += f": {name}"
                landmarks.append(label)
                continue

            # Live regions — capture alert/status text
            if role in _LIVE_REGION_ROLES:
                text = name or value
                if text and len(text.strip()) > 0:
                    alerts.append(text.strip()[:200])
                continue

            # Headings — include for page structure context
            if role == "heading":
                if name:
                    elem = AXElement(
                        role="heading",
                        name=name[:100],
                        level=props.get("level"),
                        ax_node_id=node.get("nodeId", ""),
                        backend_dom_node_id=node.get("backendDOMNodeId"),
                    )
                    elements.append(elem)
                continue

            # Interactive elements
            if role in _INTERACTIVE_ROLES:
                # Resolve nearest heading ancestor for disambiguation
                ctx = self._find_ancestor_heading(
                    node.get("nodeId", ""), parent_map, heading_map
                )
                elem = AXElement(
                    role=role,
                    name=name[:100] if name else "",
                    value=value[:200] if value else "",
                    description=props.get("description", ""),
                    disabled=props.get("disabled", False),
                    required=props.get("required", False),
                    expanded=props.get("expanded"),
                    selected=props.get("selected"),
                    checked=props.get("checked"),
                    focused=props.get("focused", False),
                    level=props.get("level"),
                    autocomplete=props.get("autocomplete", ""),
                    haspopup=props.get("haspopup", ""),
                    readonly=props.get("readonly", False),
                    multiline=props.get("multiline", False),
                    invalid=props.get("invalid", ""),
                    backend_dom_node_id=node.get("backendDOMNodeId"),
                    ax_node_id=node.get("nodeId", ""),
                    children_ids=node.get("childIds", []),
                    context=ctx,
                )
                elements.append(elem)
                continue

            # Image with alt text — include for context
            if role == "image" or role == "img":
                if name and len(name) > 3:
                    elem = AXElement(
                        role="image",
                        name=name[:100],
                        ax_node_id=node.get("nodeId", ""),
                        backend_dom_node_id=node.get("backendDOMNodeId"),
                    )
                    elements.append(elem)
                continue

        return AXTreeResult(
            elements=elements,
            landmarks=landmarks,
            alerts=alerts,
            total_ax_nodes=len(nodes),
        )

    def _extract_properties(self, node: dict[str, Any]) -> dict[str, Any]:
        """Extract typed properties from a CDP AX node."""
        props: dict[str, Any] = {}

        for prop in node.get("properties", []):
            name = prop.get("name", "")
            val_obj = prop.get("value", {})
            val = val_obj.get("value") if isinstance(val_obj, dict) else val_obj

            if name in ("disabled", "required", "focused", "readonly",
                        "multiline"):
                props[name] = bool(val)
            elif name in ("expanded", "selected"):
                props[name] = bool(val) if val is not None else None
            elif name == "checked":
                # Can be "true", "false", "mixed"
                if isinstance(val, bool):
                    props[name] = "true" if val else "false"
                elif isinstance(val, str):
                    props[name] = val
            elif name in ("level",):
                try:
                    props[name] = int(val)
                except (TypeError, ValueError):
                    pass
            elif name in ("autocomplete", "haspopup", "invalid",
                          "description", "roledescription"):
                props[name] = str(val) if val else ""

        return props

    @staticmethod
    def _find_ancestor_heading(
        node_id: str,
        parent_map: dict[str, str],
        heading_map: dict[str, str],
        max_depth: int = 15,
    ) -> str:
        """Walk up the AX tree to find the nearest heading ancestor."""
        current = node_id
        for _ in range(max_depth):
            parent = parent_map.get(current)
            if not parent:
                break
            heading = heading_map.get(parent)
            if heading:
                return heading
            current = parent
        return ""

    async def augment_from_dom(
        self,
        page: "Page",
        elements: list[AXElement],
    ) -> list[AXElement]:
        """
        Augment AX elements with data the accessibility tree lacks:
        - tag name and input type
        - placeholder text
        - <select> option values
        - href for links
        - CSS selector fallback

        Also performs DOM gap-fill: returns a list of AXElement for
        interactive DOM elements (submit buttons, etc.) that the
        CDP accessibility tree missed entirely.
        """
        # Collect all backend DOM node IDs
        node_ids = [
            e.backend_dom_node_id for e in elements
            if e.backend_dom_node_id is not None
        ]

        if not node_ids:
            return []

        try:
            # Single JS evaluate that resolves DOM info for all nodes at once
            # We use CDP to resolve nodes first, then evaluate JS
            dom_data = await page.evaluate("""(nodeIds) => {
                // Build results for each element by walking all DOM elements
                // and matching them. Since we can't directly resolve
                // backendDOMNodeIds from JS, we use a different approach:
                // query all interactive elements and build a lookup.
                const results = {};
                
                // Query ALL interactive + link elements
                const allEls = document.querySelectorAll(
                    'input, textarea, select, button, a[href], ' +
                    '[role="button"], [role="link"], [role="textbox"], ' +
                    '[role="combobox"], [role="searchbox"], [role="checkbox"], ' +
                    '[role="radio"], [role="tab"], [role="menuitem"], ' +
                    '[role="option"], [role="switch"], [role="slider"], ' +
                    '[role="spinbutton"], [role="treeitem"], ' +
                    '[contenteditable="true"], img[alt]'
                );
                
                let idx = 0;
                for (const el of allEls) {
                    // Skip invisible
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 && rect.height === 0) continue;
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') continue;
                    
                    const tag = el.tagName.toLowerCase();
                    
                    // Build a CSS selector
                    let css = '';
                    if (el.id) {
                        css = '#' + CSS.escape(el.id);
                    } else if (el.getAttribute('aria-label')) {
                        const label = el.getAttribute('aria-label');
                        css = tag + '[aria-label="' + label.replace(/"/g, '\\\\"') + '"]';
                    } else if (el.name) {
                        css = tag + '[name="' + el.name + '"]';
                    }
                    
                    // Collect options for select
                    const options = [];
                    if (tag === 'select') {
                        for (const opt of el.options) {
                            if (opt.value) options.push(opt.textContent.trim());
                        }
                    }
                    
                    // Build label
                    let name = '';
                    if (el.labels && el.labels.length > 0) name = el.labels[0].textContent.trim();
                    if (!name) name = el.getAttribute('aria-label') || '';
                    if (!name) name = el.getAttribute('placeholder') || '';
                    if (!name && (tag === 'input' || tag === 'button')) name = el.value || '';
                    if (!name) name = (el.textContent || '').trim().substring(0, 80);
                    
                    results[idx] = {
                        tag: tag,
                        type: el.type || '',
                        placeholder: el.getAttribute('placeholder') || '',
                        options: options.slice(0, 20),
                        href: el.href || '',
                        selector: css,
                        name: name.substring(0, 100),
                        value: (el.value || (tag !== 'input' && tag !== 'textarea' && tag !== 'select' ? (el.textContent || '').trim().substring(0, 200) : '') || ''),
                    };
                    idx++;
                }
                return results;
            }""", [])

            # Match AX elements to DOM elements by name/role similarity
            dom_list = list(dom_data.values()) if isinstance(dom_data, dict) else []
            unmatched = self._match_ax_to_dom(elements, dom_list)
            return self._build_gap_elements(unmatched, elements)

        except Exception as e:
            logger.warning(f"DOM augmentation failed: {e}")
            return []

    def _match_ax_to_dom(
        self,
        ax_elements: list[AXElement],
        dom_elements: list[dict],
    ) -> list[dict]:
        """
        Match AX elements to DOM elements by name and role similarity.
        
        Since we can't resolve backendDOMNodeId from JS directly,
        we match by accessible name + role as the primary key.

        Returns unmatched DOM elements for gap-fill processing.
        """
        # Build DOM lookup by name
        used_indices: set[int] = set()

        for ax_el in ax_elements:
            if not ax_el.name and not ax_el.value:
                continue

            best_idx = -1
            best_score = 0.0
            ax_name = (ax_el.name or ax_el.value or "").lower().strip()

            for i, dom_el in enumerate(dom_elements):
                if i in used_indices:
                    continue

                dom_name = dom_el.get("name", "").lower().strip()
                # For input/button elements, also consider value as name
                if not dom_name:
                    dom_tag = dom_el.get("tag", "")
                    if dom_tag in ("input", "button"):
                        dom_name = dom_el.get("value", "").lower().strip()
                if not dom_name:
                    continue

                # Exact match
                if ax_name == dom_name:
                    score = 1.0
                elif ax_name in dom_name or dom_name in ax_name:
                    score = 0.8
                else:
                    continue

                # Boost if role matches tag
                tag = dom_el.get("tag", "")
                role_tag_map = {
                    "textbox": ("input", "textarea"),
                    "combobox": ("select", "input", "div"),
                    "searchbox": ("input",),
                    "button": ("button", "input"),
                    "link": ("a",),
                    "checkbox": ("input",),
                    "radio": ("input",),
                    "spinbutton": ("input",),
                    "slider": ("input",),
                }
                expected_tags = role_tag_map.get(ax_el.role, ())
                if tag in expected_tags:
                    score += 0.1

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0 and best_score >= 0.7:
                used_indices.add(best_idx)
                dom_el = dom_elements[best_idx]
                ax_el.tag = dom_el.get("tag", "")
                ax_el.input_type = dom_el.get("type", "")
                ax_el.placeholder = dom_el.get("placeholder", "")
                ax_el.options = dom_el.get("options", [])
                ax_el.href = dom_el.get("href", "")
                ax_el.selector = dom_el.get("selector", "")

        return [
            dom_elements[i]
            for i in range(len(dom_elements))
            if i not in used_indices
        ]

    def _build_gap_elements(
        self,
        unmatched_dom: list[dict],
        ax_elements: list[AXElement],
    ) -> list[AXElement]:
        """
        Convert unmatched DOM elements into AXElement entries for
        interactive elements that the CDP accessibility tree missed.

        Focuses on submit buttons and button inputs — the types most
        commonly dropped by the AX tree on complex pages.  Generic
        <button> elements are skipped because the AX tree rarely
        misses them; unmatched <button>s are usually noise (size
        filters, pagination, etc.).
        """
        _GAP_FILL_INPUT_TYPES = {"submit", "button", "image"}

        # Cap to avoid flooding the snapshot on huge pages
        _MAX_GAP_ELEMENTS = 10

        # Known AX names for dedup (case-insensitive)
        known_names: set[str] = set()
        known_selectors: set[str] = set()
        for ax_el in ax_elements:
            if ax_el.name:
                known_names.add(ax_el.name.lower().strip())
            if ax_el.selector:
                known_selectors.add(ax_el.selector)

        gaps: list[AXElement] = []
        for dom_el in unmatched_dom:
            tag = dom_el.get("tag", "")
            input_type = dom_el.get("type", "")
            name = dom_el.get("name", "")
            value = dom_el.get("value", "")
            selector = dom_el.get("selector", "")

            # Only accept submit/button input elements (not generic <button>)
            is_candidate = (
                tag == "input" and input_type in _GAP_FILL_INPUT_TYPES
            )
            if not is_candidate:
                continue

            # Use value as label for inputs, name for buttons
            label = name or value
            if not label and not selector:
                continue  # un-identifiable, skip

            # Skip if already represented in the AX tree
            if selector and selector in known_selectors:
                continue
            if label:
                label_lower = label.lower().strip()
                if any(
                    label_lower in k or k in label_lower
                    for k in known_names
                ):
                    continue

            gaps.append(AXElement(
                role="button",
                name=label or "",
                tag=tag,
                input_type=input_type,
                selector=selector,
                disabled=dom_el.get("disabled", False),
            ))
            if len(gaps) >= _MAX_GAP_ELEMENTS:
                break

        if gaps:
            logger.info(
                f"DOM gap-fill: added {len(gaps)} element(s) missing "
                f"from AX tree: {[g.name for g in gaps]}"
            )
        return gaps

    async def resolve_locators(
        self,
        page: "Page",
        elements: list[AXElement],
    ) -> dict[int, tuple[str, str]]:
        """
        Derive the best Playwright locator for each AX element.

        Returns a dict mapping element index → (strategy, value).
        Strategy is one of: "role", "label", "selector", "name", "skip".

        This method assigns locators based on AX metadata alone —
        no browser round-trips.  Locator uniqueness is verified
        lazily at action time by _find_element's fallback cascade.
        """
        locators: dict[int, tuple[str, str]] = {}

        for i, el in enumerate(elements):
            # Skip non-interactive (headings, images)
            if el.role not in _INTERACTIVE_ROLES:
                locators[i] = ("skip", "")
                continue

            # Strategy 1: Role-based locator (preferred — survives redesigns)
            if el.role in _PLAYWRIGHT_ROLES and el.name:
                locators[i] = ("role", f'{el.role}::{el.name}')
                continue

            # Strategy 2: Label-based locator (for form inputs)
            if el.name and el.role in ("textbox", "combobox", "searchbox",
                                        "spinbutton", "slider"):
                locators[i] = ("label", el.name)
                continue

            # Strategy 3: CSS selector fallback
            if el.selector:
                locators[i] = ("selector", el.selector)
                continue

            # No locator found — will rely on text/fuzzy matching
            locators[i] = ("name", el.name or el.value or "")

        return locators


# ━━━ Helpers ━━━

def _get_str(node: dict, key: str) -> str:
    """Extract a string value from a CDP AX node field."""
    obj = node.get(key)
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return str(obj.get("value", ""))
    return str(obj)
