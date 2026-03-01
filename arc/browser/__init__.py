"""
Browser control engine — interactive browser automation for Arc.

Provides a smart browser engine that handles page analysis, element
finding, form filling, and interactive navigation. The LLM makes
page-level decisions; the engine executes actions mechanically.

Components:
    BrowserEngine   — Playwright lifecycle, navigation, action dispatch
    PageAnalyzer    — converts raw DOM / accessibility tree to structured text
    ActionExecutor  — smart fill/click/select strategies per input type
    HumanAssist     — escalation to user for CAPTCHAs, login walls, banners
"""

from arc.browser.engine import BrowserEngine
from arc.browser.snapshot import PageAnalyzer
from arc.browser.actions import ActionExecutor
from arc.browser.human import HumanAssist

__all__ = ["BrowserEngine", "PageAnalyzer", "ActionExecutor", "HumanAssist"]
