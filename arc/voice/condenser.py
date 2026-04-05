"""
Speech Condenser — extracts spoken text from LLM responses.

The agent includes a ``[spoken]...[/spoken]`` tag in every response
containing a 1–2 sentence conversational summary meant to be read
aloud.  This module:

    1. Parses the ``[spoken]`` tag and returns its content.
    2. Strips the tag from the chat-facing text so users never see it.
    3. Falls back gracefully when the tag is absent:
       - Short responses (≤ 150 chars) are spoken as-is.
       - Longer responses produce a generic nudge:
         "Your response is ready. Check chat for the full details."

Zero LLM calls happen here — the LLM already did the summarisation
work as part of its normal generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Limits ───────────────────────────────────────────────────────

SHORT_THRESHOLD = 150  # responses this short are spoken verbatim

# ── Tag pattern ──────────────────────────────────────────────────

# Match the LAST [spoken]...[/spoken] block (greedy start so we
# skip any accidental earlier occurrences).
_SPOKEN_TAG = re.compile(
    r"\[spoken\](.*?)\[/spoken\]",
    re.DOTALL,
)

_FALLBACK_LONG = "Your response is ready. Check chat for the full details."

# Lightweight markdown stripping for the short-response path.
_MD_STRIP = re.compile(r"[*_`#~\[\]!]")
_MULTI_SPACE = re.compile(r"  +")


@dataclass(slots=True)
class CondensedSpeech:
    """Result of condensing an LLM response for speech."""

    spoken_text: str
    chat_text: str
    source: str  # "tag", "verbatim", "fallback"
    original_length: int


def condense(text: str) -> CondensedSpeech:
    """
    Extract the spoken summary from an LLM response.

    Priority:
        1. ``[spoken]...[/spoken]`` tag  → use tag content, strip from chat.
        2. Short response (≤ 150 chars)  → speak entire response verbatim.
        3. Long response without tag      → generic "check chat" nudge.
    """
    if not text or not text.strip():
        return CondensedSpeech(
            spoken_text="",
            chat_text=text or "",
            source="fallback",
            original_length=0,
        )

    original_length = len(text)

    # ── Try to extract [spoken] tag ──────────────────────────
    matches = list(_SPOKEN_TAG.finditer(text))
    if matches:
        # Use the last match (agent might reference the tag format earlier).
        last = matches[-1]
        spoken = last.group(1).strip()
        # Remove the tag from chat-facing text.
        chat = (text[: last.start()] + text[last.end() :]).strip()
        if spoken:
            return CondensedSpeech(
                spoken_text=spoken,
                chat_text=chat,
                source="tag",
                original_length=original_length,
            )

    # ── Fallback: short response → speak verbatim ────────────
    stripped = _clean_for_speech(text)
    if len(stripped) <= SHORT_THRESHOLD:
        return CondensedSpeech(
            spoken_text=stripped,
            chat_text=text,
            source="verbatim",
            original_length=original_length,
        )

    # ── Fallback: long response → generic nudge ──────────────
    return CondensedSpeech(
        spoken_text=_FALLBACK_LONG,
        chat_text=text,
        source="fallback",
        original_length=original_length,
    )


def strip_spoken_tag(text: str) -> str:
    """Remove all ``[spoken]...[/spoken]`` tags from text.

    Useful for platforms (CLI, WebChat) that display the raw response
    and should never show the spoken tag to the user.
    """
    return _SPOKEN_TAG.sub("", text).strip()


# ── Internal ─────────────────────────────────────────────────────


def _clean_for_speech(text: str) -> str:
    """Light cleanup of text for the verbatim-short path."""
    out = _MD_STRIP.sub("", text)
    out = _MULTI_SPACE.sub(" ", out)
    out = out.strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out
