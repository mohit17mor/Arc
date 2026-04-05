"""Tests for the speech condenser — [spoken] tag extraction."""

import pytest

from arc.voice.condenser import CondensedSpeech, condense, strip_spoken_tag


# ── Tag extraction ───────────────────────────────────────────────


class TestSpokenTagParsing:

    def test_extracts_spoken_tag(self):
        text = (
            "Here is a long explanation about X.\n\n"
            "[spoken]Fixed the bug in main.py. Check chat for details.[/spoken]"
        )
        result = condense(text)
        assert result.spoken_text == "Fixed the bug in main.py. Check chat for details."
        assert result.source == "tag"
        assert "[spoken]" not in result.chat_text
        assert "[/spoken]" not in result.chat_text
        assert "Here is a long explanation" in result.chat_text

    def test_uses_last_tag_if_multiple(self):
        text = (
            "I mentioned [spoken]not this one[/spoken] earlier.\n\n"
            "Final answer here.\n"
            "[spoken]This is the real summary.[/spoken]"
        )
        result = condense(text)
        assert result.spoken_text == "This is the real summary."
        assert result.source == "tag"

    def test_multiline_tag_content(self):
        text = (
            "Details here.\n"
            "[spoken]Two sentences.\nSpanning lines.[/spoken]"
        )
        result = condense(text)
        assert result.spoken_text == "Two sentences.\nSpanning lines."
        assert result.source == "tag"

    def test_empty_tag_falls_through(self):
        text = "Short response. [spoken][/spoken]"
        result = condense(text)
        # Empty tag → falls to verbatim (short text)
        assert result.source == "verbatim"
        assert result.spoken_text  # should speak the main text

    def test_tag_stripped_from_chat_text(self):
        text = "Main content.\n[spoken]Summary here.[/spoken]"
        result = condense(text)
        assert "[spoken]" not in result.chat_text
        assert "Main content." in result.chat_text


# ── Fallback: short response ────────────────────────────────────


class TestShortResponseVerbatim:

    def test_short_text_spoken_verbatim(self):
        text = "The answer is 42."
        result = condense(text)
        assert result.source == "verbatim"
        assert "42" in result.spoken_text

    def test_short_text_strips_markdown(self):
        text = "The **answer** is `42`."
        result = condense(text)
        assert result.source == "verbatim"
        assert "**" not in result.spoken_text
        assert "`" not in result.spoken_text

    def test_short_text_adds_period(self):
        text = "Done"
        result = condense(text)
        assert result.spoken_text.endswith(".")

    def test_exactly_at_threshold_is_verbatim(self):
        text = "a" * 150
        result = condense(text)
        assert result.source == "verbatim"


# ── Fallback: long response ─────────────────────────────────────


class TestLongResponseFallback:

    def test_long_text_without_tag_gives_generic(self):
        text = "x " * 200  # way over 150 chars
        result = condense(text)
        assert result.source == "fallback"
        assert "response is ready" in result.spoken_text.lower()
        assert "chat" in result.spoken_text.lower()

    def test_long_text_chat_text_unchanged(self):
        text = "x " * 200
        result = condense(text)
        assert result.chat_text == text


# ── Edge cases ───────────────────────────────────────────────────


class TestEdgeCases:

    def test_empty_string(self):
        result = condense("")
        assert result.spoken_text == ""
        assert result.source == "fallback"

    def test_none_like_empty(self):
        # condense explicitly handles empty/whitespace
        result = condense("   ")
        assert result.spoken_text == ""
        assert result.source == "fallback"

    def test_original_length_tracked(self):
        text = "Hello world."
        result = condense(text)
        assert result.original_length == len(text)


# ── strip_spoken_tag utility ─────────────────────────────────────


class TestStripSpokenTag:

    def test_removes_tag(self):
        text = "Content here.\n[spoken]Summary.[/spoken]"
        assert strip_spoken_tag(text) == "Content here."

    def test_removes_multiple_tags(self):
        text = "[spoken]A[/spoken] middle [spoken]B[/spoken]"
        assert strip_spoken_tag(text) == "middle"

    def test_no_tag_returns_original(self):
        text = "No tags here."
        assert strip_spoken_tag(text) == text

    def test_empty_input(self):
        assert strip_spoken_tag("") == ""
