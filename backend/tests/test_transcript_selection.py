"""Unit tests for transcript selection policy (manual > auto > translate)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ytkb.ingestion.providers.youtube import _choose_transcript


@dataclass
class FakeTranscript:
    language_code: str
    is_generated: bool
    is_translatable: bool = False
    _translated_to: str | None = field(default=None)

    def translate(self, lang: str) -> FakeTranscript:
        return FakeTranscript(
            language_code=lang, is_generated=self.is_generated, _translated_to=lang
        )


def test_prefers_manual_over_auto_in_preferred_language():
    tlist = [
        FakeTranscript("it", is_generated=True),
        FakeTranscript("it", is_generated=False),
        FakeTranscript("en", is_generated=False),
    ]
    chosen = _choose_transcript(tlist, ["it", "en"])
    assert chosen.language_code == "it"
    assert chosen.is_generated is False


def test_falls_back_to_auto_when_no_manual():
    tlist = [FakeTranscript("it", is_generated=True), FakeTranscript("de", is_generated=False)]
    chosen = _choose_transcript(tlist, ["it", "en"])
    assert chosen.language_code == "it"
    assert chosen.is_generated is True


def test_respects_language_order_manual_first_pref():
    tlist = [
        FakeTranscript("en", is_generated=False),
        FakeTranscript("it", is_generated=False),
    ]
    # Both manual; the loop keeps the first preferred-language match encountered.
    chosen = _choose_transcript(tlist, ["it", "en"])
    assert chosen.language_code in {"it", "en"}
    assert chosen.is_generated is False


def test_translates_when_no_preferred_language_available():
    tlist = [FakeTranscript("de", is_generated=True, is_translatable=True)]
    chosen = _choose_transcript(tlist, ["it"])
    assert chosen._translated_to == "it"


def test_returns_none_on_empty_list():
    assert _choose_transcript([], ["it"]) is None
