"""Golden tests for knowledge unit extraction.

Structural verification (per plan): we do NOT exact-match on unit text, but we
assert the shape of the extraction — enough units, all typed, expected types
present, valid chunk provenance, marketing/intro excluded. The offline
HeuristicExtractor is deterministic, so these lock behaviour without a cloud LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ytkb.knowledge import prompts
from ytkb.knowledge.extraction import HeuristicExtractor, parse_units
from ytkb.knowledge.prompts import EXTRACTION_SYSTEM, build_extraction_user
from ytkb.knowledge.types import ExtractionWindow, WindowChunk

FIXTURES = Path(__file__).parent / "fixtures"
CASES = ["transcript_agents.json", "transcript_cooking.json"]


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _windows_from_segments(segments: list[dict]) -> list[ExtractionWindow]:
    return [
        ExtractionWindow(
            [
                WindowChunk(
                    chunk_id=i,
                    start_s=s["start_s"],
                    end_s=s["end_s"],
                    text=s["text"],
                )
            ]
        )
        for i, s in enumerate(segments)
    ]


@pytest.mark.parametrize("case", CASES)
async def test_extraction_structural(case):
    data = _load(case)
    windows = _windows_from_segments(data["segments"])
    valid_ids = {c.chunk_id for w in windows for c in w.chunks}
    exp = data["expected"]

    units = await HeuristicExtractor().extract(data["video_title"], windows)

    assert len(units) >= exp["min_units"]
    produced_types = {u.type.value for u in units}
    for t in exp["types_present"]:
        assert t in produced_types, f"expected type {t} missing in {produced_types}"

    for u in units:
        assert u.text.strip()
        assert u.chunk_refs, "every unit must carry provenance"
        assert all(r in valid_ids for r in u.chunk_refs)
        assert 0.0 <= u.confidence <= 1.0
        low = u.text.lower()
        for banned in exp["must_exclude_substrings"]:
            assert banned not in low, f"marketing leaked: {u.text!r}"


@pytest.mark.parametrize("case", CASES)
async def test_units_are_atomic(case):
    data = _load(case)
    windows = _windows_from_segments(data["segments"])
    units = await HeuristicExtractor().extract(data["video_title"], windows)
    # Atomic ≈ single sentence: no unit should span multiple sentence enders.
    for u in units:
        assert u.text.count(".") <= 1


def test_prompt_contains_binding_rules():
    # Guard the prompt: golden tests must be updated if these rules change.
    for token in ("atomiche", "chunk_refs", "JSON", "sponsor", "confidence"):
        assert token in EXTRACTION_SYSTEM
    assert set(prompts.UNIT_TYPES) == {
        "technique",
        "claim",
        "opinion",
        "tool",
        "workflow",
        "example",
        "definition",
    }


def test_user_prompt_includes_refs_and_timestamps():
    windows = _windows_from_segments(
        [{"start_s": 65.0, "end_s": 70.0, "text": "Esempio di contenuto."}]
    )
    prompt = build_extraction_user("Titolo", windows)
    assert "ref:0" in prompt
    assert "01:05" in prompt  # 65s → 01:05
    assert "Titolo" in prompt


def test_parse_units_tolerates_fenced_json():
    unit = '{"type": "claim", "text": "x y z", "chunk_refs": [1], "confidence": 0.7}'
    raw = f'```json\n{{"units": [{unit}]}}\n```'
    units = parse_units(raw)
    assert len(units) == 1
    assert units[0].type.value == "claim"


def test_parse_units_handles_garbage():
    assert parse_units("no json here") == []
    assert parse_units('{"units": [{"bad": true}]}') == []
