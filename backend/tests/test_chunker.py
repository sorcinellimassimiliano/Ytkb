"""Property tests for the timestamp-aware chunker."""

from __future__ import annotations

from ytkb.indexing.chunker import chunk_segments, estimate_tokens
from ytkb.ingestion.providers.base import TranscriptSegment


def _make_segments(n: int, words_per_seg: int = 8) -> list[TranscriptSegment]:
    segs = []
    for i in range(n):
        text = " ".join(f"word{i}_{j}" for j in range(words_per_seg))
        # Terminate some segments with a period to create sentence boundaries.
        if i % 3 == 2:
            text += "."
        segs.append(TranscriptSegment(start_s=float(i), end_s=float(i) + 1.0, text=text))
    return segs


def test_empty_input_returns_empty():
    assert chunk_segments([]) == []


def test_single_segment_becomes_one_chunk():
    segs = [TranscriptSegment(0.0, 2.0, "hello world.")]
    chunks = chunk_segments(segs, target_tokens=500)
    assert len(chunks) == 1
    assert chunks[0].start_s == 0.0
    assert chunks[0].end_s == 2.0
    assert chunks[0].text == "hello world."


def test_coverage_first_and_last_timestamps_preserved():
    segs = _make_segments(60)
    chunks = chunk_segments(segs, target_tokens=40, overlap_ratio=0.15)
    assert chunks, "expected multiple chunks"
    assert chunks[0].start_s == segs[0].start_s
    assert chunks[-1].end_s == segs[-1].end_s


def test_chunks_are_ordered_and_wellformed():
    segs = _make_segments(80)
    chunks = chunk_segments(segs, target_tokens=50)
    for c in chunks:
        assert c.start_s <= c.end_s
        assert c.text
        assert c.token_count == estimate_tokens(c.text)
    # start times are non-decreasing
    starts = [c.start_s for c in chunks]
    assert starts == sorted(starts)


def test_target_tokens_respected_within_max():
    segs = _make_segments(120, words_per_seg=10)
    target = 60
    chunks = chunk_segments(segs, target_tokens=target, overlap_ratio=0.1)
    max_tokens = int(target * 1.4)
    largest_seg = max(estimate_tokens(s.text) for s in segs)
    # Segments are atomic (timestamp integrity), so a chunk may overshoot the
    # hard max by at most one segment when finishing on a sentence boundary.
    for c in chunks[:-1]:
        assert c.token_count <= max_tokens + largest_seg


def test_overlap_repeats_previous_content():
    segs = _make_segments(40)
    chunks = chunk_segments(segs, target_tokens=40, overlap_ratio=0.3)
    if len(chunks) >= 2:
        # Consecutive chunks should overlap in time (overlap carries segments).
        assert chunks[1].start_s <= chunks[0].end_s
