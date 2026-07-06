"""Unit test for Reciprocal Rank Fusion (no DB)."""

from __future__ import annotations

from ytkb.db.search import ChunkHit, _rrf


def _hit(cid: int) -> ChunkHit:
    return ChunkHit(id=cid, video_id=1, start_s=0.0, end_s=1.0, text=f"c{cid}", score=0.0)


def test_rrf_rewards_items_ranked_high_in_both_lists():
    # Chunk 2 is top of list B and second in list A → should win overall.
    list_a = [_hit(1), _hit(2), _hit(3)]
    list_b = [_hit(2), _hit(4), _hit(1)]
    fused = _rrf([list_a, list_b], limit=10)
    ids = [h.id for h in fused]
    assert ids[0] == 2
    assert set(ids) == {1, 2, 3, 4}  # union of both lists


def test_rrf_respects_limit_and_dedups():
    fused = _rrf([[_hit(1), _hit(2)], [_hit(1), _hit(2)]], limit=1)
    assert len(fused) == 1
    assert fused[0].id in {1, 2}


def test_rrf_empty():
    assert _rrf([[], []], limit=5) == []
