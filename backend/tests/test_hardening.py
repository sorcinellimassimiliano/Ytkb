"""Hardening tests: pipeline stats, cost estimate, evaluation harness."""

from __future__ import annotations

from pathlib import Path

from ytkb.indexing.embeddings import FakeEmbeddingClient
from ytkb.indexing.pipeline import index_video
from ytkb.knowledge.assignment import assign_pending_units
from ytkb.knowledge.extraction import HeuristicExtractor, extract_units
from ytkb.knowledge.merge import merge_dirty_topics
from ytkb.observability.evaluation import evaluate, load_gold
from ytkb.observability.stats import cost_report, pipeline_stats

from .conftest import requires_db
from .fakes import seed_transcribed_video

pytestmark = requires_db

EMB = FakeEmbeddingClient(dim=1536)
FIXTURES = Path(__file__).parent / "fixtures"

SEGMENTS = [
    (
        0.0,
        12.0,
        "Il context management significa gestire le informazioni nel contesto. "
        "Come puoi ridurre i token conviene riassumere i messaggi vecchi. "
        "Un tool utile e il prompt caching che riusa il contesto tra chiamate.",
    ),
]


async def _seed(session):
    video = await seed_transcribed_video(session, yt_video_id="hd1", segments=SEGMENTS)
    await index_video(session, video, embedder=EMB)
    await extract_units(session, video, extractor=HeuristicExtractor(), embedder=EMB)
    await assign_pending_units(session)
    await merge_dirty_topics(session)
    return video


async def test_pipeline_stats_shape(session):
    await _seed(session)
    s = await pipeline_stats(session)
    assert s["videos_total"] >= 1
    assert s["units"] >= 1
    assert s["articles"] >= 1
    assert 0.0 <= s["failure_rate"] <= 1.0
    assert "merge_queue" in s


async def test_cost_report_positive(session):
    await _seed(session)
    c = await cost_report(session)
    assert c["embedding_tokens"] > 0
    assert c["estimated_cost_usd"] >= 0.0


async def test_evaluation_recall(session):
    await _seed(session)
    gold = load_gold(str(FIXTURES / "gold_questions.json"))
    res = await evaluate(session, gold, k=5, embedder=EMB)
    assert res.total == 3
    # FTS should find the keyword-based gold items in the seeded transcript.
    assert res.recall >= 0.6, f"low recall, misses={res.misses}"
