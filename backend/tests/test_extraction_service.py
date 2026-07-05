"""Integration tests for the extraction service against real Postgres."""

from __future__ import annotations

from sqlalchemy import func, select

from ytkb.db.models import Assignment, Chunk, IngestStatus, KnowledgeUnit
from ytkb.indexing.embeddings import FakeEmbeddingClient
from ytkb.indexing.pipeline import index_video
from ytkb.knowledge.extraction import (
    HeuristicExtractor,
    _is_semantic_near_dup,
    extract_units,
)

from .conftest import requires_db
from .fakes import seed_transcribed_video

pytestmark = requires_db

EMBEDDER = FakeEmbeddingClient(dim=1536)

SEGMENTS = [
    (
        0.0,
        12.0,
        "Il context management significa gestire le informazioni nel contesto. "
        "Come puoi ridurre i token basta riassumere i messaggi vecchi in un sommario.",
    ),
    (
        12.0,
        24.0,
        "Un tool utile e la libreria di prompt caching che riusa il contesto. "
        "Per esempio un agente mantiene solo gli ultimi tre file aperti.",
    ),
]


async def _embedded_video(session, yt_video_id="ku1"):
    video = await seed_transcribed_video(session, yt_video_id=yt_video_id, segments=SEGMENTS)
    await index_video(session, video, embedder=EMBEDDER)
    return video


async def test_extract_creates_typed_units_with_provenance(session):
    video = await _embedded_video(session)
    n = await extract_units(session, video, extractor=HeuristicExtractor(), embedder=EMBEDDER)
    assert n >= 3

    valid_chunk_ids = {
        c for (c,) in await session.execute(select(Chunk.id).where(Chunk.video_id == video.id))
    }
    units = (
        (await session.execute(select(KnowledgeUnit).where(KnowledgeUnit.video_id == video.id)))
        .scalars()
        .all()
    )
    assert len(units) == n
    for u in units:
        assert u.chunk_ids, "unit must reference chunks"
        assert all(cid in valid_chunk_ids for cid in u.chunk_ids)
        assert u.embedding is not None
        assert u.assignment == Assignment.pending
    await session.refresh(video)
    assert video.ingest_status == IngestStatus.units_extracted


async def test_extract_is_idempotent_on_rerun(session):
    video = await _embedded_video(session)
    first = await extract_units(session, video, extractor=HeuristicExtractor(), embedder=EMBEDDER)
    total_after_first = await session.scalar(select(func.count(KnowledgeUnit.id)))
    second = await extract_units(session, video, extractor=HeuristicExtractor(), embedder=EMBEDDER)
    total_after_second = await session.scalar(select(func.count(KnowledgeUnit.id)))
    assert first > 0
    assert second == 0  # content_hash dedup: nothing new
    assert total_after_first == total_after_second


async def test_semantic_near_dup_detection(session):
    video = await _embedded_video(session)
    await extract_units(session, video, extractor=HeuristicExtractor(), embedder=EMBEDDER)
    # An embedding identical to an existing unit is a near-dup...
    existing = (
        (
            await session.execute(
                select(KnowledgeUnit).where(KnowledgeUnit.embedding.is_not(None)).limit(1)
            )
        )
        .scalars()
        .first()
    )
    assert await _is_semantic_near_dup(session, existing.embedding, 0.93) is True
    # ...an unrelated embedding is not.
    far = await EMBEDDER.embed_one("argomento completamente diverso zebra quantistica")
    assert await _is_semantic_near_dup(session, far, 0.93) is False
