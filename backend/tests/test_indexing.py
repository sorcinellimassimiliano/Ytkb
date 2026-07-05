"""Integration tests for the chunk + embed pipeline against real Postgres."""

from __future__ import annotations

from sqlalchemy import func, select

from ytkb.db.models import Chunk, IngestStatus, Video
from ytkb.indexing.embeddings import FakeEmbeddingClient
from ytkb.indexing.pipeline import (
    index_transcribed,
    index_video,
    reindex_video,
)

from .conftest import requires_db
from .fakes import seed_transcribed_video

pytestmark = requires_db

EMBEDDER = FakeEmbeddingClient(dim=1536)

SEGMENTS = [
    (0.0, 5.0, "Il context management negli agenti LLM riduce i costi."),
    (5.0, 10.0, "Il prompt caching mantiene il contesto tra le chiamate."),
]


async def test_index_video_creates_chunks_and_embeddings(session):
    video = await seed_transcribed_video(session, yt_video_id="idx1", segments=SEGMENTS)
    res = await index_video(session, video, embedder=EMBEDDER)
    assert res["chunks"] >= 1
    assert res["embedded"] == res["chunks"]

    count = await session.scalar(select(func.count(Chunk.id)).where(Chunk.video_id == video.id))
    assert count == res["chunks"]
    # Every chunk has an embedding and the video advanced to embedded.
    missing = await session.scalar(
        select(func.count(Chunk.id)).where(Chunk.video_id == video.id, Chunk.embedding.is_(None))
    )
    assert missing == 0
    await session.refresh(video)
    assert video.ingest_status == IngestStatus.embedded


async def test_reindex_is_idempotent(session):
    video = await seed_transcribed_video(session, yt_video_id="idx2", segments=SEGMENTS)
    first = await index_video(session, video, embedder=EMBEDDER)
    count1 = await session.scalar(select(func.count(Chunk.id)).where(Chunk.video_id == video.id))
    # Reindex wipes and rebuilds — count stays stable, no duplicate rows.
    second = await reindex_video(session, "idx2", embedder=EMBEDDER)
    count2 = await session.scalar(select(func.count(Chunk.id)).where(Chunk.video_id == video.id))
    assert count1 == count2 == first["chunks"] == second["chunks"]


async def test_index_transcribed_processes_all_pending(session):
    await seed_transcribed_video(session, yt_video_id="idx3", segments=SEGMENTS)
    await seed_transcribed_video(session, yt_video_id="idx4", segments=SEGMENTS)
    totals = await index_transcribed(session, embedder=EMBEDDER)
    assert totals["videos"] >= 2
    # No transcribed videos remain (all advanced past transcribed).
    remaining = await session.scalar(
        select(func.count(Video.id)).where(Video.ingest_status == IngestStatus.transcribed)
    )
    assert remaining == 0
