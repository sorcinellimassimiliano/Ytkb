"""Integration tests for chunk search (semantic / FTS / hybrid) on real PG."""

from __future__ import annotations

from ytkb.db import search
from ytkb.indexing.embeddings import FakeEmbeddingClient
from ytkb.indexing.pipeline import index_video

from .conftest import requires_db
from .fakes import seed_transcribed_video

pytestmark = requires_db

EMBEDDER = FakeEmbeddingClient(dim=1536)

AGENTS_TEXT = "Il context management negli agenti riduce i token del contesto."
COOKING_TEXT = "La ricetta della carbonara richiede uova guanciale e pecorino."


async def _seed_two_topics(session):
    v1 = await seed_transcribed_video(session, yt_video_id="s1", segments=[(0.0, 6.0, AGENTS_TEXT)])
    v2 = await seed_transcribed_video(
        session, yt_video_id="s2", segments=[(0.0, 6.0, COOKING_TEXT)]
    )
    await index_video(session, v1, embedder=EMBEDDER)
    await index_video(session, v2, embedder=EMBEDDER)
    return v1, v2


async def test_semantic_ranks_matching_chunk_first(session):
    v1, _ = await _seed_two_topics(session)
    qvec = await EMBEDDER.embed_one(AGENTS_TEXT)
    hits = await search.semantic_chunks(session, qvec, limit=5)
    assert hits
    assert hits[0].video_id == v1.id
    assert hits[0].score > 0.9  # near-identical text → high cosine similarity


async def test_fts_finds_by_keyword(session):
    _, v2 = await _seed_two_topics(session)
    hits = await search.fts_chunks(session, "carbonara guanciale", limit=5)
    assert hits
    assert all("carbonara" in h.text.lower() for h in hits)
    assert hits[0].video_id == v2.id


async def test_fts_english_dictionary(session):
    await seed_transcribed_video(
        session,
        yt_video_id="s3",
        segments=[(0.0, 4.0, "A vector database enables semantic search over embeddings.")],
    )
    from sqlalchemy import select

    from ytkb.db.models import Video

    v3 = await session.scalar(select(Video).where(Video.yt_video_id == "s3"))
    await index_video(session, v3, embedder=EMBEDDER)
    hits = await search.fts_chunks(session, "databases", limit=5)
    assert any(h.video_id == v3.id for h in hits)


async def test_hybrid_returns_fused_results(session):
    v1, _ = await _seed_two_topics(session)
    qvec = await EMBEDDER.embed_one(AGENTS_TEXT)
    hits = await search.hybrid_chunks(session, "context agenti", qvec, limit=5)
    assert hits
    assert hits[0].video_id == v1.id
