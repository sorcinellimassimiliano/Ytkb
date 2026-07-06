"""Chunking + embedding pipeline: advances videos transcribed → chunked →
embedded, writing chunks (with pgvector embeddings) into Postgres.

Idempotent by construction:
- chunking deletes any existing chunks for the video before inserting, so a
  rerun (or `reindex-video`) never hits the (video_id, start_s) unique
  constraint and never leaves stale rows;
- embedding only touches chunks whose embedding is still NULL.
"""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.config import Settings, get_settings
from ytkb.db.models import Chunk, IngestStatus, Transcript, TranscriptSource, Video
from ytkb.indexing.chunker import chunk_segments
from ytkb.indexing.embeddings import EmbeddingClient, get_embedding_client
from ytkb.ingestion.providers.base import TranscriptSegment
from ytkb.logging import get_logger

log = get_logger("indexing.pipeline")

EMBED_BATCH = 128


def segments_from_transcript(transcript: Transcript) -> list[TranscriptSegment]:
    raw = transcript.raw_json or {}
    return [
        TranscriptSegment(
            start_s=float(s["start_s"]),
            end_s=float(s["end_s"]),
            text=s["text"],
        )
        for s in raw.get("segments", [])
        if s.get("text")
    ]


async def _best_transcript(session: AsyncSession, video_id: int) -> Transcript | None:
    """Prefer a manual transcript; otherwise the most recent one."""
    result = await session.execute(
        select(Transcript)
        .where(Transcript.video_id == video_id)
        .order_by(
            (Transcript.source == TranscriptSource.yt_manual).desc(),
            Transcript.id.desc(),
        )
    )
    return result.scalars().first()


async def chunk_video(
    session: AsyncSession, video: Video, *, settings: Settings | None = None
) -> int:
    settings = settings or get_settings()
    transcript = await _best_transcript(session, video.id)
    if transcript is None:
        video.ingest_status = IngestStatus.no_transcript
        await session.flush()
        return 0

    segments = segments_from_transcript(transcript)
    chunks = chunk_segments(
        segments,
        target_tokens=settings.chunk_target_tokens,
        overlap_ratio=settings.chunk_overlap_ratio,
    )

    # Idempotent: wipe any prior chunks for this video, then insert fresh.
    await session.execute(delete(Chunk).where(Chunk.video_id == video.id))
    for c in chunks:
        session.add(
            Chunk(
                video_id=video.id,
                start_s=c.start_s,
                end_s=c.end_s,
                text=c.text,
                token_count=c.token_count,
            )
        )
    video.ingest_status = IngestStatus.chunked
    await session.flush()
    log.info("video_chunked", video=video.yt_video_id, chunks=len(chunks))
    return len(chunks)


async def embed_video(
    session: AsyncSession,
    video: Video,
    *,
    embedder: EmbeddingClient | None = None,
    settings: Settings | None = None,
) -> int:
    embedder = embedder or get_embedding_client(settings)
    result = await session.execute(
        select(Chunk)
        .where(Chunk.video_id == video.id, Chunk.embedding.is_(None))
        .order_by(Chunk.id)
    )
    pending = list(result.scalars())
    embedded = 0
    for i in range(0, len(pending), EMBED_BATCH):
        batch = pending[i : i + EMBED_BATCH]
        vectors = await embedder.embed([c.text for c in batch])
        for chunk, vec in zip(batch, vectors, strict=True):
            chunk.embedding = vec
            embedded += 1
        await session.flush()

    remaining = await session.scalar(
        select(func.count(Chunk.id)).where(Chunk.video_id == video.id, Chunk.embedding.is_(None))
    )
    if not remaining:
        video.ingest_status = IngestStatus.embedded
    await session.flush()
    log.info("video_embedded", video=video.yt_video_id, embedded=embedded)
    return embedded


async def index_video(
    session: AsyncSession,
    video: Video,
    *,
    embedder: EmbeddingClient | None = None,
    settings: Settings | None = None,
) -> dict:
    n_chunks = await chunk_video(session, video, settings=settings)
    n_embed = await embed_video(session, video, embedder=embedder, settings=settings)
    return {"video": video.yt_video_id, "chunks": n_chunks, "embedded": n_embed}


async def index_transcribed(
    session: AsyncSession,
    *,
    embedder: EmbeddingClient | None = None,
    settings: Settings | None = None,
    limit: int | None = None,
) -> dict:
    """Process every video sitting in `transcribed` through to `embedded`."""
    settings = settings or get_settings()
    embedder = embedder or get_embedding_client(settings)
    stmt = select(Video).where(Video.ingest_status == IngestStatus.transcribed).order_by(Video.id)
    if limit:
        stmt = stmt.limit(limit)
    videos = (await session.execute(stmt)).scalars().all()
    totals = {"videos": 0, "chunks": 0, "embedded": 0}
    for video in videos:
        res = await index_video(session, video, embedder=embedder, settings=settings)
        totals["videos"] += 1
        totals["chunks"] += res["chunks"]
        totals["embedded"] += res["embedded"]
    return totals


async def reindex_video(
    session: AsyncSession,
    yt_video_id: str,
    *,
    embedder: EmbeddingClient | None = None,
    settings: Settings | None = None,
) -> dict:
    """Force a clean re-chunk + re-embed regardless of current status."""
    video = await session.scalar(select(Video).where(Video.yt_video_id == yt_video_id))
    if video is None:
        raise ValueError(f"Unknown video: {yt_video_id}")
    return await index_video(session, video, embedder=embedder, settings=settings)
