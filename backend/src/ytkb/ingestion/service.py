"""Ingestion orchestration: register channels, discover videos, fetch
transcripts and advance the pipeline state machine.

All operations are idempotent: channels keyed by yt_channel_id, videos by
yt_video_id, transcripts skipped if the video already has one.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.db.models import (
    Channel,
    IngestStatus,
    Transcript,
    Video,
)
from ytkb.ingestion.providers.base import SourceProvider
from ytkb.logging import get_logger

log = get_logger("ingestion.service")


async def register_channel(
    session: AsyncSession, provider: SourceProvider, handle_or_id: str
) -> Channel:
    info = provider.resolve_channel(handle_or_id)
    yt_channel_id = info["yt_channel_id"]
    existing = await session.scalar(select(Channel).where(Channel.yt_channel_id == yt_channel_id))
    if existing:
        existing.handle = info.get("handle") or existing.handle
        existing.title = info.get("title") or existing.title
        await session.flush()
        return existing
    channel = Channel(
        yt_channel_id=yt_channel_id,
        handle=info.get("handle"),
        title=info.get("title"),
        active=True,
    )
    session.add(channel)
    await session.flush()
    log.info("channel_registered", yt_channel_id=yt_channel_id, title=channel.title)
    return channel


async def discover_videos(
    session: AsyncSession,
    provider: SourceProvider,
    channel: Channel,
    *,
    limit: int | None = None,
) -> list[Video]:
    created: list[Video] = []
    for ref in provider.list_channel_videos(channel.yt_channel_id, limit=limit):
        if not ref.provider_video_id:
            continue
        existing = await session.scalar(
            select(Video).where(Video.yt_video_id == ref.provider_video_id)
        )
        if existing:
            continue
        video = Video(
            yt_video_id=ref.provider_video_id,
            channel_id=channel.id,
            title=ref.title,
            published_at=ref.published_at,
            duration_s=ref.duration_s,
            kind=ref.kind,
            url=ref.url,
            ingest_status=IngestStatus.pending,
        )
        session.add(video)
        created.append(video)
    channel.last_checked_at = datetime.now(UTC)
    await session.flush()
    log.info("videos_discovered", channel=channel.yt_channel_id, new=len(created))
    return created


async def transcribe_pending(
    session: AsyncSession,
    provider: SourceProvider,
    *,
    languages: list[str],
    limit: int | None = None,
) -> int:
    stmt = select(Video).where(Video.ingest_status == IngestStatus.pending)
    if limit:
        stmt = stmt.limit(limit)
    videos = (await session.execute(stmt)).scalars().all()
    done = 0
    for video in videos:
        try:
            result = provider.fetch_transcript(video.yt_video_id, languages=languages)
        except Exception as exc:
            video.ingest_status = IngestStatus.error
            video.error_msg = str(exc)[:2000]
            log.warning("transcript_error", video=video.yt_video_id, error=str(exc))
            continue
        if result is None or result.is_empty:
            video.ingest_status = IngestStatus.no_transcript
            continue
        session.add(
            Transcript(
                video_id=video.id,
                language=result.language,
                source=result.source,
                raw_json=result.raw_json(),
            )
        )
        video.ingest_status = IngestStatus.transcribed
        done += 1
    await session.flush()
    log.info("transcribed", count=done)
    return done
