"""Ingestion orchestration: register channels, discover videos, fetch
transcripts and advance the pipeline state machine.

All operations are idempotent: channels keyed by yt_channel_id, videos by
yt_video_id, transcripts only fetched for videos still in `pending` (so a rerun
never re-fetches or duplicates a transcript).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.db import repository as repo
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
    if not yt_channel_id:
        raise ValueError(f"Could not resolve channel: {handle_or_id!r}")
    existing = await repo.get_channel_by_yt_id(session, yt_channel_id)
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
        if await repo.get_video_by_yt_id(session, ref.provider_video_id):
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
    include_errors: bool = False,
) -> int:
    """Fetch transcripts for videos still awaiting one.

    Advances pending → transcribed (with a stored Transcript row) or pending →
    no_transcript. On a fetch exception the video goes to `error` with a
    message; pass include_errors=True to retry those on a later run.
    """
    statuses = [IngestStatus.pending]
    if include_errors:
        statuses.append(IngestStatus.error)
    stmt = (
        select(Video).where(or_(*[Video.ingest_status == s for s in statuses])).order_by(Video.id)
    )
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
        video.error_msg = None
        done += 1
    await session.flush()
    log.info("transcribed", count=done)
    return done


async def ingest_channel(
    session: AsyncSession,
    provider: SourceProvider,
    handle_or_id: str,
    *,
    languages: list[str],
    limit: int | None = None,
) -> dict:
    """End-to-end for one channel: register → discover → transcribe pending."""
    channel = await register_channel(session, provider, handle_or_id)
    new_videos = await discover_videos(session, provider, channel, limit=limit)
    transcribed = await transcribe_pending(session, provider, languages=languages, limit=limit)
    return {
        "channel": channel.yt_channel_id,
        "new_videos": len(new_videos),
        "transcribed": transcribed,
    }
