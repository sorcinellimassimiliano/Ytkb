"""In-memory SourceProvider and DB seeding helpers for offline pipeline tests."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.db.models import (
    Assignment,
    Channel,
    IngestStatus,
    KnowledgeUnit,
    Transcript,
    TranscriptSource,
    UnitType,
    Video,
)
from ytkb.ingestion.providers.base import (
    SourceProvider,
    TranscriptResult,
    VideoRef,
)
from ytkb.knowledge.hashing import unit_content_hash


class FakeProvider(SourceProvider):
    name = "fake"

    def __init__(
        self,
        channel_info: dict,
        videos: list[VideoRef],
        transcripts: dict[str, TranscriptResult | Exception | None] | None = None,
    ) -> None:
        self.channel_info = channel_info
        self.videos = videos
        self.transcripts = transcripts or {}

    def resolve_channel(self, handle_or_id: str) -> dict:
        return self.channel_info

    def list_channel_videos(
        self, channel_id: str, *, limit: int | None = None
    ) -> Iterable[VideoRef]:
        vids = self.videos[:limit] if limit else self.videos
        yield from vids

    def fetch_transcript(
        self, provider_video_id: str, *, languages: list[str]
    ) -> TranscriptResult | None:
        value = self.transcripts.get(provider_video_id)
        if isinstance(value, Exception):
            raise value
        return value


async def seed_transcribed_video(
    session: AsyncSession,
    *,
    yt_video_id: str,
    segments: list[tuple[float, float, str]],
    channel_yt_id: str = "UCseed",
) -> Video:
    """Insert a channel (if needed), a video in `transcribed` state, and a
    transcript with the given segments. Returns the flushed Video."""
    channel = await session.scalar(select(Channel).where(Channel.yt_channel_id == channel_yt_id))
    if channel is None:
        channel = Channel(yt_channel_id=channel_yt_id, title="Seed")
        session.add(channel)
        await session.flush()
    channel_id = channel.id

    video = Video(
        yt_video_id=yt_video_id,
        channel_id=channel_id,
        title=f"Video {yt_video_id}",
        ingest_status=IngestStatus.transcribed,
    )
    session.add(video)
    await session.flush()

    session.add(
        Transcript(
            video_id=video.id,
            language="it",
            source=TranscriptSource.yt_manual,
            raw_json={
                "language": "it",
                "source": "yt_manual",
                "segments": [{"start_s": s, "end_s": e, "text": t} for s, e, t in segments],
            },
        )
    )
    await session.flush()
    return video


async def seed_bare_video(
    session: AsyncSession,
    *,
    yt_video_id: str,
    status: IngestStatus = IngestStatus.units_extracted,
    channel_yt_id: str = "UCunits",
) -> Video:
    """A video row with no transcript/chunks — for assignment-layer tests."""
    channel = await session.scalar(select(Channel).where(Channel.yt_channel_id == channel_yt_id))
    if channel is None:
        channel = Channel(yt_channel_id=channel_yt_id, title="Units")
        session.add(channel)
        await session.flush()
    video = Video(
        yt_video_id=yt_video_id,
        channel_id=channel.id,
        title=f"Video {yt_video_id}",
        ingest_status=status,
    )
    session.add(video)
    await session.flush()
    return video


async def seed_unit(
    session: AsyncSession,
    *,
    video: Video,
    text: str,
    embedding: list[float],
    unit_type: UnitType = UnitType.claim,
) -> KnowledgeUnit:
    """Insert a pending knowledge unit with a chosen embedding."""
    unit = KnowledgeUnit(
        content_hash=unit_content_hash(text, unit_type.value),
        video_id=video.id,
        chunk_ids=[],
        unit_type=unit_type,
        text=text,
        embedding=embedding,
        assignment=Assignment.pending,
    )
    session.add(unit)
    await session.flush()
    return unit
