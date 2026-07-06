"""Integration tests for the ingestion pipeline against a real Postgres.

Covers the Phase 1 acceptance: channel → videos in `transcribed`, and that a
rerun is idempotent (no duplicate channels, videos or transcripts).
"""

from __future__ import annotations

from sqlalchemy import func, select

from ytkb.db.models import (
    Channel,
    IngestStatus,
    Transcript,
    TranscriptSource,
    Video,
    VideoKind,
)
from ytkb.ingestion.providers.base import (
    TranscriptResult,
    TranscriptSegment,
    VideoRef,
)
from ytkb.ingestion.service import (
    discover_videos,
    ingest_channel,
    register_channel,
    transcribe_pending,
)

from .conftest import requires_db
from .fakes import FakeProvider

pytestmark = requires_db

CHANNEL = {"yt_channel_id": "UCfake123", "handle": "@fake", "title": "Fake Channel"}


def _transcript(text: str) -> TranscriptResult:
    return TranscriptResult(
        language="it",
        source=TranscriptSource.yt_manual,
        segments=[TranscriptSegment(0.0, 3.0, text)],
    )


def _provider(*, with_transcripts: bool = True) -> FakeProvider:
    videos = [
        VideoRef(provider_video_id="vidA", title="A", kind=VideoKind.video),
        VideoRef(provider_video_id="vidB", title="B", kind=VideoKind.short),
    ]
    transcripts = (
        {"vidA": _transcript("contenuto A"), "vidB": _transcript("contenuto B")}
        if with_transcripts
        else {}
    )
    return FakeProvider(CHANNEL, videos, transcripts)


async def test_register_channel_is_idempotent(session):
    provider = _provider()
    c1 = await register_channel(session, provider, "@fake")
    c2 = await register_channel(session, provider, "@fake")
    assert c1.id == c2.id
    count = await session.scalar(select(func.count(Channel.id)))
    assert count == 1


async def test_discover_videos_no_duplicates_on_rerun(session):
    provider = _provider()
    channel = await register_channel(session, provider, "@fake")
    first = await discover_videos(session, provider, channel)
    assert len(first) == 2
    second = await discover_videos(session, provider, channel)
    assert second == []
    total = await session.scalar(select(func.count(Video.id)))
    assert total == 2


async def test_full_ingest_moves_videos_to_transcribed(session):
    provider = _provider()
    result = await ingest_channel(session, provider, "@fake", languages=["it", "en"])
    assert result["new_videos"] == 2
    assert result["transcribed"] == 2

    statuses = (await session.execute(select(Video.ingest_status))).scalars().all()
    assert all(s == IngestStatus.transcribed for s in statuses)
    tcount = await session.scalar(select(func.count(Transcript.id)))
    assert tcount == 2


async def test_rerun_does_not_duplicate_transcripts(session):
    provider = _provider()
    await ingest_channel(session, provider, "@fake", languages=["it"])
    # Second full pass: no new videos, nothing pending → no new transcripts.
    result = await ingest_channel(session, provider, "@fake", languages=["it"])
    assert result["new_videos"] == 0
    assert result["transcribed"] == 0
    tcount = await session.scalar(select(func.count(Transcript.id)))
    assert tcount == 2


async def test_no_transcript_sets_status(session):
    provider = _provider(with_transcripts=False)
    channel = await register_channel(session, provider, "@fake")
    await discover_videos(session, provider, channel)
    await transcribe_pending(session, provider, languages=["it"])
    statuses = (await session.execute(select(Video.ingest_status))).scalars().all()
    assert all(s == IngestStatus.no_transcript for s in statuses)


async def test_error_then_retry_recovers(session):
    videos = [VideoRef(provider_video_id="vidX", title="X")]
    failing = FakeProvider(CHANNEL, videos, {"vidX": RuntimeError("rate limited")})
    channel = await register_channel(session, failing, "@fake")
    await discover_videos(session, failing, channel)
    await transcribe_pending(session, failing, languages=["it"])
    video = await session.scalar(select(Video).where(Video.yt_video_id == "vidX"))
    assert video.ingest_status == IngestStatus.error
    assert "rate limited" in (video.error_msg or "")

    # Provider recovers; retry_errors picks the video back up.
    recovered = FakeProvider(CHANNEL, videos, {"vidX": _transcript("ok now")})
    n = await transcribe_pending(session, recovered, languages=["it"], include_errors=True)
    assert n == 1
    await session.refresh(video)
    assert video.ingest_status == IngestStatus.transcribed
    assert video.error_msg is None
