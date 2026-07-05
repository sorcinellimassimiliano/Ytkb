"""ASR fallback tests with a fake provider (no audio, no whisper)."""

from __future__ import annotations

from sqlalchemy import func, select

from ytkb.config import Settings
from ytkb.db.models import IngestStatus, Transcript, TranscriptSource, Video
from ytkb.ingestion.asr import ASRProvider, NoopASR, get_asr_provider, transcribe_missing
from ytkb.ingestion.providers.base import TranscriptResult, TranscriptSegment

from .conftest import requires_db
from .fakes import seed_bare_video

pytestmark = requires_db


class FakeASR(ASRProvider):
    source = TranscriptSource.cloud_asr

    def __init__(self, *, fail: bool = False, empty: bool = False) -> None:
        self.fail = fail
        self.empty = empty

    async def transcribe(self, video, *, settings) -> TranscriptResult | None:
        if self.fail:
            raise RuntimeError("asr boom")
        if self.empty:
            return None
        return TranscriptResult(
            language="it",
            source=self.source,
            segments=[TranscriptSegment(0.0, 4.0, f"trascrizione ASR di {video.yt_video_id}")],
        )


def test_default_provider_is_noop():
    assert isinstance(get_asr_provider(Settings(asr_provider="none")), NoopASR)


async def test_transcribes_no_transcript_videos(session):
    v = await seed_bare_video(session, yt_video_id="asr1", status=IngestStatus.no_transcript)
    n = await transcribe_missing(session, provider=FakeASR())
    assert n == 1
    await session.refresh(v)
    assert v.ingest_status == IngestStatus.transcribed
    tr = await session.scalar(select(Transcript).where(Transcript.video_id == v.id))
    assert tr is not None
    assert tr.source == TranscriptSource.cloud_asr


async def test_idempotent_rerun_skips_transcribed(session):
    await seed_bare_video(session, yt_video_id="asr2", status=IngestStatus.no_transcript)
    first = await transcribe_missing(session, provider=FakeASR())
    second = await transcribe_missing(session, provider=FakeASR())
    assert first == 1
    assert second == 0  # no more no_transcript videos
    total = await session.scalar(select(func.count(Transcript.id)))
    assert total == 1


async def test_empty_result_leaves_status(session):
    v = await seed_bare_video(session, yt_video_id="asr3", status=IngestStatus.no_transcript)
    n = await transcribe_missing(session, provider=FakeASR(empty=True))
    assert n == 0
    await session.refresh(v)
    assert v.ingest_status == IngestStatus.no_transcript  # can retry later


async def test_failure_marks_error(session):
    v = await seed_bare_video(session, yt_video_id="asr4", status=IngestStatus.no_transcript)
    await transcribe_missing(session, provider=FakeASR(fail=True))
    await session.refresh(v)
    assert v.ingest_status == IngestStatus.error
    assert "asr" in (v.error_msg or "")


async def test_shorts_prioritized(session):
    from ytkb.db.models import VideoKind

    long_v = Video(
        yt_video_id="long",
        channel_id=(await seed_bare_video(session, yt_video_id="seed")).channel_id,
        ingest_status=IngestStatus.no_transcript,
        kind=VideoKind.video,
    )
    short_v = Video(
        yt_video_id="short",
        channel_id=long_v.channel_id,
        ingest_status=IngestStatus.no_transcript,
        kind=VideoKind.short,
    )
    session.add_all([long_v, short_v])
    await session.flush()

    seen: list[str] = []

    class OrderASR(FakeASR):
        async def transcribe(self, video, *, settings):
            seen.append(video.yt_video_id)
            return await super().transcribe(video, settings=settings)

    await transcribe_missing(session, provider=OrderASR())
    # 'short' kind sorts before 'video' under our ordering.
    assert seen.index("short") < seen.index("long")
