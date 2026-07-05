"""ASR fallback (Phase 7): transcribe videos that have no YouTube captions.

A no_transcript video's audio is pulled with yt-dlp and transcribed on CPU with
faster-whisper (int8) or via a cloud provider, then re-enters the pipeline as
`transcribed`. Heavy/optional deps are imported lazily; the default provider is
`none`, so nothing runs unless configured.

Idempotent: only videos in `no_transcript` without a transcript are processed.
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.config import Settings, get_settings
from ytkb.db.models import IngestStatus, Transcript, TranscriptSource, Video
from ytkb.ingestion.providers.base import TranscriptResult, TranscriptSegment
from ytkb.logging import get_logger

log = get_logger("ingestion.asr")


class ASRProvider(ABC):
    source: TranscriptSource

    @abstractmethod
    async def transcribe(self, video: Video, *, settings: Settings) -> TranscriptResult | None:
        """Return a transcript for the video, or None if it can't be produced."""


class NoopASR(ASRProvider):
    source = TranscriptSource.whisper

    async def transcribe(self, video: Video, *, settings: Settings) -> TranscriptResult | None:
        log.info("asr_noop", video=video.yt_video_id)
        return None


def _download_audio(yt_video_id: str, dest_dir: str, settings: Settings) -> str | None:
    """yt-dlp -x: fetch audio only. Returns the file path or None."""
    import yt_dlp

    outtmpl = os.path.join(dest_dir, "%(id)s.%(ext)s")
    opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    if settings.yt_dlp_cookies_file:
        opts["cookiefile"] = settings.yt_dlp_cookies_file
    if settings.yt_dlp_proxy:
        opts["proxy"] = settings.yt_dlp_proxy
    url = f"https://www.youtube.com/watch?v={yt_video_id}"
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    path = os.path.join(dest_dir, f"{yt_video_id}.wav")
    return path if os.path.exists(path) else None


class FasterWhisperASR(ASRProvider):
    source = TranscriptSource.whisper

    async def transcribe(self, video: Video, *, settings: Settings) -> TranscriptResult | None:
        if video.duration_s and video.duration_s > settings.asr_max_duration_s:
            log.info("asr_skip_too_long", video=video.yt_video_id, duration=video.duration_s)
            return None
        from faster_whisper import WhisperModel

        with tempfile.TemporaryDirectory() as tmp:
            audio = _download_audio(video.yt_video_id, tmp, settings)
            if audio is None:
                return None
            model = WhisperModel(settings.asr_whisper_model, device="cpu", compute_type="int8")
            segments_iter, info = model.transcribe(audio, vad_filter=True)
            segments = [
                TranscriptSegment(start_s=float(s.start), end_s=float(s.end), text=s.text.strip())
                for s in segments_iter
                if s.text.strip()
            ]
        if not segments:
            return None
        return TranscriptResult(
            language=getattr(info, "language", None), source=self.source, segments=segments
        )


def get_asr_provider(settings: Settings | None = None) -> ASRProvider:
    settings = settings or get_settings()
    if settings.asr_provider == "faster_whisper":
        return FasterWhisperASR()
    # assemblyai / openai cloud providers plug in here; default is no-op.
    return NoopASR()


async def transcribe_missing(
    session: AsyncSession,
    *,
    provider: ASRProvider | None = None,
    settings: Settings | None = None,
    limit: int | None = None,
) -> int:
    """Run ASR over videos stuck in no_transcript. Shorts first (cheapest)."""
    settings = settings or get_settings()
    provider = provider or get_asr_provider(settings)

    stmt = (
        select(Video)
        .where(Video.ingest_status == IngestStatus.no_transcript)
        .order_by(Video.kind.desc(), Video.id)  # 'short' > 'video' lexically → shorts first
    )
    if limit:
        stmt = stmt.limit(limit)
    videos = (await session.execute(stmt)).scalars().all()

    done = 0
    for video in videos:
        try:
            result = await provider.transcribe(video, settings=settings)
        except Exception as exc:
            video.ingest_status = IngestStatus.error
            video.error_msg = f"asr: {exc}"[:2000]
            log.warning("asr_error", video=video.yt_video_id, error=str(exc))
            continue
        if result is None or result.is_empty:
            continue  # stays no_transcript; may retry later
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
    log.info("asr_transcribed", count=done)
    return done
