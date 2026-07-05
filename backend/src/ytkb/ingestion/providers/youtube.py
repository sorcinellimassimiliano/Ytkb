"""YouTube provider: channel listing via yt-dlp, transcripts via
youtube-transcript-api (manual > auto), with backoff + jitter.

Network dependencies are imported lazily so the module stays importable (and
unit-testable) in environments without them installed.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ytkb.config import get_settings
from ytkb.db.models import TranscriptSource, VideoKind
from ytkb.ingestion.providers.base import (
    SourceProvider,
    TranscriptResult,
    TranscriptSegment,
    VideoRef,
)
from ytkb.logging import get_logger

log = get_logger("ingestion.youtube")


class TranscriptRateLimited(Exception):
    """Raised on retriable transcript fetch failures."""


def _parse_upload_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError:
        return None


class YouTubeProvider(SourceProvider):
    name = "youtube"

    def _ydl_opts(self, extra: dict | None = None) -> dict:
        settings = get_settings()
        opts: dict = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
        }
        if settings.yt_dlp_cookies_file:
            opts["cookiefile"] = settings.yt_dlp_cookies_file
        if settings.yt_dlp_proxy:
            opts["proxy"] = settings.yt_dlp_proxy
        if extra:
            opts.update(extra)
        return opts

    def resolve_channel(self, handle_or_id: str) -> dict:
        import yt_dlp

        ref = handle_or_id if handle_or_id.startswith("http") else _channel_url(handle_or_id)
        with yt_dlp.YoutubeDL(self._ydl_opts({"playlist_items": "0"})) as ydl:
            info = ydl.extract_info(ref, download=False)
        return {
            "yt_channel_id": info.get("channel_id") or info.get("id"),
            "handle": info.get("uploader_id") or handle_or_id,
            "title": info.get("channel") or info.get("title"),
        }

    def list_channel_videos(
        self, channel_id: str, *, limit: int | None = None
    ) -> Iterable[VideoRef]:
        import yt_dlp

        base = _channel_url(channel_id)
        extra: dict = {}
        if limit:
            extra["playlist_items"] = f"1:{limit}"
        # Both the /videos and /shorts tabs.
        for tab, kind in (("videos", VideoKind.video), ("shorts", VideoKind.short)):
            url = f"{base}/{tab}"
            try:
                with yt_dlp.YoutubeDL(self._ydl_opts(extra)) as ydl:
                    info = ydl.extract_info(url, download=False)
            except Exception as exc:  # tab may not exist
                log.warning("channel_tab_failed", tab=tab, error=str(exc))
                continue
            for entry in info.get("entries") or []:
                if not entry:
                    continue
                yield VideoRef(
                    provider_video_id=entry.get("id"),
                    title=entry.get("title"),
                    url=entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id')}",
                    published_at=_parse_upload_date(entry.get("upload_date")),
                    duration_s=entry.get("duration"),
                    kind=kind,
                )

    @retry(
        retry=retry_if_exception_type(TranscriptRateLimited),
        wait=wait_random_exponential(multiplier=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def fetch_transcript(
        self, provider_video_id: str, *, languages: list[str]
    ) -> TranscriptResult | None:
        from youtube_transcript_api import (
            NoTranscriptFound,
            TranscriptsDisabled,
            YouTubeTranscriptApi,
        )

        api = YouTubeTranscriptApi()
        try:
            transcript_list = api.list(provider_video_id)
        except TranscriptsDisabled:
            return None
        except Exception as exc:  # network / rate limit
            raise TranscriptRateLimited(str(exc)) from exc

        manual, auto = None, None
        for t in transcript_list:
            if t.language_code not in languages:
                continue
            if t.is_generated:
                auto = auto or t
            else:
                manual = manual or t
        chosen = manual or auto
        if chosen is None:
            # Fall back to translating an available transcript into the first pref.
            try:
                any_t = next(iter(transcript_list))
                chosen = (
                    any_t.translate(languages[0]) if languages and any_t.is_translatable else any_t
                )
            except (StopIteration, NoTranscriptFound):
                return None

        source = TranscriptSource.yt_auto if chosen.is_generated else TranscriptSource.yt_manual
        try:
            fetched = chosen.fetch()
        except Exception as exc:
            raise TranscriptRateLimited(str(exc)) from exc

        segments = [
            TranscriptSegment(
                start_s=float(snippet.start),
                end_s=float(snippet.start) + float(snippet.duration),
                text=snippet.text.strip(),
            )
            for snippet in fetched
            if snippet.text.strip()
        ]
        return TranscriptResult(language=chosen.language_code, source=source, segments=segments)


def _channel_url(handle_or_id: str) -> str:
    if handle_or_id.startswith("http"):
        return handle_or_id.rstrip("/")
    if handle_or_id.startswith("@"):
        return f"https://www.youtube.com/{handle_or_id}"
    if handle_or_id.startswith("UC"):
        return f"https://www.youtube.com/channel/{handle_or_id}"
    return f"https://www.youtube.com/@{handle_or_id}"
