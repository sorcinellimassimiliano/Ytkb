"""In-memory SourceProvider for deterministic, offline pipeline tests."""

from __future__ import annotations

from collections.abc import Iterable

from ytkb.ingestion.providers.base import (
    SourceProvider,
    TranscriptResult,
    VideoRef,
)


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
