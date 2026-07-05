"""Abstract source provider interface and transport DTOs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from ytkb.db.models import TranscriptSource, VideoKind


@dataclass(slots=True)
class VideoRef:
    """A discovered source video, before ingestion."""

    provider_video_id: str
    title: str | None = None
    url: str | None = None
    published_at: datetime | None = None
    duration_s: int | None = None
    kind: VideoKind = VideoKind.video


@dataclass(slots=True)
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str


@dataclass(slots=True)
class TranscriptResult:
    language: str | None
    source: TranscriptSource
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.segments

    def raw_json(self) -> dict:
        return {
            "language": self.language,
            "source": self.source.value,
            "segments": [
                {"start_s": s.start_s, "end_s": s.end_s, "text": s.text} for s in self.segments
            ],
        }


class SourceProvider(ABC):
    """A content source (YouTube today; Reels tomorrow)."""

    name: str

    @abstractmethod
    def resolve_channel(self, handle_or_id: str) -> dict:
        """Return {yt_channel_id, handle, title} for a channel reference."""

    @abstractmethod
    def list_channel_videos(
        self, channel_id: str, *, limit: int | None = None
    ) -> Iterable[VideoRef]:
        """Yield videos (and shorts) for a channel, newest first."""

    @abstractmethod
    def fetch_transcript(
        self, provider_video_id: str, *, languages: list[str]
    ) -> TranscriptResult | None:
        """Fetch the best available transcript, or None if none exists."""
