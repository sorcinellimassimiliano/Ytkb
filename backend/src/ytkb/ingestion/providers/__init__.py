"""Source providers. Abstract `SourceProvider` keeps the door open for other
platforms (e.g. Instagram Reels) without touching the pipeline."""

from ytkb.ingestion.providers.base import (
    SourceProvider,
    TranscriptResult,
    TranscriptSegment,
    VideoRef,
)

__all__ = [
    "SourceProvider",
    "TranscriptResult",
    "TranscriptSegment",
    "VideoRef",
]
