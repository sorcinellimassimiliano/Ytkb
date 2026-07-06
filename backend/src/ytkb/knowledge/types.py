"""Shared DTOs for the knowledge layer (extraction I/O)."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ytkb.db.models import UnitType


@dataclass(slots=True)
class WindowChunk:
    chunk_id: int
    start_s: float
    end_s: float
    text: str


@dataclass(slots=True)
class ExtractionWindow:
    chunks: list[WindowChunk] = field(default_factory=list)

    @property
    def chunk_ids(self) -> list[int]:
        return [c.chunk_id for c in self.chunks]


class RawUnit(BaseModel):
    """A unit as produced by an extractor, before dedup / persistence."""

    type: UnitType
    text: str = Field(min_length=1)
    chunk_refs: list[int] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RawUnitBatch(BaseModel):
    units: list[RawUnit] = Field(default_factory=list)
