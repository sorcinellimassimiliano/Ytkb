"""Knowledge unit extraction (Phase 3).

A video that is `embedded` is decomposed into atomic, typed knowledge units:
1. its chunks are grouped into windows;
2. an extractor turns each window into RawUnits (LLM when a key is configured,
   otherwise a deterministic offline heuristic so the pipeline stays CPU-only);
3. units are deduplicated (exact by content_hash + semantic near-dup), embedded,
   and persisted with valid chunk provenance;
4. the video advances embedded → units_extracted.

Idempotent: content_hash is UNIQUE and already-seen units are skipped, so reruns
never duplicate.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.config import Settings, get_settings
from ytkb.db.models import (
    Assignment,
    Chunk,
    IngestStatus,
    KnowledgeUnit,
    UnitType,
    Video,
)
from ytkb.indexing.embeddings import EmbeddingClient, get_embedding_client
from ytkb.knowledge.hashing import unit_content_hash
from ytkb.knowledge.llm import LLMClient, get_llm_client
from ytkb.knowledge.prompts import EXTRACTION_SYSTEM, build_extraction_user
from ytkb.knowledge.types import ExtractionWindow, RawUnit, RawUnitBatch, WindowChunk
from ytkb.logging import get_logger

log = get_logger("knowledge.extraction")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_MARKETING = re.compile(
    r"\b(iscriviti|iscrivetevi|sponsor|sponsorizzat|canale|campanell|"
    r"like|commenta|benvenut|link in descrizione|subscribe|patreon)\b",
    re.IGNORECASE,
)

# Keyword → type heuristics for the offline extractor.
_TYPE_HINTS: list[tuple[re.Pattern, UnitType]] = [
    (re.compile(r"\b(per esempio|ad esempio|esempio:)\b", re.I), UnitType.example),
    (
        re.compile(r"\b(significa|si definisce|è definit|in altre parole|ovvero)\b", re.I),
        UnitType.definition,
    ),
    (
        re.compile(r"\b(secondo me|penso che|preferisco|credo che|a mio avviso)\b", re.I),
        UnitType.opinion,
    ),
    (
        re.compile(r"\b(prima|poi|infine|step|passaggi|workflow|processo)\b", re.I),
        UnitType.workflow,
    ),
    (re.compile(r"\b(tool|libreria|framework|software|app|strumento)\b", re.I), UnitType.tool),
    (re.compile(r"\b(come|puoi|basta|devi|conviene|usa|imposta)\b", re.I), UnitType.technique),
]


def build_windows(chunks: list[Chunk], window_size: int) -> list[ExtractionWindow]:
    windows: list[ExtractionWindow] = []
    for i in range(0, len(chunks), window_size):
        group = chunks[i : i + window_size]
        windows.append(
            ExtractionWindow([WindowChunk(c.id, c.start_s, c.end_s, c.text) for c in group])
        )
    return windows


# --------------------------------------------------------------------------
# Extractors
# --------------------------------------------------------------------------
class UnitExtractor(ABC):
    @abstractmethod
    async def extract(self, video_title: str, windows: list[ExtractionWindow]) -> list[RawUnit]: ...


def _infer_type(text: str) -> UnitType:
    for pattern, unit_type in _TYPE_HINTS:
        if pattern.search(text):
            return unit_type
    return UnitType.claim


class HeuristicExtractor(UnitExtractor):
    """Deterministic, offline extractor: one unit per substantive sentence,
    typed by keyword, marketing/intro filtered. Good enough to exercise the
    whole pipeline and to anchor golden tests without a cloud LLM."""

    def __init__(self, min_words: int = 6) -> None:
        self.min_words = min_words

    async def extract(self, video_title: str, windows: list[ExtractionWindow]) -> list[RawUnit]:
        units: list[RawUnit] = []
        for window in windows:
            for chunk in window.chunks:
                for sentence in _SENTENCE_SPLIT.split(chunk.text):
                    s = sentence.strip()
                    if len(s.split()) < self.min_words:
                        continue
                    if _MARKETING.search(s):
                        continue
                    units.append(
                        RawUnit(
                            type=_infer_type(s),
                            text=s,
                            chunk_refs=[chunk.chunk_id],
                            confidence=0.6,
                        )
                    )
        return units


class LLMExtractor(UnitExtractor):
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def extract(self, video_title: str, windows: list[ExtractionWindow]) -> list[RawUnit]:
        prompt = build_extraction_user(video_title, windows)
        raw = await self.client.complete(system=EXTRACTION_SYSTEM, prompt=prompt, max_tokens=4096)
        return parse_units(raw)


def parse_units(raw: str) -> list[RawUnit]:
    """Parse the model's JSON (tolerating markdown fences / surrounding text)."""
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    try:
        return RawUnitBatch.model_validate(payload).units
    except Exception:
        return []


def get_extractor(settings: Settings | None = None) -> UnitExtractor:
    settings = settings or get_settings()
    client = get_llm_client(settings, model=settings.llm_extraction_model)
    if client is not None:
        return LLMExtractor(client)
    return HeuristicExtractor()


# --------------------------------------------------------------------------
# Persistence + dedup
# --------------------------------------------------------------------------
async def _is_semantic_near_dup(
    session: AsyncSession, embedding: list[float], threshold: float
) -> bool:
    dist = KnowledgeUnit.embedding.cosine_distance(embedding).label("dist")
    nearest = await session.scalar(
        select(dist).where(KnowledgeUnit.embedding.is_not(None)).order_by(dist).limit(1)
    )
    if nearest is None:
        return False
    return (1.0 - float(nearest)) >= threshold


async def extract_units(
    session: AsyncSession,
    video: Video,
    *,
    extractor: UnitExtractor | None = None,
    embedder: EmbeddingClient | None = None,
    settings: Settings | None = None,
) -> int:
    settings = settings or get_settings()
    extractor = extractor or get_extractor(settings)
    embedder = embedder or get_embedding_client(settings)

    chunks = list(
        (
            await session.execute(
                select(Chunk).where(Chunk.video_id == video.id).order_by(Chunk.start_s)
            )
        ).scalars()
    )
    if not chunks:
        return 0
    valid_chunk_ids = {c.id for c in chunks}

    windows = build_windows(chunks, settings.extraction_window_size)
    raw_units = await extractor.extract(video.title or "", windows)

    inserted = 0
    for unit in raw_units:
        if unit.confidence < settings.extraction_min_confidence:
            continue
        text = unit.text.strip()
        if not text:
            continue
        content_hash = unit_content_hash(text, unit.type.value)
        # Exact dedup (also enforced by the UNIQUE constraint).
        if await session.scalar(
            select(KnowledgeUnit.id).where(KnowledgeUnit.content_hash == content_hash)
        ):
            continue
        refs = [r for r in unit.chunk_refs if r in valid_chunk_ids]
        if not refs:
            # Fall back to the window's chunks so provenance is never empty.
            refs = [c.id for c in chunks]
        embedding = await embedder.embed_one(text)
        if await _is_semantic_near_dup(session, embedding, settings.unit_near_dup_threshold):
            continue
        session.add(
            KnowledgeUnit(
                content_hash=content_hash,
                video_id=video.id,
                chunk_ids=refs,
                unit_type=unit.type,
                text=text,
                embedding=embedding,
                assignment=Assignment.pending,
                confidence=unit.confidence,
            )
        )
        await session.flush()  # so subsequent near-dup checks see this unit
        inserted += 1

    video.ingest_status = IngestStatus.units_extracted
    await session.flush()
    log.info("units_extracted", video=video.yt_video_id, units=inserted)
    return inserted


async def extract_pending(
    session: AsyncSession,
    *,
    extractor: UnitExtractor | None = None,
    embedder: EmbeddingClient | None = None,
    settings: Settings | None = None,
    limit: int | None = None,
) -> dict:
    settings = settings or get_settings()
    extractor = extractor or get_extractor(settings)
    embedder = embedder or get_embedding_client(settings)
    stmt = select(Video).where(Video.ingest_status == IngestStatus.embedded).order_by(Video.id)
    if limit:
        stmt = stmt.limit(limit)
    videos = (await session.execute(stmt)).scalars().all()
    totals = {"videos": 0, "units": 0}
    for video in videos:
        n = await extract_units(
            session, video, extractor=extractor, embedder=embedder, settings=settings
        )
        totals["videos"] += 1
        totals["units"] += n
    return totals
