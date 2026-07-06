"""Taxonomy maintenance: detect near-duplicate topics and merge topics manually.

Fusions are never automatic (per plan): `find_similar_topics` only *proposes*
pairs; `merge_topics` performs an explicit, admin-approved merge with unit remap.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.config import Settings, get_settings
from ytkb.db.models import KnowledgeUnit, Topic, TopicStatus
from ytkb.knowledge.taxonomy import update_centroid
from ytkb.logging import get_logger

log = get_logger("knowledge.review")


@dataclass(slots=True)
class TopicPair:
    topic_a_id: int
    topic_a_slug: str
    topic_b_id: int
    topic_b_slug: str
    similarity: float


async def find_similar_topics(
    session: AsyncSession, *, settings: Settings | None = None
) -> list[TopicPair]:
    """Propose merges: pairs of active/proposed topics whose centroids are closer
    than the review threshold. Detection only — never mutates."""
    settings = settings or get_settings()
    threshold = settings.topic_similarity_review_threshold
    topics = (
        (
            await session.execute(
                select(Topic).where(
                    Topic.centroid.is_not(None),
                    Topic.status != TopicStatus.merged_into,
                )
            )
        )
        .scalars()
        .all()
    )

    pairs: list[TopicPair] = []
    for i, a in enumerate(topics):
        for b in topics[i + 1 :]:
            if a.centroid is None or b.centroid is None:
                continue
            sim = _cosine(list(a.centroid), list(b.centroid))
            if sim >= threshold:
                pairs.append(TopicPair(a.id, a.slug, b.id, b.slug, round(sim, 4)))
    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return pairs


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


async def merge_topics(session: AsyncSession, from_slug: str, into_slug: str) -> dict:
    """Remap every unit of `from` onto `into`, fold the centroid, and mark
    `from` as merged_into. The `into` topic becomes dirty and will get a new
    article version on the next merge run."""
    if from_slug == into_slug:
        raise ValueError("cannot merge a topic into itself")
    source = await session.scalar(select(Topic).where(Topic.slug == from_slug))
    target = await session.scalar(select(Topic).where(Topic.slug == into_slug))
    if source is None or target is None:
        raise ValueError("source or target topic not found")

    units = (
        (await session.execute(select(KnowledgeUnit).where(KnowledgeUnit.topic_id == source.id)))
        .scalars()
        .all()
    )
    centroid = list(target.centroid) if target.centroid is not None else None
    count = target.units_count
    for unit in units:
        unit.topic_id = target.id
        if unit.embedding is not None:
            centroid = update_centroid(centroid, count, list(unit.embedding))
            count += 1
    target.centroid = centroid
    target.units_count = count
    source.status = TopicStatus.merged_into
    source.merged_into_id = target.id
    source.units_count = 0
    await session.flush()

    remapped = len(units)
    log.info("topics_merged", frm=from_slug, into=into_slug, units=remapped)
    return {"from": from_slug, "into": into_slug, "units_remapped": remapped}


async def topic_count(session: AsyncSession) -> int:
    return await session.scalar(select(func.count(Topic.id))) or 0
