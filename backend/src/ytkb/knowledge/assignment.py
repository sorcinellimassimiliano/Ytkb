"""Topic assignment (Phase 4a) — the matcher.

For each pending knowledge unit:
- kNN its embedding against topic centroids (top-3 candidates);
- similarity >= θ_high      → assign to the nearest topic (auto);
- θ_low <= similarity < θ_high → arbitration (LLM when configured, else an
  offline heuristic) decides among candidates or proposes a new topic;
- similarity < θ_low        → propose a new topic.

Centroids are updated incrementally on every assignment; a `proposed` topic is
promoted to `active` once it reaches `topic_promote_after_units` units.

Idempotent: only units with assignment == pending are processed.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.config import Settings, get_settings
from ytkb.db.models import (
    Assignment,
    IngestStatus,
    KnowledgeUnit,
    Topic,
    TopicStatus,
    Video,
)
from ytkb.knowledge.llm import LLMClient, get_llm_client
from ytkb.knowledge.prompts import ARBITER_SYSTEM, build_arbiter_user
from ytkb.knowledge.taxonomy import slugify, title_from_unit, update_centroid
from ytkb.logging import get_logger

log = get_logger("knowledge.assignment")

Candidate = tuple[Topic, float]  # (topic, cosine similarity)


@dataclass(slots=True)
class ArbiterDecision:
    assign_topic_id: int | None = None
    new_title: str | None = None


# --------------------------------------------------------------------------
# Arbiters
# --------------------------------------------------------------------------
class TopicArbiter(ABC):
    @abstractmethod
    async def arbitrate(self, unit_text: str, candidates: list[Candidate]) -> ArbiterDecision: ...


class HeuristicArbiter(TopicArbiter):
    """Offline arbiter: assign to the nearest candidate if it clears the midpoint
    of the uncertainty band, otherwise propose a new topic. Deterministic."""

    def __init__(self, settings: Settings) -> None:
        self.midpoint = (settings.theta_low + settings.theta_high) / 2

    async def arbitrate(self, unit_text: str, candidates: list[Candidate]) -> ArbiterDecision:
        if candidates and candidates[0][1] >= self.midpoint:
            return ArbiterDecision(assign_topic_id=candidates[0][0].id)
        return ArbiterDecision(new_title=None)


class LLMArbiter(TopicArbiter):
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def arbitrate(self, unit_text: str, candidates: list[Candidate]) -> ArbiterDecision:
        valid_ids = {t.id for t, _ in candidates}
        prompt = build_arbiter_user(unit_text, [(t.id, t.title, t.summary) for t, _ in candidates])
        raw = await self.client.complete(system=ARBITER_SYSTEM, prompt=prompt, max_tokens=256)
        decision = _parse_decision(raw)
        # Guard against hallucinated topic ids.
        if decision.assign_topic_id is not None and decision.assign_topic_id not in valid_ids:
            return ArbiterDecision(new_title=None)
        return decision


def _parse_decision(raw: str) -> ArbiterDecision:
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return ArbiterDecision(new_title=None)
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return ArbiterDecision(new_title=None)
    if payload.get("decision") == "assign" and isinstance(payload.get("topic_id"), int):
        return ArbiterDecision(assign_topic_id=payload["topic_id"])
    return ArbiterDecision(new_title=payload.get("title") or None)


def get_arbiter(settings: Settings | None = None) -> TopicArbiter:
    settings = settings or get_settings()
    client = get_llm_client(settings, model=settings.llm_merge_model)
    if client is not None:
        return LLMArbiter(client)
    return HeuristicArbiter(settings)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
async def _unique_slug(session: AsyncSession, base: str) -> str:
    slug, i = base, 2
    while await session.scalar(select(Topic.id).where(Topic.slug == slug)):
        slug = f"{base}-{i}"
        i += 1
    return slug


async def _candidates(
    session: AsyncSession, embedding: list[float], *, k: int = 3
) -> list[Candidate]:
    dist = Topic.centroid.cosine_distance(embedding).label("dist")
    rows = await session.execute(
        select(Topic, dist)
        .where(Topic.centroid.is_not(None), Topic.status != TopicStatus.merged_into)
        .order_by(dist)
        .limit(k)
    )
    return [(topic, 1.0 - float(d)) for topic, d in rows]


async def _create_topic(session: AsyncSession, title: str) -> Topic:
    slug = await _unique_slug(session, slugify(title))
    topic = Topic(slug=slug, title=title, status=TopicStatus.proposed, units_count=0)
    session.add(topic)
    await session.flush()
    return topic


def _apply_assignment(
    unit: KnowledgeUnit,
    topic: Topic,
    similarity: float,
    source: Assignment,
    settings: Settings,
) -> None:
    assert unit.embedding is not None
    centroid = list(topic.centroid) if topic.centroid is not None else None
    topic.centroid = update_centroid(centroid, topic.units_count, list(unit.embedding))
    topic.units_count += 1
    unit.topic_id = topic.id
    unit.assignment = source
    unit.confidence = similarity
    if (
        topic.status == TopicStatus.proposed
        and topic.units_count >= settings.topic_promote_after_units
    ):
        topic.status = TopicStatus.active


async def assign_unit(
    session: AsyncSession,
    unit: KnowledgeUnit,
    *,
    arbiter: TopicArbiter,
    settings: Settings,
) -> Topic:
    assert unit.embedding is not None
    candidates = await _candidates(session, list(unit.embedding))
    best_sim = candidates[0][1] if candidates else -1.0

    if candidates and best_sim >= settings.theta_high:
        topic, sim, source = candidates[0][0], best_sim, Assignment.auto
    elif candidates and best_sim >= settings.theta_low:
        decision = await arbiter.arbitrate(unit.text, candidates)
        source = Assignment.llm
        if decision.assign_topic_id is not None:
            topic = next(t for t, _ in candidates if t.id == decision.assign_topic_id)
            sim = next(s for t, s in candidates if t.id == decision.assign_topic_id)
        else:
            topic = await _create_topic(session, decision.new_title or title_from_unit(unit.text))
            sim = best_sim
    else:
        topic = await _create_topic(session, title_from_unit(unit.text))
        sim, source = max(best_sim, 0.0), Assignment.auto

    _apply_assignment(unit, topic, sim, source, settings)
    await session.flush()
    return topic


async def assign_pending_units(
    session: AsyncSession,
    *,
    arbiter: TopicArbiter | None = None,
    settings: Settings | None = None,
    limit: int | None = None,
) -> dict:
    settings = settings or get_settings()
    arbiter = arbiter or get_arbiter(settings)

    stmt = (
        select(KnowledgeUnit)
        .where(
            KnowledgeUnit.assignment == Assignment.pending,
            KnowledgeUnit.embedding.is_not(None),
        )
        .order_by(KnowledgeUnit.id)
    )
    if limit:
        stmt = stmt.limit(limit)
    units = (await session.execute(stmt)).scalars().all()

    topics_before = await session.scalar(select(func.count(Topic.id))) or 0
    assigned = 0
    for unit in units:
        await assign_unit(session, unit, arbiter=arbiter, settings=settings)
        assigned += 1
    topics_after = await session.scalar(select(func.count(Topic.id))) or 0

    await _advance_fully_assigned_videos(session)
    await session.flush()
    result = {
        "units_assigned": assigned,
        "topics_created": topics_after - topics_before,
    }
    log.info("units_assigned", **result)
    return result


async def _advance_fully_assigned_videos(session: AsyncSession) -> None:
    """Videos whose units are all assigned move units_extracted → assigned."""
    videos = (
        (
            await session.execute(
                select(Video).where(Video.ingest_status == IngestStatus.units_extracted)
            )
        )
        .scalars()
        .all()
    )
    for video in videos:
        pending = await session.scalar(
            select(func.count(KnowledgeUnit.id)).where(
                KnowledgeUnit.video_id == video.id,
                KnowledgeUnit.assignment == Assignment.pending,
            )
        )
        if not pending:
            video.ingest_status = IngestStatus.assigned
