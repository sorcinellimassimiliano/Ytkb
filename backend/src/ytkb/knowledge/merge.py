"""Incremental article merge (Phase 4b) — the living topic articles.

For each dirty topic (one with units not yet in its latest article version) the
merger produces a new markdown article that folds the new units in. Every claim
must cite [unit:ID]; a binding validator rejects any article that cites unknown
units or silently drops new ones. Articles are append-only: each merge writes a
new version, the highest version is the current article.

Idempotent: a topic with no new units is skipped (not dirty).
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.config import Settings, get_settings
from ytkb.db.models import (
    Assignment,
    IngestStatus,
    KnowledgeUnit,
    Topic,
    TopicArticle,
    TopicStatus,
    UnitType,
    Video,
)
from ytkb.knowledge.llm import LLMClient, get_llm_client
from ytkb.knowledge.prompts import MERGE_SYSTEM, build_merge_user
from ytkb.logging import get_logger

log = get_logger("knowledge.merge")

_CITATION = re.compile(r"\[unit:(\d+)\]")

# Section ordering + labels for the offline merger.
_SECTIONS: list[tuple[UnitType, str]] = [
    (UnitType.definition, "Definizioni"),
    (UnitType.technique, "Tecniche"),
    (UnitType.workflow, "Workflow"),
    (UnitType.tool, "Strumenti"),
    (UnitType.example, "Esempi"),
    (UnitType.claim, "Affermazioni"),
]


@dataclass(slots=True)
class MergeResult:
    content_md: str
    change_summary: str


# --------------------------------------------------------------------------
# Validator (binding — see CLAUDE.md)
# --------------------------------------------------------------------------
def cited_unit_ids(content_md: str) -> set[int]:
    return {int(m) for m in _CITATION.findall(content_md)}


def validate_article(content_md: str, allowed_ids: set[int], required_ids: set[int]) -> list[str]:
    """Return a list of validation errors (empty == valid).

    - every [unit:ID] must be an allowed unit of this topic;
    - every new (required) unit must be cited, unless explicitly listed under an
      "Unità omesse" note.
    """
    errors: list[str] = []
    cited = cited_unit_ids(content_md)
    invalid = cited - allowed_ids
    if invalid:
        errors.append(f"citazioni non valide: {sorted(invalid)}")
    omitted_note = "unità omesse" in content_md.lower()
    missing = required_ids - cited
    if missing and not omitted_note:
        errors.append(f"unità nuove non citate: {sorted(missing)}")
    return errors


# --------------------------------------------------------------------------
# Mergers
# --------------------------------------------------------------------------
class ArticleMerger(ABC):
    @abstractmethod
    async def merge(
        self,
        *,
        topic_title: str,
        current_md: str | None,
        current_units: list[KnowledgeUnit],
        new_units: list[KnowledgeUnit],
    ) -> MergeResult: ...


class HeuristicMerger(ArticleMerger):
    """Deterministic offline merger: rebuild the article from all the topic's
    units, grouped by type, every bullet citing its unit. Opinions go under a
    "Punti di disaccordo" section. Guarantees the validator passes."""

    async def merge(
        self,
        *,
        topic_title: str,
        current_md: str | None,
        current_units: list[KnowledgeUnit],
        new_units: list[KnowledgeUnit],
    ) -> MergeResult:
        units = list(current_units) + list(new_units)
        lines = [f"# {topic_title}", ""]
        for unit_type, label in _SECTIONS:
            group = [u for u in units if u.unit_type == unit_type]
            if not group:
                continue
            lines.append(f"## {label}")
            for u in group:
                lines.append(f"- {u.text.rstrip('.')} [unit:{u.id}]")
            lines.append("")

        opinions = [u for u in units if u.unit_type == UnitType.opinion]
        if opinions:
            lines.append("## Punti di disaccordo")
            for u in opinions:
                lines.append(f"- {u.text.rstrip('.')} [unit:{u.id}]")
            lines.append("")

        summary = (
            f"Aggiunte {len(new_units)} unità"
            if current_md
            else f"Prima versione con {len(new_units)} unità"
        )
        return MergeResult(content_md="\n".join(lines).strip() + "\n", change_summary=summary)


class LLMMerger(ArticleMerger):
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def merge(
        self,
        *,
        topic_title: str,
        current_md: str | None,
        current_units: list[KnowledgeUnit],
        new_units: list[KnowledgeUnit],
    ) -> MergeResult:
        prompt = build_merge_user(
            topic_title,
            current_md,
            [(u.id, u.unit_type.value, u.text) for u in new_units],
        )
        raw = await self.client.complete(system=MERGE_SYSTEM, prompt=prompt, max_tokens=4096)
        return _parse_merge(raw)


def _parse_merge(raw: str) -> MergeResult:
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return MergeResult(content_md="", change_summary="")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return MergeResult(content_md="", change_summary="")
    return MergeResult(
        content_md=str(payload.get("content_md", "")),
        change_summary=str(payload.get("change_summary", "")),
    )


def get_merger(settings: Settings | None = None) -> ArticleMerger:
    settings = settings or get_settings()
    client = get_llm_client(settings, model=settings.llm_merge_model)
    if client is not None:
        return LLMMerger(client)
    return HeuristicMerger()


# --------------------------------------------------------------------------
# Merge worker
# --------------------------------------------------------------------------
async def _latest_article(session: AsyncSession, topic_id: int) -> TopicArticle | None:
    return (
        (
            await session.execute(
                select(TopicArticle)
                .where(TopicArticle.topic_id == topic_id)
                .order_by(TopicArticle.version.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )


async def _topic_units(session: AsyncSession, topic_id: int) -> list[KnowledgeUnit]:
    return list(
        (
            await session.execute(
                select(KnowledgeUnit)
                .where(
                    KnowledgeUnit.topic_id == topic_id,
                    KnowledgeUnit.assignment != Assignment.pending,
                )
                .order_by(KnowledgeUnit.id)
            )
        ).scalars()
    )


async def merge_topic(
    session: AsyncSession,
    topic: Topic,
    *,
    merger: ArticleMerger | None = None,
    settings: Settings | None = None,
    force: bool = False,
) -> TopicArticle | None:
    """Merge a topic's new units into a new article version. Returns the new
    article, or None if the topic is not dirty (and not forced)."""
    settings = settings or get_settings()
    merger = merger or get_merger(settings)

    units = await _topic_units(session, topic.id)
    if not units:
        return None
    latest = await _latest_article(session, topic.id)
    included = set() if force else set(latest.units_included) if latest else set()
    new_units = [u for u in units if u.id not in included]
    if not new_units and not force:
        return None

    current_units = [u for u in units if u.id in included]
    current_md = None if force else (latest.content_md if latest else None)
    allowed = {u.id for u in units}
    required = {u.id for u in new_units}

    result = await _merge_with_validation(
        merger, topic, current_md, current_units, new_units, allowed, required
    )

    next_version = (latest.version + 1) if latest else 1
    article = TopicArticle(
        topic_id=topic.id,
        version=next_version,
        content_md=result.content_md,
        units_included=sorted(allowed),
        change_summary=result.change_summary,
    )
    session.add(article)
    topic.last_merged_at = datetime.now(UTC)
    await session.flush()
    log.info("topic_merged", topic=topic.slug, version=next_version, new_units=len(new_units))
    return article


async def _merge_with_validation(
    merger: ArticleMerger,
    topic: Topic,
    current_md: str | None,
    current_units: list[KnowledgeUnit],
    new_units: list[KnowledgeUnit],
    allowed: set[int],
    required: set[int],
) -> MergeResult:
    result = await merger.merge(
        topic_title=topic.title,
        current_md=current_md,
        current_units=current_units,
        new_units=new_units,
    )
    errors = validate_article(result.content_md, allowed, required)
    if not errors:
        return result
    log.warning("merge_invalid", topic=topic.slug, errors=errors)
    # Fall back to the deterministic merger, which always satisfies the validator.
    fallback = await HeuristicMerger().merge(
        topic_title=topic.title,
        current_md=current_md,
        current_units=current_units,
        new_units=new_units,
    )
    errors = validate_article(fallback.content_md, allowed, required)
    if errors:  # pragma: no cover - heuristic is constructed to pass
        raise ValueError(f"merge validation failed for {topic.slug}: {errors}")
    return fallback


async def merge_dirty_topics(
    session: AsyncSession,
    *,
    merger: ArticleMerger | None = None,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    merger = merger or get_merger(settings)
    topics = (
        (await session.execute(select(Topic).where(Topic.status != TopicStatus.merged_into)))
        .scalars()
        .all()
    )
    merged = 0
    for topic in topics:
        article = await merge_topic(session, topic, merger=merger, settings=settings)
        if article is not None:
            merged += 1
    await _advance_merged_videos(session)
    await session.flush()
    result = {"topics_merged": merged}
    log.info("merge_run", **result)
    return result


async def rebuild_topic(
    session: AsyncSession,
    slug: str,
    *,
    merger: ArticleMerger | None = None,
    settings: Settings | None = None,
) -> TopicArticle | None:
    topic = await session.scalar(select(Topic).where(Topic.slug == slug))
    if topic is None:
        raise ValueError(f"Unknown topic: {slug}")
    return await merge_topic(session, topic, merger=merger, settings=settings, force=True)


async def _advance_merged_videos(session: AsyncSession) -> None:
    """A video moves assigned → merged once every one of its units is included
    in its topic's latest article."""
    videos = (
        (await session.execute(select(Video).where(Video.ingest_status == IngestStatus.assigned)))
        .scalars()
        .all()
    )
    article_cache: dict[int, set[int]] = {}
    for video in videos:
        units = await _topic_units_for_video(session, video.id)
        if not units:
            continue
        all_merged = True
        for unit in units:
            if unit.topic_id is None:
                all_merged = False
                break
            if unit.topic_id not in article_cache:
                latest = await _latest_article(session, unit.topic_id)
                article_cache[unit.topic_id] = set(latest.units_included) if latest else set()
            if unit.id not in article_cache[unit.topic_id]:
                all_merged = False
                break
        if all_merged:
            video.ingest_status = IngestStatus.merged


async def _topic_units_for_video(session: AsyncSession, video_id: int) -> list[KnowledgeUnit]:
    return list(
        (
            await session.execute(select(KnowledgeUnit).where(KnowledgeUnit.video_id == video_id))
        ).scalars()
    )
