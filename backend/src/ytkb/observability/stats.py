"""Pipeline stats and a rough LLM/embedding cost estimate.

Costs are ESTIMATES derived from stored token counts and configurable per-Mtok
prices — enough for an ingestion budget report, not billing.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.config import Settings, get_settings
from ytkb.db.models import (
    Chunk,
    IngestStatus,
    KnowledgeUnit,
    Topic,
    TopicArticle,
    TopicStatus,
    Video,
)


async def pipeline_stats(session: AsyncSession) -> dict:
    videos_total = await session.scalar(select(func.count(Video.id))) or 0
    errors = (
        await session.scalar(
            select(func.count(Video.id)).where(Video.ingest_status == IngestStatus.error)
        )
        or 0
    )
    units = await session.scalar(select(func.count(KnowledgeUnit.id))) or 0
    topics_active = (
        await session.scalar(select(func.count(Topic.id)).where(Topic.status == TopicStatus.active))
        or 0
    )
    topics_proposed = (
        await session.scalar(
            select(func.count(Topic.id)).where(Topic.status == TopicStatus.proposed)
        )
        or 0
    )
    articles = await session.scalar(select(func.count(TopicArticle.id))) or 0

    # Merge queue: topics whose units_count exceeds the units in their latest
    # article (i.e. dirty). Approximated as topics with units but no article,
    # plus a coarse count is enough for a health signal.
    topics_with_units = (
        await session.scalar(select(func.count(Topic.id)).where(Topic.units_count > 0)) or 0
    )
    topics_with_article = (
        await session.scalar(select(func.count(func.distinct(TopicArticle.topic_id)))) or 0
    )
    merge_queue = max(0, topics_with_units - topics_with_article)

    return {
        "videos_total": videos_total,
        "videos_error": errors,
        "failure_rate": round(errors / videos_total, 4) if videos_total else 0.0,
        "units": units,
        "topics_active": topics_active,
        "topics_proposed": topics_proposed,
        "articles": articles,
        "merge_queue": merge_queue,
    }


async def cost_report(session: AsyncSession, *, settings: Settings | None = None) -> dict:
    """Estimate ingestion cost from stored token counts.

    - embedding: chunk tokens + unit tokens (approx) at embedding price;
    - extraction (Haiku): ~chunk tokens in;
    - merge (Sonnet): ~unit tokens in.
    """
    settings = settings or get_settings()
    chunk_tokens = await session.scalar(select(func.coalesce(func.sum(Chunk.token_count), 0))) or 0
    units = await session.scalar(select(func.count(KnowledgeUnit.id))) or 0
    # Rough: an average unit is ~30 tokens.
    unit_tokens = units * 30

    embed_tokens = chunk_tokens + unit_tokens
    extraction_tokens = chunk_tokens
    merge_tokens = unit_tokens

    def usd(tokens: int, per_mtok: float) -> float:
        return round(tokens / 1_000_000 * per_mtok, 4)

    est = (
        usd(embed_tokens, settings.price_embedding_per_mtok)
        + usd(extraction_tokens, settings.price_extraction_per_mtok)
        + usd(merge_tokens, settings.price_merge_per_mtok)
    )
    return {
        "embedding_tokens": embed_tokens,
        "extraction_tokens_in": extraction_tokens,
        "merge_tokens_in": merge_tokens,
        "estimated_cost_usd": round(est, 4),
        "note": "stima approssimata da token contati; non è fatturazione",
    }
