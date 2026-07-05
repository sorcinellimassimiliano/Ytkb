"""KB browser endpoints — pure REST, NEVER call an LLM (see CLAUDE.md).

Phase 5 will flesh these out (topic tree, articles, hybrid search). The stubs
establish the router surface and the no-LLM contract.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.db import search
from ytkb.db.base import get_session
from ytkb.db.models import Chunk, KnowledgeUnit, Topic, TopicArticle, TopicStatus, Video
from ytkb.indexing.embeddings import get_embedding_client

router = APIRouter(prefix="/kb", tags=["kb"])


@router.get("/topics")
async def list_topics(
    session: Annotated[AsyncSession, Depends(get_session)],
    status: TopicStatus | None = None,
    limit: Annotated[int, Query(le=200)] = 100,
) -> list[dict]:
    stmt = select(Topic).order_by(Topic.units_count.desc()).limit(limit)
    if status is not None:
        stmt = stmt.where(Topic.status == status)
    result = await session.execute(stmt)
    return [
        {
            "id": t.id,
            "slug": t.slug,
            "title": t.title,
            "summary": t.summary,
            "parent_id": t.parent_id,
            "status": t.status.value,
            "units_count": t.units_count,
        }
        for t in result.scalars()
    ]


@router.get("/topics/{slug}")
async def get_topic(
    slug: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    """Topic page: metadata + the current (latest) article version."""
    topic = await session.scalar(select(Topic).where(Topic.slug == slug))
    if topic is None:
        raise HTTPException(status_code=404, detail="topic not found")
    article = (
        (
            await session.execute(
                select(TopicArticle)
                .where(TopicArticle.topic_id == topic.id)
                .order_by(TopicArticle.version.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    return {
        "id": topic.id,
        "slug": topic.slug,
        "title": topic.title,
        "status": topic.status.value,
        "units_count": topic.units_count,
        "article": None
        if article is None
        else {
            "version": article.version,
            "content_md": article.content_md,
            "change_summary": article.change_summary,
            "units_included": article.units_included,
            "created_at": article.created_at,
        },
    }


@router.get("/topics/{slug}/versions")
async def get_topic_versions(
    slug: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[dict]:
    topic = await session.scalar(select(Topic).where(Topic.slug == slug))
    if topic is None:
        raise HTTPException(status_code=404, detail="topic not found")
    rows = await session.execute(
        select(TopicArticle)
        .where(TopicArticle.topic_id == topic.id)
        .order_by(TopicArticle.version.desc())
    )
    return [
        {
            "version": a.version,
            "change_summary": a.change_summary,
            "units_included": len(a.units_included),
            "created_at": a.created_at,
        }
        for a in rows.scalars()
    ]


@router.get("/units/{unit_id}")
async def get_unit(
    unit_id: int,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    """Resolve a [unit:ID] citation to its provenance: video + chunk timestamps,
    so the renderer can link to YouTube at the exact moment."""
    unit = await session.get(KnowledgeUnit, unit_id)
    if unit is None:
        raise HTTPException(status_code=404, detail="unit not found")
    video = await session.get(Video, unit.video_id)
    chunks = []
    if unit.chunk_ids:
        rows = await session.execute(
            select(Chunk).where(Chunk.id.in_(unit.chunk_ids)).order_by(Chunk.start_s)
        )
        chunks = [{"id": c.id, "start_s": c.start_s, "end_s": c.end_s} for c in rows.scalars()]
    return {
        "id": unit.id,
        "text": unit.text,
        "unit_type": unit.unit_type.value,
        "topic_id": unit.topic_id,
        "video": None
        if video is None
        else {
            "yt_video_id": video.yt_video_id,
            "title": video.title,
            "url": video.url,
        },
        "chunks": chunks,
    }


@router.get("/search")
async def search_chunks(
    session: Annotated[AsyncSession, Depends(get_session)],
    q: Annotated[str, Query(min_length=1)],
    mode: Literal["hybrid", "semantic", "fts"] = "hybrid",
    limit: Annotated[int, Query(le=100)] = 10,
) -> dict:
    """Search transcript chunks. No LLM: semantic modes embed the query once."""
    if mode == "fts":
        hits = await search.fts_chunks(session, q, limit=limit)
    else:
        query_vec = await get_embedding_client().embed_one(q)
        if mode == "semantic":
            hits = await search.semantic_chunks(session, query_vec, limit=limit)
        else:
            hits = await search.hybrid_chunks(session, q, query_vec, limit=limit)
    return {"query": q, "mode": mode, "results": [h.as_dict() for h in hits]}


@router.get("/stats")
async def kb_stats(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    total = await session.scalar(select(func.count(Topic.id)))
    active = await session.scalar(
        select(func.count(Topic.id)).where(Topic.status == TopicStatus.active)
    )
    return {"topics_total": total or 0, "topics_active": active or 0}
