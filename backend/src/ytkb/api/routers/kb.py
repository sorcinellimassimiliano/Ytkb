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
from ytkb.db.models import (
    Channel,
    Chunk,
    KnowledgeUnit,
    Topic,
    TopicArticle,
    TopicRelation,
    TopicStatus,
    Transcript,
    Video,
)
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


@router.get("/tree")
async def topic_tree(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[dict]:
    """Topic hierarchy (parent_id) as a nested tree; merged topics excluded."""
    topics = (
        (
            await session.execute(
                select(Topic).where(Topic.status != TopicStatus.merged_into).order_by(Topic.title)
            )
        )
        .scalars()
        .all()
    )

    def node(t: Topic) -> dict:
        return {
            "id": t.id,
            "slug": t.slug,
            "title": t.title,
            "status": t.status.value,
            "units_count": t.units_count,
            "children": [],
        }

    by_id = {t.id: node(t) for t in topics}
    roots: list[dict] = []
    for t in topics:
        n = by_id[t.id]
        if t.parent_id and t.parent_id in by_id:
            by_id[t.parent_id]["children"].append(n)
        else:
            roots.append(n)
    return roots


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
    # Related topics (either direction of the relation).
    rel_rows = await session.execute(
        select(TopicRelation).where(
            (TopicRelation.topic_a == topic.id) | (TopicRelation.topic_b == topic.id)
        )
    )
    related_ids = set()
    for rel in rel_rows.scalars():
        related_ids.add(rel.topic_b if rel.topic_a == topic.id else rel.topic_a)
    related = []
    if related_ids:
        rows = await session.execute(select(Topic).where(Topic.id.in_(related_ids)))
        related = [{"id": t.id, "slug": t.slug, "title": t.title} for t in rows.scalars()]

    return {
        "id": topic.id,
        "slug": topic.slug,
        "title": topic.title,
        "status": topic.status.value,
        "units_count": topic.units_count,
        "related": related,
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


@router.get("/topics/{slug}/versions/{version}")
async def get_topic_version(
    slug: str,
    version: int,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    """A specific article version — lets the frontend diff two versions."""
    topic = await session.scalar(select(Topic).where(Topic.slug == slug))
    if topic is None:
        raise HTTPException(status_code=404, detail="topic not found")
    article = await session.scalar(
        select(TopicArticle).where(
            TopicArticle.topic_id == topic.id, TopicArticle.version == version
        )
    )
    if article is None:
        raise HTTPException(status_code=404, detail="version not found")
    return {
        "version": article.version,
        "content_md": article.content_md,
        "change_summary": article.change_summary,
        "units_included": article.units_included,
        "created_at": article.created_at,
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
async def search_kb(
    session: Annotated[AsyncSession, Depends(get_session)],
    q: Annotated[str, Query(min_length=1)],
    chunk_mode: Literal["hybrid", "semantic", "fts"] = "hybrid",
    limit: Annotated[int, Query(le=100)] = 10,
) -> dict:
    """Unified KB search, grouped by type (topics / units / chunks). No LLM:
    the query is embedded once (an embedding model, not an LLM)."""
    query_vec = await get_embedding_client().embed_one(q)
    topics = await search.hybrid_topics(session, q, query_vec, limit=limit)
    units = await search.hybrid_units(session, q, query_vec, limit=limit)
    if chunk_mode == "fts":
        chunks = await search.fts_chunks(session, q, limit=limit)
    elif chunk_mode == "semantic":
        chunks = await search.semantic_chunks(session, query_vec, limit=limit)
    else:
        chunks = await search.hybrid_chunks(session, q, query_vec, limit=limit)
    return {
        "query": q,
        "results": {
            "topics": [h.as_dict() for h in topics],
            "units": [h.as_dict() for h in units],
            "chunks": [h.as_dict() for h in chunks],
        },
    }


@router.get("/sources")
async def list_sources(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[dict]:
    """Source index: channels with video counts (channels → videos → units)."""
    channels = (await session.execute(select(Channel).order_by(Channel.title))).scalars().all()
    out = []
    for c in channels:
        total = await session.scalar(select(func.count(Video.id)).where(Video.channel_id == c.id))
        out.append(
            {
                "id": c.id,
                "yt_channel_id": c.yt_channel_id,
                "handle": c.handle,
                "title": c.title,
                "videos_total": total or 0,
            }
        )
    return out


@router.get("/videos/{yt_video_id}")
async def get_video(
    yt_video_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    """A video's page: metadata, transcript segments, and where its content
    ended up (the knowledge units extracted from it, with their topic)."""
    video = await session.scalar(select(Video).where(Video.yt_video_id == yt_video_id))
    if video is None:
        raise HTTPException(status_code=404, detail="video not found")
    transcript = (
        (
            await session.execute(
                select(Transcript)
                .where(Transcript.video_id == video.id)
                .order_by(Transcript.id.desc())
            )
        )
        .scalars()
        .first()
    )
    segments = (transcript.raw_json or {}).get("segments", []) if transcript else []

    unit_rows = await session.execute(
        select(KnowledgeUnit, Topic.slug)
        .join(Topic, Topic.id == KnowledgeUnit.topic_id, isouter=True)
        .where(KnowledgeUnit.video_id == video.id)
        .order_by(KnowledgeUnit.id)
    )
    units = [
        {
            "id": u.id,
            "text": u.text,
            "unit_type": u.unit_type.value,
            "topic_slug": slug,
        }
        for u, slug in unit_rows
    ]
    return {
        "yt_video_id": video.yt_video_id,
        "title": video.title,
        "url": video.url,
        "ingest_status": video.ingest_status.value,
        "segments": segments,
        "units": units,
    }


@router.get("/stats")
async def kb_stats(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    total = await session.scalar(select(func.count(Topic.id)))
    active = await session.scalar(
        select(func.count(Topic.id)).where(Topic.status == TopicStatus.active)
    )
    return {"topics_total": total or 0, "topics_active": active or 0}
