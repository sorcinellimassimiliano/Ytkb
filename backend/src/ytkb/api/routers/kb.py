"""KB browser endpoints — pure REST, NEVER call an LLM (see CLAUDE.md).

Phase 5 will flesh these out (topic tree, articles, hybrid search). The stubs
establish the router surface and the no-LLM contract.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.db.base import get_session
from ytkb.db.models import Topic, TopicStatus

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


@router.get("/stats")
async def kb_stats(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    total = await session.scalar(select(func.count(Topic.id)))
    active = await session.scalar(
        select(func.count(Topic.id)).where(Topic.status == TopicStatus.active)
    )
    return {"topics_total": total or 0, "topics_active": active or 0}
