"""Admin endpoints — watchlist, pipeline status, proposed topics."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.db.base import get_session
from ytkb.db.models import Channel, IngestStatus, Video

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/channels")
async def list_channels(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[dict]:
    result = await session.execute(select(Channel).order_by(Channel.title))
    return [
        {
            "id": c.id,
            "yt_channel_id": c.yt_channel_id,
            "handle": c.handle,
            "title": c.title,
            "active": c.active,
            "last_checked_at": c.last_checked_at,
        }
        for c in result.scalars()
    ]


@router.get("/stats")
async def admin_stats(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    rows = await session.execute(
        select(Video.ingest_status, func.count(Video.id)).group_by(Video.ingest_status)
    )
    by_status = {status.value: count for status, count in rows}
    return {
        "videos_by_status": {s.value: by_status.get(s.value, 0) for s in IngestStatus},
        "videos_total": sum(by_status.values()),
    }
