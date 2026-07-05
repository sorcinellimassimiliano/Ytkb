"""Admin endpoints — watchlist, pipeline status, proposed topics.

Channel writes here are metadata-only (no network): they add/update a channel on
the watchlist. Actual discovery + transcription runs via the CLI/scheduler.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.db import repository as repo
from ytkb.db.base import get_session
from ytkb.db.models import Channel, IngestStatus, Video

router = APIRouter(prefix="/admin", tags=["admin"])


def _channel_dict(c: Channel) -> dict:
    return {
        "id": c.id,
        "yt_channel_id": c.yt_channel_id,
        "handle": c.handle,
        "title": c.title,
        "active": c.active,
        "last_checked_at": c.last_checked_at,
    }


class ChannelUpsert(BaseModel):
    yt_channel_id: str = Field(min_length=1)
    handle: str | None = None
    title: str | None = None
    active: bool = True


@router.get("/channels")
async def list_channels(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[dict]:
    result = await session.execute(select(Channel).order_by(Channel.title))
    return [_channel_dict(c) for c in result.scalars()]


@router.post("/channels", status_code=201)
async def upsert_channel(
    payload: ChannelUpsert,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    channel = await repo.get_channel_by_yt_id(session, payload.yt_channel_id)
    if channel is None:
        channel = Channel(yt_channel_id=payload.yt_channel_id)
        session.add(channel)
    channel.handle = payload.handle or channel.handle
    channel.title = payload.title or channel.title
    channel.active = payload.active
    await session.commit()
    await session.refresh(channel)
    return _channel_dict(channel)


@router.get("/videos")
async def list_videos(
    session: Annotated[AsyncSession, Depends(get_session)],
    status: IngestStatus | None = None,
    limit: int = 100,
) -> list[dict]:
    stmt = select(Video).order_by(Video.id.desc()).limit(min(limit, 500))
    if status is not None:
        stmt = stmt.where(Video.ingest_status == status)
    result = await session.execute(stmt)
    return [
        {
            "id": v.id,
            "yt_video_id": v.yt_video_id,
            "title": v.title,
            "kind": v.kind.value,
            "ingest_status": v.ingest_status.value,
            "error_msg": v.error_msg,
        }
        for v in result.scalars()
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
