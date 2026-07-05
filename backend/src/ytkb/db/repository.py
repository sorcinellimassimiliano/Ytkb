"""Thin repository helpers. Keep raw query knowledge in one place so services
and API routers don't duplicate SELECTs."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.db.models import Channel, IngestStatus, Video


async def get_channel_by_yt_id(session: AsyncSession, yt_channel_id: str) -> Channel | None:
    return await session.scalar(select(Channel).where(Channel.yt_channel_id == yt_channel_id))


async def list_active_channels(session: AsyncSession) -> list[Channel]:
    result = await session.execute(select(Channel).where(Channel.active.is_(True)))
    return list(result.scalars())


async def get_video_by_yt_id(session: AsyncSession, yt_video_id: str) -> Video | None:
    return await session.scalar(select(Video).where(Video.yt_video_id == yt_video_id))


async def videos_by_status(
    session: AsyncSession, status: IngestStatus, *, limit: int | None = None
) -> list[Video]:
    stmt = select(Video).where(Video.ingest_status == status).order_by(Video.id)
    if limit:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars())
