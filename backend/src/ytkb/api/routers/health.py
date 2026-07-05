"""Health endpoint — verifies DB connectivity and required extensions."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb import __version__
from ytkb.config import get_settings
from ytkb.db.base import get_session

router = APIRouter(tags=["health"])

REQUIRED_EXTENSIONS = ("vector", "pg_trgm")


@router.get("/health")
async def health(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    settings = get_settings()
    status: dict = {
        "status": "ok",
        "version": __version__,
        "chat_enabled": settings.chat_enabled,
        "embedding_dim": settings.embedding_dim,
        "db": "unknown",
        "extensions": {},
    }

    try:
        await session.execute(text("SELECT 1"))
        status["db"] = "ok"
        rows = await session.execute(
            text("SELECT extname FROM pg_extension WHERE extname = ANY(:names)").bindparams(
                names=list(REQUIRED_EXTENSIONS)
            )
        )
        installed = {r[0] for r in rows}
        status["extensions"] = {ext: (ext in installed) for ext in REQUIRED_EXTENSIONS}
        if not all(status["extensions"].values()):
            status["status"] = "degraded"
    except Exception as exc:  # pragma: no cover - depends on live DB
        status["status"] = "error"
        status["db"] = "error"
        status["detail"] = str(exc)

    return status
