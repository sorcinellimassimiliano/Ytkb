"""Chat endpoint (optional layer) — only mounted when CHAT_ENABLED=true.

Isolated in rag/ per CLAUDE.md. Full SSE implementation lands in Phase 6.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/status")
async def chat_status() -> dict:
    return {"enabled": True, "note": "Chat layer active (Phase 6 pending)."}
