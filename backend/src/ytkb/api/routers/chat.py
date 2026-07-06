"""Chat endpoint (optional layer) — only mounted when CHAT_ENABLED=true.

Isolated in rag/ per CLAUDE.md. Streams answers via SSE and persists sessions.
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.db.base import get_session
from ytkb.db.models import ChatSession
from ytkb.rag import service

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: int | None = None


@router.get("/status")
async def chat_status() -> dict:
    return {"enabled": True}


@router.post("/sessions")
async def new_session(
    session: Annotated[AsyncSession, Depends(get_session)],
    title: str | None = None,
) -> dict:
    chat = await service.create_session(session, title=title)
    await session.commit()
    return {"id": chat.id, "title": chat.title}


@router.get("/sessions/{chat_id}/messages")
async def session_messages(
    chat_id: int,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[dict]:
    messages = await service.get_messages(session, chat_id)
    return [
        {"id": m.id, "role": m.role, "content": m.content, "sources": m.sources} for m in messages
    ]


@router.post("")
async def chat(
    payload: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> StreamingResponse:
    """Stream a grounded answer over SSE. Creates a session if none is given."""
    if payload.session_id is not None:
        chat_session = await session.get(ChatSession, payload.session_id)
        if chat_session is None:
            raise HTTPException(status_code=404, detail="session not found")
        chat_id = chat_session.id
    else:
        chat_id = (await service.create_session(session)).id

    async def event_stream():
        yield f"data: {json.dumps({'type': 'session', 'session_id': chat_id})}\n\n"
        async for event in service.answer(session, chat_id, payload.message):
            yield f"data: {json.dumps(event)}\n\n"
        await session.commit()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
