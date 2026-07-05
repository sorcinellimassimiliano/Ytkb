"""Chat orchestration: retrieve context, stream an answer, persist the session.

Kept in rag/ and only reachable when CHAT_ENABLED. The /kb/* endpoints never
import this module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.config import Settings, get_settings
from ytkb.db.models import ChatMessage, ChatSession
from ytkb.indexing.embeddings import EmbeddingClient
from ytkb.logging import get_logger
from ytkb.rag.engine import ChatResponder, OfflineResponder, get_responder
from ytkb.rag.prompts import CHAT_SYSTEM, build_chat_prompt
from ytkb.rag.retrieval import retrieve

log = get_logger("rag.service")


async def create_session(session: AsyncSession, *, title: str | None = None) -> ChatSession:
    chat = ChatSession(title=title)
    session.add(chat)
    await session.flush()
    return chat


async def get_messages(session: AsyncSession, chat_id: int) -> list[ChatMessage]:
    rows = await session.execute(
        select(ChatMessage).where(ChatMessage.session_id == chat_id).order_by(ChatMessage.id)
    )
    return list(rows.scalars())


async def answer(
    session: AsyncSession,
    chat_id: int,
    query: str,
    *,
    responder: ChatResponder | None = None,
    embedder: EmbeddingClient | None = None,
    settings: Settings | None = None,
) -> AsyncIterator[dict]:
    """Yield events: {"type": "token", "text": ...} then
    {"type": "done", "sources": [...]}. Persists the user question and the full
    assistant answer (with sources) once streaming completes."""
    settings = settings or get_settings()

    context = await retrieve(session, query, embedder=embedder, settings=settings)
    responder = responder or get_responder(settings) or OfflineResponder(context)

    session.add(ChatMessage(session_id=chat_id, role="user", content=query))
    await session.flush()

    prompt = build_chat_prompt(query, context.context_md)
    parts: list[str] = []
    async for token in responder.stream(system=CHAT_SYSTEM, prompt=prompt):
        parts.append(token)
        yield {"type": "token", "text": token}

    answer_text = "".join(parts)
    sources = [{"label": s.label, "kind": s.kind, "ref": s.ref} for s in context.sources]
    session.add(
        ChatMessage(
            session_id=chat_id,
            role="assistant",
            content=answer_text,
            sources={"sources": sources},
        )
    )
    await session.flush()
    log.info("chat_answered", chat=chat_id, sources=len(sources))
    yield {"type": "done", "sources": sources}
