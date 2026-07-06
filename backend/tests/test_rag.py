"""Integration tests for the optional chat layer (rag/) against real Postgres.

Uses the offline responder — no cloud LLM — so retrieval, streaming and session
persistence are all exercised deterministically.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import func, select

from ytkb.db.models import ChatMessage
from ytkb.indexing.embeddings import FakeEmbeddingClient
from ytkb.indexing.pipeline import index_video
from ytkb.knowledge.assignment import assign_pending_units
from ytkb.knowledge.extraction import HeuristicExtractor, extract_units
from ytkb.knowledge.merge import merge_dirty_topics
from ytkb.rag import service
from ytkb.rag.engine import ChatResponder, OfflineResponder
from ytkb.rag.retrieval import retrieve

from .conftest import requires_db
from .fakes import seed_transcribed_video

pytestmark = requires_db

EMB = FakeEmbeddingClient(dim=1536)

SEGMENTS = [
    (
        0.0,
        12.0,
        "Il context management significa gestire le informazioni nel contesto. "
        "Un tool utile e il prompt caching che riusa il contesto tra chiamate.",
    ),
]


async def _seed(session):
    video = await seed_transcribed_video(session, yt_video_id="rag1", segments=SEGMENTS)
    await index_video(session, video, embedder=EMB)
    await extract_units(session, video, extractor=HeuristicExtractor(), embedder=EMB)
    await assign_pending_units(session)
    await merge_dirty_topics(session)
    return video


async def test_retrieve_builds_context_with_sources(session):
    await _seed(session)
    ctx = await retrieve(session, "context management", embedder=EMB)
    assert not ctx.is_empty
    assert ctx.sources, "retrieval should surface sources"
    # Sources carry citation labels (video title — mm:ss for units/chunks).
    assert any("—" in s.label for s in ctx.sources if s.kind in {"unit", "chunk"})


async def test_answer_streams_and_persists(session):
    chat = await service.create_session(session, title="t")
    events = [
        ev
        async for ev in service.answer(
            session, chat.id, "cos'è il context management?", embedder=EMB
        )
    ]
    kinds = [e["type"] for e in events]
    assert "token" in kinds and kinds[-1] == "done"
    assert isinstance(events[-1]["sources"], list)

    # User + assistant messages persisted.
    count = await session.scalar(
        select(func.count(ChatMessage.id)).where(ChatMessage.session_id == chat.id)
    )
    assert count == 2
    assistant = (
        (
            await session.execute(
                select(ChatMessage).where(
                    ChatMessage.session_id == chat.id, ChatMessage.role == "assistant"
                )
            )
        )
        .scalars()
        .one()
    )
    assert assistant.content
    assert assistant.sources is not None


async def test_offline_responder_honest_when_no_context(session):
    # Empty KB → no context → the responder admits it can't answer.
    chat = await service.create_session(session)
    events = [ev async for ev in service.answer(session, chat.id, "qualcosa", embedder=EMB)]
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert "non ho abbastanza contesto" in text.lower()


async def test_custom_responder_is_used(session):
    await _seed(session)
    chat = await service.create_session(session)

    class Echo(ChatResponder):
        async def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]:
            yield "RISPOSTA-ECHO"

    events = [
        ev
        async for ev in service.answer(session, chat.id, "domanda", responder=Echo(), embedder=EMB)
    ]
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert text == "RISPOSTA-ECHO"


def test_offline_responder_unit():
    from ytkb.rag.retrieval import RetrievedContext

    empty = OfflineResponder(RetrievedContext(context_md=""))
    assert empty.context.is_empty
