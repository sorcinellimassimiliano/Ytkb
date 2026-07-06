"""Chat retrieval: assemble grounded context, articles first.

Order (per plan): topic articles (dense, already synthesized) → knowledge units
→ transcript chunks (for pinpoint detail). No LLM here — the query is embedded
once. Every unit/chunk source carries a [titolo — mm:ss] label so the model can
cite and the client can link back to YouTube.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.config import Settings, get_settings
from ytkb.db import search
from ytkb.db.models import Chunk, KnowledgeUnit, TopicArticle, Video
from ytkb.indexing.embeddings import EmbeddingClient, get_embedding_client
from ytkb.rag.prompts import Source


@dataclass(slots=True)
class RetrievedContext:
    context_md: str
    sources: list[Source] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.context_md.strip()


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


async def _video_titles(session: AsyncSession, video_ids: set[int]) -> dict[int, Video]:
    if not video_ids:
        return {}
    rows = await session.execute(select(Video).where(Video.id.in_(video_ids)))
    return {v.id: v for v in rows.scalars()}


async def retrieve(
    session: AsyncSession,
    query: str,
    *,
    embedder: EmbeddingClient | None = None,
    settings: Settings | None = None,
) -> RetrievedContext:
    settings = settings or get_settings()
    embedder = embedder or get_embedding_client(settings)
    qvec = await embedder.embed_one(query)

    topics = await search.hybrid_topics(session, query, qvec, limit=settings.chat_topic_k)
    units = await search.hybrid_units(session, query, qvec, limit=settings.chat_unit_k)
    chunks = await search.hybrid_chunks(session, query, qvec, limit=settings.chat_chunk_k)

    sources: list[Source] = []
    blocks: list[str] = []

    # --- Topic articles (dense) --------------------------------------------
    if topics:
        blocks.append("=== ARGOMENTI (sintesi) ===")
        for t in topics:
            article = (
                (
                    await session.execute(
                        select(TopicArticle)
                        .where(TopicArticle.topic_id == t.id)
                        .order_by(TopicArticle.version.desc())
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )
            body = article.content_md.strip() if article else ""
            blocks.append(f"## {t.title}\n{body}")
            sources.append(Source(label=t.title, kind="topic", ref=t.slug))

    # Provenance for units + chunks (video title + timestamp).
    unit_rows = {
        u.id: u
        for u in (
            await session.execute(
                select(KnowledgeUnit).where(KnowledgeUnit.id.in_([u.id for u in units] or [-1]))
            )
        ).scalars()
    }
    chunk_ids = {c for u in unit_rows.values() for c in (u.chunk_ids or [])}
    chunk_rows = {
        c.id: c
        for c in (
            await session.execute(select(Chunk).where(Chunk.id.in_(chunk_ids or {-1})))
        ).scalars()
    }
    video_ids = {u.video_id for u in unit_rows.values()} | {c.video_id for c in chunks}
    videos = await _video_titles(session, video_ids)

    def label_for(video_id: int, start_s: float | None) -> str:
        v = videos.get(video_id)
        title = (v.title if v else None) or (v.yt_video_id if v else "video")
        return f"{title} — {_ts(start_s)}" if start_s is not None else title

    # --- Units -------------------------------------------------------------
    if units:
        blocks.append("=== UNITÀ ===")
        for u in units:
            row = unit_rows.get(u.id)
            start = None
            if row and row.chunk_ids:
                first = chunk_rows.get(row.chunk_ids[0])
                start = first.start_s if first else None
            lbl = label_for(row.video_id, start) if row else "fonte"
            blocks.append(f"- {u.text}  [{lbl}]")
            sources.append(Source(label=lbl, kind="unit", ref=str(u.id)))

    # --- Chunks ------------------------------------------------------------
    if chunks:
        blocks.append("=== TRASCRIZIONI ===")
        for c in chunks:
            lbl = label_for(c.video_id, c.start_s)
            blocks.append(f'- "{c.text}"  [{lbl}]')
            sources.append(Source(label=lbl, kind="chunk", ref=str(c.id)))

    return RetrievedContext(context_md="\n\n".join(blocks), sources=sources)
