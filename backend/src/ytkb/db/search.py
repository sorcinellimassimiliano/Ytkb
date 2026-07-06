"""Search over chunks — all in Postgres, no LLM.

Three modes:
- semantic: kNN on the pgvector embedding (cosine), query embedded once;
- fts: full-text over the generated tsvector (italian + english dictionaries);
- hybrid: Reciprocal Rank Fusion (RRF) of the two rankings.

The query embedding is produced by the embedding client (an embedding model, or
the offline `fake` provider) — never an LLM — so callers on /kb/* stay compliant.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import bindparam, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.db.models import Chunk, KnowledgeUnit, Topic, TopicStatus

RRF_K = 60


@dataclass(slots=True)
class ChunkHit:
    id: int
    video_id: int
    start_s: float
    end_s: float
    text: str
    score: float

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "video_id": self.video_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "text": self.text,
            "score": round(self.score, 6),
        }


async def semantic_chunks(
    session: AsyncSession, query_vec: list[float], *, limit: int = 10
) -> list[ChunkHit]:
    dist = Chunk.embedding.cosine_distance(query_vec).label("dist")
    stmt = select(Chunk, dist).where(Chunk.embedding.is_not(None)).order_by(dist).limit(limit)
    rows = await session.execute(stmt)
    hits: list[ChunkHit] = []
    for chunk, distance in rows:
        hits.append(
            ChunkHit(
                id=chunk.id,
                video_id=chunk.video_id,
                start_s=chunk.start_s,
                end_s=chunk.end_s,
                text=chunk.text,
                score=1.0 - float(distance),  # cosine similarity
            )
        )
    return hits


_FTS_SQL = text(
    """
    WITH q AS (
        SELECT websearch_to_tsquery('italian', :query)
               || websearch_to_tsquery('english', :query) AS tsq
    )
    SELECT c.id, c.video_id, c.start_s, c.end_s, c.text,
           ts_rank(c.tsv, q.tsq) AS rank
    FROM chunks c, q
    WHERE c.tsv @@ q.tsq
    ORDER BY rank DESC
    LIMIT :limit
    """
).bindparams(bindparam("query"), bindparam("limit"))


async def fts_chunks(session: AsyncSession, query: str, *, limit: int = 10) -> list[ChunkHit]:
    rows = await session.execute(_FTS_SQL, {"query": query, "limit": limit})
    return [
        ChunkHit(
            id=r.id,
            video_id=r.video_id,
            start_s=r.start_s,
            end_s=r.end_s,
            text=r.text,
            score=float(r.rank),
        )
        for r in rows
    ]


def _rrf(rankings: list[list[ChunkHit]], *, k: int = RRF_K, limit: int = 10) -> list[ChunkHit]:
    scores: dict[int, float] = {}
    by_id: dict[int, ChunkHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
            by_id.setdefault(hit.id, hit)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    out: list[ChunkHit] = []
    for chunk_id, score in ordered:
        hit = by_id[chunk_id]
        out.append(
            ChunkHit(
                id=hit.id,
                video_id=hit.video_id,
                start_s=hit.start_s,
                end_s=hit.end_s,
                text=hit.text,
                score=score,
            )
        )
    return out


async def hybrid_chunks(
    session: AsyncSession,
    query: str,
    query_vec: list[float],
    *,
    limit: int = 10,
) -> list[ChunkHit]:
    # Pull a wider candidate set from each mode, then fuse.
    pool = max(limit * 3, 20)
    sem = await semantic_chunks(session, query_vec, limit=pool)
    fts = await fts_chunks(session, query, limit=pool)
    return _rrf([sem, fts], limit=limit)


# --------------------------------------------------------------------------
# Units and topics (grouped KB search)
# --------------------------------------------------------------------------
@dataclass(slots=True)
class UnitHit:
    id: int
    text: str
    unit_type: str
    topic_id: int | None
    score: float

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "unit_type": self.unit_type,
            "topic_id": self.topic_id,
            "score": round(self.score, 6),
        }


@dataclass(slots=True)
class TopicHit:
    id: int
    slug: str
    title: str
    score: float

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "slug": self.slug,
            "title": self.title,
            "score": round(self.score, 6),
        }


def _rrf_ids(rankings: list[list[int]], *, k: int = RRF_K, limit: int = 10) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [item_id for item_id, _ in ordered]


_UNIT_FTS_SQL = text(
    """
    WITH q AS (
        SELECT websearch_to_tsquery('italian', :query)
               || websearch_to_tsquery('english', :query) AS tsq
    )
    SELECT u.id
    FROM knowledge_units u, q
    WHERE u.tsv @@ q.tsq
    ORDER BY ts_rank(u.tsv, q.tsq) DESC
    LIMIT :limit
    """
).bindparams(bindparam("query"), bindparam("limit"))


async def hybrid_units(
    session: AsyncSession, query: str, query_vec: list[float], *, limit: int = 10
) -> list[UnitHit]:
    pool = max(limit * 3, 20)
    fts_ids = [r.id for r in await session.execute(_UNIT_FTS_SQL, {"query": query, "limit": pool})]
    dist = KnowledgeUnit.embedding.cosine_distance(query_vec).label("dist")
    sem_ids = [
        r.id
        for r in await session.execute(
            select(KnowledgeUnit.id)
            .where(KnowledgeUnit.embedding.is_not(None))
            .order_by(dist)
            .limit(pool)
        )
    ]
    ordered = _rrf_ids([sem_ids, fts_ids], limit=limit)
    if not ordered:
        return []
    units = {
        u.id: u
        for u in (
            await session.execute(select(KnowledgeUnit).where(KnowledgeUnit.id.in_(ordered)))
        ).scalars()
    }
    return [
        UnitHit(
            id=uid,
            text=units[uid].text,
            unit_type=units[uid].unit_type.value,
            topic_id=units[uid].topic_id,
            score=1.0 / (rank + 1),
        )
        for rank, uid in enumerate(ordered)
        if uid in units
    ]


_TOPIC_FTS_SQL = text(
    """
    WITH q AS (
        SELECT websearch_to_tsquery('italian', :query)
               || websearch_to_tsquery('english', :query) AS tsq
    )
    SELECT t.id
    FROM topics t, q
    WHERE to_tsvector('italian', coalesce(t.title, '') || ' ' || coalesce(t.summary, '')) @@ q.tsq
      AND t.status <> 'merged_into'
    ORDER BY ts_rank(
        to_tsvector('italian', coalesce(t.title, '') || ' ' || coalesce(t.summary, '')), q.tsq
    ) DESC
    LIMIT :limit
    """
).bindparams(bindparam("query"), bindparam("limit"))


async def hybrid_topics(
    session: AsyncSession, query: str, query_vec: list[float], *, limit: int = 10
) -> list[TopicHit]:
    pool = max(limit * 3, 20)
    fts_ids = [r.id for r in await session.execute(_TOPIC_FTS_SQL, {"query": query, "limit": pool})]
    dist = Topic.centroid.cosine_distance(query_vec).label("dist")
    sem_ids = [
        r.id
        for r in await session.execute(
            select(Topic.id)
            .where(Topic.centroid.is_not(None), Topic.status != TopicStatus.merged_into)
            .order_by(dist)
            .limit(pool)
        )
    ]
    ordered = _rrf_ids([sem_ids, fts_ids], limit=limit)
    if not ordered:
        return []
    topics = {
        t.id: t
        for t in (await session.execute(select(Topic).where(Topic.id.in_(ordered)))).scalars()
    }
    return [
        TopicHit(id=tid, slug=topics[tid].slug, title=topics[tid].title, score=1.0 / (rank + 1))
        for rank, tid in enumerate(ordered)
        if tid in topics
    ]
