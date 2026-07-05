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

from ytkb.db.models import Chunk

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
