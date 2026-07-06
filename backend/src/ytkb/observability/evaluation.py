"""Evaluation harness: gold questions → is the expected item in the top-k?

Baseline before tuning thresholds (per plan). A gold item names a query, the
result group to look in (topic / unit / chunk), and a substring the correct
result must contain. Recall@k = fraction of gold items whose expected item is in
the top-k of that group.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from ytkb.config import Settings, get_settings
from ytkb.db import search
from ytkb.indexing.embeddings import EmbeddingClient, get_embedding_client


@dataclass(slots=True)
class GoldItem:
    query: str
    kind: str  # topic | unit | chunk
    contains: str  # substring expected in the matching result (case-insensitive)


@dataclass(slots=True)
class EvalResult:
    total: int
    hits: int
    misses: list[str]

    @property
    def recall(self) -> float:
        return round(self.hits / self.total, 4) if self.total else 0.0


def _texts(kind: str, results) -> list[str]:
    if kind == "topic":
        return [h.title for h in results["topics"]]
    if kind == "unit":
        return [h.text for h in results["units"]]
    return [h.text for h in results["chunks"]]


async def evaluate(
    session: AsyncSession,
    gold: list[GoldItem],
    *,
    k: int = 5,
    embedder: EmbeddingClient | None = None,
    settings: Settings | None = None,
) -> EvalResult:
    settings = settings or get_settings()
    embedder = embedder or get_embedding_client(settings)

    hits = 0
    misses: list[str] = []
    for item in gold:
        qvec = await embedder.embed_one(item.query)
        results = {
            "topics": await search.hybrid_topics(session, item.query, qvec, limit=k),
            "units": await search.hybrid_units(session, item.query, qvec, limit=k),
            "chunks": await search.hybrid_chunks(session, item.query, qvec, limit=k),
        }
        needle = item.contains.lower()
        if any(needle in t.lower() for t in _texts(item.kind, results)):
            hits += 1
        else:
            misses.append(item.query)
    return EvalResult(total=len(gold), hits=hits, misses=misses)


def load_gold(path: str) -> list[GoldItem]:
    import json

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [GoldItem(query=d["query"], kind=d["kind"], contains=d["contains"]) for d in data]
