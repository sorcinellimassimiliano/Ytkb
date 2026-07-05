"""Integration tests for the KB browser API (no LLM) against real Postgres.

Runs the full offline pipeline (index → extract → assign → merge) then exercises
the read endpoints the frontend depends on.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from ytkb.api.app import create_app
from ytkb.db.base import get_session
from ytkb.indexing.embeddings import FakeEmbeddingClient
from ytkb.indexing.pipeline import index_video
from ytkb.knowledge.assignment import assign_pending_units
from ytkb.knowledge.extraction import HeuristicExtractor, extract_units
from ytkb.knowledge.merge import merge_dirty_topics

from .conftest import requires_db
from .fakes import seed_transcribed_video

pytestmark = requires_db

EMB = FakeEmbeddingClient(dim=1536)

SEGMENTS = [
    (
        0.0,
        12.0,
        "Il context management significa gestire le informazioni nel contesto. "
        "Come puoi ridurre i token conviene riassumere i messaggi vecchi in un sommario.",
    ),
]


def _client(session) -> httpx.AsyncClient:
    app = create_app()

    async def _override() -> AsyncIterator:
        yield session

    app.dependency_overrides[get_session] = _override
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _run_pipeline(session, yt_video_id="kb1"):
    video = await seed_transcribed_video(session, yt_video_id=yt_video_id, segments=SEGMENTS)
    await index_video(session, video, embedder=EMB)
    await extract_units(session, video, extractor=HeuristicExtractor(), embedder=EMB)
    await assign_pending_units(session)
    await merge_dirty_topics(session)
    return video


async def test_tree_and_topic_article(session):
    await _run_pipeline(session)
    async with _client(session) as client:
        tree = (await client.get("/kb/tree")).json()
        assert len(tree) >= 1
        slug = tree[0]["slug"]

        topic = (await client.get(f"/kb/topics/{slug}")).json()
        assert topic["article"] is not None
        assert topic["article"]["version"] >= 1
        assert "[unit:" in topic["article"]["content_md"]

        versions = (await client.get(f"/kb/topics/{slug}/versions")).json()
        assert versions[0]["version"] == topic["article"]["version"]

        v1 = (await client.get(f"/kb/topics/{slug}/versions/1")).json()
        assert v1["version"] == 1


async def test_grouped_search(session):
    await _run_pipeline(session)
    async with _client(session) as client:
        body = (await client.get("/kb/search", params={"q": "context management"})).json()
    assert set(body["results"].keys()) == {"topics", "units", "chunks"}
    assert body["results"]["chunks"], "expected chunk hits for a term in the transcript"


async def test_unit_provenance_and_video_page(session):
    video = await _run_pipeline(session)
    async with _client(session) as client:
        sources = (await client.get("/kb/sources")).json()
        assert any(s["videos_total"] >= 1 for s in sources)

        page = (await client.get(f"/kb/videos/{video.yt_video_id}")).json()
        assert page["segments"], "video page shows transcript segments"
        assert page["units"], "video page shows extracted units"

        unit_id = page["units"][0]["id"]
        unit = (await client.get(f"/kb/units/{unit_id}")).json()
        assert unit["video"]["yt_video_id"] == video.yt_video_id
        assert unit["chunks"], "citation resolves to chunk timestamps"


async def test_missing_topic_returns_404(session):
    async with _client(session) as client:
        assert (await client.get("/kb/topics/does-not-exist")).status_code == 404
        assert (await client.get("/kb/units/999999")).status_code == 404
