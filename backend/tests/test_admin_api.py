"""Admin API tests against a real DB via dependency override.

Uses httpx.ASGITransport (not TestClient) so the app runs on the same event
loop as the async DB session fixture — asyncpg connections are loop-bound.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from ytkb.api.app import create_app
from ytkb.db.base import get_session

from .conftest import requires_db

pytestmark = requires_db


def _make_client(session) -> httpx.AsyncClient:
    app = create_app()

    async def _override() -> AsyncIterator:
        yield session

    app.dependency_overrides[get_session] = _override
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_upsert_and_list_channel(session):
    async with _make_client(session) as client:
        resp = await client.post(
            "/admin/channels", json={"yt_channel_id": "UCapi1", "title": "API Channel"}
        )
        assert resp.status_code == 201
        assert resp.json()["yt_channel_id"] == "UCapi1"

        resp2 = await client.post(
            "/admin/channels",
            json={"yt_channel_id": "UCapi1", "title": "Renamed", "active": False},
        )
        assert resp2.status_code == 201
        assert resp2.json()["title"] == "Renamed"
        assert resp2.json()["active"] is False

        listed = (await client.get("/admin/channels")).json()
        matches = [c for c in listed if c["yt_channel_id"] == "UCapi1"]
        assert len(matches) == 1


async def test_admin_stats_shape(session):
    async with _make_client(session) as client:
        body = (await client.get("/admin/stats")).json()
    assert "videos_by_status" in body
    assert "pending" in body["videos_by_status"]
