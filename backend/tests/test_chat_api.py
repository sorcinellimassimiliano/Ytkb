"""Chat API tests: mounted only when enabled, SSE streaming when it is."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from ytkb.api.app import create_app
from ytkb.config import get_settings
from ytkb.db.base import get_session

from .conftest import requires_db

pytestmark = requires_db


def _client(session, app) -> httpx.AsyncClient:
    async def _override() -> AsyncIterator:
        yield session

    app.dependency_overrides[get_session] = _override
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_chat_route_absent_when_disabled(session):
    get_settings.cache_clear()
    app = create_app()  # CHAT_ENABLED defaults to false
    try:
        async with _client(session, app) as client:
            resp = await client.post("/chat", json={"message": "ciao"})
        assert resp.status_code == 404
    finally:
        get_settings.cache_clear()


@pytest.fixture
def chat_enabled(monkeypatch):
    monkeypatch.setenv("CHAT_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_chat_streams_sse_when_enabled(session, chat_enabled):
    app = create_app()
    async with _client(session, app) as client:
        resp = await client.post("/chat", json={"message": "cos'è il context management?"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text

    assert "data:" in body
    assert '"type": "session"' in body
    assert '"type": "done"' in body


async def test_session_lifecycle(session, chat_enabled):
    app = create_app()
    async with _client(session, app) as client:
        created = (await client.post("/chat/sessions")).json()
        sid = created["id"]
        await client.post("/chat", json={"message": "domanda", "session_id": sid})
        messages = (await client.get(f"/chat/sessions/{sid}/messages")).json()
    roles = [m["role"] for m in messages]
    assert "user" in roles and "assistant" in roles
