"""Health endpoint test with the DB session overridden (no live Postgres)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from ytkb.api.app import create_app
from ytkb.db.base import get_session


class _FakeResult:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    async def execute(self, statement, *args, **kwargs):
        compiled = str(statement).lower()
        if "pg_extension" in compiled:
            return _FakeResult([("vector",), ("pg_trgm",)])
        return _FakeResult([(1,)])


async def _override_session() -> AsyncIterator[_FakeSession]:
    yield _FakeSession()


def test_health_reports_ok_with_extensions():
    app = create_app()
    app.dependency_overrides[get_session] = _override_session
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["db"] == "ok"
    assert body["embedding_dim"] == 1536
    assert body["extensions"] == {"vector": True, "pg_trgm": True}


def test_health_degraded_when_extension_missing():
    app = create_app()

    class _MissingSession(_FakeSession):
        async def execute(self, statement, *args, **kwargs):
            compiled = str(statement).lower()
            if "pg_extension" in compiled:
                return _FakeResult([("vector",)])  # pg_trgm missing
            return _FakeResult([(1,)])

    async def _override() -> AsyncIterator[_MissingSession]:
        yield _MissingSession()

    app.dependency_overrides[get_session] = _override
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["extensions"]["pg_trgm"] is False
