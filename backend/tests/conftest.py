"""Shared test fixtures.

DB-backed tests run only when TEST_DATABASE_URL points at a Postgres with the
migrations already applied (CI applies them before pytest; locally use
`alembic upgrade head`). Each test runs in a transaction rolled back at teardown,
with `create_savepoint` so even code that commits stays isolated.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

requires_db = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(TEST_DATABASE_URL, poolclass=None)
    conn = await engine.connect()
    trans = await conn.begin()
    maker = async_sessionmaker(
        bind=conn,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    db = maker()
    try:
        yield db
    finally:
        await db.close()
        if trans.is_active:
            await trans.rollback()
        await conn.close()
        await engine.dispose()
