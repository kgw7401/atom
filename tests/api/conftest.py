"""Shared fixtures for API tests.

Uses in-memory SQLite with StaticPool so all async tasks share one connection.
Overrides server.models.db engine/session before the FastAPI app is used.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from server.models.db import Base
from src.state.constants import NUM_DIMS


# ---------------------------------------------------------------------------
# Test DB engine (in-memory SQLite, shared single connection)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def test_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def test_session_factory(test_engine):
    return async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Patch DB + provide AsyncClient
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client(test_engine, test_session_factory, monkeypatch):
    """AsyncClient with test DB patched into all modules that import async_session."""
    import server.models.db as db_mod
    import server.routers.users as users_mod
    import server.routers.sessions as sessions_mod
    import server.routers.training as training_mod
    import server.services.analysis_pipeline as pipeline_mod

    # Patch everywhere async_session was imported
    for mod in (db_mod, users_mod, sessions_mod, training_mod, pipeline_mod):
        monkeypatch.setattr(mod, "async_session", test_session_factory)
    monkeypatch.setattr(db_mod, "engine", test_engine)

    from server.main import app

    # Skip lifespan (DB already initialized above)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zeros_json(n: int = NUM_DIMS) -> str:
    return json.dumps([0.0] * n)


def zeros_int_json(n: int = NUM_DIMS) -> str:
    return json.dumps([0] * n)
