"""Tests for Session API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from atom.api.app import app
from atom.models.base import Base
from atom.seed import seed_all
import atom.models.base as base_module


@pytest.fixture(autouse=True)
async def setup_db(monkeypatch):
    """Override async_session to use in-memory DB for tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as s:
        await seed_all(s)

    monkeypatch.setattr(base_module, "async_session", factory)
    monkeypatch.setattr("atom.api.routers.sessions.async_session", factory)

    yield
    await engine.dispose()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_generate_plan_fundamentals(client: AsyncClient):
    resp = await client.post("/api/sessions/plan", json={"template": "fundamentals"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["llm_model"] == "fallback"
    assert data["plan"]["template"] == "fundamentals"
    assert len(data["plan"]["rounds"]) == 3
    assert "id" in data


async def test_generate_plan_combos(client: AsyncClient):
    resp = await client.post("/api/sessions/plan", json={"template": "combos"})
    assert resp.status_code == 201
    data = resp.json()
    assert len(data["plan"]["rounds"]) == 4


async def test_generate_plan_with_prompt(client: AsyncClient):
    resp = await client.post("/api/sessions/plan", json={
        "template": "fundamentals",
        "prompt": "잽 위주로",
    })
    assert resp.status_code == 201
    assert resp.json()["plan"]["rounds"]


async def test_generate_plan_invalid_template(client: AsyncClient):
    resp = await client.post("/api/sessions/plan", json={"template": "nonexistent"})
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


async def test_generate_plan_empty_template(client: AsyncClient):
    resp = await client.post("/api/sessions/plan", json={"template": ""})
    assert resp.status_code == 422  # Pydantic validation


async def test_plan_response_structure(client: AsyncClient):
    resp = await client.post("/api/sessions/plan", json={"template": "mixed"})
    assert resp.status_code == 201
    data = resp.json()

    # Top-level fields
    assert "id" in data
    assert "llm_model" in data
    assert "plan" in data

    plan = data["plan"]
    assert "session_type" in plan
    assert "template" in plan
    assert "focus" in plan
    assert "total_duration_minutes" in plan
    assert "rounds" in plan

    # Round structure
    rnd = plan["rounds"][0]
    assert "round_number" in rnd
    assert "duration_seconds" in rnd
    assert "rest_after_seconds" in rnd
    assert "instructions" in rnd

    # Instruction structure
    instr = rnd["instructions"][0]
    assert "timestamp_offset" in instr
    assert "combo_display_name" in instr
    assert "actions" in instr
