"""Tests for session history and profile API endpoints."""

from unittest.mock import patch, AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from atom.api.app import app
from atom.models.base import Base
from atom.seed import seed_all
from atom.services.profile_service import ProfileService
from atom.services.session_engine import SessionEngine
from atom.services.session_service import SessionService
import atom.models.base as base_module
import atom.api.routers.history as history_module


@pytest.fixture(autouse=True)
async def setup_db(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        await seed_all(s)

    monkeypatch.setattr(base_module, "async_session", factory)
    monkeypatch.setattr(history_module, "async_session", factory)
    monkeypatch.setattr("atom.api.routers.sessions.async_session", factory)

    # Store factory for helper use
    setup_db.factory = factory
    yield
    await engine.dispose()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _make_session_log():
    """Create a completed session log via service layer."""
    factory = setup_db.factory
    async with factory() as db:
        plan_svc = SessionService(db)
        result = await plan_svc.generate_plan("fundamentals")
        engine = SessionEngine(plan=result["plan"], plan_id=result["id"], tts_enabled=False)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await engine.run()
        return await engine.save_log(db)


# ── Templates ─────────────────────────────────────────────────────────

async def test_list_templates(client: AsyncClient):
    resp = await client.get("/api/templates")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    names = {t["name"] for t in data}
    assert names == {"fundamentals", "combos", "mixed"}


async def test_template_fields(client: AsyncClient):
    resp = await client.get("/api/templates")
    t = next(t for t in resp.json() if t["name"] == "fundamentals")
    assert t["default_rounds"] == 3
    assert t["default_round_duration_sec"] == 120
    assert t["combo_include_defense"] is False


# ── Session history ───────────────────────────────────────────────────

async def test_list_sessions_empty(client: AsyncClient):
    resp = await client.get("/api/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_sessions_after_run(client: AsyncClient):
    await _make_session_log()
    await _make_session_log()

    resp = await client.get("/api/sessions")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_list_sessions_limit(client: AsyncClient):
    for _ in range(5):
        await _make_session_log()

    resp = await client.get("/api/sessions", params={"limit": 3})
    assert resp.status_code == 200
    assert len(resp.json()) == 3


async def test_get_session_by_id(client: AsyncClient):
    log = await _make_session_log()

    resp = await client.get(f"/api/sessions/{log.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == log.id
    assert data["template_name"] == "fundamentals"
    assert data["status"] == "completed"
    assert "delivery_log_json" in data
    assert data["delivery_log_json"]["events"]


async def test_get_session_not_found(client: AsyncClient):
    resp = await client.get("/api/sessions/nonexistent-id")
    assert resp.status_code == 404


# ── Profile ───────────────────────────────────────────────────────────

async def test_get_profile(client: AsyncClient):
    resp = await client.get("/api/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["experience_level"] == "beginner"
    assert "total_sessions" in data
    assert "combo_exposure_json" in data


async def test_update_profile_experience(client: AsyncClient):
    resp = await client.put("/api/profile", json={"experience_level": "intermediate"})
    assert resp.status_code == 200
    assert resp.json()["experience_level"] == "intermediate"


async def test_update_profile_goal(client: AsyncClient):
    resp = await client.put("/api/profile", json={"goal": "스파링 준비"})
    assert resp.status_code == 200
    assert resp.json()["goal"] == "스파링 준비"


async def test_update_profile_invalid_experience(client: AsyncClient):
    resp = await client.put("/api/profile", json={"experience_level": "expert"})
    assert resp.status_code == 422


async def test_update_profile_empty_body(client: AsyncClient):
    resp = await client.put("/api/profile", json={})
    assert resp.status_code == 400
