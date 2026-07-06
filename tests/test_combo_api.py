"""Tests for Combo API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from atom.api.app import app
from atom.models.base import Base, async_session as _orig_session
from atom.seed import seed_all
import atom.models.base as base_module


@pytest.fixture(autouse=True)
async def setup_db(monkeypatch):
    """Override async_session to use in-memory DB for tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Seed data
    async with factory() as s:
        await seed_all(s)

    # Monkeypatch the session factory used by the routers
    monkeypatch.setattr(base_module, "async_session", factory)
    monkeypatch.setattr("atom.api.routers.combos.async_session", factory)

    yield
    await engine.dispose()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_list_combos(client: AsyncClient):
    resp = await client.get("/api/combos")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 12


async def test_list_combos_filter_complexity(client: AsyncClient):
    resp = await client.get("/api/combos", params={"complexity": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert all(c["complexity"] == 2 for c in data)


async def test_create_combo(client: AsyncClient):
    resp = await client.post("/api/combos", json={
        "display_name": "API테스트",
        "actions": ["jab", "cross"],
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["display_name"] == "API테스트"
    assert data["complexity"] == 2
    assert data["is_system"] is False


async def test_create_combo_invalid_action(client: AsyncClient):
    resp = await client.post("/api/combos", json={
        "display_name": "잘못된",
        "actions": ["jab", "fake_punch"],
    })
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"]


async def test_get_combo(client: AsyncClient):
    # Create first
    create_resp = await client.post("/api/combos", json={
        "display_name": "조회테스트",
        "actions": ["jab"],
    })
    combo_id = create_resp.json()["id"]

    resp = await client.get(f"/api/combos/{combo_id}")
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "조회테스트"


async def test_get_combo_not_found(client: AsyncClient):
    resp = await client.get("/api/combos/nonexistent-id")
    assert resp.status_code == 404


async def test_update_combo(client: AsyncClient):
    create_resp = await client.post("/api/combos", json={
        "display_name": "수정전",
        "actions": ["jab"],
    })
    combo_id = create_resp.json()["id"]

    resp = await client.put(f"/api/combos/{combo_id}", json={
        "display_name": "수정후",
        "actions": ["jab", "cross"],
    })
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "수정후"
    assert resp.json()["complexity"] == 2


async def test_update_system_combo_fails(client: AsyncClient):
    # Get a system combo
    list_resp = await client.get("/api/combos")
    system_combo = next(c for c in list_resp.json() if c["is_system"])

    resp = await client.put(f"/api/combos/{system_combo['id']}", json={
        "display_name": "변경시도",
    })
    assert resp.status_code == 403


async def test_delete_combo(client: AsyncClient):
    create_resp = await client.post("/api/combos", json={
        "display_name": "삭제대상",
        "actions": ["jab"],
    })
    combo_id = create_resp.json()["id"]

    resp = await client.delete(f"/api/combos/{combo_id}")
    assert resp.status_code == 204

    # Verify deleted
    resp = await client.get(f"/api/combos/{combo_id}")
    assert resp.status_code == 404


async def test_delete_system_combo_fails(client: AsyncClient):
    list_resp = await client.get("/api/combos")
    system_combo = next(c for c in list_resp.json() if c["is_system"])

    resp = await client.delete(f"/api/combos/{system_combo['id']}")
    assert resp.status_code == 403
