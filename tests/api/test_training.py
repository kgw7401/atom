"""Tests for /api/v1/training endpoints."""

from __future__ import annotations

import json

import pytest

from server.models.db import UserState
from src.state.constants import NUM_DIMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _create_user(client, device_id: str = "train-device") -> str:
    r = await client.post("/api/v1/users", json={"device_id": device_id})
    return r.json()["user_id"]


# ---------------------------------------------------------------------------
# POST /api/v1/training/generate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_plan_zeros(client, test_session_factory):
    """Zero state → all dims weak → targeted plan."""
    user_id = await _create_user(client)

    # Manually bump confidence so detect_weaknesses fires
    async with test_session_factory() as db:
        user = await db.get(UserState, user_id)
        user.confidence_json = json.dumps([0.8] * NUM_DIMS)
        user.obs_counts_json = json.dumps([10] * NUM_DIMS)
        await db.commit()

    resp = await client.post("/api/v1/training/generate", json={"user_id": user_id})
    assert resp.status_code == 201
    data = resp.json()
    assert "plan_id" in data
    assert data["plan"]["plan_type"] in ("targeted", "maintenance")
    assert "rounds" in data["plan"]


@pytest.mark.asyncio
async def test_generate_plan_unknown_user(client):
    resp = await client.post(
        "/api/v1/training/generate", json={"user_id": "nonexistent"}
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_plan_not_found(client):
    resp = await client.get("/api/v1/training/plans/nonexistent-plan-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_generate_then_get_plan(client):
    """Generate a plan, then retrieve it by ID."""
    user_id = await _create_user(client, "plan-get-device")

    resp = await client.post("/api/v1/training/generate", json={"user_id": user_id})
    assert resp.status_code == 201
    plan_id = resp.json()["plan_id"]

    resp2 = await client.get(f"/api/v1/training/plans/{plan_id}")
    assert resp2.status_code == 200
    assert resp2.json()["plan_id"] == plan_id
    assert resp2.json()["plan"] == resp.json()["plan"]
