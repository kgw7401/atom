"""Tests for /api/v1/users endpoints."""

from __future__ import annotations

import json

import pytest

from server.models.db import UserState
from src.state.constants import NUM_DIMS
from tests.api.conftest import zeros_int_json, zeros_json


# ---------------------------------------------------------------------------
# POST /api/v1/users
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_user(client):
    resp = await client.post("/api/v1/users", json={"device_id": "device-abc"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["device_id"] == "device-abc"
    assert "user_id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_create_user_duplicate_device_returns_existing(client):
    r1 = await client.post("/api/v1/users", json={"device_id": "dup-device"})
    r2 = await client.post("/api/v1/users", json={"device_id": "dup-device"})
    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json()["user_id"] == r2.json()["user_id"]


@pytest.mark.asyncio
async def test_create_user_empty_device_id(client):
    resp = await client.post("/api/v1/users", json={"device_id": ""})
    assert resp.status_code == 422  # validation error


# ---------------------------------------------------------------------------
# GET /api/v1/users/{user_id}/state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_state(client):
    # Create user first
    r = await client.post("/api/v1/users", json={"device_id": "state-device"})
    user_id = r.json()["user_id"]

    resp = await client.get(f"/api/v1/users/{user_id}/state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == user_id
    assert len(data["values"]) == NUM_DIMS
    assert len(data["confidence"]) == NUM_DIMS
    assert len(data["obs_counts"]) == NUM_DIMS
    assert data["version"] == 0
    assert len(data["dim_names"]) == NUM_DIMS


@pytest.mark.asyncio
async def test_get_state_not_found(client):
    resp = await client.get("/api/v1/users/nonexistent-id/state")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/users/{user_id}/history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_history_empty(client):
    r = await client.post("/api/v1/users", json={"device_id": "hist-device"})
    user_id = r.json()["user_id"]

    resp = await client.get(f"/api/v1/users/{user_id}/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == user_id
    assert data["transitions"] == []


@pytest.mark.asyncio
async def test_get_history_not_found(client):
    resp = await client.get("/api/v1/users/nonexistent-id/history")
    assert resp.status_code == 404
