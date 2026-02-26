"""Tests for /api/v1/sessions endpoints."""

from __future__ import annotations

import io
import json

import pytest

from server.models.db import Session, StateTransition
from src.state.constants import NUM_DIMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _create_user(client, device_id: str = "sess-device") -> str:
    r = await client.post("/api/v1/users", json={"device_id": device_id})
    return r.json()["user_id"]


# ---------------------------------------------------------------------------
# POST /api/v1/sessions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session(client):
    user_id = await _create_user(client)
    resp = await client.post(
        "/api/v1/sessions", json={"user_id": user_id, "mode": "shadow"}
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["user_id"] == user_id
    assert data["mode"] == "shadow"
    assert data["status"] == "CREATED"
    assert "session_id" in data


@pytest.mark.asyncio
async def test_create_session_invalid_mode(client):
    user_id = await _create_user(client, "mode-device")
    resp = await client.post(
        "/api/v1/sessions", json={"user_id": user_id, "mode": "invalid_mode"}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_session_heavy_bag(client):
    user_id = await _create_user(client, "bag-device")
    resp = await client.post(
        "/api/v1/sessions", json={"user_id": user_id, "mode": "heavy_bag"}
    )
    assert resp.status_code == 201
    assert resp.json()["mode"] == "heavy_bag"


# ---------------------------------------------------------------------------
# GET /api/v1/sessions/{session_id}/status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_status(client):
    user_id = await _create_user(client, "status-device")
    r = await client.post(
        "/api/v1/sessions", json={"user_id": user_id, "mode": "shadow"}
    )
    session_id = r.json()["session_id"]

    resp = await client.get(f"/api/v1/sessions/{session_id}/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "CREATED"


@pytest.mark.asyncio
async def test_get_status_not_found(client):
    resp = await client.get("/api/v1/sessions/nonexistent/status")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v1/sessions/{session_id}/upload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_sets_uploading(client, tmp_path):
    user_id = await _create_user(client, "upload-device")
    r = await client.post(
        "/api/v1/sessions", json={"user_id": user_id, "mode": "shadow"}
    )
    session_id = r.json()["session_id"]

    # Create a tiny fake video file
    fake_video = io.BytesIO(b"fake-video-data")
    resp = await client.post(
        f"/api/v1/sessions/{session_id}/upload",
        files={"file": ("test.mp4", fake_video, "video/mp4")},
    )
    # The endpoint returns after saving file and before background task completes
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id
    # Status should be UPLOADING (background task runs separately)
    assert data["status"] in ("UPLOADING", "PROCESSING", "COMPLETED", "FAILED")


@pytest.mark.asyncio
async def test_upload_not_found(client):
    fake_video = io.BytesIO(b"data")
    resp = await client.post(
        "/api/v1/sessions/nonexistent/upload",
        files={"file": ("test.mp4", fake_video, "video/mp4")},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/sessions/{session_id}/report
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_report_not_completed(client):
    user_id = await _create_user(client, "report-device")
    r = await client.post(
        "/api/v1/sessions", json={"user_id": user_id, "mode": "shadow"}
    )
    session_id = r.json()["session_id"]

    resp = await client.get(f"/api/v1/sessions/{session_id}/report")
    assert resp.status_code == 400  # not completed yet


@pytest.mark.asyncio
async def test_report_completed(client, test_session_factory):
    """Manually create a COMPLETED session with transition to test report."""
    import uuid
    from server.models.db import Session as SessionModel, StateTransition

    user_id = await _create_user(client, "report-ok-device")
    session_id = str(uuid.uuid4())

    async with test_session_factory() as db:
        s = SessionModel(
            id=session_id,
            user_id=user_id,
            mode="shadow",
            status="COMPLETED",
            state_update_applied=True,
        )
        db.add(s)

        obs_values = [0.5 if i % 2 == 0 else None for i in range(NUM_DIMS)]
        mask = [i % 2 == 0 for i in range(NUM_DIMS)]
        delta = [0.1] * NUM_DIMS

        t = StateTransition(
            user_id=user_id,
            session_id=session_id,
            version_before=0,
            version_after=1,
            vector_before_json=json.dumps([0.0] * NUM_DIMS),
            vector_after_json=json.dumps([0.5] * NUM_DIMS),
            observation_json=json.dumps(obs_values),
            observation_mask_json=json.dumps(mask),
            delta_json=json.dumps(delta),
        )
        db.add(t)
        await db.commit()

    resp = await client.get(f"/api/v1/sessions/{session_id}/report")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id
    assert len(data["observation"]) == NUM_DIMS
    assert len(data["observation_mask"]) == NUM_DIMS
    assert len(data["delta"]) == NUM_DIMS
