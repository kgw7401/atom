"""Tests for server/services/report_generator.py."""

from __future__ import annotations

import json
import uuid

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from server.models.db import Base, Session as SessionModel, StateTransition, UserState
from server.services.report_generator import (
    generate_progress_report,
    generate_session_report,
)
from src.state.constants import DIM_NAMES, EPSILON, NUM_DIMS
from src.state.types import StateVector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_state(values: list, confidence: list = None, obs_counts: list = None) -> StateVector:
    """Create a StateVector from lists."""
    v = np.array(values, dtype=np.float64)
    c = np.array(confidence or [0.8] * NUM_DIMS, dtype=np.float64)
    n = np.array(obs_counts or [10] * NUM_DIMS, dtype=np.float64)
    return StateVector(values=v, confidence=c, obs_counts=n, version=1, schema_version="v1")


def _all_observed_mask():
    return np.ones(NUM_DIMS, dtype=bool)


def _partial_mask(observed_indices: list):
    mask = np.zeros(NUM_DIMS, dtype=bool)
    for i in observed_indices:
        mask[i] = True
    return mask


# ---------------------------------------------------------------------------
# Session Report Tests
# ---------------------------------------------------------------------------

class TestSessionReportAllImproved:
    def test_all_improved(self):
        """All observed dims improved → 'all improved' summary."""
        state = _make_state([0.7] * NUM_DIMS)
        delta = np.full(NUM_DIMS, 0.05)  # all positive
        mask = _all_observed_mask()

        report = generate_session_report("s1", delta, mask, state)

        assert "좋은 세션" in report.coaching["summary"]
        assert len(report.state_delta["improved"]) > 0
        assert len(report.state_delta["regressed"]) == 0


class TestSessionReportMixed:
    def test_mixed(self):
        """Some improved, some regressed → mixed summary."""
        state = _make_state([0.5] * NUM_DIMS)
        delta = np.zeros(NUM_DIMS)
        delta[0] = 0.1   # improved
        delta[1] = 0.08  # improved
        delta[4] = -0.05  # regressed
        mask = _all_observed_mask()

        report = generate_session_report("s2", delta, mask, state)

        assert "향상" in report.coaching["summary"]
        assert "하락" in report.coaching["summary"]
        assert len(report.state_delta["improved"]) >= 2
        assert len(report.state_delta["regressed"]) >= 1


class TestSessionReportAllRegressed:
    def test_all_regressed(self):
        """All dims regressed → rest recommendation."""
        state = _make_state([0.3] * NUM_DIMS)
        delta = np.full(NUM_DIMS, -0.05)
        mask = _all_observed_mask()

        report = generate_session_report("s3", delta, mask, state)

        assert "휴식" in report.coaching["summary"]
        assert len(report.state_delta["regressed"]) > 0
        assert len(report.state_delta["improved"]) == 0


class TestSessionReportWithWeaknesses:
    def test_weaknesses_generate_coaching(self):
        """Low state with high confidence → weaknesses detected with coaching."""
        # All values at 0.2 (well below thresholds), high confidence
        state = _make_state([0.2] * NUM_DIMS)
        delta = np.zeros(NUM_DIMS)
        mask = _all_observed_mask()

        report = generate_session_report("s4", delta, mask, state)

        assert len(report.weaknesses) > 0
        for w in report.weaknesses:
            assert "coaching_hint" in w
            assert len(w["coaching_hint"]) > 0


class TestSessionReportNoWeaknesses:
    def test_no_weaknesses(self):
        """High state → no weaknesses, 'balance good' hint."""
        state = _make_state([0.9] * NUM_DIMS)
        delta = np.zeros(NUM_DIMS)
        mask = _all_observed_mask()

        report = generate_session_report("s5", delta, mask, state)

        assert len(report.weaknesses) == 0
        assert "밸런스" in report.coaching["next_session_hint"]


class TestSessionReportPartialObservation:
    def test_partial_observation(self):
        """Only some dims observed → only those in report."""
        state = _make_state([0.5] * NUM_DIMS)
        delta = np.zeros(NUM_DIMS)
        delta[0] = 0.1
        delta[4] = -0.05
        # Only observe dims 0 and 4
        mask = _partial_mask([0, 4])

        report = generate_session_report("s6", delta, mask, state)

        total_classified = (
            len(report.state_delta["improved"])
            + len(report.state_delta["regressed"])
            + len(report.state_delta["unchanged"])
        )
        assert total_classified == 2


class TestDeltaClassification:
    def test_epsilon_threshold(self):
        """Deltas within epsilon → unchanged."""
        state = _make_state([0.5] * NUM_DIMS)
        delta = np.full(NUM_DIMS, EPSILON * 0.5)  # below threshold
        mask = _all_observed_mask()

        report = generate_session_report("s7", delta, mask, state)

        assert len(report.state_delta["improved"]) == 0
        assert len(report.state_delta["regressed"]) == 0
        assert len(report.state_delta["unchanged"]) == NUM_DIMS


class TestKoreanText:
    def test_all_text_is_korean(self):
        """All coaching text should contain Korean characters."""
        state = _make_state([0.3] * NUM_DIMS)
        delta = np.full(NUM_DIMS, 0.05)
        mask = _all_observed_mask()

        report = generate_session_report("s8", delta, mask, state)

        # Check summary has Korean
        assert any('\uac00' <= c <= '\ud7a3' for c in report.coaching["summary"])
        # Check next session hint has Korean
        assert any('\uac00' <= c <= '\ud7a3' for c in report.coaching["next_session_hint"])
        # Check improved descriptions have Korean
        for item in report.state_delta["improved"]:
            assert any('\uac00' <= c <= '\ud7a3' for c in item["description"])


class TestNextSessionHint:
    def test_hint_with_weakness(self):
        """With weaknesses → hint mentions top weakness."""
        state = _make_state([0.2] * NUM_DIMS)
        delta = np.zeros(NUM_DIMS)
        mask = _all_observed_mask()

        report = generate_session_report("s9", delta, mask, state)

        assert "집중" in report.coaching["next_session_hint"]
        assert "추천" in report.coaching["next_session_hint"]


# ---------------------------------------------------------------------------
# Progress Report Tests
# ---------------------------------------------------------------------------

class TestProgressTrendingUp:
    def test_consistent_improvement(self):
        """5 sessions with same dim improving → trending up."""
        state = _make_state([0.7] * NUM_DIMS)
        mask = _all_observed_mask()

        # 5 sessions: dim 0 always improves
        deltas = []
        masks = []
        for _ in range(5):
            d = np.zeros(NUM_DIMS)
            d[0] = 0.05  # always positive
            deltas.append(d)
            masks.append(mask)

        report = generate_progress_report("u1", deltas, masks, state)

        assert report.num_sessions == 5
        up_names = [x["dim_name"] for x in report.trending_up]
        assert "repertoire_entropy" in up_names


class TestProgressPlateau:
    def test_no_change(self):
        """5 sessions with tiny deltas → plateau."""
        state = _make_state([0.5] * NUM_DIMS)
        mask = _all_observed_mask()

        deltas = [np.full(NUM_DIMS, 0.001) for _ in range(5)]
        masks = [mask for _ in range(5)]

        report = generate_progress_report("u2", deltas, masks, state)

        # All dims should be plateau (no clear trend)
        assert len(report.plateau) > 0


class TestDeterminism:
    def test_same_input_same_output(self):
        """Same inputs → identical outputs."""
        state = _make_state([0.4] * NUM_DIMS)
        delta = np.array([0.05 if i % 2 == 0 else -0.03 for i in range(NUM_DIMS)])
        mask = _all_observed_mask()

        r1 = generate_session_report("det1", delta, mask, state)
        r2 = generate_session_report("det1", delta, mask, state)

        assert r1.coaching == r2.coaching
        assert r1.state_delta == r2.state_delta
        assert r1.weaknesses == r2.weaknesses


# ---------------------------------------------------------------------------
# API Integration Tests (coaching endpoint)
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


@pytest_asyncio.fixture
async def client(test_engine, test_session_factory, monkeypatch):
    import server.models.db as db_mod
    import server.routers.users as users_mod
    import server.routers.sessions as sessions_mod
    import server.routers.training as training_mod
    import server.services.analysis_pipeline as pipeline_mod

    for mod in (db_mod, users_mod, sessions_mod, training_mod, pipeline_mod):
        monkeypatch.setattr(mod, "async_session", test_session_factory)
    monkeypatch.setattr(db_mod, "engine", test_engine)

    from server.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_coaching_endpoint(client, test_session_factory):
    """GET /sessions/{id}/coaching returns enriched report."""
    # Create user
    r = await client.post("/api/v1/users", json={"device_id": "coach-dev"})
    user_id = r.json()["user_id"]

    # Manually create completed session + transition
    session_id = str(uuid.uuid4())
    async with test_session_factory() as db:
        s = SessionModel(
            id=session_id, user_id=user_id, mode="shadow",
            status="COMPLETED", state_update_applied=True,
        )
        db.add(s)

        delta = [0.05 if i < 9 else -0.03 for i in range(NUM_DIMS)]
        mask = [True] * NUM_DIMS
        t = StateTransition(
            user_id=user_id, session_id=session_id,
            version_before=0, version_after=1,
            vector_before_json=json.dumps([0.0] * NUM_DIMS),
            vector_after_json=json.dumps([0.5] * NUM_DIMS),
            observation_json=json.dumps([0.5] * NUM_DIMS),
            observation_mask_json=json.dumps(mask),
            delta_json=json.dumps(delta),
        )
        db.add(t)
        await db.commit()

    resp = await client.get(f"/api/v1/sessions/{session_id}/coaching")
    assert resp.status_code == 200
    data = resp.json()
    assert "coaching" in data
    assert "summary" in data["coaching"]
    assert "focus_areas" in data["coaching"]
    assert "next_session_hint" in data["coaching"]
    assert "state_delta" in data
    assert "raw" in data


@pytest.mark.asyncio
async def test_progress_endpoint(client, test_session_factory):
    """GET /users/{id}/progress returns progress report."""
    r = await client.post("/api/v1/users", json={"device_id": "progress-dev"})
    user_id = r.json()["user_id"]

    resp = await client.get(f"/api/v1/users/{user_id}/progress")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == user_id
    assert data["num_sessions"] == 0
    assert "coaching" in data
