"""Tests for state update functions (spec/state-vector.md §5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.state.constants import ALPHA, NUM_DIMS
from src.state.types import ObservationVector, StateVector
from src.state.update import apply_observation, compute_delta, ema_update, initialize_state


def _make_obs(values: list[float], mask: list[bool] | None = None) -> ObservationVector:
    v = np.array(values, dtype=np.float64)
    if mask is None:
        m = ~np.isnan(v)
        v = np.where(np.isnan(v), 0.0, v)
    else:
        m = np.array(mask, dtype=bool)
    return ObservationVector(values=v, mask=m)


def _full_obs(value: float = 0.6) -> ObservationVector:
    return _make_obs([value] * NUM_DIMS, [True] * NUM_DIMS)


class TestInitializeState:
    def test_first_observation_applied(self):
        obs = _full_obs(0.8)
        state = initialize_state(obs)
        np.testing.assert_allclose(state.values, 0.8)
        assert state.version == 1

    def test_unobserved_gets_neutral_prior(self):
        values = [0.7] * NUM_DIMS
        mask = [True] * NUM_DIMS
        mask[5] = False  # dim 5 unobserved
        mask[11] = False  # dim 11 unobserved
        obs = _make_obs(values, mask)
        state = initialize_state(obs)
        assert state.values[5] == 0.5  # neutral prior
        assert state.values[11] == 0.5
        assert state.values[0] == 0.7  # observed

    def test_bounds_valid(self):
        obs = _full_obs(0.3)
        state = initialize_state(obs)
        assert state.validate_bounds()

    def test_obs_counts_initialized(self):
        mask = [True] * NUM_DIMS
        mask[3] = False
        obs = _make_obs([0.5] * NUM_DIMS, mask)
        state = initialize_state(obs)
        assert state.obs_counts[0] == 1
        assert state.obs_counts[3] == 0


class TestEmaUpdate:
    def test_basic_ema(self):
        state = StateVector(
            values=np.full(NUM_DIMS, 0.5, dtype=np.float64),
            confidence=np.zeros(NUM_DIMS, dtype=np.float64),
            obs_counts=np.ones(NUM_DIMS, dtype=np.int64),
            version=1,
        )
        obs = _full_obs(1.0)
        new_state = ema_update(state, obs)
        # S = 0.7 * 0.5 + 0.3 * 1.0 = 0.65
        np.testing.assert_allclose(new_state.values, 0.65)
        assert new_state.version == 2

    def test_unobserved_dims_unchanged(self):
        state = StateVector(
            values=np.full(NUM_DIMS, 0.5, dtype=np.float64),
            confidence=np.zeros(NUM_DIMS, dtype=np.float64),
            obs_counts=np.ones(NUM_DIMS, dtype=np.int64),
            version=1,
        )
        mask = [True] * NUM_DIMS
        mask[5] = False
        mask[11] = False
        obs = _make_obs([0.8] * NUM_DIMS, mask)
        new_state = ema_update(state, obs)
        assert new_state.values[5] == 0.5  # unchanged
        assert new_state.values[11] == 0.5  # unchanged
        assert new_state.values[0] != 0.5  # changed

    def test_bounds_preserved(self):
        """EMA of values in [0,1] stays in [0,1]."""
        state = StateVector(
            values=np.full(NUM_DIMS, 1.0, dtype=np.float64),
            confidence=np.zeros(NUM_DIMS, dtype=np.float64),
            obs_counts=np.ones(NUM_DIMS, dtype=np.int64),
            version=1,
        )
        obs = _full_obs(1.0)
        new_state = ema_update(state, obs)
        assert new_state.validate_bounds()

    def test_bounds_at_zero(self):
        state = StateVector(
            values=np.zeros(NUM_DIMS, dtype=np.float64),
            confidence=np.zeros(NUM_DIMS, dtype=np.float64),
            obs_counts=np.ones(NUM_DIMS, dtype=np.int64),
            version=1,
        )
        obs = _full_obs(0.0)
        new_state = ema_update(state, obs)
        assert new_state.validate_bounds()
        np.testing.assert_allclose(new_state.values, 0.0)

    def test_version_increments(self):
        state = StateVector(
            values=np.full(NUM_DIMS, 0.5, dtype=np.float64),
            confidence=np.zeros(NUM_DIMS, dtype=np.float64),
            obs_counts=np.ones(NUM_DIMS, dtype=np.int64),
            version=5,
        )
        obs = _full_obs(0.6)
        new_state = ema_update(state, obs)
        assert new_state.version == 6

    def test_three_consecutive_updates(self):
        """State converges toward observation over repeated updates."""
        obs = _full_obs(0.9)
        state = initialize_state(obs)  # S_0 = 0.9
        for _ in range(10):
            state = ema_update(state, _full_obs(0.9))
        # After many updates with same obs, should converge to 0.9
        np.testing.assert_allclose(state.values, 0.9, atol=0.01)

    def test_obs_counts_increment(self):
        state = StateVector(
            values=np.full(NUM_DIMS, 0.5, dtype=np.float64),
            confidence=np.zeros(NUM_DIMS, dtype=np.float64),
            obs_counts=np.array([3] * NUM_DIMS, dtype=np.int64),
            version=3,
        )
        mask = [False] * NUM_DIMS
        mask[0] = True
        mask[1] = True
        obs = _make_obs([0.6] * NUM_DIMS, mask)
        new_state = ema_update(state, obs)
        assert new_state.obs_counts[0] == 4
        assert new_state.obs_counts[1] == 4
        assert new_state.obs_counts[2] == 3  # unobserved, unchanged


class TestComputeDelta:
    def test_basic_delta(self):
        s_old = StateVector(
            values=np.full(NUM_DIMS, 0.5, dtype=np.float64),
            confidence=np.zeros(NUM_DIMS, dtype=np.float64),
            obs_counts=np.zeros(NUM_DIMS, dtype=np.int64),
            version=0,
        )
        s_new = StateVector(
            values=np.full(NUM_DIMS, 0.7, dtype=np.float64),
            confidence=np.zeros(NUM_DIMS, dtype=np.float64),
            obs_counts=np.ones(NUM_DIMS, dtype=np.int64),
            version=1,
        )
        delta = compute_delta(s_new, s_old)
        np.testing.assert_allclose(delta, 0.2)

    def test_negative_delta(self):
        s_old = StateVector(
            values=np.full(NUM_DIMS, 0.8, dtype=np.float64),
            confidence=np.zeros(NUM_DIMS, dtype=np.float64),
            obs_counts=np.zeros(NUM_DIMS, dtype=np.int64),
            version=0,
        )
        s_new = StateVector(
            values=np.full(NUM_DIMS, 0.6, dtype=np.float64),
            confidence=np.zeros(NUM_DIMS, dtype=np.float64),
            obs_counts=np.ones(NUM_DIMS, dtype=np.int64),
            version=1,
        )
        delta = compute_delta(s_new, s_old)
        np.testing.assert_allclose(delta, -0.2)


class TestApplyObservation:
    def test_first_session_initializes(self):
        obs = _full_obs(0.6)
        state = apply_observation(None, obs)
        np.testing.assert_allclose(state.values, 0.6)

    def test_subsequent_session_updates(self):
        obs1 = _full_obs(0.6)
        state = apply_observation(None, obs1)
        obs2 = _full_obs(0.8)
        state = apply_observation(state, obs2)
        # S = 0.7 * 0.6 + 0.3 * 0.8 = 0.66
        np.testing.assert_allclose(state.values, 0.66)

    def test_empty_observation_raises(self):
        obs = ObservationVector(
            values=np.full(NUM_DIMS, np.nan, dtype=np.float64),
            mask=np.zeros(NUM_DIMS, dtype=bool),
        )
        with pytest.raises(ValueError, match="empty observation"):
            apply_observation(None, obs)
