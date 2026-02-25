"""Tests for state engine invariants (spec/state-vector.md §12)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.state.constants import NUM_DIMS
from src.state.observation import compute_observation
from src.state.types import ObservationVector, StateVector
from src.state.update import apply_observation, ema_update, initialize_state

from .conftest import make_keypoint_frame, make_segment


class TestDeterminism:
    def test_same_input_same_output(self, diverse_segments, diverse_keypoints):
        """Invariant 3: Same RawData → same O_t → same S_{t+1}."""
        duration = diverse_segments[-1].t_end + 1.0
        results = []
        for _ in range(20):
            obs = compute_observation(diverse_segments, diverse_keypoints, duration)
            state = initialize_state(obs)
            results.append(state.values.copy())

        for r in results[1:]:
            np.testing.assert_array_equal(results[0], r)


class TestBounds:
    def test_observation_bounds(self, diverse_segments, diverse_keypoints):
        """Invariant: All observed values in [0, 1]."""
        duration = diverse_segments[-1].t_end + 1.0
        obs = compute_observation(diverse_segments, diverse_keypoints, duration)
        observed = obs.values[obs.mask]
        assert np.all(observed >= 0.0), f"Values below 0: {observed[observed < 0]}"
        assert np.all(observed <= 1.0), f"Values above 1: {observed[observed > 1]}"

    def test_state_bounds_after_updates(self, diverse_segments, diverse_keypoints):
        """Invariant 2: S_t ∈ [0,1]^18 after multiple updates."""
        duration = diverse_segments[-1].t_end + 1.0
        state = None
        for _ in range(10):
            obs = compute_observation(diverse_segments, diverse_keypoints, duration)
            state = apply_observation(state, obs)
            assert state.validate_bounds(), f"Bounds violated: {state.values}"

    def test_random_fuzzing(self):
        """Fuzz test: random observations always produce valid state."""
        rng = np.random.RandomState(42)
        state = None
        for _ in range(50):
            values = rng.rand(NUM_DIMS)
            mask = rng.rand(NUM_DIMS) > 0.3  # ~70% observed
            if not np.any(mask):
                mask[0] = True  # ensure at least one observed
            obs = ObservationVector(values=values, mask=mask)
            state = apply_observation(state, obs)
            assert state.validate_bounds()


class TestFixedDimension:
    def test_dimension_always_18(self, diverse_segments, diverse_keypoints):
        """Invariant 1: d = 18."""
        duration = diverse_segments[-1].t_end + 1.0
        obs = compute_observation(diverse_segments, diverse_keypoints, duration)
        assert obs.values.shape == (NUM_DIMS,)
        state = initialize_state(obs)
        assert state.values.shape == (NUM_DIMS,)
        assert state.confidence.shape == (NUM_DIMS,)
        assert state.obs_counts.shape == (NUM_DIMS,)


class TestRoundTrip:
    def test_json_round_trip(self):
        """Serialize → deserialize → equal (no floating point drift)."""
        rng = np.random.RandomState(123)
        state = StateVector(
            values=rng.rand(NUM_DIMS),
            confidence=rng.rand(NUM_DIMS),
            obs_counts=rng.randint(0, 20, NUM_DIMS).astype(np.int64),
            version=7,
        )
        data = state.to_json()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = StateVector.from_json(restored_data)

        np.testing.assert_array_equal(state.values, restored.values)
        np.testing.assert_array_equal(state.confidence, restored.confidence)
        np.testing.assert_array_equal(state.obs_counts, restored.obs_counts)
        assert state.version == restored.version
        assert state.schema_version == restored.schema_version


class TestMonotonicConfidence:
    def test_confidence_never_decreases(self):
        """Invariant 4: C_t is non-decreasing."""
        state = None
        rng = np.random.RandomState(99)
        prev_confidence = np.zeros(NUM_DIMS)

        for _ in range(20):
            values = rng.rand(NUM_DIMS)
            mask = rng.rand(NUM_DIMS) > 0.4
            if not np.any(mask):
                mask[0] = True
            obs = ObservationVector(values=values, mask=mask)
            state = apply_observation(state, obs)
            assert np.all(state.confidence >= prev_confidence - 1e-10)
            prev_confidence = state.confidence.copy()


class TestGracefulMissingData:
    def test_unobserved_preserved(self):
        """Invariant 5: Unobserved dims preserved, never zeroed."""
        # Initialize with full observation
        obs1 = ObservationVector(
            values=np.full(NUM_DIMS, 0.7, dtype=np.float64),
            mask=np.ones(NUM_DIMS, dtype=bool),
        )
        state = initialize_state(obs1)
        assert state.values[5] == 0.7

        # Update with partial observation (dim 5 missing)
        values = np.full(NUM_DIMS, 0.3, dtype=np.float64)
        mask = np.ones(NUM_DIMS, dtype=bool)
        mask[5] = False
        obs2 = ObservationVector(values=values, mask=mask)
        new_state = ema_update(state, obs2)

        # Dim 5 should be preserved at 0.7
        assert new_state.values[5] == 0.7
