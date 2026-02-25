"""Tests for weakness detection.

Verification criteria from roadmap:
1. Low-dim state → correct weaknesses identified
2. Confidence filtering: low-confidence dims excluded
"""

from __future__ import annotations

import numpy as np
import pytest

from src.policy.weakness import (
    DEFAULT_THRESHOLDS_BY_GROUP,
    MIN_CONFIDENCE,
    Weakness,
    _build_threshold_vector,
    detect_weaknesses,
)
from src.state.constants import DIM_GROUPS, DIM_NAMES, NUM_DIMS
from src.state.types import StateVector


class TestBuildThresholdVector:
    def test_default_thresholds_shape(self):
        tau = _build_threshold_vector()
        assert tau.shape == (NUM_DIMS,)

    def test_default_thresholds_values(self):
        tau = _build_threshold_vector()
        # Offensive profile dims (0-3) → 0.4
        for i in DIM_GROUPS["offensive_profile"]:
            assert tau[i] == 0.4
        # Technique dims (4-7) → 0.5
        for i in DIM_GROUPS["technique"]:
            assert tau[i] == 0.5
        # Defense dims (8-11) → 0.5
        for i in DIM_GROUPS["defense"]:
            assert tau[i] == 0.5
        # Rhythm dims (12-14) → 0.4
        for i in DIM_GROUPS["rhythm"]:
            assert tau[i] == 0.4
        # Conditioning dims (15-17) → 0.6
        for i in DIM_GROUPS["conditioning"]:
            assert tau[i] == 0.6

    def test_custom_thresholds(self):
        custom = {
            "offensive_profile": 0.3,
            "technique": 0.6,
            "defense": 0.7,
            "rhythm": 0.5,
            "conditioning": 0.8,
        }
        tau = _build_threshold_vector(custom)
        for i in DIM_GROUPS["technique"]:
            assert tau[i] == 0.6


class TestDetectWeaknesses:
    def test_balanced_state_no_weaknesses(self, balanced_state):
        """Roadmap verification #6: Balanced state → no weaknesses."""
        weaknesses = detect_weaknesses(balanced_state)
        assert len(weaknesses) == 0

    def test_weak_hooks_detected(self, weak_hooks_state):
        """Roadmap verification #1: Low-dim state → correct weakness identified."""
        weaknesses = detect_weaknesses(weak_hooks_state)
        assert len(weaknesses) == 1
        w = weaknesses[0]
        assert w.dim_index == 5
        assert w.dim_name == "tech_hook"
        assert w.group == "technique"
        assert w.value == 0.25
        assert w.threshold == 0.5
        assert w.gap == pytest.approx(0.25)

    def test_multiple_weaknesses(self, multi_weakness_state):
        weaknesses = detect_weaknesses(multi_weakness_state)
        dim_indices = {w.dim_index for w in weaknesses}
        assert dim_indices == {0, 5, 8, 12, 15}

    def test_low_confidence_excluded(self, low_confidence_state):
        """Roadmap verification #2: Low-confidence dims excluded from weaknesses."""
        weaknesses = detect_weaknesses(low_confidence_state)
        assert len(weaknesses) == 0

    def test_mixed_confidence_filtering(self, mixed_confidence_state):
        """Only high-confidence weak dims are flagged."""
        weaknesses = detect_weaknesses(mixed_confidence_state)
        dim_indices = {w.dim_index for w in weaknesses}
        assert 5 in dim_indices   # tech_hook: C=0.8 → included
        assert 6 not in dim_indices  # tech_uppercut: C=0.1 → excluded

    def test_exactly_at_threshold_not_weak(self):
        """Value exactly at τ_i is NOT a weakness (s_i < τ_i, not <=)."""
        values = np.full(NUM_DIMS, 0.75, dtype=np.float64)
        values[4] = 0.5  # exactly at technique threshold
        state = StateVector(
            values=values,
            confidence=np.full(NUM_DIMS, 0.8, dtype=np.float64),
            obs_counts=np.full(NUM_DIMS, 10, dtype=np.int64),
        )
        weaknesses = detect_weaknesses(state)
        assert all(w.dim_index != 4 for w in weaknesses)

    def test_exactly_at_confidence_threshold(self):
        """Confidence exactly at 0.3 IS included (C >= 0.3)."""
        values = np.full(NUM_DIMS, 0.75, dtype=np.float64)
        values[5] = 0.25
        confidence = np.full(NUM_DIMS, 0.8, dtype=np.float64)
        confidence[5] = 0.3  # exactly at min_confidence
        state = StateVector(
            values=values,
            confidence=confidence,
            obs_counts=np.full(NUM_DIMS, 10, dtype=np.int64),
        )
        weaknesses = detect_weaknesses(state)
        assert any(w.dim_index == 5 for w in weaknesses)

    def test_gap_always_positive(self, multi_weakness_state):
        """Gap (τ_i - s_i) is always > 0 for weaknesses."""
        weaknesses = detect_weaknesses(multi_weakness_state)
        for w in weaknesses:
            assert w.gap > 0

    def test_weakness_fields_complete(self, weak_hooks_state):
        weaknesses = detect_weaknesses(weak_hooks_state)
        w = weaknesses[0]
        assert isinstance(w.dim_index, int)
        assert isinstance(w.dim_name, str)
        assert isinstance(w.group, str)
        assert isinstance(w.value, float)
        assert isinstance(w.threshold, float)
        assert isinstance(w.confidence, float)
        assert isinstance(w.gap, float)

    def test_custom_min_confidence(self):
        """Custom min_confidence parameter."""
        values = np.full(NUM_DIMS, 0.20, dtype=np.float64)
        confidence = np.full(NUM_DIMS, 0.15, dtype=np.float64)
        state = StateVector(
            values=values,
            confidence=confidence,
            obs_counts=np.full(NUM_DIMS, 1, dtype=np.int64),
        )
        # Default min_confidence=0.3 → no weaknesses
        assert len(detect_weaknesses(state)) == 0
        # Lower min_confidence → weaknesses detected
        weaknesses = detect_weaknesses(state, min_confidence=0.1)
        assert len(weaknesses) > 0
