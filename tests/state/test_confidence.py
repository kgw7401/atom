"""Tests for confidence model (spec/state-vector.md §6)."""

from __future__ import annotations

import numpy as np

from src.state.constants import N_REF
from src.state.update import compute_confidence


class TestConfidence:
    def test_zero_observations(self):
        c = compute_confidence(np.array([0]))
        assert c[0] == 0.0

    def test_n_ref_observations(self):
        """At n_ref sessions, confidence ≈ 0.632."""
        c = compute_confidence(np.array([N_REF]))
        assert abs(c[0] - 0.632) < 0.01

    def test_many_observations(self):
        """At 20 sessions, confidence ≈ 0.982."""
        c = compute_confidence(np.array([20]))
        assert abs(c[0] - 0.982) < 0.01

    def test_monotonic_increasing(self):
        """Confidence must be monotonically increasing with n."""
        counts = np.arange(0, 50, dtype=np.int64)
        confidences = compute_confidence(counts)
        diffs = np.diff(confidences)
        assert np.all(diffs >= 0), "Confidence must be monotonically non-decreasing"

    def test_bounded_zero_one(self):
        """Confidence is always in [0, 1]."""
        counts = np.array([0, 1, 5, 10, 50, 100, 1000], dtype=np.int64)
        c = compute_confidence(counts)
        assert np.all(c >= 0.0)
        assert np.all(c <= 1.0)

    def test_vector_shape(self):
        """Works with multi-dimensional input."""
        counts = np.array([0, 3, 5, 10, 0, 7, 2, 1, 4, 6, 8, 0, 3, 5, 2, 1, 9, 15], dtype=np.int64)
        c = compute_confidence(counts)
        assert c.shape == (18,)
