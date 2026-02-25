"""Tests for observation function (spec/state-vector.md §4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.state.constants import (
    CLASS_CROSS,
    CLASS_GUARD,
    CLASS_JAB,
    CLASS_LEAD_BODYSHOT,
    CLASS_LEAD_HOOK,
    CLASS_LEAD_UPPERCUT,
    CLASS_REAR_BODYSHOT,
    CLASS_REAR_HOOK,
    CLASS_REAR_UPPERCUT,
    MIN_SESSION_DURATION,
    NUM_DIMS,
    NUM_PUNCH_TYPES,
)
from src.state.observation import (
    _compute_combo_diversity,
    _compute_lead_rear_balance,
    _compute_level_change_ratio,
    _compute_repertoire_entropy,
    _compute_work_rate,
    compute_observation,
)
from src.state.types import ActionSegment, ObservationVector

from .conftest import make_segment


class TestRepertoireEntropy:
    def test_uniform_distribution(self):
        """All 8 punch types equally → entropy = 1.0."""
        segs = []
        for cls_id in range(1, NUM_PUNCH_TYPES + 1):
            for i in range(5):
                t = cls_id * 10 + i
                segs.append(make_segment(cls_id, t, t + 0.3))
        result = _compute_repertoire_entropy(segs)
        assert abs(result - 1.0) < 0.01

    def test_single_type(self):
        """Only jabs → entropy = 0.0."""
        segs = [make_segment(CLASS_JAB, i, i + 0.3) for i in range(10)]
        result = _compute_repertoire_entropy(segs)
        assert result == 0.0

    def test_two_types(self):
        """Two types equally → entropy between 0 and 1."""
        segs = [make_segment(CLASS_JAB, i, i + 0.3) for i in range(10)]
        segs += [make_segment(CLASS_CROSS, i + 20, i + 20.3) for i in range(10)]
        result = _compute_repertoire_entropy(segs)
        expected = np.log(2) / np.log(NUM_PUNCH_TYPES)
        assert abs(result - expected) < 0.01


class TestLevelChangeRatio:
    def test_no_body_shots(self):
        segs = [make_segment(CLASS_JAB, i, i + 0.3) for i in range(10)]
        assert _compute_level_change_ratio(segs) == 0.0

    def test_all_body_shots(self):
        segs = [make_segment(CLASS_LEAD_BODYSHOT, i, i + 0.3) for i in range(5)]
        segs += [make_segment(CLASS_REAR_BODYSHOT, i + 10, i + 10.3) for i in range(5)]
        assert _compute_level_change_ratio(segs) == 1.0

    def test_mixed(self):
        segs = [make_segment(CLASS_JAB, i, i + 0.3) for i in range(8)]
        segs += [make_segment(CLASS_LEAD_BODYSHOT, i + 20, i + 20.3) for i in range(2)]
        assert abs(_compute_level_change_ratio(segs) - 0.2) < 0.01


class TestLeadRearBalance:
    def test_perfect_balance(self):
        segs = [make_segment(CLASS_JAB, i, i + 0.3) for i in range(5)]
        segs += [make_segment(CLASS_CROSS, i + 10, i + 10.3) for i in range(5)]
        assert abs(_compute_lead_rear_balance(segs) - 1.0) < 0.01

    def test_all_lead(self):
        segs = [make_segment(CLASS_JAB, i, i + 0.3) for i in range(10)]
        assert abs(_compute_lead_rear_balance(segs) - 0.0) < 0.01

    def test_all_rear(self):
        segs = [make_segment(CLASS_CROSS, i, i + 0.3) for i in range(10)]
        assert abs(_compute_lead_rear_balance(segs) - 0.0) < 0.01


class TestComboDiversity:
    def test_insufficient_combos(self):
        segs = [make_segment(CLASS_JAB, 0, 0.3), make_segment(CLASS_CROSS, 0.5, 0.8)]
        assert _compute_combo_diversity(segs) is None

    def test_identical_combos(self):
        """Repeating same combo → low diversity."""
        segs = []
        for i in range(6):
            t = i * 3
            segs.append(make_segment(CLASS_JAB, t, t + 0.3))
            segs.append(make_segment(CLASS_CROSS, t + 0.5, t + 0.8))
        result = _compute_combo_diversity(segs)
        assert result is not None
        assert result < 0.5  # mostly same combos


class TestWorkRate:
    def test_basic(self):
        segs = [make_segment(CLASS_JAB, i, i + 0.3) for i in range(40)]
        # 40 punches in ~40 seconds = 60 punches/min → 60/80 = 0.75
        result = _compute_work_rate(segs, 40.0)
        assert abs(result - 0.75) < 0.01

    def test_zero_duration(self):
        segs = [make_segment(CLASS_JAB, 0, 0.3)]
        assert _compute_work_rate(segs, 0.0) == 0.0


class TestComputeObservation:
    def test_no_punches(self):
        """No punches → empty observation."""
        segs = [make_segment(CLASS_GUARD, 0, 5)]
        obs = compute_observation(segs, [], 5.0)
        assert obs.is_empty

    def test_all_values_bounded(self, diverse_segments, diverse_keypoints):
        """All observed values must be in [0, 1]."""
        duration = diverse_segments[-1].t_end + 1.0
        obs = compute_observation(diverse_segments, diverse_keypoints, duration)
        observed_values = obs.values[obs.mask]
        assert np.all(observed_values >= 0.0)
        assert np.all(observed_values <= 1.0)

    def test_shape(self, diverse_segments, diverse_keypoints):
        duration = diverse_segments[-1].t_end + 1.0
        obs = compute_observation(diverse_segments, diverse_keypoints, duration)
        assert obs.values.shape == (NUM_DIMS,)
        assert obs.mask.shape == (NUM_DIMS,)

    def test_short_session_no_conditioning(self, short_segments, short_keypoints):
        """Session < 90s → conditioning dims (15-17) not observed."""
        obs = compute_observation(short_segments, short_keypoints, 30.0)
        # Conditioning dims should be unobserved
        assert not obs.mask[15]  # volume_endurance
        assert not obs.mask[16]  # technique_endurance
        assert not obs.mask[17]  # rhythm_stability

    def test_defensive_reaction_only_ai_session(self, diverse_segments, diverse_keypoints):
        """defensive_reaction (dim 11) only in ai_session mode."""
        duration = diverse_segments[-1].t_end + 1.0
        obs_shadow = compute_observation(diverse_segments, diverse_keypoints, duration, mode="shadow")
        assert not obs_shadow.mask[11]

    def test_offensive_profile_observed(self, diverse_segments, diverse_keypoints):
        """Diverse segments → offensive profile dims (0-2) observed."""
        duration = diverse_segments[-1].t_end + 1.0
        obs = compute_observation(diverse_segments, diverse_keypoints, duration)
        assert obs.mask[0]  # repertoire_entropy
        assert obs.mask[1]  # level_change_ratio
        assert obs.mask[2]  # lead_rear_balance

    def test_deterministic(self, diverse_segments, diverse_keypoints):
        """Same inputs → same output."""
        duration = diverse_segments[-1].t_end + 1.0
        obs1 = compute_observation(diverse_segments, diverse_keypoints, duration)
        obs2 = compute_observation(diverse_segments, diverse_keypoints, duration)
        np.testing.assert_array_equal(obs1.values[obs1.mask], obs2.values[obs2.mask])
        np.testing.assert_array_equal(obs1.mask, obs2.mask)
