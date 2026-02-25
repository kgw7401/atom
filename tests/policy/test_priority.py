"""Tests for priority scoring.

Verification criteria from roadmap:
3. Priority ordering: Defense weakness ranked above repertoire weakness
"""

from __future__ import annotations

import numpy as np
import pytest

from src.policy.priority import DEFAULT_GROUP_WEIGHTS, PrioritizedWeakness, score_priorities
from src.policy.weakness import Weakness, detect_weaknesses
from src.state.constants import NUM_DIMS
from src.state.types import StateVector


class TestScorePriorities:
    def test_empty_weaknesses(self):
        result = score_priorities([])
        assert result == []

    def test_single_weakness(self):
        w = Weakness(
            dim_index=5, dim_name="tech_hook", group="technique",
            value=0.25, threshold=0.5, confidence=0.8, gap=0.25,
        )
        result = score_priorities([w])
        assert len(result) == 1
        pw = result[0]
        assert pw.rank == 1
        assert pw.weakness is w
        # priority = w_group(technique=1.2) * gap(0.25) * C(0.8) = 0.24
        assert pw.priority == pytest.approx(1.2 * 0.25 * 0.8)

    def test_defense_ranked_above_offensive(self):
        """Roadmap verification #3: Defense weakness ranked above repertoire weakness."""
        w_defense = Weakness(
            dim_index=8, dim_name="guard_consistency", group="defense",
            value=0.30, threshold=0.5, confidence=0.8, gap=0.20,
        )
        w_offensive = Weakness(
            dim_index=0, dim_name="repertoire_entropy", group="offensive_profile",
            value=0.20, threshold=0.4, confidence=0.8, gap=0.20,
        )
        # Same gap, same confidence → defense wins due to higher w_group
        result = score_priorities([w_offensive, w_defense])
        assert result[0].weakness.group == "defense"
        assert result[1].weakness.group == "offensive_profile"

    def test_priority_formula_correct(self):
        """priority_i = w_group(i) * (τ_i - s_i) * C_{t,i}"""
        w = Weakness(
            dim_index=15, dim_name="volume_endurance", group="conditioning",
            value=0.30, threshold=0.6, confidence=0.632, gap=0.30,
        )
        result = score_priorities([w])
        expected = DEFAULT_GROUP_WEIGHTS["conditioning"] * 0.30 * 0.632
        assert result[0].priority == pytest.approx(expected)

    def test_ranking_by_priority_descending(self, multi_weakness_state):
        weaknesses = detect_weaknesses(multi_weakness_state)
        prioritized = score_priorities(weaknesses)
        for i in range(len(prioritized) - 1):
            assert prioritized[i].priority >= prioritized[i + 1].priority

    def test_ranks_are_sequential(self, multi_weakness_state):
        weaknesses = detect_weaknesses(multi_weakness_state)
        prioritized = score_priorities(weaknesses)
        ranks = [pw.rank for pw in prioritized]
        assert ranks == list(range(1, len(prioritized) + 1))

    def test_custom_group_weights(self):
        w = Weakness(
            dim_index=0, dim_name="repertoire_entropy", group="offensive_profile",
            value=0.10, threshold=0.4, confidence=0.8, gap=0.30,
        )
        custom_weights = {"offensive_profile": 5.0}
        result = score_priorities([w], group_weights=custom_weights)
        assert result[0].priority == pytest.approx(5.0 * 0.30 * 0.8)

    def test_higher_gap_higher_priority(self):
        """Larger gap → higher priority (same group, same confidence)."""
        w_big_gap = Weakness(
            dim_index=4, dim_name="tech_straight", group="technique",
            value=0.10, threshold=0.5, confidence=0.8, gap=0.40,
        )
        w_small_gap = Weakness(
            dim_index=5, dim_name="tech_hook", group="technique",
            value=0.40, threshold=0.5, confidence=0.8, gap=0.10,
        )
        result = score_priorities([w_small_gap, w_big_gap])
        assert result[0].weakness.dim_name == "tech_straight"

    def test_higher_confidence_higher_priority(self):
        """Higher confidence → higher priority (same group, same gap)."""
        w_high_c = Weakness(
            dim_index=4, dim_name="tech_straight", group="technique",
            value=0.25, threshold=0.5, confidence=0.9, gap=0.25,
        )
        w_low_c = Weakness(
            dim_index=5, dim_name="tech_hook", group="technique",
            value=0.25, threshold=0.5, confidence=0.4, gap=0.25,
        )
        result = score_priorities([w_low_c, w_high_c])
        assert result[0].weakness.dim_name == "tech_straight"

    def test_multi_weakness_priority_order(self, multi_weakness_state):
        """Full integration: detect + score with multi_weakness_state.

        Expected order by priority = w_group * gap * C:
        - guard_consistency: 1.5 * 0.30 * 0.8 = 0.360 (defense)
        - work_rate:         0.8 * 0.30 * 0.8 = 0.192 (rhythm)
        - volume_endurance:  1.0 * 0.30 * 0.8 = 0.240 (conditioning)
        - tech_hook:         1.2 * 0.25 * 0.8 = 0.240 (technique)
        - repertoire_entropy: 0.8 * 0.25 * 0.8 = 0.160 (offensive)

        Order: guard > tech_hook = volume_endurance > work_rate > repertoire
        (tie between tech_hook and volume_endurance resolved by insertion order)
        """
        weaknesses = detect_weaknesses(multi_weakness_state)
        prioritized = score_priorities(weaknesses)
        assert prioritized[0].weakness.dim_name == "guard_consistency"
        assert prioritized[-1].weakness.dim_name == "repertoire_entropy"
