"""Tests for session planner.

Verification criteria from roadmap:
4. Session plan structure: valid plan with rounds, drills, durations
5. Targeted session: state with weak hooks → plan includes hook drills
6. Balanced state: no weaknesses → maintenance/general session
7. Determinism: same S_t → same session plan
"""

from __future__ import annotations

import numpy as np
import pytest

from src.policy.drill_library import clear_cache, load_drills
from src.policy.session_planner import (
    SessionPlan,
    plan_session,
)
from src.state.constants import NUM_DIMS
from src.state.types import StateVector


@pytest.fixture(autouse=True)
def _clear_drill_cache():
    """Clear drill cache before each test."""
    clear_cache()
    yield
    clear_cache()


class TestSessionPlanStructure:
    def test_plan_has_required_fields(self, weak_hooks_state):
        """Roadmap verification #4: Valid plan with rounds, drills, durations."""
        plan = plan_session(weak_hooks_state)
        assert isinstance(plan, SessionPlan)
        assert plan.plan_type in ("targeted", "maintenance")
        assert plan.total_duration_seconds > 0
        assert plan.num_rounds > 0
        assert plan.rest_duration_seconds >= 0
        assert len(plan.rounds) == plan.num_rounds

    def test_rounds_have_drills(self, weak_hooks_state):
        plan = plan_session(weak_hooks_state)
        for r in plan.rounds:
            assert r.round_number >= 1
            assert len(r.drills) >= 1
            assert r.duration_seconds > 0
            assert len(r.focus_dims) >= 1

    def test_drills_have_complete_fields(self, weak_hooks_state):
        plan = plan_session(weak_hooks_state)
        for r in plan.rounds:
            for d in r.drills:
                assert d.drill_name
                assert d.drill_type in ("single", "combo", "defense", "conditioning")
                assert len(d.actions) > 0
                assert len(d.target_dims) > 0
                assert len(d.target_dim_names) > 0
                assert d.level in (1, 2, 3)
                assert d.duration_seconds > 0

    def test_to_dict_serializable(self, weak_hooks_state):
        """Plan can be serialized to JSON-compatible dict."""
        plan = plan_session(weak_hooks_state)
        d = plan.to_dict()
        assert isinstance(d, dict)
        assert "plan_type" in d
        assert "rounds" in d
        assert isinstance(d["rounds"], list)
        # Verify nested structure
        round_0 = d["rounds"][0]
        assert "drills" in round_0
        assert "round_number" in round_0


class TestTargetedSession:
    def test_weak_hooks_includes_hook_drills(self, weak_hooks_state):
        """Roadmap verification #5: State with weak hooks → plan includes hook drills."""
        plan = plan_session(weak_hooks_state)
        assert plan.plan_type == "targeted"

        # At least one drill should target dim 5 (tech_hook)
        all_target_dims = set()
        for r in plan.rounds:
            for d in r.drills:
                all_target_dims.update(d.target_dims)
        assert 5 in all_target_dims

    def test_target_weaknesses_populated(self, weak_hooks_state):
        plan = plan_session(weak_hooks_state)
        assert len(plan.target_weaknesses) > 0
        tw = plan.target_weaknesses[0]
        assert "dim_name" in tw
        assert "priority" in tw
        assert "rank" in tw

    def test_focus_summary_mentions_weakness(self, weak_hooks_state):
        plan = plan_session(weak_hooks_state)
        assert "tech_hook" in plan.focus_summary

    def test_multi_weakness_plan(self, multi_weakness_state):
        plan = plan_session(multi_weakness_state, max_weaknesses=3)
        assert plan.plan_type == "targeted"
        assert len(plan.target_weaknesses) <= 3
        # Top weakness (guard_consistency) should be first
        assert plan.target_weaknesses[0]["dim_name"] == "guard_consistency"


class TestMaintenanceSession:
    def test_balanced_state_maintenance(self, balanced_state):
        """Roadmap verification #6: No weaknesses → maintenance/general session."""
        plan = plan_session(balanced_state)
        assert plan.plan_type == "maintenance"
        assert len(plan.target_weaknesses) == 0
        assert "maintenance" in plan.focus_summary.lower() or "no" in plan.focus_summary.lower()

    def test_maintenance_still_has_rounds(self, balanced_state):
        plan = plan_session(balanced_state)
        assert plan.num_rounds > 0
        assert len(plan.rounds) > 0
        for r in plan.rounds:
            assert len(r.drills) >= 1


class TestDeterminism:
    def test_same_state_same_plan(self, weak_hooks_state):
        """Roadmap verification #7: Same S_t → same session plan."""
        plan_1 = plan_session(weak_hooks_state)
        plan_2 = plan_session(weak_hooks_state)
        d1 = plan_1.to_dict()
        d2 = plan_2.to_dict()
        assert d1 == d2

    def test_determinism_multi_weakness(self, multi_weakness_state):
        plan_1 = plan_session(multi_weakness_state)
        plan_2 = plan_session(multi_weakness_state)
        assert plan_1.to_dict() == plan_2.to_dict()


class TestPlanParameters:
    def test_custom_rounds(self, weak_hooks_state):
        plan = plan_session(weak_hooks_state, num_rounds=5)
        assert plan.num_rounds == 5
        assert len(plan.rounds) == 5

    def test_custom_max_weaknesses(self, multi_weakness_state):
        plan = plan_session(multi_weakness_state, max_weaknesses=1)
        assert len(plan.target_weaknesses) == 1

    def test_total_duration_calculation(self, weak_hooks_state):
        plan = plan_session(
            weak_hooks_state,
            num_rounds=3,
            round_duration=180,
            rest_duration=60,
        )
        expected = 3 * 180 + 2 * 60  # rounds + rests between rounds
        assert plan.total_duration_seconds == expected
