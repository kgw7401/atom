"""Tests for B3 Task 13: Track A data integration interface."""

import pytest

from track_b.b3.feedback import ComboStats, DrillFeedback
from track_b.b3.integration import (
    ComboMasteryUpdate,
    SessionContext,
    drill_feedback_to_llm_context,
    drill_feedback_to_profile_updates,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_feedback(
    session_id: str = "sess1",
    combo_stats: dict | None = None,
    overall_accuracy: float = 0.75,
    missed_actions: list[str] | None = None,
    notes: list[str] | None = None,
) -> DrillFeedback:
    return DrillFeedback(
        session_id=session_id,
        combo_stats=combo_stats or {},
        overall_accuracy=overall_accuracy,
        missed_actions=missed_actions or [],
        notes=notes or [],
    )


def make_rich_feedback() -> DrillFeedback:
    return DrillFeedback(
        session_id="sess42",
        combo_stats={
            "jab-cross": ComboStats(attempts=10, successes=9, partials=1, misses=0),
            "jab-cross-lead_hook": ComboStats(attempts=10, successes=2, partials=5, misses=3),
        },
        overall_accuracy=0.55,
        missed_actions=["lead_hook", "cross"],
        notes=["lead_hook is frequently missed."],
    )


# ── drill_feedback_to_profile_updates ────────────────────────────────────────


class TestDrillFeedbackToProfileUpdates:
    def test_returns_list(self):
        updates = drill_feedback_to_profile_updates(make_rich_feedback())
        assert isinstance(updates, list)

    def test_all_updates_are_correct_type(self):
        updates = drill_feedback_to_profile_updates(make_rich_feedback())
        assert all(isinstance(u, ComboMasteryUpdate) for u in updates)

    def test_one_update_per_combo(self):
        updates = drill_feedback_to_profile_updates(make_rich_feedback())
        assert len(updates) == 2

    def test_combo_names_present(self):
        updates = drill_feedback_to_profile_updates(make_rich_feedback())
        names = {u.combo_name for u in updates}
        assert "jab-cross" in names
        assert "jab-cross-lead_hook" in names

    def test_success_rate_correct(self):
        updates = drill_feedback_to_profile_updates(make_rich_feedback())
        jab_cross = next(u for u in updates if u.combo_name == "jab-cross")
        assert jab_cross.success_rate == pytest.approx(0.9)

    def test_attempts_correct(self):
        updates = drill_feedback_to_profile_updates(make_rich_feedback())
        jab_cross = next(u for u in updates if u.combo_name == "jab-cross")
        assert jab_cross.attempts == 10

    def test_last_result_success_for_good_combo(self):
        updates = drill_feedback_to_profile_updates(make_rich_feedback())
        jab_cross = next(u for u in updates if u.combo_name == "jab-cross")
        assert jab_cross.last_result == "success"

    def test_last_result_for_weak_combo(self):
        updates = drill_feedback_to_profile_updates(make_rich_feedback())
        hook = next(u for u in updates if u.combo_name == "jab-cross-lead_hook")
        assert hook.last_result in ("partial", "miss")

    def test_empty_feedback_returns_empty(self):
        updates = drill_feedback_to_profile_updates(make_feedback())
        assert updates == []

    def test_all_miss_combo_result(self):
        feedback = make_feedback(combo_stats={
            "jab": ComboStats(attempts=3, successes=0, partials=0, misses=3),
        })
        updates = drill_feedback_to_profile_updates(feedback)
        assert updates[0].last_result == "miss"

    def test_partial_combo_result(self):
        feedback = make_feedback(combo_stats={
            "jab": ComboStats(attempts=3, successes=0, partials=3, misses=0),
        })
        updates = drill_feedback_to_profile_updates(feedback)
        assert updates[0].last_result == "partial"


# ── drill_feedback_to_llm_context ─────────────────────────────────────────────


class TestDrillFeedbackToLlmContext:
    def test_returns_session_context(self):
        ctx = drill_feedback_to_llm_context(make_rich_feedback())
        assert isinstance(ctx, SessionContext)

    def test_session_id_preserved(self):
        ctx = drill_feedback_to_llm_context(make_rich_feedback())
        assert ctx.session_id == "sess42"

    def test_summary_is_string(self):
        ctx = drill_feedback_to_llm_context(make_rich_feedback())
        assert isinstance(ctx.summary, str)
        assert len(ctx.summary) > 10

    def test_summary_contains_accuracy(self):
        ctx = drill_feedback_to_llm_context(make_feedback(overall_accuracy=0.75))
        assert "75%" in ctx.summary

    def test_weak_combos_identified(self):
        ctx = drill_feedback_to_llm_context(make_rich_feedback())
        assert "jab-cross-lead_hook" in ctx.weak_combos

    def test_strong_combos_identified(self):
        ctx = drill_feedback_to_llm_context(make_rich_feedback())
        assert "jab-cross" in ctx.strong_combos

    def test_missed_actions_preserved(self):
        ctx = drill_feedback_to_llm_context(make_rich_feedback())
        assert "lead_hook" in ctx.missed_actions

    def test_suggested_focus_nonempty(self):
        ctx = drill_feedback_to_llm_context(make_rich_feedback())
        assert len(ctx.suggested_focus) >= 1

    def test_overall_accuracy_preserved(self):
        ctx = drill_feedback_to_llm_context(make_feedback(overall_accuracy=0.55))
        assert ctx.overall_accuracy == pytest.approx(0.55)

    def test_empty_combos_no_weak_strong(self):
        ctx = drill_feedback_to_llm_context(make_feedback())
        assert ctx.weak_combos == []
        assert ctx.strong_combos == []

    def test_perfect_session_suggests_complexity(self):
        """All combos successful → suggest increasing complexity."""
        feedback = DrillFeedback(
            session_id="perfect",
            combo_stats={"jab-cross": ComboStats(attempts=5, successes=5, partials=0, misses=0)},
            overall_accuracy=1.0,
            missed_actions=[],
            notes=[],
        )
        ctx = drill_feedback_to_llm_context(feedback)
        full_text = " ".join(ctx.suggested_focus).lower()
        assert "complex" in full_text or "maintain" in full_text

    def test_weak_combo_suggested_for_focus(self):
        ctx = drill_feedback_to_llm_context(make_rich_feedback())
        full_text = " ".join(ctx.suggested_focus).lower()
        assert "jab-cross-lead_hook" in full_text or "drill" in full_text

    def test_summary_contains_session_id(self):
        ctx = drill_feedback_to_llm_context(make_rich_feedback())
        assert "sess42" in ctx.summary
