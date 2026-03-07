"""Tests for B3 Task 12: DrillFeedback generation."""

import pytest

from track_b.b3.feedback import ComboStats, DrillFeedback, generate_feedback
from track_b.b3.matching import InstructionMatch, SessionMatch


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_match(
    instructed: list[str],
    detected: list[str],
    result: str,
    detail: str = "",
    time: float = 0.0,
) -> InstructionMatch:
    return InstructionMatch(
        instruction_time=time,
        instructed_combo=instructed,
        detected_combo=detected,
        result=result,
        detail=detail,
    )


def make_session(matches: list[InstructionMatch], session_id: str = "sess1") -> SessionMatch:
    return SessionMatch(session_id=session_id, matches=matches)


def make_20_instruction_session() -> SessionMatch:
    """Session with 20 instructions: mixed success/partial/miss."""
    matches = []
    # 10 x jab-cross → 8 success, 2 partial
    for _ in range(8):
        matches.append(make_match(["jab", "cross"], ["jab", "cross"], "success"))
    for _ in range(2):
        matches.append(make_match(["jab", "cross"], ["jab"], "partial", "cross missing"))
    # 10 x jab-cross-hook → 2 success, 5 partial, 3 miss
    for _ in range(2):
        matches.append(make_match(
            ["jab", "cross", "lead_hook"], ["jab", "cross", "lead_hook"], "success"
        ))
    for _ in range(5):
        matches.append(make_match(
            ["jab", "cross", "lead_hook"], ["jab", "cross"], "partial", "lead_hook missing"
        ))
    for _ in range(3):
        matches.append(make_match(
            ["jab", "cross", "lead_hook"], [], "miss", "no combo detected"
        ))
    return make_session(matches)


# ── ComboStats ─────────────────────────────────────────────────────────────────


class TestComboStats:
    def test_success_rate_perfect(self):
        s = ComboStats(attempts=10, successes=10, partials=0, misses=0)
        assert s.success_rate == pytest.approx(1.0)

    def test_success_rate_zero(self):
        s = ComboStats(attempts=5, successes=0, partials=3, misses=2)
        assert s.success_rate == pytest.approx(0.0)

    def test_success_rate_half(self):
        s = ComboStats(attempts=4, successes=2, partials=1, misses=1)
        assert s.success_rate == pytest.approx(0.5)

    def test_success_rate_no_attempts(self):
        s = ComboStats()
        assert s.success_rate == pytest.approx(0.0)

    def test_defaults_zero(self):
        s = ComboStats()
        assert s.attempts == 0
        assert s.successes == 0
        assert s.partials == 0
        assert s.misses == 0


# ── generate_feedback ─────────────────────────────────────────────────────────


class TestGenerateFeedback:
    def test_returns_drill_feedback(self):
        result = generate_feedback(make_session([]))
        assert isinstance(result, DrillFeedback)

    def test_session_id_preserved(self):
        result = generate_feedback(make_session([], session_id="test_sess"))
        assert result.session_id == "test_sess"

    def test_overall_accuracy_all_success(self):
        matches = [make_match(["jab"], ["jab"], "success") for _ in range(5)]
        result = generate_feedback(make_session(matches))
        assert result.overall_accuracy == pytest.approx(1.0)

    def test_overall_accuracy_all_miss(self):
        matches = [make_match(["jab"], [], "miss") for _ in range(5)]
        result = generate_feedback(make_session(matches))
        assert result.overall_accuracy == pytest.approx(0.0)

    def test_overall_accuracy_mixed(self):
        matches = [
            make_match(["jab"], ["jab"], "success"),
            make_match(["jab"], ["jab"], "success"),
            make_match(["jab"], [], "miss"),
            make_match(["jab"], [], "miss"),
        ]
        result = generate_feedback(make_session(matches))
        assert result.overall_accuracy == pytest.approx(0.5)

    def test_empty_session_accuracy_zero(self):
        result = generate_feedback(make_session([]))
        assert result.overall_accuracy == pytest.approx(0.0)

    def test_combo_stats_keys(self):
        matches = [
            make_match(["jab", "cross"], ["jab", "cross"], "success"),
            make_match(["jab", "cross", "lead_hook"], ["jab", "cross"], "partial"),
        ]
        result = generate_feedback(make_session(matches))
        assert "jab-cross" in result.combo_stats
        assert "jab-cross-lead_hook" in result.combo_stats

    def test_combo_stats_counts(self):
        matches = [
            make_match(["jab", "cross"], ["jab", "cross"], "success"),
            make_match(["jab", "cross"], ["jab", "cross"], "success"),
            make_match(["jab", "cross"], ["jab"], "partial"),
            make_match(["jab", "cross"], [], "miss"),
        ]
        result = generate_feedback(make_session(matches))
        stats = result.combo_stats["jab-cross"]
        assert stats.attempts == 4
        assert stats.successes == 2
        assert stats.partials == 1
        assert stats.misses == 1

    def test_missed_actions_identified(self):
        matches = [
            make_match(["jab", "cross", "lead_hook"], ["jab", "cross"], "partial") for _ in range(3)
        ]
        result = generate_feedback(make_session(matches))
        assert "lead_hook" in result.missed_actions

    def test_missed_actions_sorted_by_frequency(self):
        matches = [
            make_match(["jab", "cross", "lead_hook"], ["jab"], "partial"),  # cross + lead_hook missing
            make_match(["jab", "cross", "lead_hook"], ["jab"], "partial"),  # cross + lead_hook missing
            make_match(["jab", "cross", "lead_hook"], ["jab", "cross"], "partial"),  # lead_hook missing
        ]
        result = generate_feedback(make_session(matches))
        # lead_hook missed 3 times, cross missed 2 times
        assert result.missed_actions[0] == "lead_hook"

    def test_twenty_instructions_has_combo_breakdown(self):
        result = generate_feedback(make_20_instruction_session())
        assert len(result.combo_stats) == 2

    def test_twenty_instructions_has_notes(self):
        result = generate_feedback(make_20_instruction_session())
        assert len(result.notes) >= 2

    def test_notes_are_nonempty_strings(self):
        result = generate_feedback(make_20_instruction_session())
        assert all(len(n) > 0 for n in result.notes)

    def test_success_rate_in_combo_stats(self):
        matches = [make_match(["jab"], ["jab"], "success") for _ in range(3)]
        result = generate_feedback(make_session(matches))
        assert result.combo_stats["jab"].success_rate == pytest.approx(1.0)

    def test_no_missed_actions_on_perfect_session(self):
        matches = [
            make_match(["jab", "cross"], ["jab", "cross"], "success") for _ in range(5)
        ]
        result = generate_feedback(make_session(matches))
        assert result.missed_actions == []
