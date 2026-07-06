"""Tests for Pipeline Stage 4: Session Matching."""

import pytest

from ml.configs import BoxingConfig
from ml.pipeline.session_matcher import SessionMatcher
from ml.pipeline.types import DetectedCombo, TTSInstruction


@pytest.fixture
def config():
    return BoxingConfig()


@pytest.fixture
def matcher(config):
    return SessionMatcher(config)


def _instr(ts: float, actions: list, name: str = "test") -> TTSInstruction:
    return TTSInstruction(timestamp=ts, combo_name=name, expected_actions=actions)


def _combo(start: float, end: float, actions: list) -> DetectedCombo:
    return DetectedCombo(start_time=start, end_time=end, actions=actions)


# ── Exact match ──

class TestExactMatch:
    def test_single_exact_match(self, matcher):
        instrs = [_instr(1.0, ["jab", "cross"])]
        combos = [_combo(1.5, 2.0, ["jab", "cross"])]
        result = matcher.match(instrs, combos)
        assert result.total_success == 1
        assert result.total_miss == 0
        assert result.matches[0].result == "success"
        assert result.matches[0].detail == "Exact match"

    def test_multiple_exact_matches(self, matcher):
        instrs = [
            _instr(1.0, ["jab", "cross"]),
            _instr(5.0, ["lead_hook", "cross"]),
        ]
        combos = [
            _combo(1.5, 2.0, ["jab", "cross"]),
            _combo(5.5, 6.0, ["lead_hook", "cross"]),
        ]
        result = matcher.match(instrs, combos)
        assert result.total_success == 2
        assert result.total_instructions == 2


# ── Partial match ──

class TestPartialMatch:
    def test_partial_overlap(self, matcher):
        """2 of 3 actions match → partial (score ≥ 0.5)."""
        instrs = [_instr(1.0, ["jab", "cross", "lead_hook"])]
        combos = [_combo(1.5, 2.0, ["jab", "cross"])]
        result = matcher.match(instrs, combos)
        assert result.matches[0].result == "partial"
        assert result.total_partial == 1

    def test_half_match_is_partial(self, matcher):
        """Exactly 50% overlap → partial."""
        instrs = [_instr(1.0, ["jab", "cross"])]
        combos = [_combo(1.5, 2.0, ["jab", "lead_hook"])]
        result = matcher.match(instrs, combos)
        assert result.matches[0].result == "partial"


# ── Miss ──

class TestMiss:
    def test_no_combo_in_window(self, matcher):
        """No combo detected within match window → miss."""
        instrs = [_instr(1.0, ["jab", "cross"])]
        combos = [_combo(10.0, 10.5, ["jab", "cross"])]
        result = matcher.match(instrs, combos)
        assert result.total_miss == 1
        assert result.matches[0].result == "miss"
        assert result.matches[0].combo_idx is None

    def test_completely_different_combo(self, matcher):
        """Combo in window but totally different actions → miss."""
        instrs = [_instr(1.0, ["jab", "cross"])]
        combos = [_combo(1.5, 2.0, ["lead_hook", "rear_hook"])]
        result = matcher.match(instrs, combos)
        assert result.matches[0].result == "miss"

    def test_empty_combos(self, matcher):
        instrs = [_instr(1.0, ["jab", "cross"])]
        result = matcher.match(instrs, [])
        assert result.total_miss == 1

    def test_empty_instructions(self, matcher):
        combos = [_combo(1.0, 2.0, ["jab"])]
        result = matcher.match([], combos)
        assert result.total_instructions == 0
        assert len(result.matches) == 0


# ── Match window ──

class TestMatchWindow:
    def test_combo_at_window_edge(self, matcher):
        """Combo starting just before window ends → still matched."""
        # Window: [1.0, 4.0] (match_window = 3.0s)
        instrs = [_instr(1.0, ["jab"])]
        combos = [_combo(3.9, 4.5, ["jab"])]
        result = matcher.match(instrs, combos)
        assert result.matches[0].result == "success"

    def test_combo_after_window(self, matcher):
        """Combo starting after window → miss."""
        instrs = [_instr(1.0, ["jab"])]
        combos = [_combo(4.1, 4.5, ["jab"])]
        result = matcher.match(instrs, combos)
        assert result.matches[0].result == "miss"

    def test_combo_before_instruction(self, matcher):
        """Combo ending before instruction time → still overlaps if end >= start."""
        instrs = [_instr(2.0, ["jab"])]
        combos = [_combo(1.5, 2.0, ["jab"])]
        result = matcher.match(instrs, combos)
        assert result.matches[0].result == "success"

    def test_combo_ending_before_window_start(self, matcher):
        """Combo ending before window start → miss."""
        instrs = [_instr(5.0, ["jab"])]
        combos = [_combo(1.0, 4.9, ["jab"])]
        result = matcher.match(instrs, combos)
        assert result.matches[0].result == "miss"


# ── Best match selection ──

class TestBestMatch:
    def test_picks_best_among_candidates(self, matcher):
        """When multiple combos in window, pick the best match."""
        instrs = [_instr(1.0, ["jab", "cross"])]
        combos = [
            _combo(1.2, 1.5, ["lead_hook"]),          # bad match
            _combo(1.5, 2.0, ["jab", "cross"]),         # exact match
            _combo(2.0, 2.5, ["jab"]),                   # partial match
        ]
        result = matcher.match(instrs, combos)
        assert result.matches[0].result == "success"
        assert result.matches[0].combo_idx == 1


# ── Combo stats aggregation ──

class TestComboStats:
    def test_stats_aggregated_by_key(self, matcher):
        instrs = [
            _instr(1.0, ["jab", "cross"], "원-투"),
            _instr(5.0, ["jab", "cross"], "원-투"),
            _instr(10.0, ["jab", "cross"], "원-투"),
        ]
        combos = [
            _combo(1.5, 2.0, ["jab", "cross"]),
            _combo(5.5, 6.0, ["jab", "cross"]),
            # third has no combo → miss
        ]
        result = matcher.match(instrs, combos)

        key = "jab-cross"
        assert key in result.combo_stats
        stat = result.combo_stats[key]
        assert stat.attempts == 3
        assert stat.successes == 2
        assert stat.misses == 1
        assert stat.combo_name == "원-투"

    def test_multiple_combo_types(self, matcher):
        instrs = [
            _instr(1.0, ["jab", "cross"], "원-투"),
            _instr(5.0, ["jab", "cross", "lead_hook"], "원-투-쓰리"),
        ]
        combos = [
            _combo(1.5, 2.0, ["jab", "cross"]),
            _combo(5.5, 6.5, ["jab", "cross", "lead_hook"]),
        ]
        result = matcher.match(instrs, combos)
        assert len(result.combo_stats) == 2
        assert "jab-cross" in result.combo_stats
        assert "jab-cross-lead_hook" in result.combo_stats

    def test_success_rate(self, matcher):
        instrs = [
            _instr(1.0, ["jab"]),
            _instr(3.0, ["jab"]),  # out of window of first combo
        ]
        combos = [_combo(1.5, 1.8, ["jab"])]
        result = matcher.match(instrs, combos)
        stat = result.combo_stats["jab"]
        assert stat.success_rate == 0.5


# ── DrillResult properties ──

class TestDrillResult:
    def test_overall_success_rate(self, matcher):
        instrs = [
            _instr(1.0, ["jab"]),
            _instr(5.0, ["jab"]),
            _instr(10.0, ["jab"]),
            _instr(15.0, ["jab"]),
        ]
        combos = [
            _combo(1.5, 1.8, ["jab"]),
            _combo(5.5, 5.8, ["jab"]),
        ]
        result = matcher.match(instrs, combos)
        assert result.overall_success_rate == 0.5

    def test_totals_sum(self, matcher):
        instrs = [
            _instr(1.0, ["jab", "cross"]),
            _instr(5.0, ["jab", "cross"]),
            _instr(10.0, ["jab", "cross"]),
        ]
        combos = [
            _combo(1.5, 2.0, ["jab", "cross"]),      # success
            _combo(5.5, 6.0, ["jab"]),                  # partial
            # third → miss
        ]
        result = matcher.match(instrs, combos)
        assert result.total_success + result.total_partial + result.total_miss == 3


# ── Overlap score (LCS-based) ──

class TestOverlapScore:
    def test_exact_score(self, matcher):
        score = matcher._overlap_score(["jab", "cross"], ["jab", "cross"])
        assert score == 1.0

    def test_empty_expected(self, matcher):
        assert matcher._overlap_score([], ["jab"]) == 0.0

    def test_empty_actual(self, matcher):
        assert matcher._overlap_score(["jab"], []) == 0.0

    def test_subsequence_match(self, matcher):
        """LCS: jab, cross appear in order within actual."""
        score = matcher._overlap_score(
            ["jab", "cross"],
            ["jab", "lead_hook", "cross"],
        )
        assert score == 1.0

    def test_partial_score(self, matcher):
        """Only 1 of 2 expected in actual."""
        score = matcher._overlap_score(
            ["jab", "cross"],
            ["jab", "lead_hook"],
        )
        assert score == 0.5

    def test_no_overlap(self, matcher):
        score = matcher._overlap_score(
            ["jab", "cross"],
            ["lead_hook", "rear_hook"],
        )
        assert score == 0.0
