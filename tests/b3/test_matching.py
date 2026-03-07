"""Tests for B3 Task 11: Session matching — instruction vs execution."""

import pytest

from track_b.b3.combo import ComboInstance, ComboSequence
from track_b.b3.matching import (
    InstructionMatch,
    SessionMatch,
    TTSInstruction,
    TTSInstructionLog,
    _classify_match,
    match_session,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_log(
    instructions: list[tuple[float, str, list[str]]],
    session_id: str = "sess1",
) -> TTSInstructionLog:
    return TTSInstructionLog(
        session_id=session_id,
        instructions=[
            TTSInstruction(timestamp=t, combo_name=name, combo_actions=actions)
            for t, name, actions in instructions
        ],
    )


def make_combo_seq(
    combos: list[tuple[list[str], float, float]],
) -> ComboSequence:
    return ComboSequence(
        video_id="v1",
        fighter_id="user",
        combos=[
            ComboInstance(actions=a, start_time=s, end_time=e)
            for a, s, e in combos
        ],
    )


def spec_log() -> TTSInstructionLog:
    return make_log([(12.0, "jab-cross-hook", ["jab", "cross", "lead_hook"])])


def spec_combo() -> ComboSequence:
    return make_combo_seq([(["jab", "cross"], 12.3, 12.9)])


# ── _classify_match ────────────────────────────────────────────────────────────


class TestClassifyMatch:
    def test_success_exact_match(self):
        result, _ = _classify_match(["jab", "cross"], ["jab", "cross"])
        assert result == "success"

    def test_success_detail_empty(self):
        _, detail = _classify_match(["jab", "cross"], ["jab", "cross"])
        assert detail == ""

    def test_partial_missing_one(self):
        result, _ = _classify_match(["jab", "cross", "lead_hook"], ["jab", "cross"])
        assert result == "partial"

    def test_partial_detail_names_missing(self):
        _, detail = _classify_match(["jab", "cross", "lead_hook"], ["jab", "cross"])
        assert "lead_hook" in detail

    def test_miss_empty_detected(self):
        result, _ = _classify_match(["jab", "cross"], [])
        assert result == "miss"

    def test_miss_no_overlap(self):
        result, _ = _classify_match(["jab", "cross"], ["rear_hook", "rear_uppercut"])
        assert result == "miss"

    def test_partial_multiple_missing(self):
        result, detail = _classify_match(["jab", "cross", "lead_hook"], ["jab"])
        assert result == "partial"
        assert "cross" in detail
        assert "lead_hook" in detail

    def test_success_with_extra_actions(self):
        """All instructed actions present even with extras → success."""
        result, _ = _classify_match(["jab", "cross"], ["jab", "cross", "lead_hook"])
        assert result == "success"

    def test_miss_detail_no_matching(self):
        _, detail = _classify_match(["jab"], ["rear_hook"])
        assert "no matching" in detail or detail != ""


# ── match_session ─────────────────────────────────────────────────────────────


class TestMatchSession:
    def test_spec_case_partial(self):
        """Spec: jab-cross-hook instructed, jab-cross detected → partial."""
        result = match_session(spec_log(), spec_combo())
        assert result.matches[0].result == "partial"

    def test_spec_case_detail(self):
        """Spec: lead_hook missing."""
        result = match_session(spec_log(), spec_combo())
        assert "lead_hook" in result.matches[0].detail

    def test_spec_case_detected_combo(self):
        result = match_session(spec_log(), spec_combo())
        assert result.matches[0].detected_combo == ["jab", "cross"]

    def test_spec_case_instructed_combo(self):
        result = match_session(spec_log(), spec_combo())
        assert result.matches[0].instructed_combo == ["jab", "cross", "lead_hook"]

    def test_returns_session_match(self):
        result = match_session(spec_log(), spec_combo())
        assert isinstance(result, SessionMatch)

    def test_session_id_preserved(self):
        log = make_log([], session_id="my_session")
        result = match_session(log, make_combo_seq([]))
        assert result.session_id == "my_session"

    def test_success_match(self):
        log = make_log([(10.0, "jab-cross", ["jab", "cross"])])
        cs = make_combo_seq([(["jab", "cross"], 10.5, 11.0)])
        result = match_session(log, cs)
        assert result.matches[0].result == "success"

    def test_miss_no_combo_in_window(self):
        log = make_log([(10.0, "jab-cross", ["jab", "cross"])])
        cs = make_combo_seq([(["jab", "cross"], 20.0, 20.5)])  # outside window
        result = match_session(log, cs)
        assert result.matches[0].result == "miss"

    def test_multiple_instructions(self):
        log = make_log([
            (10.0, "jab-cross", ["jab", "cross"]),
            (25.0, "jab-cross-hook", ["jab", "cross", "lead_hook"]),
        ])
        cs = make_combo_seq([
            (["jab", "cross"], 10.3, 10.8),
            (["jab", "cross"], 25.2, 25.7),
        ])
        result = match_session(log, cs)
        assert len(result.matches) == 2
        assert result.matches[0].result == "success"
        assert result.matches[1].result == "partial"

    def test_empty_instruction_log(self):
        log = make_log([])
        result = match_session(log, make_combo_seq([(["jab"], 1.0, 1.1)]))
        assert result.matches == []

    def test_window_boundary_start_included(self):
        """Combo starting exactly at instruction_time is within window."""
        log = make_log([(10.0, "jab", ["jab"])])
        cs = make_combo_seq([(["jab"], 10.0, 10.1)])
        result = match_session(log, cs)
        assert result.matches[0].result == "success"

    def test_combo_at_window_end_excluded(self):
        """Combo starting at instruction_time + window is outside window."""
        log = make_log([(10.0, "jab", ["jab"])])
        cs = make_combo_seq([(["jab"], 13.0, 13.1)])  # exactly window_end
        result = match_session(log, cs, window=3.0)
        assert result.matches[0].result == "miss"

    def test_best_candidate_selected(self):
        """Multiple combos in window — pick one with most overlap."""
        log = make_log([(10.0, "jab-cross-hook", ["jab", "cross", "lead_hook"])])
        cs = make_combo_seq([
            (["jab"], 10.1, 10.2),                         # 1 overlap
            (["jab", "cross", "lead_hook"], 10.5, 11.0),  # 3 overlap → best
        ])
        result = match_session(log, cs)
        assert result.matches[0].result == "success"

    def test_custom_window(self):
        """Combo at t+1.5 with window=1.0 should miss."""
        log = make_log([(10.0, "jab", ["jab"])])
        cs = make_combo_seq([(["jab"], 11.5, 11.6)])
        result = match_session(log, cs, window=1.0)
        assert result.matches[0].result == "miss"

    def test_instruction_match_type(self):
        log = make_log([(10.0, "jab-cross", ["jab", "cross"])])
        cs = make_combo_seq([(["jab", "cross"], 10.3, 10.8)])
        result = match_session(log, cs)
        assert isinstance(result.matches[0], InstructionMatch)

    def test_instruction_time_preserved(self):
        log = make_log([(10.0, "jab-cross", ["jab", "cross"])])
        cs = make_combo_seq([(["jab", "cross"], 10.3, 10.8)])
        result = match_session(log, cs)
        assert result.matches[0].instruction_time == pytest.approx(10.0)

    def test_empty_combos_all_miss(self):
        log = make_log([
            (10.0, "jab", ["jab"]),
            (20.0, "cross", ["cross"]),
        ])
        result = match_session(log, make_combo_seq([]))
        assert all(m.result == "miss" for m in result.matches)
