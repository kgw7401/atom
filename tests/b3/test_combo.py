"""Tests for B3 Task 10: Combo sequence recognition."""

import pytest

from track_b.b2.tad import ActionDetection, ActionTimeline
from track_b.b3.combo import ComboInstance, ComboSequence, group_into_combos


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_det(action: str, start: float, end: float, conf: float = 0.9) -> ActionDetection:
    return ActionDetection(start_time=start, end_time=end, action_class=action, confidence=conf)


def make_timeline(
    dets: list[ActionDetection],
    video_id: str = "v1",
    fighter_id: str = "user",
) -> ActionTimeline:
    return ActionTimeline(video_id=video_id, fighter_id=fighter_id, actions=dets)


def spec_timeline() -> ActionTimeline:
    """Spec verification case from the spec doc."""
    return make_timeline([
        make_det("jab", 12.3, 12.4),
        make_det("cross", 12.5, 12.6),
        make_det("lead_hook", 12.9, 13.1),
        make_det("jab", 15.1, 15.2),
        make_det("cross", 15.3, 15.4),
    ])


# ── ComboInstance ──────────────────────────────────────────────────────────────


class TestComboInstance:
    def test_duration(self):
        ci = ComboInstance(actions=["jab", "cross"], start_time=1.0, end_time=1.5)
        assert ci.duration == pytest.approx(0.5)

    def test_duration_zero_if_inverted(self):
        ci = ComboInstance(actions=["jab"], start_time=1.5, end_time=1.0)
        assert ci.duration == pytest.approx(0.0)

    def test_length(self):
        ci = ComboInstance(actions=["jab", "cross", "lead_hook"], start_time=0.0, end_time=1.0)
        assert ci.length == 3

    def test_length_empty(self):
        ci = ComboInstance(actions=[], start_time=0.0, end_time=0.0)
        assert ci.length == 0


# ── ComboSequence ─────────────────────────────────────────────────────────────


class TestComboSequence:
    def test_len(self):
        cs = ComboSequence(
            video_id="v1", fighter_id="user",
            combos=[
                ComboInstance(actions=["jab"], start_time=0.0, end_time=0.1),
                ComboInstance(actions=["cross"], start_time=2.0, end_time=2.1),
            ],
        )
        assert len(cs) == 2

    def test_len_empty(self):
        cs = ComboSequence(video_id="v1", fighter_id="user")
        assert len(cs) == 0

    def test_filter_by_length_min_2(self):
        cs = ComboSequence(
            video_id="v1", fighter_id="user",
            combos=[
                ComboInstance(actions=["jab"], start_time=0.0, end_time=0.1),
                ComboInstance(actions=["jab", "cross"], start_time=2.0, end_time=2.2),
                ComboInstance(actions=["jab", "cross", "lead_hook"], start_time=4.0, end_time=4.4),
            ],
        )
        filtered = cs.filter_by_length(2)
        assert len(filtered) == 2
        assert all(c.length >= 2 for c in filtered)

    def test_filter_by_length_keeps_all(self):
        cs = ComboSequence(
            video_id="v1", fighter_id="user",
            combos=[
                ComboInstance(actions=["jab", "cross"], start_time=0.0, end_time=0.5),
            ],
        )
        assert len(cs.filter_by_length(1)) == 1

    def test_filter_by_length_excludes_all(self):
        cs = ComboSequence(
            video_id="v1", fighter_id="user",
            combos=[
                ComboInstance(actions=["jab"], start_time=0.0, end_time=0.1),
            ],
        )
        assert cs.filter_by_length(3) == []


# ── group_into_combos ─────────────────────────────────────────────────────────


class TestGroupIntoCombos:
    def test_spec_case_two_combos(self):
        """Spec: [jab@12.3, cross@12.5, lead_hook@12.9, gap, jab@15.1, cross@15.3] → 2 combos."""
        result = group_into_combos(spec_timeline())
        assert len(result.combos) == 2

    def test_spec_case_first_combo_actions(self):
        result = group_into_combos(spec_timeline())
        assert result.combos[0].actions == ["jab", "cross", "lead_hook"]

    def test_spec_case_second_combo_actions(self):
        result = group_into_combos(spec_timeline())
        assert result.combos[1].actions == ["jab", "cross"]

    def test_spec_case_first_combo_start(self):
        result = group_into_combos(spec_timeline())
        assert result.combos[0].start_time == pytest.approx(12.3)

    def test_spec_case_first_combo_end(self):
        result = group_into_combos(spec_timeline())
        assert result.combos[0].end_time == pytest.approx(13.1)

    def test_spec_case_second_combo_start(self):
        result = group_into_combos(spec_timeline())
        assert result.combos[1].start_time == pytest.approx(15.1)

    def test_spec_case_second_combo_end(self):
        result = group_into_combos(spec_timeline())
        assert result.combos[1].end_time == pytest.approx(15.4)

    def test_empty_timeline_no_combos(self):
        result = group_into_combos(make_timeline([]))
        assert len(result.combos) == 0

    def test_empty_preserves_video_id(self):
        result = group_into_combos(make_timeline([], video_id="fight_42"))
        assert result.video_id == "fight_42"

    def test_empty_preserves_fighter_id(self):
        result = group_into_combos(make_timeline([], fighter_id="fighter_a"))
        assert result.fighter_id == "fighter_a"

    def test_single_action_one_combo(self):
        result = group_into_combos(make_timeline([make_det("jab", 1.0, 1.1)]))
        assert len(result.combos) == 1
        assert result.combos[0].actions == ["jab"]

    def test_all_close_together_one_combo(self):
        """Actions 0.1s apart → all in one combo (default threshold 0.8s)."""
        result = group_into_combos(make_timeline([
            make_det("jab", 0.0, 0.1),
            make_det("cross", 0.2, 0.3),
            make_det("lead_hook", 0.4, 0.5),
        ]))
        assert len(result.combos) == 1

    def test_gap_at_threshold_splits(self):
        """Gap of exactly 0.8s should split (>= threshold)."""
        result = group_into_combos(make_timeline([
            make_det("jab", 0.0, 0.5),
            make_det("cross", 1.3, 1.4),   # gap = 1.3 - 0.5 = 0.8
        ]), gap_threshold=0.8)
        assert len(result.combos) == 2

    def test_gap_just_below_threshold_no_split(self):
        """Gap just below threshold stays in same combo."""
        result = group_into_combos(make_timeline([
            make_det("jab", 0.0, 0.5),
            make_det("cross", 1.29, 1.4),  # gap = 0.79 < 0.8
        ]), gap_threshold=0.8)
        assert len(result.combos) == 1

    def test_custom_threshold(self):
        """Custom gap_threshold of 2.0s."""
        result = group_into_combos(make_timeline([
            make_det("jab", 0.0, 0.1),
            make_det("cross", 1.5, 1.6),    # gap 1.4 < 2.0 → same combo
            make_det("lead_hook", 4.0, 4.1), # gap 2.4 >= 2.0 → new combo
        ]), gap_threshold=2.0)
        assert len(result.combos) == 2
        assert result.combos[0].actions == ["jab", "cross"]

    def test_returns_combo_sequence(self):
        result = group_into_combos(spec_timeline())
        assert isinstance(result, ComboSequence)

    def test_preserves_video_id_and_fighter_id(self):
        result = group_into_combos(
            make_timeline([make_det("jab", 0.0, 0.1)], video_id="fight_42", fighter_id="fighter_b")
        )
        assert result.video_id == "fight_42"
        assert result.fighter_id == "fighter_b"

    def test_combos_sorted_by_start_time(self):
        """Even if input is unsorted, output combos should be time-ordered."""
        result = group_into_combos(make_timeline([
            make_det("cross", 15.0, 15.1),
            make_det("jab", 12.0, 12.1),
        ]))
        times = [c.start_time for c in result.combos]
        assert times == sorted(times)

    def test_many_separate_combos(self):
        """Ten actions with 5s gaps → 10 separate combos."""
        dets = [make_det("jab", i * 5.0, i * 5.0 + 0.1) for i in range(10)]
        result = group_into_combos(make_timeline(dets))
        assert len(result.combos) == 10

    def test_three_combo_groups(self):
        result = group_into_combos(make_timeline([
            make_det("jab", 0.0, 0.1),
            make_det("cross", 0.2, 0.3),   # same combo
            make_det("jab", 2.0, 2.1),     # new combo (gap 1.7s)
            make_det("cross", 2.2, 2.3),   # same combo
            make_det("lead_hook", 5.0, 5.1), # new combo (gap 2.7s)
        ]))
        assert len(result.combos) == 3

    def test_combo_actions_in_time_order(self):
        """Actions in combo are stored in start_time order."""
        result = group_into_combos(make_timeline([
            make_det("cross", 0.3, 0.4),
            make_det("jab", 0.0, 0.1),
        ]))
        assert result.combos[0].actions == ["jab", "cross"]
