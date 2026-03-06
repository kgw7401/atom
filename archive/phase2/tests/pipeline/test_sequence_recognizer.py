"""Tests for Pipeline Stage 3: Sequence Recognition."""

import pytest

from ml.configs import BoxingConfig
from ml.pipeline.sequence_recognizer import SequenceRecognizer
from ml.pipeline.types import DetectedAction


@pytest.fixture
def config():
    return BoxingConfig()


@pytest.fixture
def recognizer(config):
    return SequenceRecognizer(config)


def _action(ts: float, action: str = "jab", conf: float = 0.9) -> DetectedAction:
    """Helper to create a DetectedAction."""
    return DetectedAction(
        timestamp=ts,
        action=action,
        confidence=conf,
        window_start=ts - 0.5,
        window_end=ts + 0.5,
    )


# ── Empty input ──

class TestEmpty:
    def test_empty_list(self, recognizer):
        assert recognizer.recognize([]) == []


# ── Single action ──

class TestSingleAction:
    def test_single_action_becomes_single_combo(self, recognizer):
        actions = [_action(1.0, "jab")]
        combos = recognizer.recognize(actions)
        assert len(combos) == 1
        assert combos[0].actions == ["jab"]
        assert combos[0].start_time == 1.0
        assert combos[0].end_time == 1.0

    def test_single_combo_key(self, recognizer):
        actions = [_action(1.0, "cross")]
        combos = recognizer.recognize(actions)
        assert combos[0].key == "cross"


# ── Grouping by time gap ──

class TestGrouping:
    def test_two_actions_within_gap_same_combo(self, recognizer):
        """Actions within combo_gap_threshold (0.8s) → same combo."""
        actions = [_action(1.0, "jab"), _action(1.5, "cross")]
        combos = recognizer.recognize(actions)
        assert len(combos) == 1
        assert combos[0].actions == ["jab", "cross"]

    def test_three_action_combo(self, recognizer):
        actions = [
            _action(1.0, "jab"),
            _action(1.3, "cross"),
            _action(1.6, "lead_hook"),
        ]
        combos = recognizer.recognize(actions)
        assert len(combos) == 1
        assert combos[0].actions == ["jab", "cross", "lead_hook"]
        assert combos[0].key == "jab-cross-lead_hook"

    def test_gap_at_threshold_splits(self, recognizer):
        """Gap exactly at threshold (0.8s) → separate combos."""
        actions = [_action(1.0, "jab"), _action(1.8, "cross")]
        combos = recognizer.recognize(actions)
        assert len(combos) == 2

    def test_gap_just_below_threshold_same(self, recognizer):
        """Gap just below threshold → same combo."""
        actions = [_action(1.0, "jab"), _action(1.79, "cross")]
        combos = recognizer.recognize(actions)
        assert len(combos) == 1

    def test_large_gap_splits(self, recognizer):
        """Gap >> threshold → separate combos."""
        actions = [_action(1.0, "jab"), _action(5.0, "cross")]
        combos = recognizer.recognize(actions)
        assert len(combos) == 2
        assert combos[0].actions == ["jab"]
        assert combos[1].actions == ["cross"]

    def test_multiple_combos(self, recognizer):
        """Multiple combos separated by gaps."""
        actions = [
            _action(1.0, "jab"),
            _action(1.3, "cross"),
            # gap
            _action(5.0, "lead_hook"),
            _action(5.3, "cross"),
            _action(5.6, "lead_hook"),
        ]
        combos = recognizer.recognize(actions)
        assert len(combos) == 2
        assert combos[0].actions == ["jab", "cross"]
        assert combos[1].actions == ["lead_hook", "cross", "lead_hook"]


# ── Sorting ──

class TestSorting:
    def test_unsorted_input_sorted_output(self, recognizer):
        """Input doesn't need to be sorted; output is sorted by start_time."""
        actions = [
            _action(5.0, "cross"),
            _action(1.0, "jab"),
            _action(1.3, "cross"),
        ]
        combos = recognizer.recognize(actions)
        assert len(combos) == 2
        assert combos[0].start_time == 1.0
        assert combos[1].start_time == 5.0

    def test_reverse_order_input(self, recognizer):
        actions = [
            _action(3.0, "lead_hook"),
            _action(2.5, "cross"),
            _action(2.0, "jab"),
        ]
        combos = recognizer.recognize(actions)
        assert len(combos) == 1
        assert combos[0].actions == ["jab", "cross", "lead_hook"]


# ── Combo properties ──

class TestComboProperties:
    def test_start_end_times(self, recognizer):
        actions = [
            _action(1.0, "jab"),
            _action(1.3, "cross"),
            _action(1.6, "lead_hook"),
        ]
        combos = recognizer.recognize(actions)
        assert combos[0].start_time == 1.0
        assert combos[0].end_time == 1.6

    def test_combo_key_format(self, recognizer):
        actions = [
            _action(1.0, "jab"),
            _action(1.3, "cross"),
            _action(1.6, "lead_hook"),
        ]
        combos = recognizer.recognize(actions)
        assert combos[0].key == "jab-cross-lead_hook"


# ── Custom config ──

class TestCustomConfig:
    def test_custom_gap_threshold(self):
        """Custom gap threshold changes grouping behavior."""
        raw = BoxingConfig()._raw.copy()
        raw["pipeline"] = dict(raw["pipeline"])
        raw["pipeline"]["combo_gap_threshold"] = 2.0
        config = BoxingConfig(raw=raw)
        rec = SequenceRecognizer(config)

        # 1.5s gap — would split with default 0.8, but not with 2.0
        actions = [_action(1.0, "jab"), _action(2.5, "cross")]
        combos = rec.recognize(actions)
        assert len(combos) == 1


# ── Edge cases ──

class TestEdgeCases:
    def test_same_timestamp(self, recognizer):
        """Two actions at exact same time → same combo."""
        actions = [_action(1.0, "jab"), _action(1.0, "cross")]
        combos = recognizer.recognize(actions)
        assert len(combos) == 1

    def test_many_single_combos(self, recognizer):
        """Many widely spaced actions → many single-action combos."""
        actions = [_action(i * 10.0, "jab") for i in range(10)]
        combos = recognizer.recognize(actions)
        assert len(combos) == 10
        for c in combos:
            assert len(c.actions) == 1

    def test_long_combo(self, recognizer):
        """Many actions in rapid succession → one long combo."""
        actions = [_action(1.0 + i * 0.2, "jab") for i in range(20)]
        combos = recognizer.recognize(actions)
        assert len(combos) == 1
        assert len(combos[0].actions) == 20
