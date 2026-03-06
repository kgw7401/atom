"""Tests for Combo Mastery Updater."""

import pytest

from ml.configs import BoxingConfig
from ml.pipeline.types import ComboStat
from ml.services.mastery_updater import ComboMastery, MasteryUpdater


@pytest.fixture
def config():
    return BoxingConfig()


@pytest.fixture
def updater(config):
    return MasteryUpdater(config)


def _stat(attempts: int = 1, successes: int = 1, name: str = "test") -> ComboStat:
    """Helper to build a ComboStat."""
    return ComboStat(
        combo_name=name,
        combo_key="jab-cross",
        attempts=attempts,
        successes=successes,
        partials=0,
        misses=attempts - successes,
    )


# ── New combo initialization ──

class TestNewCombo:
    def test_new_combo_created(self, updater):
        result = updater.update({}, {"jab-cross": _stat()})
        assert "jab-cross" in result
        assert isinstance(result["jab-cross"], ComboMastery)

    def test_new_combo_starts_learning(self, updater):
        """First attempt transitions new → learning."""
        result = updater.update({}, {"jab-cross": _stat()})
        assert result["jab-cross"].status == "learning"

    def test_initial_rate_set_directly(self, updater):
        """First session: rate = session_rate (no EMA)."""
        stat = _stat(attempts=10, successes=8)
        result = updater.update({}, {"jab-cross": stat})
        assert result["jab-cross"].drill_success_rate == 0.8


# ── EMA update ──

class TestEMA:
    def test_ema_formula(self, updater):
        """EMA: α * new + (1-α) * current, α=0.3."""
        current = {
            "jab-cross": ComboMastery(
                combo_key="jab-cross",
                status="learning",
                drill_success_rate=0.5,
                sessions_attempted=1,
            )
        }
        stat = _stat(attempts=10, successes=10)  # session_rate = 1.0
        result = updater.update(current, {"jab-cross": stat})
        # EMA: 0.3 * 1.0 + 0.7 * 0.5 = 0.65
        assert abs(result["jab-cross"].drill_success_rate - 0.65) < 1e-6

    def test_ema_multiple_sessions(self, updater):
        """Multiple updates apply EMA iteratively."""
        current = {}
        # Session 1: rate = 0.5
        stat1 = _stat(attempts=10, successes=5)
        current = updater.update(current, {"jab-cross": stat1})
        assert abs(current["jab-cross"].drill_success_rate - 0.5) < 1e-6

        # Session 2: rate = 1.0
        stat2 = _stat(attempts=10, successes=10)
        current = updater.update(current, {"jab-cross": stat2})
        # EMA: 0.3 * 1.0 + 0.7 * 0.5 = 0.65
        assert abs(current["jab-cross"].drill_success_rate - 0.65) < 1e-6

        # Session 3: rate = 1.0
        current = updater.update(current, {"jab-cross": stat2})
        # EMA: 0.3 * 1.0 + 0.7 * 0.65 = 0.755
        assert abs(current["jab-cross"].drill_success_rate - 0.755) < 1e-6


# ── Counter updates ──

class TestCounters:
    def test_attempts_accumulate(self, updater):
        stat = _stat(attempts=5, successes=3)
        result = updater.update({}, {"jab-cross": stat})
        assert result["jab-cross"].total_attempts == 5
        assert result["jab-cross"].total_successes == 3

        stat2 = _stat(attempts=10, successes=7)
        result = updater.update(result, {"jab-cross": stat2})
        assert result["jab-cross"].total_attempts == 15
        assert result["jab-cross"].total_successes == 10

    def test_sessions_increment(self, updater):
        current = {}
        for i in range(5):
            current = updater.update(current, {"jab-cross": _stat()})
        assert current["jab-cross"].sessions_attempted == 5


# ── Consecutive tracking ──

class TestConsecutive:
    def test_consecutive_above_threshold(self, updater):
        """Sessions with rate ≥ mastered_rate increment consecutive counter."""
        current = {}
        high_stat = _stat(attempts=10, successes=10)  # rate = 1.0

        for _ in range(3):
            current = updater.update(current, {"jab-cross": high_stat})

        assert current["jab-cross"].consecutive_above_threshold == 3

    def test_consecutive_resets_on_low(self, updater):
        """A session below mastered_rate resets consecutive counter."""
        mastery = ComboMastery(
            combo_key="jab-cross",
            status="proficient",
            drill_success_rate=0.85,
            sessions_attempted=5,
            consecutive_above_threshold=2,
        )
        current = {"jab-cross": mastery}
        low_stat = _stat(attempts=10, successes=3)  # rate = 0.3
        result = updater.update(current, {"jab-cross": low_stat})
        assert result["jab-cross"].consecutive_above_threshold == 0


# ── State transitions ──

class TestStateTransitions:
    def test_new_to_learning(self, updater):
        result = updater.update({}, {"jab-cross": _stat()})
        assert result["jab-cross"].status == "learning"

    def test_learning_stays_learning_few_sessions(self, updater):
        """Need ≥3 sessions for proficient, even with high rate."""
        current = {}
        high_stat = _stat(attempts=10, successes=10)
        current = updater.update(current, {"jab-cross": high_stat})
        assert current["jab-cross"].status == "learning"
        current = updater.update(current, {"jab-cross": high_stat})
        assert current["jab-cross"].status == "learning"

    def test_learning_to_proficient(self, updater):
        """rate ≥ 0.5 AND sessions ≥ 3 → proficient."""
        current = {}
        stat = _stat(attempts=10, successes=8)
        for _ in range(3):
            current = updater.update(current, {"jab-cross": stat})
        assert current["jab-cross"].status == "proficient"

    def test_learning_stays_with_low_rate(self, updater):
        """Even with 3+ sessions, low rate stays learning."""
        current = {}
        low_stat = _stat(attempts=10, successes=2)  # rate = 0.2
        for _ in range(5):
            current = updater.update(current, {"jab-cross": low_stat})
        assert current["jab-cross"].status == "learning"

    def test_proficient_to_mastered(self, updater):
        """rate ≥ 0.8 AND consecutive ≥ 3 → mastered."""
        mastery = ComboMastery(
            combo_key="jab-cross",
            status="proficient",
            drill_success_rate=0.75,
            sessions_attempted=5,
            consecutive_above_threshold=2,
        )
        current = {"jab-cross": mastery}
        high_stat = _stat(attempts=10, successes=10)  # rate = 1.0
        result = updater.update(current, {"jab-cross": high_stat})
        # EMA: 0.3 * 1.0 + 0.7 * 0.75 = 0.825 ≥ 0.8
        # consecutive: 2 + 1 = 3 ≥ 3
        assert result["jab-cross"].status == "mastered"

    def test_mastered_stays_mastered(self, updater):
        """Once mastered, never regresses."""
        mastery = ComboMastery(
            combo_key="jab-cross",
            status="mastered",
            drill_success_rate=0.9,
            sessions_attempted=10,
            consecutive_above_threshold=5,
        )
        current = {"jab-cross": mastery}
        low_stat = _stat(attempts=10, successes=1)  # rate = 0.1
        result = updater.update(current, {"jab-cross": low_stat})
        assert result["jab-cross"].status == "mastered"

    def test_full_progression(self, updater):
        """Complete journey: new → learning → proficient → mastered."""
        current = {}
        high_stat = _stat(attempts=10, successes=10)

        # Session 1: new → learning
        current = updater.update(current, {"jab-cross": high_stat})
        assert current["jab-cross"].status == "learning"

        # Session 2: still learning (need 3 sessions)
        current = updater.update(current, {"jab-cross": high_stat})
        assert current["jab-cross"].status == "learning"

        # Session 3: learning → proficient (rate=1.0 ≥ 0.5, sessions=3)
        current = updater.update(current, {"jab-cross": high_stat})
        assert current["jab-cross"].status == "proficient"

        # Session 4-5: consecutive building
        current = updater.update(current, {"jab-cross": high_stat})
        current = updater.update(current, {"jab-cross": high_stat})
        # consecutive at session 3 became 3 (sessions 1,2,3 all high)
        # but we need to check — depends on when proficient started counting
        # After becoming proficient at session 3: consecutive was already 3
        # Session 4: consecutive = 4, rate still high → should be mastered
        # Actually let's check: at proficient transition, consecutive was already 3
        # So at session 4 check: rate ≥ 0.8 (yes) AND consecutive ≥ 3 (4, yes)
        # It should have become mastered at session 4
        assert current["jab-cross"].status == "mastered"


# ── Multiple combos ──

class TestMultipleCombos:
    def test_independent_tracking(self, updater):
        """Different combos tracked independently."""
        stats = {
            "jab-cross": _stat(attempts=10, successes=10),
            "lead_hook-cross": ComboStat(
                combo_name="hook-cross",
                combo_key="lead_hook-cross",
                attempts=10,
                successes=2,
            ),
        }
        result = updater.update({}, stats)
        assert result["jab-cross"].drill_success_rate == 1.0
        assert result["lead_hook-cross"].drill_success_rate == 0.2

    def test_partial_update(self, updater):
        """Only combos in session_stats are updated."""
        current = {
            "jab-cross": ComboMastery(
                combo_key="jab-cross",
                status="learning",
                drill_success_rate=0.5,
                sessions_attempted=2,
            )
        }
        stats = {"lead_hook-cross": _stat()}
        result = updater.update(current, stats)
        # jab-cross unchanged
        assert result["jab-cross"].sessions_attempted == 2
        # new combo added
        assert "lead_hook-cross" in result


# ── to_dict ──

class TestToDict:
    def test_serialization(self):
        m = ComboMastery(
            combo_key="jab-cross",
            status="learning",
            drill_success_rate=0.12345678,
            total_attempts=5,
            total_successes=3,
            sessions_attempted=2,
            consecutive_above_threshold=1,
        )
        d = m.to_dict()
        assert d["combo_key"] == "jab-cross"
        assert d["status"] == "learning"
        assert d["drill_success_rate"] == 0.1235  # rounded to 4 decimals
        assert d["total_attempts"] == 5
        assert d["sessions_attempted"] == 2
