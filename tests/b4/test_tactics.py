"""Tests for B4 Task 15: Situational tactic extraction."""

import json

import pytest

from track_b.b2.tad import ActionDetection, ActionTimeline
from track_b.b4.gemini_client import GeminiAnalysisClient, SituationalTactic
from track_b.b4.tactics import (
    FightAnalysisRequest,
    FightAnalysisResult,
    analyze_fight_tactics,
    filter_actionable_tactics,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_timeline(
    actions: list[tuple[str, float, float]],
    fighter_id: str = "fighter_a",
) -> ActionTimeline:
    return ActionTimeline(
        video_id="v1",
        fighter_id=fighter_id,
        actions=[
            ActionDetection(start_time=s, end_time=e, action_class=a, confidence=0.9)
            for a, s, e in actions
        ],
    )


def make_tactic(
    situation: str = "Opponent throws jab",
    response: str = "counter with cross",
    frequency: int = 3,
    success_rate: float = 0.7,
) -> SituationalTactic:
    return SituationalTactic(
        video_id="v1",
        situation=situation,
        effective_response=response,
        frequency=frequency,
        success_rate=success_rate,
    )


class MockClient:
    def __init__(self, tactics: list[SituationalTactic]) -> None:
        self._tactics = tactics

    def analyze_fight(
        self,
        fighter_a_timeline: ActionTimeline,
        fighter_b_timeline: ActionTimeline,
        video_id: str,
    ) -> list[SituationalTactic]:
        return self._tactics


# ── FightAnalysisResult ───────────────────────────────────────────────────────


class TestFightAnalysisResult:
    def test_top_tactics_sorted_by_frequency(self):
        tactics = [
            make_tactic(frequency=2),
            make_tactic(frequency=5, situation="B"),
            make_tactic(frequency=1, situation="C"),
        ]
        result = FightAnalysisResult(video_id="v1", tactics=tactics)
        freqs = [t.frequency for t in result.top_tactics]
        assert freqs == sorted(freqs, reverse=True)

    def test_high_confidence_tactics(self):
        tactics = [
            make_tactic(success_rate=0.8),
            make_tactic(success_rate=0.4, situation="B"),
            make_tactic(success_rate=0.6, situation="C"),
        ]
        result = FightAnalysisResult(video_id="v1", tactics=tactics)
        high = result.high_confidence_tactics
        assert all(t.success_rate >= 0.6 for t in high)
        assert len(high) == 2

    def test_empty_tactics(self):
        result = FightAnalysisResult(video_id="v1", tactics=[])
        assert result.top_tactics == []
        assert result.high_confidence_tactics == []


# ── analyze_fight_tactics ─────────────────────────────────────────────────────


class TestAnalyzeFightTactics:
    def test_returns_fight_analysis_result(self):
        mock = MockClient([make_tactic()])
        request = FightAnalysisRequest(
            video_id="fight_1",
            fighter_a_timeline=make_timeline([("jab", 1.0, 1.1)]),
            fighter_b_timeline=make_timeline([("cross", 1.3, 1.4)], fighter_id="fighter_b"),
        )
        result = analyze_fight_tactics(request, mock)
        assert isinstance(result, FightAnalysisResult)

    def test_video_id_preserved(self):
        mock = MockClient([])
        request = FightAnalysisRequest(
            video_id="my_fight",
            fighter_a_timeline=make_timeline([]),
            fighter_b_timeline=make_timeline([], fighter_id="fighter_b"),
        )
        result = analyze_fight_tactics(request, mock)
        assert result.video_id == "my_fight"

    def test_tactics_forwarded(self):
        tactics = [make_tactic(), make_tactic(situation="B")]
        mock = MockClient(tactics)
        request = FightAnalysisRequest(
            video_id="v1",
            fighter_a_timeline=make_timeline([]),
            fighter_b_timeline=make_timeline([], fighter_id="fighter_b"),
        )
        result = analyze_fight_tactics(request, mock)
        assert len(result.tactics) == 2

    def test_empty_tactics_on_no_patterns(self):
        mock = MockClient([])
        request = FightAnalysisRequest(
            video_id="v1",
            fighter_a_timeline=make_timeline([]),
            fighter_b_timeline=make_timeline([], fighter_id="fighter_b"),
        )
        result = analyze_fight_tactics(request, mock)
        assert result.tactics == []


# ── filter_actionable_tactics ─────────────────────────────────────────────────


class TestFilterActionableTactics:
    def test_filters_by_frequency(self):
        tactics = [
            make_tactic(frequency=1),
            make_tactic(frequency=3, situation="B"),
        ]
        result = filter_actionable_tactics(tactics, min_frequency=2)
        assert len(result) == 1
        assert result[0].frequency == 3

    def test_filters_by_success_rate(self):
        tactics = [
            make_tactic(success_rate=0.3),
            make_tactic(success_rate=0.7, situation="B"),
        ]
        result = filter_actionable_tactics(tactics, min_success_rate=0.5)
        assert len(result) == 1
        assert result[0].success_rate == pytest.approx(0.7)

    def test_sorted_by_frequency(self):
        tactics = [
            make_tactic(frequency=2),
            make_tactic(frequency=5, situation="B"),
            make_tactic(frequency=3, situation="C"),
        ]
        result = filter_actionable_tactics(tactics, min_frequency=1)
        freqs = [t.frequency for t in result]
        assert freqs == sorted(freqs, reverse=True)

    def test_all_pass_filter(self):
        tactics = [make_tactic(frequency=3, success_rate=0.8) for _ in range(3)]
        result = filter_actionable_tactics(tactics, min_frequency=2, min_success_rate=0.5)
        assert len(result) == 3

    def test_none_pass_filter(self):
        tactics = [make_tactic(frequency=1, success_rate=0.3)]
        result = filter_actionable_tactics(tactics, min_frequency=5, min_success_rate=0.9)
        assert result == []

    def test_empty_input(self):
        result = filter_actionable_tactics([])
        assert result == []
