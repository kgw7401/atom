"""Tests for B4 Task 14: Gemini 2.5 Pro integration."""

import json

import pytest

from track_b.b2.tad import ActionDetection, ActionTimeline
from track_b.b4.gemini_client import (
    GeminiAnalysisClient,
    SituationalTactic,
    build_analysis_prompt,
    parse_tactics_response,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_timeline(
    actions: list[tuple[str, float, float]],
    video_id: str = "v1",
    fighter_id: str = "fighter_a",
) -> ActionTimeline:
    return ActionTimeline(
        video_id=video_id,
        fighter_id=fighter_id,
        actions=[
            ActionDetection(start_time=s, end_time=e, action_class=a, confidence=0.9)
            for a, s, e in actions
        ],
    )


def make_tactic_json(
    situation: str = "Opponent throws jab-cross",
    response: str = "Slip outside, counter with cross-hook",
    frequency: int = 3,
    success_rate: float = 0.7,
    evidence: list | None = None,
) -> dict:
    return {
        "situation": situation,
        "effective_response": response,
        "frequency": frequency,
        "success_rate": success_rate,
        "evidence": evidence or [{"timestamp": 10.5, "outcome": "landed"}],
    }


class MockGeminiClient:
    """Mock Gemini client that returns preset responses."""

    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response


# ── SituationalTactic ─────────────────────────────────────────────────────────


class TestSituationalTactic:
    def test_fields(self):
        t = SituationalTactic(
            video_id="v1",
            situation="Opponent jab-cross",
            effective_response="Slip, counter cross",
            frequency=3,
            success_rate=0.7,
        )
        assert t.video_id == "v1"
        assert t.frequency == 3
        assert t.success_rate == pytest.approx(0.7)

    def test_evidence_default_empty(self):
        t = SituationalTactic(
            video_id="v1",
            situation="x",
            effective_response="y",
            frequency=1,
            success_rate=0.5,
        )
        assert t.evidence == []


# ── build_analysis_prompt ─────────────────────────────────────────────────────


class TestBuildAnalysisPrompt:
    def test_returns_string(self):
        a = make_timeline([("jab", 1.0, 1.1)])
        b = make_timeline([("cross", 1.2, 1.3)], fighter_id="fighter_b")
        result = build_analysis_prompt(a, b)
        assert isinstance(result, str)

    def test_contains_fighter_ids(self):
        a = make_timeline([("jab", 1.0, 1.1)], fighter_id="fighter_a")
        b = make_timeline([("cross", 1.2, 1.3)], fighter_id="fighter_b")
        result = build_analysis_prompt(a, b)
        assert "fighter_a" in result
        assert "fighter_b" in result

    def test_contains_action_names(self):
        a = make_timeline([("jab", 1.0, 1.1)])
        b = make_timeline([("cross", 2.0, 2.1)], fighter_id="fighter_b")
        result = build_analysis_prompt(a, b)
        assert "jab" in result
        assert "cross" in result

    def test_contains_json_instruction(self):
        a = make_timeline([])
        b = make_timeline([], fighter_id="fighter_b")
        result = build_analysis_prompt(a, b)
        assert "JSON" in result

    def test_empty_timeline_handled(self):
        a = make_timeline([])
        b = make_timeline([], fighter_id="fighter_b")
        result = build_analysis_prompt(a, b)
        assert "No actions detected" in result


# ── parse_tactics_response ────────────────────────────────────────────────────


class TestParseTacticsResponse:
    def test_parses_valid_json(self):
        data = [make_tactic_json()]
        result = parse_tactics_response(json.dumps(data), "v1")
        assert len(result) == 1

    def test_returns_situational_tactic_type(self):
        data = [make_tactic_json()]
        result = parse_tactics_response(json.dumps(data), "v1")
        assert isinstance(result[0], SituationalTactic)

    def test_video_id_attached(self):
        data = [make_tactic_json()]
        result = parse_tactics_response(json.dumps(data), "fight_42")
        assert result[0].video_id == "fight_42"

    def test_situation_parsed(self):
        data = [make_tactic_json(situation="Opponent jabs")]
        result = parse_tactics_response(json.dumps(data), "v1")
        assert result[0].situation == "Opponent jabs"

    def test_frequency_parsed(self):
        data = [make_tactic_json(frequency=5)]
        result = parse_tactics_response(json.dumps(data), "v1")
        assert result[0].frequency == 5

    def test_success_rate_parsed(self):
        data = [make_tactic_json(success_rate=0.8)]
        result = parse_tactics_response(json.dumps(data), "v1")
        assert result[0].success_rate == pytest.approx(0.8)

    def test_multiple_tactics(self):
        data = [make_tactic_json(), make_tactic_json(situation="Opponent hooks")]
        result = parse_tactics_response(json.dumps(data), "v1")
        assert len(result) == 2

    def test_empty_array(self):
        result = parse_tactics_response("[]", "v1")
        assert result == []

    def test_invalid_json_returns_empty(self):
        result = parse_tactics_response("not json at all", "v1")
        assert result == []

    def test_non_array_returns_empty(self):
        result = parse_tactics_response('{"situation": "x"}', "v1")
        assert result == []

    def test_strips_markdown_code_block(self):
        data = [make_tactic_json()]
        wrapped = f"```json\n{json.dumps(data)}\n```"
        result = parse_tactics_response(wrapped, "v1")
        assert len(result) == 1

    def test_strips_plain_code_block(self):
        data = [make_tactic_json()]
        wrapped = f"```\n{json.dumps(data)}\n```"
        result = parse_tactics_response(wrapped, "v1")
        assert len(result) == 1

    def test_skips_malformed_items(self):
        data = [make_tactic_json(), "not a dict", {"bad": "item"}]
        result = parse_tactics_response(json.dumps(data), "v1")
        # At least the valid one parsed; malformed ones skipped
        assert len(result) >= 1


# ── GeminiAnalysisClient ──────────────────────────────────────────────────────


class TestGeminiAnalysisClient:
    def test_analyze_fight_returns_list(self):
        data = [make_tactic_json()]
        mock = MockGeminiClient(json.dumps(data))
        client = GeminiAnalysisClient(mock_client=mock)
        a = make_timeline([("jab", 1.0, 1.1)])
        b = make_timeline([("cross", 1.3, 1.4)], fighter_id="fighter_b")
        result = client.analyze_fight(a, b, "v1")
        assert isinstance(result, list)

    def test_analyze_fight_returns_situational_tactics(self):
        data = [make_tactic_json()]
        mock = MockGeminiClient(json.dumps(data))
        client = GeminiAnalysisClient(mock_client=mock)
        a = make_timeline([("jab", 1.0, 1.1)])
        b = make_timeline([("cross", 1.3, 1.4)], fighter_id="fighter_b")
        result = client.analyze_fight(a, b, "v1")
        assert all(isinstance(t, SituationalTactic) for t in result)

    def test_analyze_fight_passes_video_id(self):
        data = [make_tactic_json()]
        mock = MockGeminiClient(json.dumps(data))
        client = GeminiAnalysisClient(mock_client=mock)
        a = make_timeline([])
        b = make_timeline([], fighter_id="fighter_b")
        result = client.analyze_fight(a, b, "my_fight")
        assert result[0].video_id == "my_fight"

    def test_analyze_fight_empty_response(self):
        mock = MockGeminiClient("[]")
        client = GeminiAnalysisClient(mock_client=mock)
        a = make_timeline([])
        b = make_timeline([], fighter_id="fighter_b")
        result = client.analyze_fight(a, b, "v1")
        assert result == []

    def test_analyze_fight_multiple_tactics(self):
        data = [make_tactic_json(), make_tactic_json(situation="Opponent hooks")]
        mock = MockGeminiClient(json.dumps(data))
        client = GeminiAnalysisClient(mock_client=mock)
        a = make_timeline([("jab", 1.0, 1.1)])
        b = make_timeline([("cross", 1.3, 1.4)], fighter_id="fighter_b")
        result = client.analyze_fight(a, b, "v1")
        assert len(result) == 2
