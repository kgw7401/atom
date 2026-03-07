"""Tests for B4 Task 16: DrillTemplate export to Track A."""

import pytest

from track_b.b4.export import (
    ACTION_ALIASES,
    TRACK_A_ACTIONS,
    DrillTemplate,
    ExportResult,
    _make_scenario,
    _parse_actions_from_text,
    export_tactics_to_templates,
    tactic_to_drill_template,
)
from track_b.b4.gemini_client import SituationalTactic


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_tactic(
    situation: str = "Opponent throws jab-cross",
    response: str = "slip outside, counter with cross hook",
    frequency: int = 3,
    success_rate: float = 0.7,
    video_id: str = "fight_1",
) -> SituationalTactic:
    return SituationalTactic(
        video_id=video_id,
        situation=situation,
        effective_response=response,
        frequency=frequency,
        success_rate=success_rate,
    )


# ── TRACK_A_ACTIONS ───────────────────────────────────────────────────────────


class TestTrackAActions:
    def test_contains_core_punches(self):
        for action in ("jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut"):
            assert action in TRACK_A_ACTIONS

    def test_contains_defensive(self):
        assert "slip" in TRACK_A_ACTIONS
        assert "block" in TRACK_A_ACTIONS

    def test_is_frozenset(self):
        assert isinstance(TRACK_A_ACTIONS, frozenset)


# ── _make_scenario ────────────────────────────────────────────────────────────


class TestMakeScenario:
    def test_adds_exclamation(self):
        result = _make_scenario("Opponent throws jab-cross")
        assert result.endswith("!")

    def test_no_double_exclamation(self):
        result = _make_scenario("Opponent jabs!")
        assert result == "Opponent jabs!"

    def test_empty_returns_default(self):
        result = _make_scenario("")
        assert "Opponent" in result and result.endswith("!")

    def test_strips_whitespace(self):
        result = _make_scenario("  Opponent jabs  ")
        assert not result.startswith(" ")


# ── _parse_actions_from_text ──────────────────────────────────────────────────


class TestParseActionsFromText:
    def test_parses_jab(self):
        result = _parse_actions_from_text("throw a jab")
        assert "jab" in result

    def test_parses_cross(self):
        result = _parse_actions_from_text("follow with cross")
        assert "cross" in result

    def test_parses_hook_to_lead_hook(self):
        result = _parse_actions_from_text("counter with hook")
        assert "lead_hook" in result

    def test_parses_slip(self):
        result = _parse_actions_from_text("slip outside")
        assert "slip" in result

    def test_parses_multiple_actions(self):
        result = _parse_actions_from_text("slip, cross, hook")
        assert "slip" in result
        assert "cross" in result
        assert "lead_hook" in result

    def test_deduplicates(self):
        result = _parse_actions_from_text("jab jab jab")
        assert result.count("jab") == 1

    def test_unknown_words_ignored(self):
        result = _parse_actions_from_text("move forward aggressively")
        # None of these are Track A actions
        assert all(a in TRACK_A_ACTIONS for a in result)

    def test_empty_string(self):
        result = _parse_actions_from_text("")
        assert result == []

    def test_hyphenated_lead_hook(self):
        result = _parse_actions_from_text("lead-hook counter")
        assert "lead_hook" in result


# ── tactic_to_drill_template ──────────────────────────────────────────────────


class TestTacticToDrillTemplate:
    def test_returns_export_result(self):
        result = tactic_to_drill_template(make_tactic())
        assert isinstance(result, ExportResult)

    def test_template_type(self):
        result = tactic_to_drill_template(make_tactic())
        assert isinstance(result.template, DrillTemplate)

    def test_source_is_track_b(self):
        result = tactic_to_drill_template(make_tactic())
        assert result.template.source == "track_b_analysis"

    def test_source_video_id(self):
        result = tactic_to_drill_template(make_tactic(video_id="fight_42"))
        assert result.template.source_video_id == "fight_42"

    def test_scenario_contains_situation(self):
        result = tactic_to_drill_template(make_tactic(situation="Opponent jabs"))
        assert "Opponent jabs" in result.template.scenario

    def test_scenario_ends_exclamation(self):
        result = tactic_to_drill_template(make_tactic())
        assert result.template.scenario.endswith("!")

    def test_target_combo_nonempty_for_known_response(self):
        result = tactic_to_drill_template(make_tactic(response="slip then cross"))
        assert len(result.template.target_combo) >= 1

    def test_target_combo_contains_known_actions(self):
        result = tactic_to_drill_template(make_tactic(response="slip then cross"))
        assert all(a in TRACK_A_ACTIONS for a in result.template.target_combo)

    def test_context_note_contains_frequency(self):
        result = tactic_to_drill_template(make_tactic(frequency=4))
        assert "4" in result.template.context_note

    def test_context_note_contains_success_rate(self):
        result = tactic_to_drill_template(make_tactic(success_rate=0.8))
        assert "80%" in result.template.context_note

    def test_is_complete_true_when_all_known(self):
        result = tactic_to_drill_template(make_tactic(response="jab cross"))
        assert result.template.is_complete is True

    def test_is_complete_false_when_no_actions(self):
        result = tactic_to_drill_template(make_tactic(response="move and evade rapidly"))
        assert result.template.is_complete is False

    def test_warnings_list(self):
        result = tactic_to_drill_template(make_tactic())
        assert isinstance(result.warnings, list)

    def test_warning_on_no_parseable_actions(self):
        result = tactic_to_drill_template(make_tactic(response="move forward quickly"))
        assert len(result.warnings) >= 1

    def test_combo_name_nonempty_for_known_combo(self):
        result = tactic_to_drill_template(make_tactic(response="jab cross"))
        assert result.template.combo_name != "" and result.template.combo_name != "Unknown Combo"


# ── export_tactics_to_templates ───────────────────────────────────────────────


class TestExportTacticsToTemplates:
    def test_returns_list(self):
        tactics = [make_tactic(), make_tactic(situation="B")]
        result = export_tactics_to_templates(tactics)
        assert isinstance(result, list)

    def test_one_per_tactic(self):
        tactics = [make_tactic(), make_tactic(situation="B")]
        result = export_tactics_to_templates(tactics)
        assert len(result) == 2

    def test_all_export_results(self):
        tactics = [make_tactic()]
        result = export_tactics_to_templates(tactics)
        assert all(isinstance(r, ExportResult) for r in result)

    def test_min_frequency_filters(self):
        tactics = [
            make_tactic(frequency=1),
            make_tactic(frequency=3, situation="B"),
        ]
        result = export_tactics_to_templates(tactics, min_frequency=2)
        assert len(result) == 1

    def test_empty_input(self):
        result = export_tactics_to_templates([])
        assert result == []

    def test_all_filtered_returns_empty(self):
        tactics = [make_tactic(frequency=1)]
        result = export_tactics_to_templates(tactics, min_frequency=5)
        assert result == []
