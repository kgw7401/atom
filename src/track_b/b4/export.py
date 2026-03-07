"""B4 Task 16: DrillTemplate export to Track A.

Converts SituationalTactics into DrillTemplates compatible with
Track A's Combo Registry. Unknown actions are flagged for user review.
All exports are user-curated (no auto-import v1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from track_b.b4.gemini_client import SituationalTactic


# ── Track A vocabulary ────────────────────────────────────────────────────────

TRACK_A_ACTIONS: frozenset[str] = frozenset({
    # Core punches (BoxingVI)
    "jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut",
    # Defensive (future)
    "slip", "duck", "block",
    # Extended (Olympic Boxing additions)
    "body_jab", "body_cross", "body_hook",
})

# Aliases: maps common variant spellings → canonical Track A action names
ACTION_ALIASES: dict[str, str] = {
    "jab": "jab",
    "cross": "cross",
    "hook": "lead_hook",
    "lead_hook": "lead_hook",
    "rear_hook": "rear_hook",
    "uppercut": "lead_uppercut",
    "lead_uppercut": "lead_uppercut",
    "rear_uppercut": "rear_uppercut",
    "slip": "slip",
    "duck": "duck",
    "block": "block",
    "body_jab": "body_jab",
    "body_cross": "body_cross",
    "body_hook": "body_hook",
}


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class DrillTemplate:
    """A drill template compatible with Track A's Combo Registry."""

    source: str = "track_b_analysis"
    source_video_id: str = ""
    scenario: str = ""              # TTS-ready: "Opponent jab-cross!"
    target_combo: list[str] = field(default_factory=list)
    combo_name: str = ""
    context_note: str = ""
    unknown_actions: list[str] = field(default_factory=list)  # flagged for review
    is_complete: bool = True        # False if unknown_actions is non-empty


@dataclass
class ExportResult:
    """Result of exporting a SituationalTactic to a DrillTemplate."""

    template: DrillTemplate
    warnings: list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_scenario(situation: str) -> str:
    """Convert a situation description to a TTS-ready scenario string."""
    situation = situation.strip()
    if not situation:
        return "Opponent attacks!"
    if not situation.endswith("!"):
        situation = situation + "!"
    return situation


def _parse_actions_from_text(text: str) -> list[str]:
    """Parse action words from a response string using alias lookup.

    Tokenizes the text and matches tokens to Track A action aliases.
    Returns known actions in order of appearance, deduplicated.

    Args:
        text: Natural-language response description.

    Returns:
        List of canonical Track A action names in order of appearance.
    """
    # Normalize: lowercase, replace hyphens with underscores for compound names
    normalized = text.lower().replace("-", "_")
    # Tokenize: split on whitespace and punctuation
    tokens = []
    for word in normalized.split():
        tokens.append(word.strip("!.,;:()"))

    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        canonical = ACTION_ALIASES.get(token)
        if canonical and canonical not in seen:
            result.append(canonical)
            seen.add(canonical)

    return result


# ── Export ────────────────────────────────────────────────────────────────────


def tactic_to_drill_template(tactic: SituationalTactic) -> ExportResult:
    """Convert a SituationalTactic to a DrillTemplate.

    Parses the effective_response for known Track A actions.
    Unknown actions are flagged for user review.

    Args:
        tactic: SituationalTactic from Gemini analysis.

    Returns:
        ExportResult with the DrillTemplate and any warnings.
    """
    target_actions = _parse_actions_from_text(tactic.effective_response)

    unknown_actions = [a for a in target_actions if a not in TRACK_A_ACTIONS]
    known_actions = [a for a in target_actions if a in TRACK_A_ACTIONS]

    warnings: list[str] = []
    if not known_actions:
        warnings.append(
            f"Could not parse actions from response: '{tactic.effective_response}'"
        )
    for action in unknown_actions:
        warnings.append(f"Unknown action '{action}' not in Track A vocabulary")

    combo_name = (
        " ".join(a.replace("_", " ").title() for a in known_actions)
        if known_actions
        else "Unknown Combo"
    )

    context_note = (
        f"Effective counter to: {tactic.situation} "
        f"(observed {tactic.frequency}x, ~{tactic.success_rate:.0%} success)"
    )

    template = DrillTemplate(
        source="track_b_analysis",
        source_video_id=tactic.video_id,
        scenario=_make_scenario(tactic.situation),
        target_combo=known_actions,
        combo_name=combo_name,
        context_note=context_note,
        unknown_actions=unknown_actions,
        is_complete=len(unknown_actions) == 0 and len(known_actions) > 0,
    )
    return ExportResult(template=template, warnings=warnings)


def export_tactics_to_templates(
    tactics: list[SituationalTactic],
    min_frequency: int = 1,
) -> list[ExportResult]:
    """Convert a list of SituationalTactics to DrillTemplates.

    Args:
        tactics: Tactics from fight analysis.
        min_frequency: Minimum frequency to include (default 1).

    Returns:
        List of ExportResult (template + warnings), one per qualifying tactic.
        User must review all templates before importing to Track A.
    """
    results = []
    for tactic in tactics:
        if tactic.frequency < min_frequency:
            continue
        results.append(tactic_to_drill_template(tactic))
    return results
