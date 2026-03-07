"""B3 Task 12: DrillFeedback generation.

Aggregates SessionMatch results into structured DrillFeedback.
Per-combo statistics, overall accuracy, missed actions, and actionable notes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from track_b.b3.matching import InstructionMatch, SessionMatch


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class ComboStats:
    """Per-combo statistics."""

    attempts: int = 0
    successes: int = 0
    partials: int = 0
    misses: int = 0

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts


@dataclass
class DrillFeedback:
    """Aggregated drill feedback for Track A's LLM context."""

    session_id: str
    combo_stats: dict[str, ComboStats] = field(default_factory=dict)
    overall_accuracy: float = 0.0
    missed_actions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _combo_key(combo_actions: list[str]) -> str:
    """Create a human-readable key from combo action list."""
    return "-".join(combo_actions)


def _find_missed_actions(matches: list[InstructionMatch]) -> list[str]:
    """Find actions that are frequently missed across instructions.

    Returns actions sorted by miss frequency descending.
    """
    miss_count: dict[str, int] = defaultdict(int)
    for m in matches:
        if m.result in ("partial", "miss"):
            detected_set = set(m.detected_combo)
            for action in m.instructed_combo:
                if action not in detected_set:
                    miss_count[action] += 1

    return sorted(miss_count, key=lambda a: miss_count[a], reverse=True)


def _generate_notes(
    combo_stats: dict[str, ComboStats],
    missed_actions: list[str],
    overall_accuracy: float,
) -> list[str]:
    """Generate actionable notes from session statistics."""
    notes: list[str] = []

    # Note: low success combos
    for combo_name, stats in combo_stats.items():
        if stats.attempts >= 2 and stats.success_rate < 0.5:
            notes.append(
                f"Low success rate on {combo_name} "
                f"({stats.success_rate:.0%} of {stats.attempts} attempts). "
                f"Focus on completing the full combo."
            )

    # Note: frequently missed individual actions (top 3)
    for action in missed_actions[:3]:
        notes.append(f"Action '{action}' is frequently missed. Drill it in isolation.")

    # Note: overall performance
    if overall_accuracy >= 0.8:
        notes.append("Strong overall performance. Ready to increase combo complexity.")
    elif overall_accuracy >= 0.5:
        notes.append("Moderate performance. Consistent practice will improve completion rates.")
    else:
        notes.append(
            "Accuracy is below 50%. Slow down and focus on clean execution before speed."
        )

    return notes


# ── Main function ─────────────────────────────────────────────────────────────


def generate_feedback(session_match: SessionMatch) -> DrillFeedback:
    """Generate DrillFeedback from a SessionMatch.

    Args:
        session_match: SessionMatch from match_session().

    Returns:
        DrillFeedback with per-combo stats, accuracy, missed actions, and notes.
    """
    combo_stats: dict[str, ComboStats] = {}
    total_attempts = 0
    total_successes = 0

    for m in session_match.matches:
        key = _combo_key(m.instructed_combo)
        if key not in combo_stats:
            combo_stats[key] = ComboStats()
        stats = combo_stats[key]
        stats.attempts += 1
        total_attempts += 1

        if m.result == "success":
            stats.successes += 1
            total_successes += 1
        elif m.result == "partial":
            stats.partials += 1
        else:
            stats.misses += 1

    overall_accuracy = total_successes / total_attempts if total_attempts > 0 else 0.0
    missed_actions = _find_missed_actions(session_match.matches)
    notes = _generate_notes(combo_stats, missed_actions, overall_accuracy)

    return DrillFeedback(
        session_id=session_match.session_id,
        combo_stats=combo_stats,
        overall_accuracy=overall_accuracy,
        missed_actions=missed_actions,
        notes=notes,
    )
