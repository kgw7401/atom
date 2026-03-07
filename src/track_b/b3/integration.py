"""B3 Task 13: Track A data integration interface.

Defines the interface between DrillFeedback (Track B) and Track A's
UserProfile / LLM session planner.

This is an interface definition. Actual wiring happens when A1's data model
is finalized. B3 can be built and tested with mock Track A schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from track_b.b3.feedback import DrillFeedback


# ── Data contracts ─────────────────────────────────────────────────────────────


@dataclass
class ComboMasteryUpdate:
    """Update to a combo's mastery record in Track A's UserProfile.

    Maps DrillFeedback stats to Track A's combo_mastery field.
    """

    combo_name: str      # e.g., "jab-cross-lead_hook"
    success_rate: float  # 0.0 – 1.0
    attempts: int        # total attempts this session
    last_result: str     # "success" / "partial" / "miss" (most recent)


@dataclass
class SessionContext:
    """Formatted context for Track A's LLM session planner.

    This is passed into the LLM's system prompt / context so it can tailor
    the next drill session based on user performance.
    """

    session_id: str
    summary: str                  # plain-text performance summary
    weak_combos: list[str]        # combos below 50% success rate
    strong_combos: list[str]      # combos above 80% success rate
    missed_actions: list[str]     # most frequently missed actions
    suggested_focus: list[str]    # actionable suggestions for next session
    overall_accuracy: float


# ── Conversion functions ───────────────────────────────────────────────────────


def drill_feedback_to_profile_updates(
    feedback: DrillFeedback,
) -> list[ComboMasteryUpdate]:
    """Convert DrillFeedback to a list of UserProfile combo_mastery updates.

    Args:
        feedback: DrillFeedback from generate_feedback().

    Returns:
        List of ComboMasteryUpdate, one per combo attempted this session.
    """
    updates: list[ComboMasteryUpdate] = []
    for combo_name, stats in feedback.combo_stats.items():
        if stats.attempts == 0:
            continue

        # Determine last result: success > partial > miss
        if stats.successes > 0 and stats.successes >= stats.misses:
            last_result = "success"
        elif stats.partials > 0:
            last_result = "partial"
        else:
            last_result = "miss"

        updates.append(ComboMasteryUpdate(
            combo_name=combo_name,
            success_rate=stats.success_rate,
            attempts=stats.attempts,
            last_result=last_result,
        ))

    return updates


def drill_feedback_to_llm_context(feedback: DrillFeedback) -> SessionContext:
    """Convert DrillFeedback to a SessionContext for Track A's LLM planner.

    Args:
        feedback: DrillFeedback from generate_feedback().

    Returns:
        SessionContext with structured performance data for the LLM.
    """
    weak_combos = [
        name for name, stats in feedback.combo_stats.items()
        if stats.attempts > 0 and stats.success_rate < 0.5
    ]
    strong_combos = [
        name for name, stats in feedback.combo_stats.items()
        if stats.attempts > 0 and stats.success_rate >= 0.8
    ]

    pct = int(feedback.overall_accuracy * 100)
    summary = (
        f"Session {feedback.session_id}: "
        f"{pct}% overall accuracy. "
        f"Strong combos: {', '.join(strong_combos) or 'none'}. "
        f"Weak combos: {', '.join(weak_combos) or 'none'}. "
        f"Frequently missed actions: {', '.join(feedback.missed_actions[:3]) or 'none'}."
    )

    suggested_focus: list[str] = []
    for combo in weak_combos[:2]:
        suggested_focus.append(f"Drill {combo} until >50% success rate")
    for action in feedback.missed_actions[:2]:
        suggested_focus.append(f"Isolate {action} practice")
    if not suggested_focus:
        suggested_focus.append("Maintain current performance and increase complexity")

    return SessionContext(
        session_id=feedback.session_id,
        summary=summary,
        weak_combos=weak_combos,
        strong_combos=strong_combos,
        missed_actions=feedback.missed_actions,
        suggested_focus=suggested_focus,
        overall_accuracy=feedback.overall_accuracy,
    )
