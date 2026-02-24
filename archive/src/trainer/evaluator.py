"""Drill attempt evaluation — scores accuracy, timing, and guard discipline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionEvaluation:
    """Evaluation of a single action within a combo."""

    action: str
    expected: bool
    timing: float | None  # seconds from previous action (or instruction)
    timing_rating: str  # "fast", "good", "slow"


@dataclass
class DrillResult:
    """Complete evaluation of one drill attempt."""

    success: bool
    combo_name: str
    actions_completed: int
    actions_expected: int
    reaction_time: float | None
    action_evaluations: list[ActionEvaluation]
    guard_returned: bool
    total_time: float | None
    feedback_text: str
    score: int  # 0-100


# Timing thresholds (seconds)
REACTION_FAST = 0.5
REACTION_GOOD = 1.0

COMBO_GAP_FAST = 0.3
COMBO_GAP_GOOD = 0.6


def evaluate(attempt, config) -> DrillResult:
    """Evaluate a completed drill attempt.

    Scoring (100 pts max):
      - Accuracy: 50 pts (correct actions in order?)
      - Reaction time: 20 pts (how fast did you start?)
      - Combo timing: 20 pts (gaps between actions)
      - Guard return: 10 pts (returned to guard after?)
    """
    expected = attempt.expected_actions
    recognized = attempt.recognized_actions
    n_expected = len(expected)
    n_completed = len(recognized)

    # --- Accuracy (50 pts) ---
    if n_completed == 0:
        accuracy_score = 0
        success = False
    else:
        match_count = sum(
            1 for exp, got in zip(expected, recognized) if exp == got
        )
        accuracy_score = int(50 * match_count / n_expected)
        success = match_count == n_expected

    # --- Reaction time (20 pts) ---
    reaction_time = None
    reaction_score = 0
    reaction_rating = "missed"
    if attempt.first_action_time > 0:
        reaction_time = attempt.first_action_time - attempt.instruction_time
        if reaction_time <= REACTION_FAST:
            reaction_score = 20
            reaction_rating = "fast"
        elif reaction_time <= REACTION_GOOD:
            reaction_score = 15
            reaction_rating = "good"
        else:
            reaction_score = 8
            reaction_rating = "slow"

    # --- Combo timing (20 pts) ---
    action_evals: list[ActionEvaluation] = []
    combo_timing_score = 0
    if n_completed >= 2:
        gap_scores = []
        for i in range(1, n_completed):
            gap = attempt.action_times[i] - attempt.action_times[i - 1]
            if gap <= COMBO_GAP_FAST:
                gap_scores.append(20)
                rating = "fast"
            elif gap <= COMBO_GAP_GOOD:
                gap_scores.append(15)
                rating = "good"
            else:
                gap_scores.append(5)
                rating = "slow"
            action_evals.append(ActionEvaluation(
                action=recognized[i],
                expected=(i < n_expected and recognized[i] == expected[i]),
                timing=gap,
                timing_rating=rating,
            ))
        combo_timing_score = int(sum(gap_scores) / len(gap_scores))
    elif n_completed == 1 and n_expected == 1:
        combo_timing_score = 20  # single action, no gap to measure

    # --- Guard return (10 pts) ---
    guard_score = 10 if attempt.guard_returned else 0

    # --- Total ---
    total_score = accuracy_score + reaction_score + combo_timing_score + guard_score

    # --- Feedback text ---
    display_names = config.display_names
    feedback = _build_feedback(
        success, recognized, expected, reaction_time, reaction_rating,
        action_evals, attempt.guard_returned, display_names,
    )

    # --- Total time ---
    total_time = None
    if attempt.action_times:
        total_time = attempt.action_times[-1] - attempt.instruction_time

    return DrillResult(
        success=success,
        combo_name=attempt.combo.name,
        actions_completed=n_completed,
        actions_expected=n_expected,
        reaction_time=reaction_time,
        action_evaluations=action_evals,
        guard_returned=attempt.guard_returned,
        total_time=total_time,
        feedback_text=feedback,
        score=total_score,
    )


def _build_feedback(
    success: bool,
    recognized: list[str],
    expected: list[str],
    reaction_time: float | None,
    reaction_rating: str,
    action_evals: list[ActionEvaluation],
    guard_returned: bool,
    display_names: dict[str, str],
) -> str:
    if not recognized:
        return "Missed! No action detected."

    parts: list[str] = []

    if success:
        if len(expected) == 1:
            name = display_names.get(expected[0], expected[0])
            parts.append(f"OK {name}!")
        else:
            parts.append("OK Combo!")
    else:
        parts.append(f"Partial: {len(recognized)}/{len(expected)}")

    if reaction_time is not None:
        parts.append(f"{reaction_time:.1f}s")

    for ev in action_evals:
        if ev.timing_rating == "slow" and ev.timing is not None:
            name = display_names.get(ev.action, ev.action)
            parts.append(f"{name} +{ev.timing:.1f}s")

    if not guard_returned:
        parts.append("Guard dropped!")

    return " | ".join(parts)
