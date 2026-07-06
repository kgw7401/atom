"""Verification engine — aligns session script instructions with detected action segments.

For each instruction in the script, searches the detected action sequence
within the expected time window and scores the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from server.config import settings


@dataclass
class InstructionVerification:
    """Result of verifying one instruction against detections."""

    index: int
    t: float
    type: str  # "attack" or "defend"
    command: str
    expected_actions: list[str]
    detected_actions: list[str] = field(default_factory=list)
    reaction_time: float | None = None
    action_times: list[float] = field(default_factory=list)
    guard_returned: bool = False
    status: str = "missed"  # "success", "partial", "missed"
    score: int = 0
    feedback: str = ""


# Scoring constants (same as evaluator.py)
_REACTION_FAST = 0.5
_REACTION_GOOD = 1.0
_COMBO_GAP_FAST = 0.3
_COMBO_GAP_GOOD = 0.6


def _load_timing(level: int) -> dict:
    """Load level-specific timing from config."""
    with open(settings.config_path) as f:
        cfg = yaml.safe_load(f)
    timing = cfg.get("drills", {}).get("timing", {})
    return timing.get(f"level_{level}", {
        "reaction_window": 3.0,
        "guard_return_window": 1.2,
        "combo_gap_max": 1.5,
    })


def verify_session(
    instructions: list[dict],
    segments: list[dict],
    level: int = 1,
) -> list[InstructionVerification]:
    """Align script instructions with detected action segments and score each.

    Args:
        instructions: List of script instructions [{t, type, action, display, ...}]
        segments: List of action segments [{action, start_s, end_s, avg_confidence}]
        level: Difficulty level (affects timing windows)

    Returns:
        List of InstructionVerification, one per instruction.
    """
    timing = _load_timing(level)
    reaction_window = timing.get("reaction_window", 3.0)
    guard_return_window = timing.get("guard_return_window", 1.2)

    results: list[InstructionVerification] = []

    for i, instr in enumerate(instructions):
        instr_t = instr["t"]
        instr_type = instr.get("type", "attack")
        action_str = instr["action"]
        expected = action_str.split(",")
        display = instr.get("display", action_str)

        # Determine the next instruction's time (for window boundary)
        if i + 1 < len(instructions):
            next_t = instructions[i + 1]["t"]
        else:
            next_t = instr_t + reaction_window + 2.0

        # Search window: [instr_t, min(instr_t + reaction_window + buffer, next_t)]
        window_end = min(instr_t + reaction_window + 2.0, next_t)

        verification = InstructionVerification(
            index=i,
            t=instr_t,
            type=instr_type,
            command=display,
            expected_actions=expected,
        )

        if instr_type == "defend":
            # Defense: success if no attack detected in window (guard maintained)
            verification = _verify_defense(verification, segments, instr_t, window_end)
        else:
            # Attack: match expected actions in order
            verification = _verify_attack(
                verification, segments, instr_t, window_end,
                reaction_window, guard_return_window,
            )

        results.append(verification)

    return results


def _verify_attack(
    v: InstructionVerification,
    segments: list[dict],
    window_start: float,
    window_end: float,
    reaction_window: float,
    guard_return_window: float,
) -> InstructionVerification:
    """Verify an attack instruction against detected segments."""
    expected = v.expected_actions
    n_expected = len(expected)

    # Find non-guard segments within the time window
    candidates = [
        s for s in segments
        if s["action"] != "guard"
        and s["end_s"] >= window_start
        and s["start_s"] <= window_end
    ]

    # Try to match expected actions in order
    matched: list[str] = []
    match_times: list[float] = []
    exp_idx = 0

    for seg in candidates:
        if exp_idx >= n_expected:
            break
        if seg["action"] == expected[exp_idx]:
            matched.append(seg["action"])
            match_times.append(seg["start_s"])
            exp_idx += 1

    v.detected_actions = matched
    v.action_times = match_times
    n_matched = len(matched)

    # Reaction time (first matched action relative to instruction)
    if match_times:
        v.reaction_time = match_times[0] - v.t

    # Check guard return after last action
    if match_times:
        last_action_end = match_times[-1] + 0.5  # estimate action duration
        guard_segments = [
            s for s in segments
            if s["action"] == "guard"
            and s["start_s"] >= last_action_end
            and s["start_s"] <= last_action_end + guard_return_window
        ]
        v.guard_returned = len(guard_segments) > 0

    # --- Scoring (mirrors evaluator.py) ---

    # Accuracy (50 pts)
    if n_matched == 0:
        accuracy_score = 0
    else:
        accuracy_score = int(50 * n_matched / n_expected)

    # Reaction time (20 pts)
    reaction_score = 0
    if v.reaction_time is not None:
        if v.reaction_time <= _REACTION_FAST:
            reaction_score = 20
        elif v.reaction_time <= _REACTION_GOOD:
            reaction_score = 15
        else:
            reaction_score = 8

    # Combo timing (20 pts)
    combo_score = 0
    if n_matched >= 2:
        gap_scores = []
        for j in range(1, n_matched):
            gap = match_times[j] - match_times[j - 1]
            if gap <= _COMBO_GAP_FAST:
                gap_scores.append(20)
            elif gap <= _COMBO_GAP_GOOD:
                gap_scores.append(15)
            else:
                gap_scores.append(5)
        combo_score = int(sum(gap_scores) / len(gap_scores))
    elif n_matched == 1 and n_expected == 1:
        combo_score = 20

    # Guard return (10 pts)
    guard_score = 10 if v.guard_returned else 0

    v.score = accuracy_score + reaction_score + combo_score + guard_score

    # Status
    if n_matched == n_expected:
        v.status = "success"
    elif n_matched > 0:
        v.status = "partial"
    else:
        v.status = "missed"

    # Feedback
    v.feedback = _build_attack_feedback(v, n_matched, n_expected)

    return v


def _verify_defense(
    v: InstructionVerification,
    segments: list[dict],
    window_start: float,
    window_end: float,
) -> InstructionVerification:
    """Verify a defense instruction. Success = no attack in window (guard held)."""
    attack_in_window = [
        s for s in segments
        if s["action"] != "guard"
        and s["end_s"] >= window_start
        and s["start_s"] <= window_end
    ]

    if not attack_in_window:
        v.status = "success"
        v.score = 80  # Full marks for defense (no timing to measure)
        v.guard_returned = True
        v.feedback = "Good defense!"
    else:
        v.status = "missed"
        v.score = 0
        v.detected_actions = [s["action"] for s in attack_in_window[:3]]
        v.feedback = "Defense missed — attack detected during defense window."

    return v


def _build_attack_feedback(v: InstructionVerification, matched: int, expected: int) -> str:
    """Generate human-readable feedback for an attack instruction."""
    if matched == 0:
        return "Missed! No action detected."

    parts: list[str] = []

    if v.status == "success":
        if expected == 1:
            parts.append(f"Good {v.expected_actions[0]}!")
        else:
            parts.append("Combo complete!")
    else:
        parts.append(f"Partial: {matched}/{expected}")

    if v.reaction_time is not None:
        if v.reaction_time <= _REACTION_FAST:
            parts.append(f"Fast reaction ({v.reaction_time:.2f}s)")
        elif v.reaction_time <= _REACTION_GOOD:
            parts.append(f"Reaction {v.reaction_time:.2f}s")
        else:
            parts.append(f"Slow start ({v.reaction_time:.2f}s)")

    if not v.guard_returned:
        parts.append("Return to guard!")

    return " | ".join(parts)
