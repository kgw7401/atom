"""Report generator — aggregates verification results into a session report."""

from __future__ import annotations

import uuid

from server.models.schemas import (
    CoachingFeedback,
    InstructionResult,
    ReportSummary,
    SessionReportResponse,
)
from server.services.verification_engine import InstructionVerification


def generate_report(
    session_id: uuid.UUID,
    verifications: list[InstructionVerification],
) -> SessionReportResponse:
    """Build a complete session report from verification results."""

    instruction_results = [
        InstructionResult(
            index=v.index,
            t=v.t,
            type=v.type,
            command=v.command,
            status=v.status,
            score=v.score,
            reaction_time=v.reaction_time,
            detected_actions=v.detected_actions,
            feedback=v.feedback,
        )
        for v in verifications
    ]

    # Summary stats
    total = len(verifications)
    success_count = sum(1 for v in verifications if v.status == "success")
    partial_count = sum(1 for v in verifications if v.status == "partial")
    missed_count = sum(1 for v in verifications if v.status == "missed")
    completed = success_count + partial_count

    # Per-type accuracy
    attacks = [v for v in verifications if v.type == "attack"]
    defenses = [v for v in verifications if v.type == "defend"]

    attack_acc = (
        sum(1 for v in attacks if v.status == "success") / len(attacks)
        if attacks else 0.0
    )
    defense_acc = (
        sum(1 for v in defenses if v.status == "success") / len(defenses)
        if defenses else 0.0
    )

    # Average reaction time (attacks with measured reaction)
    reaction_times = [v.reaction_time for v in attacks if v.reaction_time is not None]
    avg_reaction = sum(reaction_times) / len(reaction_times) if reaction_times else 0.0

    summary = ReportSummary(
        total_instructions=total,
        completed=completed,
        success=success_count,
        partial=partial_count,
        missed=missed_count,
        attack_accuracy=round(attack_acc, 2),
        defense_accuracy=round(defense_acc, 2),
        avg_reaction_time=round(avg_reaction, 2),
    )

    # Overall score: weighted average of instruction scores
    overall_score = (
        round(sum(v.score for v in verifications) / total) if total > 0 else 0
    )

    # Coaching feedback
    coaching = _generate_coaching(verifications, avg_reaction)

    return SessionReportResponse(
        session_id=session_id,
        overall_score=overall_score,
        summary=summary,
        instructions=instruction_results,
        coaching=coaching,
    )


def _generate_coaching(
    verifications: list[InstructionVerification],
    avg_reaction: float,
) -> CoachingFeedback:
    """Generate rule-based coaching feedback from verification results."""
    strengths: list[str] = []
    weaknesses: list[str] = []

    # Analyze per-action performance
    action_stats: dict[str, dict] = {}
    for v in verifications:
        if v.type != "attack":
            continue
        for action in v.expected_actions:
            if action not in action_stats:
                action_stats[action] = {"attempts": 0, "detected": 0, "reactions": []}
            action_stats[action]["attempts"] += 1
            if action in v.detected_actions:
                action_stats[action]["detected"] += 1
            if v.reaction_time is not None:
                action_stats[action]["reactions"].append(v.reaction_time)

    for action, stats in action_stats.items():
        acc = stats["detected"] / stats["attempts"] if stats["attempts"] > 0 else 0
        if acc >= 0.9:
            strengths.append(f"{action} accuracy excellent ({acc:.0%})")
        elif acc < 0.6:
            weaknesses.append(f"{action} missed frequently ({acc:.0%} accuracy)")

        if stats["reactions"]:
            avg_r = sum(stats["reactions"]) / len(stats["reactions"])
            if avg_r <= 0.5:
                strengths.append(f"{action} fast reaction ({avg_r:.2f}s)")
            elif avg_r > 1.0:
                weaknesses.append(f"{action} slow reaction ({avg_r:.2f}s)")

    # Guard discipline
    guard_drops = sum(1 for v in verifications if v.type == "attack" and not v.guard_returned)
    total_attacks = sum(1 for v in verifications if v.type == "attack")
    if total_attacks > 0:
        guard_rate = 1 - guard_drops / total_attacks
        if guard_rate >= 0.9:
            strengths.append(f"Guard discipline excellent ({guard_rate:.0%})")
        elif guard_rate < 0.7:
            weaknesses.append(f"Guard drops too often ({guard_drops}/{total_attacks})")

    # Next session recommendation
    if weaknesses:
        next_session = f"Focus on: {weaknesses[0].split('(')[0].strip()}"
    elif avg_reaction > 0.8:
        next_session = "Work on reaction speed — try a harder level"
    else:
        next_session = "Great session! Ready for the next level"

    return CoachingFeedback(
        strengths=strengths or ["Keep training!"],
        weaknesses=weaknesses or ["No major issues detected"],
        next_session=next_session,
    )
