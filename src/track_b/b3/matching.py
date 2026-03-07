"""B3 Task 11: Session matching — instruction vs execution.

Compares TTS instruction log (from Track A) with detected ComboSequences.
For each instruction, finds the combo detected within a 3-second window
and classifies: success / partial / miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from track_b.b3.combo import ComboSequence


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class TTSInstruction:
    """A single TTS combo instruction from Track A."""

    timestamp: float          # seconds from session start
    combo_name: str           # human-readable, e.g., "jab-cross-hook"
    combo_actions: list[str]  # machine-readable, e.g., ["jab", "cross", "lead_hook"]


@dataclass
class TTSInstructionLog:
    """Session instruction log produced by Track A's session engine."""

    session_id: str
    instructions: list[TTSInstruction] = field(default_factory=list)


@dataclass
class InstructionMatch:
    """Result of comparing one TTS instruction vs detected combo."""

    instruction_time: float
    instructed_combo: list[str]
    detected_combo: list[str]
    result: str    # "success" / "partial" / "miss"
    detail: str    # e.g., "lead_hook missing" or "" for success


@dataclass
class SessionMatch:
    """Full session match results."""

    session_id: str
    matches: list[InstructionMatch] = field(default_factory=list)


# ── Classification ────────────────────────────────────────────────────────────


def _classify_match(
    instructed: list[str],
    detected: list[str],
) -> tuple[str, str]:
    """Classify match quality between instructed and detected combos.

    Args:
        instructed: Instructed combo actions in order.
        detected: Detected combo actions in order.

    Returns:
        Tuple of (result, detail).
        result: "success" / "partial" / "miss"
        detail: Human-readable explanation.
    """
    if not detected:
        return "miss", "no combo detected"

    detected_set = set(detected)
    missing = [a for a in instructed if a not in detected_set]

    if not missing:
        return "success", ""

    if len(missing) < len(instructed):
        detail = ", ".join(f"{a} missing" for a in missing)
        return "partial", detail

    return "miss", "no matching actions"


def _find_best_combo_in_window(
    combo_sequence: ComboSequence,
    instruction_time: float,
    instructed: list[str],
    window: float,
) -> list[str]:
    """Find the best-matching detected combo within the matching window.

    Collects all combos that start within [instruction_time, instruction_time + window),
    then picks the one with maximum overlap with the instructed combo.

    Args:
        combo_sequence: Grouped combos.
        instruction_time: TTS instruction timestamp.
        instructed: Instructed combo actions.
        window: Matching window in seconds.

    Returns:
        List of detected actions (empty if none found).
    """
    window_end = instruction_time + window
    candidates = [
        c for c in combo_sequence.combos
        if instruction_time <= c.start_time < window_end
    ]

    if not candidates:
        return []

    instructed_set = set(instructed)

    def overlap(combo):
        return sum(1 for a in combo.actions if a in instructed_set)

    best = max(candidates, key=overlap)
    return best.actions


# ── Matching ──────────────────────────────────────────────────────────────────


def match_session(
    instruction_log: TTSInstructionLog,
    combo_sequence: ComboSequence,
    window: float = 3.0,
) -> SessionMatch:
    """Compare TTS instructions vs detected combos.

    For each TTS instruction, finds the best matching detected combo within
    [instruction_time, instruction_time + window) and classifies the result.

    Args:
        instruction_log: TTS instruction log from Track A.
        combo_sequence: Detected combos from group_into_combos().
        window: Matching window in seconds (default 3.0).

    Returns:
        SessionMatch with per-instruction comparison results.
    """
    matches: list[InstructionMatch] = []

    for instr in instruction_log.instructions:
        detected = _find_best_combo_in_window(
            combo_sequence,
            instr.timestamp,
            instr.combo_actions,
            window,
        )
        result, detail = _classify_match(instr.combo_actions, detected)
        matches.append(InstructionMatch(
            instruction_time=instr.timestamp,
            instructed_combo=instr.combo_actions,
            detected_combo=detected,
            result=result,
            detail=detail,
        ))

    return SessionMatch(session_id=instruction_log.session_id, matches=matches)
