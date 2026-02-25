"""Session planner: generates training plans from prioritized weaknesses.

Reference: spec/roadmap.md Phase 2c
- Policy is 100% rule-based (no ML, no LLM)
- Session plan is a JSON-serializable structure
- Balanced state → maintenance/general session
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.policy.drill_library import Drill, find_drills_for_dim, load_drills
from src.policy.priority import PrioritizedWeakness, score_priorities
from src.policy.weakness import Weakness, detect_weaknesses
from src.state.constants import DIM_NAMES
from src.state.types import StateVector

# Session planning constants
DEFAULT_SESSION_DURATION = 600  # 10 minutes total
DEFAULT_MAX_WEAKNESSES = 3  # focus on top-N weaknesses
DEFAULT_ROUNDS = 3
DEFAULT_ROUND_DURATION = 180  # 3 minutes per round
DEFAULT_REST_DURATION = 60  # 1 minute rest between rounds


@dataclass(frozen=True)
class DrillAssignment:
    """A drill assigned within a round."""

    drill_name: str
    drill_type: str
    actions: list[str]
    target_dims: list[int]
    target_dim_names: list[str]
    level: int
    duration_seconds: int
    reps: int | None


@dataclass(frozen=True)
class Round:
    """A single round within a session plan."""

    round_number: int  # 1-based
    drills: list[DrillAssignment]
    duration_seconds: int
    focus_dims: list[int]


@dataclass
class SessionPlan:
    """A complete training session plan."""

    plan_type: str  # "targeted" or "maintenance"
    total_duration_seconds: int
    num_rounds: int
    rest_duration_seconds: int
    rounds: list[Round]
    target_weaknesses: list[dict]  # simplified weakness info
    focus_summary: str  # human-readable summary of what to focus on

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "plan_type": self.plan_type,
            "total_duration_seconds": self.total_duration_seconds,
            "num_rounds": self.num_rounds,
            "rest_duration_seconds": self.rest_duration_seconds,
            "rounds": [
                {
                    "round_number": r.round_number,
                    "drills": [
                        {
                            "drill_name": d.drill_name,
                            "drill_type": d.drill_type,
                            "actions": d.actions,
                            "target_dims": d.target_dims,
                            "target_dim_names": d.target_dim_names,
                            "level": d.level,
                            "duration_seconds": d.duration_seconds,
                            "reps": d.reps,
                        }
                        for d in r.drills
                    ],
                    "duration_seconds": r.duration_seconds,
                    "focus_dims": r.focus_dims,
                }
                for r in self.rounds
            ],
            "target_weaknesses": self.target_weaknesses,
            "focus_summary": self.focus_summary,
        }


def _select_drill_for_weakness(
    weakness: Weakness,
    drills: list[Drill],
    used_drills: set[str],
) -> Drill | None:
    """Select the best drill for a weakness, avoiding duplicates.

    Prefers drills not yet used. Among candidates, picks the lowest level
    that still targets the weakness dimension.
    """
    candidates = find_drills_for_dim(weakness.dim_index, drills)
    # Prefer unused drills
    unused = [d for d in candidates if d.name not in used_drills]
    if unused:
        return unused[0]
    # Fall back to any drill if all have been used
    if candidates:
        return candidates[0]
    return None


def _make_drill_assignment(drill: Drill) -> DrillAssignment:
    return DrillAssignment(
        drill_name=drill.name,
        drill_type=drill.type,
        actions=drill.actions,
        target_dims=drill.target_dims,
        target_dim_names=[DIM_NAMES[i] for i in drill.target_dims],
        level=drill.level,
        duration_seconds=drill.duration_seconds,
        reps=drill.reps,
    )


def _build_maintenance_plan(
    drills: list[Drill],
    num_rounds: int = DEFAULT_ROUNDS,
    round_duration: int = DEFAULT_ROUND_DURATION,
    rest_duration: int = DEFAULT_REST_DURATION,
) -> SessionPlan:
    """Build a general maintenance session when no weaknesses are detected."""
    # Pick general/freestyle drills
    general_drills = [d for d in drills if "freestyle" in d.name or "full_combo" in d.name]
    if not general_drills:
        general_drills = [d for d in drills if d.type == "combo"]
    if not general_drills:
        general_drills = drills[:3]

    rounds: list[Round] = []
    for r in range(1, num_rounds + 1):
        drill = general_drills[(r - 1) % len(general_drills)]
        assignment = _make_drill_assignment(drill)
        rounds.append(
            Round(
                round_number=r,
                drills=[assignment],
                duration_seconds=round_duration,
                focus_dims=drill.target_dims,
            )
        )

    total = num_rounds * round_duration + (num_rounds - 1) * rest_duration
    return SessionPlan(
        plan_type="maintenance",
        total_duration_seconds=total,
        num_rounds=num_rounds,
        rest_duration_seconds=rest_duration,
        rounds=rounds,
        target_weaknesses=[],
        focus_summary="No specific weaknesses detected. General maintenance session.",
    )


def plan_session(
    state: StateVector,
    drills: list[Drill] | None = None,
    max_weaknesses: int = DEFAULT_MAX_WEAKNESSES,
    num_rounds: int = DEFAULT_ROUNDS,
    round_duration: int = DEFAULT_ROUND_DURATION,
    rest_duration: int = DEFAULT_REST_DURATION,
) -> SessionPlan:
    """Generate a training session plan from current state.

    1. Detect weaknesses from S_t
    2. Score and rank by priority
    3. Select top-N weaknesses
    4. Map to drills from library
    5. Arrange into rounds

    If no weaknesses: generate a maintenance/general session.

    Args:
        state: Current state vector.
        drills: Optional pre-loaded drill list. Loads from default if None.
        max_weaknesses: Maximum weaknesses to address per session.
        num_rounds: Number of rounds in the session.
        round_duration: Duration per round in seconds.
        rest_duration: Rest between rounds in seconds.

    Returns:
        SessionPlan with rounds, drills, and focus areas.
    """
    all_drills = drills if drills is not None else load_drills()

    # Step 1-2: Detect and prioritize weaknesses
    weaknesses = detect_weaknesses(state)
    if not weaknesses:
        return _build_maintenance_plan(all_drills, num_rounds, round_duration, rest_duration)

    prioritized = score_priorities(weaknesses)

    # Step 3: Select top-N
    top = prioritized[:max_weaknesses]

    # Step 4-5: Map weaknesses to drills and arrange into rounds
    used_drills: set[str] = set()
    rounds: list[Round] = []

    for r in range(1, num_rounds + 1):
        round_drills: list[DrillAssignment] = []
        round_focus: list[int] = []

        # Cycle through top weaknesses across rounds
        pw = top[(r - 1) % len(top)]
        drill = _select_drill_for_weakness(pw.weakness, all_drills, used_drills)
        if drill is not None:
            used_drills.add(drill.name)
            round_drills.append(_make_drill_assignment(drill))
            round_focus.extend(drill.target_dims)

        # If round has room and there are more weaknesses, add a secondary drill
        if len(top) > 1:
            secondary_pw = top[r % len(top)]
            if secondary_pw.weakness.dim_index != pw.weakness.dim_index:
                secondary_drill = _select_drill_for_weakness(
                    secondary_pw.weakness, all_drills, used_drills
                )
                if secondary_drill is not None and secondary_drill.name not in {
                    d.drill_name for d in round_drills
                }:
                    used_drills.add(secondary_drill.name)
                    round_drills.append(_make_drill_assignment(secondary_drill))
                    round_focus.extend(secondary_drill.target_dims)

        rounds.append(
            Round(
                round_number=r,
                drills=round_drills,
                duration_seconds=round_duration,
                focus_dims=sorted(set(round_focus)),
            )
        )

    # Build weakness summary for the plan
    target_weaknesses = [
        {
            "dim_index": pw.weakness.dim_index,
            "dim_name": pw.weakness.dim_name,
            "group": pw.weakness.group,
            "value": round(pw.weakness.value, 3),
            "threshold": pw.weakness.threshold,
            "gap": round(pw.weakness.gap, 3),
            "priority": round(pw.priority, 4),
            "rank": pw.rank,
        }
        for pw in top
    ]

    # Focus summary
    focus_dims_str = ", ".join(pw.weakness.dim_name for pw in top)
    focus_summary = f"Targeting {len(top)} weakness(es): {focus_dims_str}"

    total = num_rounds * round_duration + (num_rounds - 1) * rest_duration
    return SessionPlan(
        plan_type="targeted",
        total_duration_seconds=total,
        num_rounds=num_rounds,
        rest_duration_seconds=rest_duration,
        rounds=rounds,
        target_weaknesses=target_weaknesses,
        focus_summary=focus_summary,
    )
