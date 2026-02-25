"""Priority scoring for detected weaknesses.

Reference: spec/state-vector.md §8.2

priority_i = w_group(i) * (τ_i - s_i) * C_{t,i}
"""

from __future__ import annotations

from dataclasses import dataclass

from src.policy.weakness import Weakness

# Group importance weights (spec/state-vector.md §8.2)
DEFAULT_GROUP_WEIGHTS: dict[str, float] = {
    "defense": 1.5,
    "technique": 1.2,
    "conditioning": 1.0,
    "offensive_profile": 0.8,
    "rhythm": 0.8,
}


@dataclass(frozen=True)
class PrioritizedWeakness:
    """A weakness with its computed priority score."""

    weakness: Weakness
    priority: float  # w_group * gap * confidence
    rank: int  # 1-based rank (1 = highest priority)


def score_priorities(
    weaknesses: list[Weakness],
    group_weights: dict[str, float] | None = None,
) -> list[PrioritizedWeakness]:
    """Score and rank weaknesses by priority.

    priority_i = w_group(i) * (τ_i - s_i) * C_{t,i}

    Args:
        weaknesses: List of detected weaknesses.
        group_weights: Optional custom group importance weights.
            Defaults to spec/state-vector.md §8.2 values.

    Returns:
        List of PrioritizedWeakness, sorted by priority descending (rank 1 = highest).
    """
    weights = group_weights or DEFAULT_GROUP_WEIGHTS

    scored: list[tuple[float, Weakness]] = []
    for w in weaknesses:
        w_group = weights.get(w.group, 1.0)
        priority = w_group * w.gap * w.confidence
        scored.append((priority, w))

    # Sort descending by priority, stable on insertion order for ties
    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        PrioritizedWeakness(weakness=w, priority=p, rank=rank)
        for rank, (p, w) in enumerate(scored, start=1)
    ]
