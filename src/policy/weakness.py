"""Weakness detection from State Vector.

Reference: spec/state-vector.md §8.1

W = { i : s_i < τ_i  AND  C_{t,i} >= 0.3 }
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.state.constants import DIM_GROUPS, DIM_NAMES, NUM_DIMS
from src.state.types import StateVector

# Default thresholds τ_i per group (spec/state-vector.md §8.1)
DEFAULT_THRESHOLDS_BY_GROUP: dict[str, float] = {
    "offensive_profile": 0.4,
    "technique": 0.5,
    "defense": 0.5,
    "rhythm": 0.4,
    "conditioning": 0.6,
}

# Minimum confidence to consider a dimension reliable
MIN_CONFIDENCE = 0.3


@dataclass(frozen=True)
class Weakness:
    """A detected weakness in the state vector."""

    dim_index: int
    dim_name: str
    group: str
    value: float  # current s_i
    threshold: float  # τ_i
    confidence: float  # C_{t,i}
    gap: float  # τ_i - s_i (always > 0 for a weakness)


def _build_threshold_vector(
    thresholds_by_group: dict[str, float] | None = None,
) -> np.ndarray:
    """Build a per-dimension threshold vector from group thresholds."""
    thresholds = thresholds_by_group or DEFAULT_THRESHOLDS_BY_GROUP
    tau = np.zeros(NUM_DIMS, dtype=np.float64)
    for group_name, dim_indices in DIM_GROUPS.items():
        for i in dim_indices:
            tau[i] = thresholds[group_name]
    return tau


def _dim_to_group(dim_index: int) -> str:
    """Map a dimension index to its group name."""
    for group_name, indices in DIM_GROUPS.items():
        if dim_index in indices:
            return group_name
    raise ValueError(f"Dimension {dim_index} not found in any group")


def detect_weaknesses(
    state: StateVector,
    thresholds_by_group: dict[str, float] | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> list[Weakness]:
    """Detect weaknesses: dimensions where s_i < τ_i AND C_{t,i} >= min_confidence.

    Args:
        state: Current state vector with values and confidence.
        thresholds_by_group: Optional custom thresholds per group.
            Defaults to spec/state-vector.md §8.1 values.
        min_confidence: Minimum confidence required. Default 0.3.

    Returns:
        List of Weakness objects, unordered.
    """
    tau = _build_threshold_vector(thresholds_by_group)

    weaknesses: list[Weakness] = []
    for i in range(NUM_DIMS):
        s_i = float(state.values[i])
        c_i = float(state.confidence[i])
        tau_i = float(tau[i])

        if s_i < tau_i and c_i >= min_confidence:
            weaknesses.append(
                Weakness(
                    dim_index=i,
                    dim_name=DIM_NAMES[i],
                    group=_dim_to_group(i),
                    value=s_i,
                    threshold=tau_i,
                    confidence=c_i,
                    gap=tau_i - s_i,
                )
            )

    return weaknesses
