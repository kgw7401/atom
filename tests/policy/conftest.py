"""Shared test fixtures for policy engine tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.state.constants import NUM_DIMS
from src.state.types import StateVector


@pytest.fixture
def balanced_state() -> StateVector:
    """A state where all dimensions are above all thresholds.

    No weaknesses should be detected.
    """
    return StateVector(
        values=np.full(NUM_DIMS, 0.75, dtype=np.float64),
        confidence=np.full(NUM_DIMS, 0.8, dtype=np.float64),
        obs_counts=np.full(NUM_DIMS, 10, dtype=np.int64),
        version=5,
    )


@pytest.fixture
def weak_hooks_state() -> StateVector:
    """A state with weak hooks (dim 5 = tech_hook = 0.25).

    tech_hook is well below technique threshold (0.5).
    """
    values = np.full(NUM_DIMS, 0.75, dtype=np.float64)
    values[5] = 0.25  # tech_hook: weak
    confidence = np.full(NUM_DIMS, 0.8, dtype=np.float64)
    return StateVector(
        values=values,
        confidence=confidence,
        obs_counts=np.full(NUM_DIMS, 10, dtype=np.int64),
        version=3,
    )


@pytest.fixture
def multi_weakness_state() -> StateVector:
    """A state with multiple weaknesses across groups.

    - dim 5 (tech_hook): 0.25 (technique, τ=0.5, gap=0.25)
    - dim 8 (guard_consistency): 0.20 (defense, τ=0.5, gap=0.30)
    - dim 15 (volume_endurance): 0.30 (conditioning, τ=0.6, gap=0.30)
    - dim 0 (repertoire_entropy): 0.15 (offensive, τ=0.4, gap=0.25)
    - dim 12 (work_rate): 0.10 (rhythm, τ=0.4, gap=0.30)
    """
    values = np.full(NUM_DIMS, 0.75, dtype=np.float64)
    values[5] = 0.25   # tech_hook
    values[8] = 0.20   # guard_consistency
    values[15] = 0.30  # volume_endurance
    values[0] = 0.15   # repertoire_entropy
    values[12] = 0.10  # work_rate
    confidence = np.full(NUM_DIMS, 0.8, dtype=np.float64)
    return StateVector(
        values=values,
        confidence=confidence,
        obs_counts=np.full(NUM_DIMS, 10, dtype=np.int64),
        version=5,
    )


@pytest.fixture
def low_confidence_state() -> StateVector:
    """A state where values are low but confidence is also low.

    Dimensions with C < 0.3 should NOT be flagged as weaknesses.
    """
    values = np.full(NUM_DIMS, 0.20, dtype=np.float64)  # all below thresholds
    confidence = np.full(NUM_DIMS, 0.15, dtype=np.float64)  # all below min_confidence
    return StateVector(
        values=values,
        confidence=confidence,
        obs_counts=np.full(NUM_DIMS, 1, dtype=np.int64),
        version=1,
    )


@pytest.fixture
def mixed_confidence_state() -> StateVector:
    """A state with some low-confidence and some high-confidence weak dims.

    - dim 5 (tech_hook): value=0.25, C=0.8 → weakness (high confidence)
    - dim 6 (tech_uppercut): value=0.25, C=0.1 → NOT weakness (low confidence)
    """
    values = np.full(NUM_DIMS, 0.75, dtype=np.float64)
    values[5] = 0.25  # tech_hook
    values[6] = 0.25  # tech_uppercut
    confidence = np.full(NUM_DIMS, 0.8, dtype=np.float64)
    confidence[6] = 0.1  # low confidence for uppercut
    return StateVector(
        values=values,
        confidence=confidence,
        obs_counts=np.full(NUM_DIMS, 10, dtype=np.int64),
        version=2,
    )
