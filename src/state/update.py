"""State update functions: EMA, initialization, confidence, delta.

Reference: spec/state-vector.md §5, §6, §7
"""

from __future__ import annotations

import numpy as np

from src.state.constants import ALPHA, N_REF, NUM_DIMS
from src.state.types import ObservationVector, StateVector


def initialize_state(obs: ObservationVector) -> StateVector:
    """Create initial state from the first observation.

    S_0,i = O_0,i  for observed dimensions
    S_0,i = 0.5    for unobserved dimensions (neutral prior)

    Reference: spec/state-vector.md §5.2
    """
    values = np.full(NUM_DIMS, 0.5, dtype=np.float64)
    obs_counts = np.zeros(NUM_DIMS, dtype=np.int64)

    # Set observed dimensions
    values[obs.mask] = obs.values[obs.mask]
    obs_counts[obs.mask] = 1

    confidence = compute_confidence(obs_counts)

    state = StateVector(
        values=values,
        confidence=confidence,
        obs_counts=obs_counts,
        version=1,
    )
    assert state.validate_bounds(), "Initial state violates bounds"
    return state


def ema_update(
    state: StateVector,
    obs: ObservationVector,
    alpha: float = ALPHA,
) -> StateVector:
    """Apply EMA update to state vector.

    For observed dims:   S_{t+1,i} = α · S_{t,i} + (1-α) · O_{t,i}
    For unobserved dims: S_{t+1,i} = S_{t,i}

    Reference: spec/state-vector.md §5.1
    """
    new_values = state.values.copy()
    new_obs_counts = state.obs_counts.copy()

    # Update only observed dimensions
    observed = obs.mask
    new_values[observed] = alpha * state.values[observed] + (1 - alpha) * obs.values[observed]
    new_obs_counts[observed] += 1

    # Clip to [0, 1] — should be inherently bounded but guard against float drift
    new_values = np.clip(new_values, 0.0, 1.0)

    new_confidence = compute_confidence(new_obs_counts)

    new_state = StateVector(
        values=new_values,
        confidence=new_confidence,
        obs_counts=new_obs_counts,
        version=state.version + 1,
    )
    assert new_state.validate_bounds(), "Updated state violates bounds"
    return new_state


def compute_confidence(obs_counts: np.ndarray, n_ref: int = N_REF) -> np.ndarray:
    """Compute confidence from observation counts.

    C_i = 1 - exp(-n_i / n_ref)

    Reference: spec/state-vector.md §6
    """
    return 1.0 - np.exp(-obs_counts.astype(np.float64) / n_ref)


def compute_delta(s_new: StateVector, s_old: StateVector) -> np.ndarray:
    """Compute state delta.

    ΔS = S_{t+1} - S_t

    Reference: spec/state-vector.md §7
    """
    return s_new.values - s_old.values


def apply_observation(state: StateVector | None, obs: ObservationVector) -> StateVector:
    """High-level entry point: apply observation to state.

    If state is None (first session), initializes. Otherwise, EMA update.
    """
    if obs.is_empty:
        raise ValueError("Cannot apply empty observation (all dimensions unobserved)")

    if state is None:
        return initialize_state(obs)
    return ema_update(state, obs)
