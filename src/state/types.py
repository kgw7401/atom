"""Core data types for the Boxing State Vector.

Reference: spec/state-vector.md §2, §3
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from src.state.constants import NUM_DIMS, SCHEMA_VERSION


@dataclass(frozen=True)
class ActionSegment:
    """A classified action segment from the LSTM pipeline."""

    class_id: int  # 0=guard, 1=jab, ..., 8=rear_bodyshot
    class_name: str
    t_start: float  # seconds
    t_end: float  # seconds

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass(frozen=True)
class KeypointFrame:
    """A single frame of pose keypoints."""

    timestamp: float  # seconds
    keypoints: np.ndarray  # shape (11, 2) — x, y in normalized coords

    def __post_init__(self) -> None:
        if self.keypoints.shape != (11, 2):
            raise ValueError(f"Expected keypoints shape (11, 2), got {self.keypoints.shape}")


@dataclass
class ObservationVector:
    """Observation from a single session. NaN = not observed."""

    values: np.ndarray  # shape (NUM_DIMS,), NaN for unobserved
    mask: np.ndarray  # shape (NUM_DIMS,), True if observed

    def __post_init__(self) -> None:
        if self.values.shape != (NUM_DIMS,):
            raise ValueError(f"Expected values shape ({NUM_DIMS},), got {self.values.shape}")
        if self.mask.shape != (NUM_DIMS,):
            raise ValueError(f"Expected mask shape ({NUM_DIMS},), got {self.mask.shape}")

    @property
    def num_observed(self) -> int:
        return int(self.mask.sum())

    @property
    def is_empty(self) -> bool:
        return self.num_observed == 0


@dataclass
class StateVector:
    """The boxer's accumulated ability state."""

    values: np.ndarray  # shape (NUM_DIMS,), all in [0, 1]
    confidence: np.ndarray  # shape (NUM_DIMS,), all in [0, 1]
    obs_counts: np.ndarray  # shape (NUM_DIMS,), int >= 0
    version: int = 0
    schema_version: str = field(default=SCHEMA_VERSION)

    def __post_init__(self) -> None:
        if self.values.shape != (NUM_DIMS,):
            raise ValueError(f"Expected values shape ({NUM_DIMS},), got {self.values.shape}")
        if self.confidence.shape != (NUM_DIMS,):
            raise ValueError(
                f"Expected confidence shape ({NUM_DIMS},), got {self.confidence.shape}"
            )
        if self.obs_counts.shape != (NUM_DIMS,):
            raise ValueError(
                f"Expected obs_counts shape ({NUM_DIMS},), got {self.obs_counts.shape}"
            )

    def validate_bounds(self) -> bool:
        """Check all values and confidence are in [0, 1]."""
        return bool(
            np.all(self.values >= 0.0)
            and np.all(self.values <= 1.0)
            and np.all(self.confidence >= 0.0)
            and np.all(self.confidence <= 1.0)
            and np.all(self.obs_counts >= 0)
        )

    def to_json(self) -> dict:
        """Serialize for DB storage."""
        return {
            "values": self.values.tolist(),
            "confidence": self.confidence.tolist(),
            "obs_counts": self.obs_counts.astype(int).tolist(),
            "version": self.version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_json(cls, data: dict) -> StateVector:
        """Deserialize from DB storage."""
        return cls(
            values=np.array(data["values"], dtype=np.float64),
            confidence=np.array(data["confidence"], dtype=np.float64),
            obs_counts=np.array(data["obs_counts"], dtype=np.int64),
            version=data["version"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    @classmethod
    def zeros(cls) -> StateVector:
        """Create a zero-initialized state (for first-time users before any session)."""
        return cls(
            values=np.full(NUM_DIMS, 0.5, dtype=np.float64),
            confidence=np.zeros(NUM_DIMS, dtype=np.float64),
            obs_counts=np.zeros(NUM_DIMS, dtype=np.int64),
            version=0,
        )


@dataclass(frozen=True)
class DefensiveCommand:
    """A defensive command in AI Session mode."""

    timestamp: float  # seconds — when the command was issued
    action: str  # e.g., "guard", "slip"
