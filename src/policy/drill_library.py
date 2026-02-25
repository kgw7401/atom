"""Drill library loaded from YAML configuration.

Each drill specifies which state vector dimensions it targets,
enabling the session planner to select drills that address weaknesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Drill:
    """A training drill from the drill library."""

    name: str
    type: str  # single | combo | defense | conditioning
    actions: list[str]
    target_dims: list[int]  # 0-indexed state vector dimensions
    level: int  # 1-3
    duration_seconds: int
    reps: int | None  # None if time-based


_DRILLS_CACHE: list[Drill] | None = None
_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "drills.yaml"


def load_drills(path: Path | None = None) -> list[Drill]:
    """Load drill definitions from YAML.

    Args:
        path: Path to drills.yaml. Defaults to configs/drills.yaml.

    Returns:
        List of Drill objects.
    """
    global _DRILLS_CACHE
    if _DRILLS_CACHE is not None and path is None:
        return _DRILLS_CACHE

    yaml_path = path or _DEFAULT_PATH
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    drills = [
        Drill(
            name=d["name"],
            type=d["type"],
            actions=d["actions"],
            target_dims=d["target_dims"],
            level=d["level"],
            duration_seconds=d["duration_seconds"],
            reps=d.get("reps"),
        )
        for d in data["drills"]
    ]

    if path is None:
        _DRILLS_CACHE = drills

    return drills


def find_drills_for_dim(dim_index: int, drills: list[Drill] | None = None) -> list[Drill]:
    """Find all drills that target a specific dimension.

    Args:
        dim_index: 0-indexed state vector dimension.
        drills: Optional pre-loaded drill list. Loads from default if None.

    Returns:
        List of drills targeting this dimension, sorted by level ascending.
    """
    all_drills = drills if drills is not None else load_drills()
    matching = [d for d in all_drills if dim_index in d.target_dims]
    matching.sort(key=lambda d: d.level)
    return matching


def clear_cache() -> None:
    """Clear the drill cache (for testing)."""
    global _DRILLS_CACHE
    _DRILLS_CACHE = None
