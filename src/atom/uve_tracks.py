"""Convert periodically identity-classified video tracks into boxer poses.

UVE (the identity stage described by BoxMind) classifies each tracked person as
red boxer, blue boxer, or non-boxer.  This module deliberately separates that
appearance classifier from temporal event detection: callers supply its
three-class probabilities, and the refiner makes a stable, distinct red/blue
track assignment every ``cadence`` frames.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np


def _validate(tracks: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ("track_ids", "joints_2d", "joints_3d", "identity_probabilities")
    missing = [key for key in required if key not in tracks]
    if missing:
        raise ValueError(f"Missing track fields: {', '.join(missing)}")
    ids = np.asarray(tracks["track_ids"])
    pose_2d = np.asarray(tracks["joints_2d"], dtype=np.float32)
    pose_3d = np.asarray(tracks["joints_3d"], dtype=np.float32)
    probabilities = np.asarray(tracks["identity_probabilities"], dtype=np.float32)
    if ids.ndim != 2 or pose_2d.ndim != 4 or pose_3d.ndim != 4 or probabilities.ndim != 3:
        raise ValueError("Expected track_ids [T,N], joints_2d [T,N,J,2], joints_3d [T,N,J,3], identity_probabilities [T,N,3].")
    if pose_2d.shape[:2] != ids.shape or pose_3d.shape[:2] != ids.shape or probabilities.shape[:2] != ids.shape:
        raise ValueError("Track arrays must have equal [T,N] dimensions.")
    if pose_2d.shape[-1] != 2 or pose_3d.shape[-1] != 3 or probabilities.shape[-1] != 3:
        raise ValueError("Expected 2D xy, 3D xyz, and red/blue/non-boxer probabilities.")
    return ids, pose_2d, pose_3d, probabilities


def _best_assignment(ids: np.ndarray, probabilities: np.ndarray) -> tuple[int | None, int | None]:
    """Return two distinct track IDs maximizing red+blue confidence in a block."""

    scores: dict[int, np.ndarray] = {}
    for frame_ids, frame_probability in zip(ids, probabilities):
        for track_id, probability in zip(frame_ids, frame_probability):
            if track_id < 0:
                continue
            scores.setdefault(int(track_id), []).append(probability)
    means = {track_id: np.mean(values, axis=0) for track_id, values in scores.items()}
    candidates = list(means)
    if len(candidates) < 2:
        return (candidates[0], None) if candidates else (None, None)
    red_id, blue_id = max(
        permutations(candidates, 2),
        key=lambda pair: float(means[pair[0]][0] + means[pair[1]][1]),
    )
    return red_id, blue_id


def refine_boxer_tracks(tracks: dict[str, np.ndarray], cadence: int = 10) -> dict[str, np.ndarray]:
    """Build frame-aligned red/blue pose arrays from UV-identity probabilities.

    ``identity_probabilities[..., 0:3]`` is ``(red, blue, non_boxer)``.  The
    selected IDs are fixed within each cadence block, as in periodic UVE
    verification, avoiding frame-by-frame identity swaps during occlusion.
    """

    if cadence < 1:
        raise ValueError("cadence must be positive")
    ids, input_2d, input_3d, probabilities = _validate(tracks)
    frames, _, joints = input_2d.shape[:3]
    red_2d = np.zeros((frames, joints, 2), dtype=np.float32)
    blue_2d = np.zeros_like(red_2d)
    red_3d = np.zeros((frames, joints, 3), dtype=np.float32)
    blue_3d = np.zeros_like(red_3d)
    red_ids = np.full(frames, -1, dtype=np.int64)
    blue_ids = np.full(frames, -1, dtype=np.int64)
    for start in range(0, frames, cadence):
        stop = min(start + cadence, frames)
        red_id, blue_id = _best_assignment(ids[start:stop], probabilities[start:stop])
        for frame in range(start, stop):
            for output_id, output_2d, output_3d, selected in (
                (red_ids, red_2d, red_3d, red_id),
                (blue_ids, blue_2d, blue_3d, blue_id),
            ):
                if selected is None:
                    continue
                matches = np.flatnonzero(ids[frame] == selected)
                if len(matches):
                    index = int(matches[0])
                    output_id[frame] = selected
                    output_2d[frame] = input_2d[frame, index]
                    output_3d[frame] = input_3d[frame, index]
    return {
        "pose_red_2d": red_2d,
        "pose_blue_2d": blue_2d,
        "pose_red_3d": red_3d,
        "pose_blue_3d": blue_3d,
        "red_track_ids": red_ids,
        "blue_track_ids": blue_ids,
    }


def load_track_npz(path: str) -> dict[str, np.ndarray]:
    """Load the compact interchange format produced after pose tracking + UVE."""

    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def canonical_pose_payload(tracks: dict[str, np.ndarray], cadence: int = 10) -> dict[str, Any]:
    """Return the pose payload consumed by the temporal detector, with metadata."""

    payload: dict[str, Any] = refine_boxer_tracks(tracks, cadence)
    payload["format"] = "atom-canonical-boxer-pose-v1"
    payload["uve_cadence"] = cadence
    return payload
