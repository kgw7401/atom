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


def _box_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Return a soft spatial-continuity score for two xyxy boxes."""

    first, second = np.asarray(first, np.float32), np.asarray(second, np.float32)
    if first.shape != (4,) or second.shape != (4,) or np.any(first[2:] <= first[:2]) or np.any(second[2:] <= second[:2]):
        return 0.0
    lower, upper = np.maximum(first[:2], second[:2]), np.minimum(first[2:], second[2:])
    intersection = float(np.maximum(upper - lower, 0).prod())
    areas = [float(np.maximum(box[2:] - box[:2], 0).prod()) for box in (first, second)]
    iou = intersection / max(areas[0] + areas[1] - intersection, 1e-6)
    centers = [0.5 * (box[:2] + box[2:]) for box in (first, second)]
    scale = max(float(np.linalg.norm(first[2:] - first[:2])), 1.0)
    center_score = max(0.0, 1.0 - float(np.linalg.norm(centers[0] - centers[1])) / (2.5 * scale))
    return iou + center_score


def _fallback_assignment(
    frame_ids: np.ndarray,
    frame_probabilities: np.ndarray,
    frame_boxes: np.ndarray,
    primary: tuple[int | None, int | None],
    last_boxes: tuple[np.ndarray | None, np.ndarray | None],
    require_two_boxers: bool,
    spatial_bonus: float,
) -> tuple[int | None, int | None]:
    """Keep available primary IDs and spatially reconnect missing identities."""

    available = [int(value) for value in frame_ids if value >= 0]
    index_by_id = {int(value): index for index, value in enumerate(frame_ids) if value >= 0}
    fixed = [track_id if track_id in index_by_id else None for track_id in primary]
    missing = [identity for identity, track_id in enumerate(fixed) if track_id is None]
    candidates = [track_id for track_id in available if track_id not in fixed]
    if not missing or not candidates:
        return fixed[0], fixed[1]
    count = min(len(missing), len(candidates))
    choices = permutations(candidates, count)

    def score(identity: int, track_id: int) -> float:
        index = index_by_id[track_id]
        probability = frame_probabilities[index]
        appearance = float(np.log(probability[identity] + 1e-6) - np.log(probability[2] + 1e-6))
        spatial = 0.0 if last_boxes[identity] is None else _box_similarity(frame_boxes[index], last_boxes[identity])
        return appearance + spatial_bonus * spatial

    best = max(choices, key=lambda values: sum(score(identity, track_id) for identity, track_id in zip(missing, values)))
    for identity, track_id in zip(missing, best):
        fixed[identity] = track_id
    if require_two_boxers and len(available) >= 2 and any(value is None for value in fixed):
        remaining = [track_id for track_id in available if track_id not in fixed]
        for identity, value in enumerate(fixed):
            if value is None and remaining:
                fixed[identity] = remaining.pop(0)
    return fixed[0], fixed[1]


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


def _best_assignment(
    ids: np.ndarray,
    probabilities: np.ndarray,
    previous: tuple[int | None, int | None] = (None, None),
    identity_threshold: float = .35,
    continuity_bonus: float = .25,
    require_two_boxers: bool = False,
    boxes: np.ndarray | None = None,
    last_boxes: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
    spatial_bonus: float = 0.0,
) -> tuple[int | None, int | None]:
    """Return distinct IDs maximizing boxer-vs-non-boxer evidence in a block."""

    scores: dict[int, np.ndarray] = {}
    for frame_ids, frame_probability in zip(ids, probabilities):
        for track_id, probability in zip(frame_ids, frame_probability):
            if track_id < 0:
                continue
            scores.setdefault(int(track_id), []).append(probability)
    means = {track_id: np.mean(values, axis=0) for track_id, values in scores.items()}
    mean_boxes: dict[int, np.ndarray] = {}
    if boxes is not None:
        for frame_ids, frame_boxes in zip(ids, boxes):
            for track_id, box in zip(frame_ids, frame_boxes):
                if track_id >= 0 and np.any(box[2:] > box[:2]):
                    mean_boxes.setdefault(int(track_id), []).append(box)
        mean_boxes = {track_id: np.mean(values, axis=0) for track_id, values in mean_boxes.items()}
    candidates = list(means)
    if not candidates:
        return None, None

    def identity_score(track_id: int, identity: int) -> float:
        probability = means[track_id]
        if not require_two_boxers and probability[identity] < identity_threshold:
            return -np.inf
        evidence = np.log(probability[identity] + 1e-6) - np.log(probability[2] + 1e-6)
        if previous[identity] == track_id:
            evidence += continuity_bonus
        if last_boxes[identity] is not None and track_id in mean_boxes:
            evidence += spatial_bonus * _box_similarity(mean_boxes[track_id], last_boxes[identity])
        return float(evidence)

    if require_two_boxers and len(candidates) >= 2:
        assignments = list(permutations(candidates, 2))
    else:
        assignments: list[tuple[int | None, int | None]] = [(None, None)]
        assignments.extend((track_id, None) for track_id in candidates)
        assignments.extend((None, track_id) for track_id in candidates)
        assignments.extend(permutations(candidates, 2))

    def assignment_score(pair: tuple[int | None, int | None]) -> float:
        values = [
            0.0 if track_id is None else identity_score(track_id, identity)
            for identity, track_id in enumerate(pair)
        ]
        return float(sum(values)) if np.isfinite(values).all() else -np.inf

    return max(assignments, key=assignment_score)


def refine_boxer_tracks(
    tracks: dict[str, np.ndarray],
    cadence: int = 10,
    identity_threshold: float = .35,
    continuity_bonus: float = .25,
    require_two_boxers: bool = False,
    fallback_missing: bool = True,
    spatial_bonus: float = 1.5,
) -> dict[str, np.ndarray]:
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
    red_boxes = np.zeros((frames, 4), dtype=np.float32)
    blue_boxes = np.zeros_like(red_boxes)
    red_ids = np.full(frames, -1, dtype=np.int64)
    blue_ids = np.full(frames, -1, dtype=np.int64)
    input_boxes = np.asarray(tracks.get("boxes", np.zeros((*ids.shape, 4))), dtype=np.float32)
    if input_boxes.shape != (*ids.shape, 4):
        raise ValueError("boxes must have shape [T,N,4] when provided")
    last_boxes: list[np.ndarray | None] = [None, None]
    previous: tuple[int | None, int | None] = (None, None)
    for start in range(0, frames, cadence):
        stop = min(start + cadence, frames)
        red_id, blue_id = _best_assignment(
            ids[start:stop], probabilities[start:stop], previous,
            identity_threshold=identity_threshold, continuity_bonus=continuity_bonus,
            require_two_boxers=require_two_boxers,
            boxes=input_boxes[start:stop], last_boxes=(last_boxes[0], last_boxes[1]),
            spatial_bonus=spatial_bonus,
        )
        previous = red_id, blue_id
        for frame in range(start, stop):
            selected_pair = (red_id, blue_id)
            available = {int(value) for value in ids[frame] if value >= 0}
            if fallback_missing and any(selected is not None and selected not in available for selected in selected_pair):
                selected_pair = _fallback_assignment(
                    ids[frame], probabilities[frame], input_boxes[frame], selected_pair,
                    (last_boxes[0], last_boxes[1]), require_two_boxers, spatial_bonus,
                )
            for identity, (output_id, output_2d, output_3d, output_box, selected) in enumerate((
                (red_ids, red_2d, red_3d, red_boxes, selected_pair[0]),
                (blue_ids, blue_2d, blue_3d, blue_boxes, selected_pair[1]),
            )):
                if selected is None:
                    continue
                matches = np.flatnonzero(ids[frame] == selected)
                if len(matches):
                    index = int(matches[0])
                    output_id[frame] = selected
                    output_2d[frame] = input_2d[frame, index]
                    output_3d[frame] = input_3d[frame, index]
                    output_box[frame] = input_boxes[frame, index]
                    if np.any(input_boxes[frame, index, 2:] > input_boxes[frame, index, :2]):
                        last_boxes[identity] = input_boxes[frame, index].copy()
    return {
        "pose_red_2d": red_2d,
        "pose_blue_2d": blue_2d,
        "pose_red_3d": red_3d,
        "pose_blue_3d": blue_3d,
        "box_red": red_boxes,
        "box_blue": blue_boxes,
        "red_track_ids": red_ids,
        "blue_track_ids": blue_ids,
    }


def load_track_npz(path: str) -> dict[str, np.ndarray]:
    """Load the compact interchange format produced after pose tracking + UVE."""

    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def canonical_pose_payload(
    tracks: dict[str, np.ndarray],
    cadence: int = 10,
    identity_threshold: float = .35,
    continuity_bonus: float = .25,
    require_two_boxers: bool = False,
    fallback_missing: bool = True,
    spatial_bonus: float = 1.5,
) -> dict[str, Any]:
    """Return the pose payload consumed by the temporal detector, with metadata."""

    payload: dict[str, Any] = refine_boxer_tracks(
        tracks, cadence, identity_threshold=identity_threshold, continuity_bonus=continuity_bonus,
        require_two_boxers=require_two_boxers,
        fallback_missing=fallback_missing,
        spatial_bonus=spatial_bonus,
    )
    payload["format"] = "atom-canonical-boxer-pose-v1"
    payload["uve_cadence"] = cadence
    payload["uve_identity_threshold"] = identity_threshold
    payload["uve_continuity_bonus"] = continuity_bonus
    payload["uve_require_two_boxers"] = require_two_boxers
    payload["uve_fallback_missing"] = fallback_missing
    payload["uve_spatial_bonus"] = spatial_bonus
    return payload
