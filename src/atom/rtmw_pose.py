"""Convert RTMW3D whole-body predictions to BoxingWeb canonical poses."""

from __future__ import annotations

from itertools import permutations

import numpy as np


# BoxingWeb's first 15 GT joints are head, neck, right arm, left arm,
# pelvis, right leg, and left leg. RTMW3D uses COCO-WholeBody's first 17
# keypoints. This table maps the non-derived BoxingWeb joints to COCO indices.
BOXINGWEB_TO_COCO = {
    0: 0,   # head / nose
    2: 6,   # right shoulder
    3: 8,   # right elbow
    4: 10,  # right wrist
    5: 5,   # left shoulder
    6: 7,   # left elbow
    7: 9,   # left wrist
    9: 12,  # right hip
    10: 14, # right knee
    11: 16, # right ankle
    12: 11, # left hip
    13: 13, # left knee
    14: 15, # left ankle
}
BODY_JOINTS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14])


def square_normalize(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map image pixels to BoxingWeb's square-padded normalized coordinates."""

    scale = float(max(width, height))
    padding = np.array(((scale - width) / 2.0, (scale - height) / 2.0), dtype=np.float32)
    return (np.asarray(points, dtype=np.float32) + padding) / scale


def square_denormalize(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map BoxingWeb square-padded normalized coordinates back to pixels."""

    scale = float(max(width, height))
    padding = np.array(((scale - width) / 2.0, (scale - height) / 2.0), dtype=np.float32)
    return np.asarray(points, dtype=np.float32) * scale - padding


def rtmw_to_boxingweb(
    keypoints_2d: np.ndarray,
    keypoints_3d: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one RTMW3D person as canonical ``[45,2]`` and ``[45,3]`` arrays."""

    points_2d = np.asarray(keypoints_2d, dtype=np.float32)
    points_3d = np.asarray(keypoints_3d, dtype=np.float32)
    if points_2d.shape != (133, 2) or points_3d.shape != (133, 3):
        raise ValueError("Expected RTMW3D keypoints with shapes [133,2] and [133,3].")
    pose_2d = np.zeros((45, 2), dtype=np.float32)
    pose_3d = np.zeros((45, 3), dtype=np.float32)
    normalized_2d = square_normalize(points_2d, width, height)
    for boxingweb_index, coco_index in BOXINGWEB_TO_COCO.items():
        pose_2d[boxingweb_index] = normalized_2d[coco_index]
        pose_3d[boxingweb_index] = points_3d[coco_index]
    # Neck and pelvis are not explicit COCO joints.
    pose_2d[1] = 0.5 * (normalized_2d[5] + normalized_2d[6])
    pose_3d[1] = 0.5 * (points_3d[5] + points_3d[6])
    pose_2d[8] = 0.5 * (normalized_2d[11] + normalized_2d[12])
    pose_3d[8] = 0.5 * (points_3d[11] + points_3d[12])
    return pose_2d, pose_3d


def pose_bbox(pose_2d: np.ndarray, width: int, height: int, margin: float = 0.25) -> np.ndarray | None:
    """Estimate an image-space boxer box from a canonical GT pose."""

    points = square_denormalize(np.asarray(pose_2d)[BODY_JOINTS], width, height)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if len(points) < 4:
        return None
    lower, upper = points.min(axis=0), points.max(axis=0)
    size = upper - lower
    if np.any(size < 2):
        return None
    box = np.concatenate((lower - margin * size, upper + margin * size))
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
    return box.astype(np.float32)


def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    lower = np.maximum(first[:2], second[:2])
    upper = np.minimum(first[2:], second[2:])
    intersection = float(np.maximum(upper - lower, 0).prod())
    first_area = float(np.maximum(first[2:] - first[:2], 0).prod())
    second_area = float(np.maximum(second[2:] - second[:2], 0).prod())
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_oracle_boxers(
    detections: np.ndarray,
    red_box: np.ndarray | None,
    blue_box: np.ndarray | None,
    minimum_iou: float = 0.05,
) -> tuple[int | None, int | None]:
    """Assign two distinct person detections to GT red/blue pose boxes."""

    boxes = np.asarray(detections, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("detections must have shape [N,4]")
    targets = (red_box, blue_box)
    if len(boxes) == 0:
        return None, None
    candidates: list[tuple[int | None, int | None]] = [(None, None)]
    indices = range(len(boxes))
    candidates.extend((index, None) for index in indices)
    candidates.extend((None, index) for index in indices)
    candidates.extend(permutations(indices, 2))

    def score(pair: tuple[int | None, int | None]) -> float:
        total = 0.0
        for detection_index, target in zip(pair, targets):
            if detection_index is not None and target is not None:
                total += bbox_iou(boxes[detection_index], target)
        return total

    selected = max(candidates, key=score)
    result: list[int | None] = []
    for detection_index, target in zip(selected, targets):
        if detection_index is None or target is None or bbox_iou(boxes[detection_index], target) < minimum_iou:
            result.append(None)
        else:
            result.append(int(detection_index))
    return result[0], result[1]


def interpolate_short_gaps(values: np.ndarray, valid: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Linearly fill only bounded missing runs no longer than ``max_gap``."""

    output = np.asarray(values, dtype=np.float32).copy()
    present = np.asarray(valid, dtype=bool)
    start = 0
    while start < len(present):
        if present[start]:
            start += 1
            continue
        stop = start
        while stop < len(present) and not present[stop]:
            stop += 1
        if start > 0 and stop < len(present) and stop - start <= max_gap:
            for frame in range(start, stop):
                weight = (frame - start + 1) / (stop - start + 1)
                output[frame] = (1.0 - weight) * output[start - 1] + weight * output[stop]
        start = stop
    return output


def interpolate_all_gaps(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Interpolate all missing frames, including nearest-value edge fill."""

    output = np.asarray(values, dtype=np.float32).copy()
    known = np.flatnonzero(np.asarray(valid, dtype=bool))
    if not len(known):
        return output
    frames = np.arange(len(output))
    flattened = output.reshape(len(output), -1)
    for column in range(flattened.shape[1]):
        flattened[:, column] = np.interp(frames, known, flattened[known, column])
    return flattened.reshape(output.shape)


def smooth_sequence(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a centered odd-width moving average along the time axis."""

    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    array = np.asarray(values, dtype=np.float32)
    if window == 1:
        return array.copy()
    padding = window // 2
    padded = np.pad(array, ((padding, padding),) + ((0, 0),) * (array.ndim - 1), mode="edge")
    cumulative = np.cumsum(padded, axis=0, dtype=np.float64)
    cumulative = np.concatenate((np.zeros_like(cumulative[:1]), cumulative), axis=0)
    return ((cumulative[window:] - cumulative[:-window]) / window).astype(np.float32)
