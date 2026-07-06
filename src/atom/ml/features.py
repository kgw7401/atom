"""Shared feature engineering for pose keypoints.

Converts raw keypoints (AlphaPose COCO or MediaPipe) into a normalized
24-dimensional feature vector per frame:
  6 joints * 4 features (x, y, dx, dy) = 24

Used by both offline training (AlphaPose) and online inference (MediaPipe).
"""

from __future__ import annotations

import numpy as np

# Joint index mappings: joint_name -> index in raw keypoint array
ALPHAPOSE_INDICES = {
    "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10,
    "l_hip": 11, "r_hip": 12,
}

MEDIAPIPE_INDICES = {
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
}

INDEX_MAP = {
    "alphapose": ALPHAPOSE_INDICES,
    "mediapipe": MEDIAPIPE_INDICES,
}

# The 6 joints we keep (in order) for the feature vector
JOINT_NAMES = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"]
# Reference joints for normalization
REF_JOINTS = ["l_hip", "r_hip"]

NUM_JOINTS = len(JOINT_NAMES)        # 6
FEATURES_PER_JOINT = 4               # x, y, dx, dy
FEATURE_DIM = NUM_JOINTS * FEATURES_PER_JOINT  # 24


def extract_joint_subset(
    keypoints: np.ndarray,
    source: str = "mediapipe",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the 6-joint subset + 2 reference joints from raw keypoints.

    Args:
        keypoints: (T, K, 2+) where K = total keypoints, last dim >= 2 (x, y, ...).
        source: "alphapose" or "mediapipe".

    Returns:
        joints: (T, 6, 2) — the 6 target joints (x, y).
        refs: (T, 2, 2) — the 2 reference hip joints (x, y).
    """
    idx = INDEX_MAP[source]
    joint_indices = [idx[name] for name in JOINT_NAMES]
    ref_indices = [idx[name] for name in REF_JOINTS]

    joints = keypoints[:, joint_indices, :2]   # (T, 6, 2)
    refs = keypoints[:, ref_indices, :2]       # (T, 2, 2)
    return joints, refs


def normalize(
    joints: np.ndarray,
    refs: np.ndarray,
) -> np.ndarray:
    """Center on hip midpoint and scale by shoulder width.

    Args:
        joints: (T, 6, 2)
        refs: (T, 2, 2) — [l_hip, r_hip]

    Returns:
        normalized: (T, 6, 2)
    """
    # Hip midpoint: (T, 1, 2)
    hip_center = refs.mean(axis=1, keepdims=True)

    # Center joints on hip midpoint
    centered = joints - hip_center

    # Shoulder width for scaling: distance between joint[0] (l_shoulder) and joint[1] (r_shoulder)
    shoulder_diff = centered[:, 0, :] - centered[:, 1, :]  # (T, 2)
    shoulder_width = np.linalg.norm(shoulder_diff, axis=1, keepdims=True)  # (T, 1)

    # Avoid division by zero
    shoulder_width = np.maximum(shoulder_width, 1e-6)

    # Scale: (T, 6, 2) / (T, 1, 1)
    normalized = centered / shoulder_width[:, :, np.newaxis]
    return normalized


def compute_velocity(positions: np.ndarray) -> np.ndarray:
    """Compute per-frame velocity (dx, dy) via finite differences.

    Args:
        positions: (T, 6, 2)

    Returns:
        velocity: (T, 6, 2) — first frame velocity is zero.
    """
    velocity = np.zeros_like(positions)
    if positions.shape[0] > 1:
        velocity[1:] = positions[1:] - positions[:-1]
    return velocity


def flatten_features(positions: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Concatenate position + velocity and flatten to (T, 24).

    Args:
        positions: (T, 6, 2)
        velocity: (T, 6, 2)

    Returns:
        features: (T, 24) — [x0, y0, dx0, dy0, x1, y1, dx1, dy1, ...]
    """
    T = positions.shape[0]
    # Interleave: (T, 6, 4) then flatten
    combined = np.concatenate([positions, velocity], axis=2)  # (T, 6, 4)
    return combined.reshape(T, FEATURE_DIM)


def extract_features(
    keypoints: np.ndarray,
    source: str = "mediapipe",
) -> np.ndarray:
    """Full pipeline: raw keypoints → normalized 24-dim feature vector per frame.

    Args:
        keypoints: (T, K, 2+) raw keypoints from AlphaPose or MediaPipe.
        source: "alphapose" or "mediapipe".

    Returns:
        features: (T, 24) normalized feature vectors.
    """
    joints, refs = extract_joint_subset(keypoints, source)
    normed = normalize(joints, refs)
    vel = compute_velocity(normed)
    return flatten_features(normed, vel)
