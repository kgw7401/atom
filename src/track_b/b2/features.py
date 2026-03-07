"""B2 Task 6: Feature engineering from pose keypoints for action classification.

Converts a 30-frame window of MediaPipe pose keypoints into a fixed-dimension
feature vector suitable for XGBoost classification.

Feature groups (120 features total, 5 summary stats each):
  - 6 joint angles (elbow L/R, shoulder L/R, hip L/R)          → 30 features
  - 6 angular velocities (frame-to-frame angle changes)         → 30 features
  - 8 relative distances (wrist-to-joint, wrist-to-nose, etc.)  → 40 features
  - 4 wrist acceleration components (x/y per wrist)             → 20 features

Summary statistics per feature series: [mean, std, min, max, range].

Usage::

    keypoints = np.zeros((30, 33, 4), dtype=np.float32)
    vector = extract_features(keypoints)  # shape (120,)
    df = build_feature_dataframe(windows, labels)  # for XGBoost training
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── MediaPipe pose landmark indices ──────────────────────────────────────────

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26

# ── Angle definitions: (vertex, arm1, arm2) triplets ─────────────────────────

_ANGLE_DEFS: list[tuple[int, int, int]] = [
    (LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST),       # left elbow angle
    (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST),    # right elbow angle
    (LEFT_SHOULDER, LEFT_HIP, LEFT_ELBOW),         # left shoulder angle
    (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ELBOW),      # right shoulder angle
    (LEFT_HIP, LEFT_SHOULDER, LEFT_KNEE),          # left hip angle
    (RIGHT_HIP, RIGHT_SHOULDER, RIGHT_KNEE),       # right hip angle
]
_ANGLE_NAMES: list[str] = [
    "angle_left_elbow",
    "angle_right_elbow",
    "angle_left_shoulder",
    "angle_right_shoulder",
    "angle_left_hip",
    "angle_right_hip",
]

# ── Distance definitions: (landmark_a, landmark_b) pairs ─────────────────────

_DIST_DEFS: list[tuple[int, int]] = [
    (LEFT_WRIST, LEFT_SHOULDER),    # left wrist extension
    (RIGHT_WRIST, RIGHT_SHOULDER),  # right wrist extension
    (LEFT_WRIST, NOSE),             # left reach toward head
    (RIGHT_WRIST, NOSE),            # right reach toward head
    (LEFT_WRIST, RIGHT_SHOULDER),   # cross-body reach (jab orthodox / cross southpaw)
    (RIGHT_WRIST, LEFT_SHOULDER),   # cross-body reach (cross orthodox / jab southpaw)
    (LEFT_WRIST, RIGHT_WRIST),      # hands separation
    (NOSE, LEFT_HIP),               # torso lean proxy
]
_DIST_NAMES: list[str] = [
    "dist_left_wrist_left_shoulder",
    "dist_right_wrist_right_shoulder",
    "dist_left_wrist_nose",
    "dist_right_wrist_nose",
    "dist_left_wrist_right_shoulder",
    "dist_right_wrist_left_shoulder",
    "dist_left_wrist_right_wrist",
    "dist_nose_left_hip",
]

# ── Wrist acceleration components ─────────────────────────────────────────────

_ACCEL_NAMES: list[str] = [
    "accel_left_wrist_x",
    "accel_left_wrist_y",
    "accel_right_wrist_x",
    "accel_right_wrist_y",
]

# ── Summary stat suffixes ─────────────────────────────────────────────────────

_STAT_SUFFIXES: list[str] = ["mean", "std", "min", "max", "range"]

# Total feature count (constant — used by downstream tasks and tests)
FEATURE_DIM: int = (
    len(_ANGLE_NAMES) * len(_STAT_SUFFIXES)        # 6 × 5 = 30
    + len(_ANGLE_NAMES) * len(_STAT_SUFFIXES)      # 6 × 5 = 30  (angular velocity)
    + len(_DIST_NAMES) * len(_STAT_SUFFIXES)       # 8 × 5 = 40
    + len(_ACCEL_NAMES) * len(_STAT_SUFFIXES)      # 4 × 5 = 20
)  # = 120

# ── Core geometry ─────────────────────────────────────────────────────────────


def _joint_angle(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Angle (radians) at `vertex` between rays vertex→a and vertex→b.

    Args:
        a, vertex, b: 2-element arrays (x, y) in normalised coordinates.

    Returns:
        Angle in [0, π]. Returns 0.0 if either ray has zero length.
    """
    va = a - vertex
    vb = b - vertex
    len_a = np.linalg.norm(va)
    len_b = np.linalg.norm(vb)
    if len_a < 1e-9 or len_b < 1e-9:
        return 0.0
    cos_theta = np.dot(va, vb) / (len_a * len_b)
    # Clamp to [-1, 1] to handle floating-point errors
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


# ── Per-window feature extractors ─────────────────────────────────────────────


def extract_angles(keypoints: np.ndarray) -> np.ndarray:
    """Compute joint angles for each frame in the window.

    Args:
        keypoints: (n_frames, 33, 4) array — (x, y, z, visibility) per landmark.

    Returns:
        (n_frames, 6) array of joint angles in radians.
    """
    n = keypoints.shape[0]
    angles = np.zeros((n, len(_ANGLE_DEFS)), dtype=np.float32)
    xy = keypoints[:, :, :2]  # (n, 33, 2) — only x, y
    for f in range(n):
        for j, (vertex, arm1, arm2) in enumerate(_ANGLE_DEFS):
            angles[f, j] = _joint_angle(xy[f, arm1], xy[f, vertex], xy[f, arm2])
    return angles


def extract_distances(keypoints: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances between landmark pairs for each frame.

    Args:
        keypoints: (n_frames, 33, 4) array.

    Returns:
        (n_frames, 8) array of distances in normalised coordinate units.
    """
    xy = keypoints[:, :, :2]  # (n, 33, 2)
    n = xy.shape[0]
    dists = np.zeros((n, len(_DIST_DEFS)), dtype=np.float32)
    for j, (lm_a, lm_b) in enumerate(_DIST_DEFS):
        diff = xy[:, lm_a, :] - xy[:, lm_b, :]  # (n, 2)
        dists[:, j] = np.linalg.norm(diff, axis=1)
    return dists


def extract_wrist_accelerations(keypoints: np.ndarray) -> np.ndarray:
    """Compute second-order finite differences (acceleration) of wrist positions.

    Args:
        keypoints: (n_frames, 33, 4) array.

    Returns:
        (max(0, n_frames - 2), 4) array — [lw_x, lw_y, rw_x, rw_y] accelerations.
        Returns zeros array of shape (1, 4) if n_frames < 3.
    """
    if keypoints.shape[0] < 3:
        return np.zeros((1, 4), dtype=np.float32)
    lw = keypoints[:, LEFT_WRIST, :2]   # (n, 2)
    rw = keypoints[:, RIGHT_WRIST, :2]  # (n, 2)
    # Second-order finite difference: a[i] = x[i+1] - 2*x[i] + x[i-1]
    lw_accel = lw[2:] - 2 * lw[1:-1] + lw[:-2]   # (n-2, 2)
    rw_accel = rw[2:] - 2 * rw[1:-1] + rw[:-2]   # (n-2, 2)
    return np.concatenate([lw_accel, rw_accel], axis=1).astype(np.float32)


def _summarize(series: np.ndarray) -> np.ndarray:
    """Compute [mean, std, min, max, range] over axis 0 for each column.

    Args:
        series: (n_frames, n_features) array.

    Returns:
        (n_features * 5,) flattened array of summary statistics.
    """
    mean = series.mean(axis=0)
    std = series.std(axis=0)
    minimum = series.min(axis=0)
    maximum = series.max(axis=0)
    rng = maximum - minimum
    return np.concatenate([mean, std, minimum, maximum, rng]).astype(np.float32)


def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract a fixed-dimension feature vector from a pose keypoint window.

    This is the main entry point for the feature engineering pipeline.

    Args:
        window: (n_frames, 33, 4) keypoint array — one sliding window.
                n_frames should be 30 (1 second at 30fps) for standard use.

    Returns:
        (FEATURE_DIM,) = (120,) float32 feature vector with components:
        - joint angle summaries        (30 features)
        - angular velocity summaries   (30 features)
        - distance summaries           (40 features)
        - wrist acceleration summaries (20 features)

    Raises:
        ValueError: If window has fewer than 2 frames or wrong landmark count.
    """
    if window.ndim != 3 or window.shape[1] != 33 or window.shape[2] < 2:
        raise ValueError(
            f"Expected window shape (n_frames, 33, 4), got {window.shape}"
        )
    if window.shape[0] < 2:
        raise ValueError(f"Window must have at least 2 frames, got {window.shape[0]}")

    angles = extract_angles(window)           # (n, 6)
    ang_vel = np.diff(angles, axis=0)         # (n-1, 6)
    distances = extract_distances(window)     # (n, 8)
    accelerations = extract_wrist_accelerations(window)  # (n-2, 4)

    parts = [
        _summarize(angles),        # 30
        _summarize(ang_vel),       # 30
        _summarize(distances),     # 40
        _summarize(accelerations), # 20
    ]
    vector = np.concatenate(parts)
    assert vector.shape == (FEATURE_DIM,), f"Feature dim mismatch: {vector.shape}"
    return vector


# ── Feature naming ─────────────────────────────────────────────────────────────


def feature_names() -> list[str]:
    """Return ordered list of feature names matching extract_features() output.

    Returns:
        List of 120 strings, e.g. ["angle_left_elbow_mean", ..., "accel_right_wrist_y_range"].
        Order matches the feature vector produced by extract_features().
    """
    names: list[str] = []
    for group, base_names in [
        (_ANGLE_NAMES, _ANGLE_NAMES),
        ([f"vel_{n}" for n in _ANGLE_NAMES], [f"vel_{n}" for n in _ANGLE_NAMES]),
        (_DIST_NAMES, _DIST_NAMES),
        (_ACCEL_NAMES, _ACCEL_NAMES),
    ]:
        for base in group:
            for stat in _STAT_SUFFIXES:
                names.append(f"{base}_{stat}")
    return names


# ── Batch processing ──────────────────────────────────────────────────────────


def build_feature_dataframe(
    windows: list[np.ndarray],
    labels: list[int],
) -> pd.DataFrame:
    """Convert a list of keypoint windows into a labeled feature DataFrame.

    Args:
        windows: List of (n_frames, 33, 4) arrays — one per training clip.
        labels: Integer class labels, one per window.

    Returns:
        DataFrame with 120 named feature columns + "label" column.
        Ready for XGBoost training.

    Raises:
        ValueError: If windows and labels have different lengths.
    """
    if len(windows) != len(labels):
        raise ValueError(
            f"windows ({len(windows)}) and labels ({len(labels)}) must have same length"
        )
    cols = feature_names()
    rows = [extract_features(w) for w in windows]
    df = pd.DataFrame(rows, columns=cols, dtype=np.float32)
    df["label"] = np.array(labels, dtype=np.int32)
    return df


def windows_from_pose_df(
    pose_df: pd.DataFrame,
    window_size: int = 30,
    stride: int = 1,
) -> list[np.ndarray]:
    """Slide a window over a PoseFrame DataFrame and extract keypoint arrays.

    Each window contains `window_size` consecutive frames of pose keypoints,
    reconstructed from the flat kp_*_x/y/z/vis columns.

    Args:
        pose_df: PoseFrame DataFrame with columns kp_0_x ... kp_32_vis.
        window_size: Number of frames per window (default 30 = 1 second at 30fps).
        stride: Step between window starts (default 1 frame).

    Returns:
        List of (window_size, 33, 4) keypoint arrays in frame order.
        Empty list if pose_df has fewer rows than window_size.
    """
    n = len(pose_df)
    if n < window_size:
        return []

    # Build (n, 33, 4) array from flat columns
    kp_array = np.zeros((n, 33, 4), dtype=np.float32)
    for lm in range(33):
        for ci, coord in enumerate(("x", "y", "z", "vis")):
            col = f"kp_{lm}_{coord}"
            if col in pose_df.columns:
                kp_array[:, lm, ci] = pose_df[col].to_numpy(dtype=np.float32)

    windows = []
    for start in range(0, n - window_size + 1, stride):
        windows.append(kp_array[start : start + window_size])
    return windows
