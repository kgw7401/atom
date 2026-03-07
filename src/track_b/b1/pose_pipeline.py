"""B1 Task 3: Quality filtering, normalization & storage.

Combines video ingestion (Task 1) and pose estimation (Task 2) into a complete
single-person pipeline. Handles:
- Confidence filtering (drop frames <0.5)
- Keypoint normalization (scale-invariant, relative to body bounding box)
- Frame rate normalization to 30fps
- Parquet serialization with PoseFrame schema

PoseFrame schema (per frame):
  video_id       : str
  fighter_id     : str   ("user" for single-person, null for training mode)
  frame_number   : int
  timestamp      : float
  keypoints      : float32[33, 4]  — flattened to 33*4=132 columns in Parquet
  confidence     : float
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from track_b.b1.pose_estimator import PoseEstimator, PoseLandmark
from track_b.b1.video_ingest import extract_frames_analysis, extract_frames_training

# Landmark indices used for body bounding box normalization
_BODY_LANDMARKS = [
    PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_ELBOW, PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST,
]

# Parquet schema — keypoints stored as 132 individual float32 columns (kp_0_x, kp_0_y, ...)
def _build_schema() -> pa.Schema:
    fields = [
        pa.field("video_id", pa.string()),
        pa.field("fighter_id", pa.string()),
        pa.field("frame_number", pa.int32()),
        pa.field("timestamp", pa.float32()),
        pa.field("confidence", pa.float32()),
    ]
    # 33 keypoints × 4 values
    for i in range(33):
        for j, dim in enumerate(["x", "y", "z", "vis"]):
            fields.append(pa.field(f"kp_{i}_{dim}", pa.float32()))
    return pa.schema(fields)


POSE_FRAME_SCHEMA = _build_schema()


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Normalize keypoints relative to body bounding box.

    Makes poses scale-invariant across different camera distances and resolutions.
    Uses left/right shoulder + hip + elbow + wrist landmarks to define the bounding box.

    Args:
        keypoints: float32 array (33, 4) with (x, y, z, visibility).

    Returns:
        Normalized float32 array (33, 4). Same shape.
    """
    body_kps = keypoints[_BODY_LANDMARKS, :2]  # (8, 2) — x, y only

    x_min, y_min = body_kps.min(axis=0)
    x_max, y_max = body_kps.max(axis=0)

    width = x_max - x_min
    height = y_max - y_min
    scale = max(width, height)

    if scale < 1e-6:
        # Degenerate case — all body landmarks at same point
        return keypoints.copy()

    normalized = keypoints.copy()
    normalized[:, 0] = (keypoints[:, 0] - x_min) / scale  # x
    normalized[:, 1] = (keypoints[:, 1] - y_min) / scale  # y
    # z: relative depth already normalized by MediaPipe; keep as-is
    # visibility: keep as-is
    return normalized


def _keypoints_to_row(keypoints: np.ndarray) -> dict:
    """Flatten (33, 4) keypoints array into column dict for Parquet."""
    row = {}
    for i in range(33):
        row[f"kp_{i}_x"] = float(keypoints[i, 0])
        row[f"kp_{i}_y"] = float(keypoints[i, 1])
        row[f"kp_{i}_z"] = float(keypoints[i, 2])
        row[f"kp_{i}_vis"] = float(keypoints[i, 3])
    return row


def _row_to_keypoints(row: dict) -> np.ndarray:
    """Reconstruct (33, 4) keypoints array from Parquet row."""
    kp = np.zeros((33, 4), dtype=np.float32)
    for i in range(33):
        kp[i, 0] = row[f"kp_{i}_x"]
        kp[i, 1] = row[f"kp_{i}_y"]
        kp[i, 2] = row[f"kp_{i}_z"]
        kp[i, 3] = row[f"kp_{i}_vis"]
    return kp


def run_training_pipeline(
    video_path: str | Path,
    video_id: str,
    estimator: PoseEstimator,
    min_confidence: float = 0.5,
) -> pd.DataFrame:
    """Training mode pipeline: extract all frames, estimate pose, filter, normalize.

    For BoxingVI clips: uses all frames, no FPS resampling, fighter_id = "".

    Args:
        video_path: Path to local video clip.
        video_id: Identifier for this video (e.g., "boxingvi_jab_001").
        estimator: Initialized PoseEstimator.
        min_confidence: Drop frames below this confidence.

    Returns:
        DataFrame with PoseFrame schema (filtered + normalized).
    """
    _, frames = extract_frames_training(video_path)
    records = []

    for frame in frames:
        pose = estimator.estimate(frame.image)
        if pose.confidence < min_confidence:
            continue
        normalized = normalize_keypoints(pose.keypoints)
        row = {
            "video_id": video_id,
            "fighter_id": "",  # no fighter identity in training mode
            "frame_number": frame.frame_number,
            "timestamp": frame.timestamp,
            "confidence": pose.confidence,
        }
        row.update(_keypoints_to_row(normalized))
        records.append(row)

    if not records:
        return pd.DataFrame(columns=[f.name for f in POSE_FRAME_SCHEMA])

    return pd.DataFrame(records)


def run_analysis_pipeline(
    source: str | Path,
    video_id: str,
    fighter_id: str,
    estimator: PoseEstimator,
    target_fps: int = 30,
    start_time: float | None = None,
    end_time: float | None = None,
    min_confidence: float = 0.5,
) -> pd.DataFrame:
    """Analysis mode pipeline (single-person): extract frames, estimate pose, filter, normalize.

    Args:
        source: Local file path or YouTube URL.
        video_id: Identifier for this video.
        fighter_id: Fighter identifier (e.g., "user").
        estimator: Initialized PoseEstimator.
        target_fps: Target frame rate for extraction.
        start_time: Start time in seconds (optional).
        end_time: End time in seconds (optional).
        min_confidence: Drop frames below this confidence.

    Returns:
        DataFrame with PoseFrame schema (filtered + normalized).
    """
    _, frames = extract_frames_analysis(source, target_fps, start_time, end_time)
    records = []

    for frame in frames:
        pose = estimator.estimate(frame.image)
        if pose.confidence < min_confidence:
            continue
        normalized = normalize_keypoints(pose.keypoints)
        row = {
            "video_id": video_id,
            "fighter_id": fighter_id,
            "frame_number": frame.frame_number,
            "timestamp": frame.timestamp,
            "confidence": pose.confidence,
        }
        row.update(_keypoints_to_row(normalized))
        records.append(row)

    if not records:
        return pd.DataFrame(columns=[f.name for f in POSE_FRAME_SCHEMA])

    return pd.DataFrame(records)


def save_pose_frames(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save PoseFrame DataFrame to Parquet.

    Args:
        df: DataFrame with PoseFrame schema.
        output_path: Destination .parquet file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=POSE_FRAME_SCHEMA, safe=False)
    pq.write_table(table, output_path, compression="snappy")


def load_pose_frames(parquet_path: str | Path) -> pd.DataFrame:
    """Load PoseFrame DataFrame from Parquet.

    Args:
        parquet_path: Path to .parquet file.

    Returns:
        DataFrame with PoseFrame schema.
    """
    return pq.read_table(str(parquet_path)).to_pandas()
