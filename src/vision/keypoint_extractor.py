"""MediaPipe wrapper → keypoint trajectories.

Extracts pose keypoints from video files using MediaPipe PoseLandmarker.
Produces both raw (N, 33, 4) arrays for LSTM classification and
(N, 11, 2) KeypointFrame list for the state engine observation function.

Heavy dependencies (cv2, mediapipe) are imported lazily inside
extract_keypoints() so the module can be imported without them.

Reference: spec/roadmap.md Phase 2b
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.state.types import KeypointFrame

# 11 upper-body landmark indices (configs/boxing.yaml)
UPPER_BODY_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]


@dataclass
class ExtractionResult:
    """Complete keypoint extraction output from a video."""

    raw_keypoints: np.ndarray  # (N, 33, 4) — full MediaPipe landmarks
    timestamps_s: np.ndarray  # (N,) — per-frame timestamps in seconds
    fps: float
    duration: float  # total video duration in seconds
    keypoint_frames: list[KeypointFrame]  # (11, 2) per frame for state engine


def extract_keypoints(
    video_path: str | Path,
    *,
    model_path: str | Path | None = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> ExtractionResult:
    """Extract pose keypoints from a video file.

    Args:
        video_path: Path to the video file.
        model_path: Path to MediaPipe pose_landmarker.task model.
            Defaults to server config setting.
        min_detection_confidence: Minimum pose detection confidence.
        min_tracking_confidence: Minimum pose tracking confidence.

    Returns:
        ExtractionResult with raw and processed keypoints.

    Raises:
        FileNotFoundError: If video or model file not found.
        ValueError: If video cannot be opened or no poses detected.
    """
    import cv2
    import mediapipe as mp

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if model_path is None:
        from server.config import settings

        model_path = settings.pose_model
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Pose model not found: {model_path}")

    # Video metadata
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    # MediaPipe PoseLandmarker
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    raw_list: list[np.ndarray] = []
    ts_list: list[float] = []
    last_ts_ms = -1

    try:
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * 1000 / fps)
            if timestamp_ms <= last_ts_ms:
                timestamp_ms = last_ts_ms + 1
            last_ts_ms = timestamp_ms

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_idx += 1

            if not result.pose_landmarks:
                continue

            landmarks = result.pose_landmarks[0]
            keypoints = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
                dtype=np.float32,
            )
            raw_list.append(keypoints)
            ts_list.append(timestamp_ms / 1000.0)

        cap.release()
    finally:
        landmarker.close()

    if not raw_list:
        raise ValueError(f"No poses detected in video: {video_path}")

    raw_keypoints = np.stack(raw_list)  # (N, 33, 4)
    timestamps_s = np.array(ts_list, dtype=np.float64)

    keypoint_frames = _to_keypoint_frames(raw_keypoints, timestamps_s)

    return ExtractionResult(
        raw_keypoints=raw_keypoints,
        timestamps_s=timestamps_s,
        fps=fps,
        duration=duration,
        keypoint_frames=keypoint_frames,
    )


def _to_keypoint_frames(
    raw: np.ndarray, timestamps_s: np.ndarray
) -> list[KeypointFrame]:
    """Convert (N, 33, 4) raw keypoints to list[KeypointFrame] with (11, 2)."""
    selected = raw[:, UPPER_BODY_INDICES, :2]  # (N, 11, 2) — x, y only
    frames = []
    for i in range(len(raw)):
        kp = selected[i].astype(np.float64)
        frames.append(KeypointFrame(timestamp=timestamps_s[i], keypoints=kp))
    return frames
