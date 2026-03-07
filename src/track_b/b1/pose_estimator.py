"""B1 Task 2: MediaPipe BlazePose integration.

Per-frame pose estimation using MediaPipe PoseLandmarker (Tasks API, IMAGE mode).
Produces 33 keypoints × (x, y, z, visibility) per frame.
Works on full frames and cropped ROIs.

Model files are downloaded separately via scripts/download_models.py.
Default model path: models/pose_landmarker_lite.task (relative to project root).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np

# MediaPipe Tasks API
_BaseOptions = mp.tasks.BaseOptions
_PoseLandmarker = mp.tasks.vision.PoseLandmarker
_PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode

# Default model path relative to project root
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent.parent / "models" / "pose_landmarker_lite.task"


@dataclass
class PoseKeypoints:
    """33 MediaPipe BlazePose keypoints for a single frame."""

    # Shape: (33, 4) — columns: (x, y, z, visibility)
    # x, y: normalized [0, 1] in image space
    # z: depth relative to hip (smaller = closer to camera)
    # visibility: confidence [0, 1]
    keypoints: np.ndarray  # float32, shape (33, 4)
    confidence: float  # max visibility score across all landmarks

    @property
    def is_valid(self) -> bool:
        """True if confidence meets the minimum threshold (≥0.5)."""
        return self.confidence >= 0.5

    def landmark_xy(self, index: int) -> tuple[float, float]:
        """Get (x, y) for a landmark by index."""
        return float(self.keypoints[index, 0]), float(self.keypoints[index, 1])

    def landmark_visibility(self, index: int) -> float:
        """Get visibility score for a landmark by index."""
        return float(self.keypoints[index, 3])


class PoseLandmarkIndex:
    """MediaPipe BlazePose 33-keypoint landmark indices."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

# Alias for backward compatibility
PoseLandmark = PoseLandmarkIndex


class PoseEstimator:
    """MediaPipe PoseLandmarker (IMAGE mode, single-person).

    Designed for use in two contexts:
    - Full video frames (single-person analysis)
    - Cropped ROI images (per-fighter crop in multi-person mode)

    Usage:
        estimator = PoseEstimator()
        keypoints = estimator.estimate(frame_bgr)
        if keypoints.is_valid:
            x, y = keypoints.landmark_xy(PoseLandmark.LEFT_SHOULDER)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """Initialize BlazePose PoseLandmarker in IMAGE mode.

        Args:
            model_path: Path to .task model file. Defaults to models/pose_landmarker_lite.task.
            min_detection_confidence: Minimum confidence for pose detection.
            min_presence_confidence: Minimum confidence for pose presence.
            min_tracking_confidence: Minimum tracking confidence.
        """
        if model_path is None:
            model_path = _DEFAULT_MODEL_PATH

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}\n"
                "Download it with: python scripts/download_models.py"
            )

        options = _PoseLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=str(model_path)),
            running_mode=_RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = _PoseLandmarker.create_from_options(options)

    def estimate(self, image_bgr: np.ndarray) -> PoseKeypoints:
        """Estimate pose for a single frame (BGR image from OpenCV).

        Args:
            image_bgr: BGR image array (H, W, 3), uint8.

        Returns:
            PoseKeypoints with 33 keypoints. Check .is_valid before using.
        """
        # MediaPipe expects RGB
        image_rgb = image_bgr[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            # No person detected — return zeros with zero confidence
            return PoseKeypoints(
                keypoints=np.zeros((33, 4), dtype=np.float32),
                confidence=0.0,
            )

        # First (and only) person detected
        landmarks = result.pose_landmarks[0]
        keypoints = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
            dtype=np.float32,
        )
        confidence = float(np.max(keypoints[:, 3]))

        return PoseKeypoints(keypoints=keypoints, confidence=confidence)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *_) -> None:
        self.close()
