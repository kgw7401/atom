"""MediaPipe pose extractor — the first stage of the cognitive engine.

Wraps MediaPipe PoseLandmarker (Tasks API) to produce (33, 4) keypoint arrays
[x, y, z, visibility] per frame. Designed to be replaceable: any extractor
yielding the same shape works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import cv2
import mediapipe as mp
import numpy as np

NUM_LANDMARKS = 33
LANDMARK_DIMS = 4  # x, y, z, visibility

_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker.task"

# MediaPipe Pose connections (33-landmark topology)
_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),      # right eye
    (0, 4), (4, 5), (5, 6), (6, 8),      # left eye
    (9, 10),                               # mouth
    (11, 12),                              # shoulders
    (11, 13), (13, 15),                    # left arm
    (12, 14), (14, 16),                    # right arm
    (15, 17), (15, 19), (15, 21),         # left hand
    (16, 18), (16, 20), (16, 22),         # right hand
    (11, 23), (12, 24),                    # torso
    (23, 24),                              # hips
    (23, 25), (25, 27),                    # left leg
    (24, 26), (26, 28),                    # right leg
    (27, 29), (29, 31),                    # left foot
    (28, 30), (30, 32),                    # right foot
]


@dataclass
class PoseFrame:
    """Single-frame pose extraction result."""

    timestamp_ms: float
    keypoints: np.ndarray  # (33, 4)
    image: np.ndarray | None = field(default=None, repr=False)


class PoseExtractor:
    """MediaPipe PoseLandmarker wrapper (Tasks API).

    Usage::

        with PoseExtractor() as ext:
            for frame in ext.process_video(0):  # webcam
                print(frame.keypoints.shape)    # (33, 4)
    """

    def __init__(
        self,
        *,
        model_path: str | Path = _DEFAULT_MODEL,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Download: curl -L -o models/pose_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            )

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1  # track last timestamp for monotonicity

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_frame(
        self, bgr_frame: np.ndarray, timestamp_ms: int
    ) -> np.ndarray | None:
        """Extract keypoints from a single BGR frame.

        Returns:
            (33, 4) float32 array, or None if no pose detected.
        """
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB),
        )
        # Ensure monotonically increasing timestamps across videos
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]  # first person
        keypoints = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
            dtype=np.float32,
        )
        return keypoints

    def process_video(
        self,
        source: str | int | Path,
        *,
        keep_images: bool = False,
    ) -> Generator[PoseFrame, None, None]:
        """Yield PoseFrame for each frame where a pose is detected.

        Args:
            source: Video file path or camera index (0 = default webcam).
            keep_images: Include the original BGR frame in each PoseFrame.
        """
        cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        try:
            frame_idx = 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = int(frame_idx * 1000 / fps)
                keypoints = self.extract_frame(frame, timestamp_ms)
                frame_idx += 1

                if keypoints is not None:
                    yield PoseFrame(
                        timestamp_ms=timestamp_ms,
                        keypoints=keypoints,
                        image=frame if keep_images else None,
                    )
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def draw_landmarks(
        self,
        bgr_frame: np.ndarray,
        keypoints: np.ndarray,
        *,
        visibility_threshold: float = 0.5,
    ) -> np.ndarray:
        """Draw skeleton overlay on a BGR frame."""
        h, w = bgr_frame.shape[:2]
        annotated = bgr_frame.copy()

        # Joints
        for x, y, _z, vis in keypoints:
            if vis < visibility_threshold:
                continue
            cv2.circle(annotated, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)

        # Bones
        for i, j in _POSE_CONNECTIONS:
            if keypoints[i, 3] < visibility_threshold or keypoints[j, 3] < visibility_threshold:
                continue
            pt1 = (int(keypoints[i, 0] * w), int(keypoints[i, 1] * h))
            pt2 = (int(keypoints[j, 0] * w), int(keypoints[j, 1] * h))
            cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

        return annotated

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
