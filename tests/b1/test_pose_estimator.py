"""Tests for B1 Task 2: MediaPipe BlazePose integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from track_b.b1.pose_estimator import PoseEstimator, PoseKeypoints, PoseLandmark

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "pose_landmarker_lite.task"


def make_blank_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Create a blank (black) BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_mock_landmark(x: float, y: float, z: float, visibility: float) -> MagicMock:
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    lm.visibility = visibility
    return lm


def make_mock_result(detected: bool = True) -> MagicMock:
    """Create mock PoseLandmarkerResult."""
    mock_result = MagicMock()
    if not detected:
        mock_result.pose_landmarks = []
        return mock_result

    landmarks = [
        make_mock_landmark(float(i) / 33, float(i) / 33 * 0.5, -0.1, 0.8 + (i % 5) * 0.04)
        for i in range(33)
    ]
    mock_result.pose_landmarks = [landmarks]
    return mock_result


@pytest.fixture(scope="session")
def estimator():
    """Create a PoseEstimator instance using the downloaded model."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found: {MODEL_PATH}. Run scripts/download_models.py first.")
    with PoseEstimator(model_path=MODEL_PATH) as est:
        yield est


class TestPoseKeypoints:
    def test_is_valid_above_threshold(self):
        kp = PoseKeypoints(
            keypoints=np.ones((33, 4), dtype=np.float32) * 0.5,
            confidence=0.8,
        )
        assert kp.is_valid is True

    def test_is_valid_below_threshold(self):
        kp = PoseKeypoints(
            keypoints=np.zeros((33, 4), dtype=np.float32),
            confidence=0.3,
        )
        assert kp.is_valid is False

    def test_is_valid_at_threshold(self):
        kp = PoseKeypoints(
            keypoints=np.zeros((33, 4), dtype=np.float32),
            confidence=0.5,
        )
        assert kp.is_valid is True

    def test_landmark_xy(self):
        kp = np.zeros((33, 4), dtype=np.float32)
        kp[11, 0] = 0.4
        kp[11, 1] = 0.3
        pose = PoseKeypoints(keypoints=kp, confidence=0.9)
        x, y = pose.landmark_xy(PoseLandmark.LEFT_SHOULDER)
        assert x == pytest.approx(0.4)
        assert y == pytest.approx(0.3)

    def test_landmark_visibility(self):
        kp = np.zeros((33, 4), dtype=np.float32)
        kp[16, 3] = 0.95
        pose = PoseKeypoints(keypoints=kp, confidence=0.95)
        assert pose.landmark_visibility(PoseLandmark.RIGHT_WRIST) == pytest.approx(0.95)


class TestPoseEstimatorNoDetection:
    """Tests using real model with blank frame (no person → no detection)."""

    def test_blank_frame_returns_keypoints(self, estimator: PoseEstimator):
        result = estimator.estimate(make_blank_frame())
        assert isinstance(result, PoseKeypoints)

    def test_blank_frame_invalid(self, estimator: PoseEstimator):
        result = estimator.estimate(make_blank_frame())
        assert result.is_valid is False

    def test_blank_frame_keypoints_shape(self, estimator: PoseEstimator):
        result = estimator.estimate(make_blank_frame())
        assert result.keypoints.shape == (33, 4)

    def test_blank_frame_keypoints_dtype(self, estimator: PoseEstimator):
        result = estimator.estimate(make_blank_frame())
        assert result.keypoints.dtype == np.float32

    def test_blank_frame_confidence_zero(self, estimator: PoseEstimator):
        result = estimator.estimate(make_blank_frame())
        assert result.confidence == pytest.approx(0.0)


class TestPoseEstimatorMocked:
    """Tests mocking MediaPipe detect() to verify coordinate handling logic."""

    def test_detected_keypoints_shape(self, estimator: PoseEstimator):
        with patch.object(estimator._landmarker, "detect", return_value=make_mock_result()):
            result = estimator.estimate(make_blank_frame())
        assert result.keypoints.shape == (33, 4)

    def test_detected_keypoints_dtype(self, estimator: PoseEstimator):
        with patch.object(estimator._landmarker, "detect", return_value=make_mock_result()):
            result = estimator.estimate(make_blank_frame())
        assert result.keypoints.dtype == np.float32

    def test_detected_is_valid(self, estimator: PoseEstimator):
        with patch.object(estimator._landmarker, "detect", return_value=make_mock_result()):
            result = estimator.estimate(make_blank_frame())
        assert result.is_valid is True

    def test_detected_confidence_is_max_visibility(self, estimator: PoseEstimator):
        mock_result = make_mock_result()
        landmarks = mock_result.pose_landmarks[0]
        expected_confidence = max(lm.visibility for lm in landmarks)

        with patch.object(estimator._landmarker, "detect", return_value=mock_result):
            result = estimator.estimate(make_blank_frame())

        assert result.confidence == pytest.approx(expected_confidence, abs=1e-5)

    def test_coordinate_columns_assigned_correctly(self, estimator: PoseEstimator):
        """Verify x=col0, y=col1, z=col2, visibility=col3."""
        mock_result = make_mock_result()
        lm0 = mock_result.pose_landmarks[0][0]

        with patch.object(estimator._landmarker, "detect", return_value=mock_result):
            result = estimator.estimate(make_blank_frame())

        assert result.keypoints[0, 0] == pytest.approx(lm0.x, abs=1e-5)
        assert result.keypoints[0, 1] == pytest.approx(lm0.y, abs=1e-5)
        assert result.keypoints[0, 2] == pytest.approx(lm0.z, abs=1e-5)
        assert result.keypoints[0, 3] == pytest.approx(lm0.visibility, abs=1e-5)

    def test_no_detection_returns_zeros(self, estimator: PoseEstimator):
        with patch.object(estimator._landmarker, "detect", return_value=make_mock_result(detected=False)):
            result = estimator.estimate(make_blank_frame())
        assert result.confidence == pytest.approx(0.0)
        assert np.all(result.keypoints == 0)

    def test_rgb_conversion_applied(self, estimator: PoseEstimator):
        """MediaPipe receives RGB input (not BGR). Verify channel swap."""
        captured = []

        def capture_detect(mp_image):
            captured.append(np.array(mp_image.numpy_view()).copy())
            return make_mock_result(detected=False)

        with patch.object(estimator._landmarker, "detect", side_effect=capture_detect):
            bgr = np.zeros((10, 10, 3), dtype=np.uint8)
            bgr[0, 0] = [255, 0, 0]  # BGR: blue
            estimator.estimate(bgr)

        # After BGR→RGB: [255, 0, 0] BGR → [0, 0, 255] RGB
        assert captured[0][0, 0, 0] == 0    # R
        assert captured[0][0, 0, 2] == 255  # B


class TestModelNotFound:
    def test_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="MediaPipe model not found"):
            PoseEstimator(model_path=tmp_path / "nonexistent.task")


class TestPoseLandmarkConstants:
    def test_key_landmark_indices(self):
        assert PoseLandmark.LEFT_SHOULDER == 11
        assert PoseLandmark.RIGHT_SHOULDER == 12
        assert PoseLandmark.LEFT_WRIST == 15
        assert PoseLandmark.RIGHT_WRIST == 16
        assert PoseLandmark.LEFT_HIP == 23
        assert PoseLandmark.RIGHT_HIP == 24
        assert PoseLandmark.NOSE == 0
        assert PoseLandmark.RIGHT_FOOT_INDEX == 32
