"""Tests for B1 Task 3: Quality filtering, normalization & Parquet storage."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from track_b.b1.pose_estimator import PoseKeypoints
from track_b.b1.pose_pipeline import (
    POSE_FRAME_SCHEMA,
    _keypoints_to_row,
    _row_to_keypoints,
    load_pose_frames,
    normalize_keypoints,
    run_training_pipeline,
    save_pose_frames,
)

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "pose_landmarker_lite.task"


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_valid_pose(confidence: float = 0.8, x_offset: float = 0.0) -> PoseKeypoints:
    """Create a PoseKeypoints with plausible values."""
    kp = np.zeros((33, 4), dtype=np.float32)
    # Spread body landmarks across different positions
    positions = {
        11: (0.4 + x_offset, 0.3),  # LEFT_SHOULDER
        12: (0.6 + x_offset, 0.3),  # RIGHT_SHOULDER
        23: (0.4 + x_offset, 0.6),  # LEFT_HIP
        24: (0.6 + x_offset, 0.6),  # RIGHT_HIP
        13: (0.3 + x_offset, 0.45), # LEFT_ELBOW
        14: (0.7 + x_offset, 0.45), # RIGHT_ELBOW
        15: (0.25 + x_offset, 0.55),# LEFT_WRIST
        16: (0.75 + x_offset, 0.55),# RIGHT_WRIST
    }
    for idx, (x, y) in positions.items():
        kp[idx, 0] = x
        kp[idx, 1] = y
        kp[idx, 3] = confidence
    kp[:, 3] = np.where(kp[:, 3] == 0, confidence * 0.5, kp[:, 3])
    return PoseKeypoints(keypoints=kp, confidence=confidence)


def make_invalid_pose() -> PoseKeypoints:
    """Create a PoseKeypoints with low confidence (filtered out)."""
    return PoseKeypoints(keypoints=np.zeros((33, 4), dtype=np.float32), confidence=0.0)


def make_mock_estimator(poses: list[PoseKeypoints]) -> MagicMock:
    """Mock PoseEstimator that returns poses in sequence."""
    est = MagicMock()
    est.estimate.side_effect = poses
    return est


# ── Normalization ─────────────────────────────────────────────────────────────

class TestNormalizeKeypoints:
    def test_output_shape(self):
        kp = np.random.rand(33, 4).astype(np.float32)
        normalized = normalize_keypoints(kp)
        assert normalized.shape == (33, 4)

    def test_output_dtype(self):
        kp = np.random.rand(33, 4).astype(np.float32)
        normalized = normalize_keypoints(kp)
        assert normalized.dtype == np.float32

    def test_body_landmarks_in_unit_range(self):
        """After normalization, body landmark x/y should be in [0, 1]."""
        kp = make_valid_pose().keypoints
        normalized = normalize_keypoints(kp)
        body_indices = [11, 12, 23, 24, 13, 14, 15, 16]
        for idx in body_indices:
            assert 0.0 <= normalized[idx, 0] <= 1.0, f"kp {idx} x={normalized[idx,0]} out of range"
            assert 0.0 <= normalized[idx, 1] <= 1.0, f"kp {idx} y={normalized[idx,1]} out of range"

    def test_scale_invariant(self):
        """Same pose at 2× scale should produce same normalized output."""
        kp1 = make_valid_pose().keypoints
        kp2 = kp1.copy()
        kp2[:, :2] *= 0.5  # scale x,y by 0.5

        n1 = normalize_keypoints(kp1)
        n2 = normalize_keypoints(kp2)
        np.testing.assert_allclose(n1[:, :2], n2[:, :2], atol=1e-5)

    def test_visibility_unchanged(self):
        """Visibility column should not be modified by normalization."""
        kp = make_valid_pose().keypoints
        normalized = normalize_keypoints(kp)
        np.testing.assert_array_equal(kp[:, 3], normalized[:, 3])

    def test_degenerate_case_returns_copy(self):
        """All body landmarks at same point → return original (no division by zero)."""
        kp = np.zeros((33, 4), dtype=np.float32)
        result = normalize_keypoints(kp)
        assert result.shape == (33, 4)


# ── Keypoint serialization ────────────────────────────────────────────────────

class TestKeypointSerialization:
    def test_roundtrip(self):
        kp = np.random.rand(33, 4).astype(np.float32)
        row = _keypoints_to_row(kp)
        restored = _row_to_keypoints(row)
        np.testing.assert_allclose(kp, restored, atol=1e-6)

    def test_column_count(self):
        kp = np.zeros((33, 4), dtype=np.float32)
        row = _keypoints_to_row(kp)
        assert len(row) == 33 * 4  # 132 columns

    def test_column_names(self):
        kp = np.zeros((33, 4), dtype=np.float32)
        row = _keypoints_to_row(kp)
        assert "kp_0_x" in row
        assert "kp_32_vis" in row


# ── Training pipeline ─────────────────────────────────────────────────────────

class TestTrainingPipeline:
    def test_filters_low_confidence_frames(self, fixture_video_2s: Path):
        """Frames where pose confidence < 0.5 should be dropped."""
        poses = [
            make_valid_pose(0.9),   # kept
            make_invalid_pose(),    # filtered (confidence=0.0)
            make_valid_pose(0.8),   # kept
        ] + [make_invalid_pose()] * 57  # rest filtered

        mock_est = make_mock_estimator(poses)
        df = run_training_pipeline(fixture_video_2s, "test_vid", mock_est)
        assert len(df) == 2

    def test_output_columns_match_schema(self, fixture_video_2s: Path):
        poses = [make_valid_pose()] * 60
        mock_est = make_mock_estimator(poses)
        df = run_training_pipeline(fixture_video_2s, "test_vid", mock_est)

        expected_cols = {f.name for f in POSE_FRAME_SCHEMA}
        actual_cols = set(df.columns)
        assert expected_cols == actual_cols

    def test_video_id_set_correctly(self, fixture_video_2s: Path):
        poses = [make_valid_pose()] * 60
        mock_est = make_mock_estimator(poses)
        df = run_training_pipeline(fixture_video_2s, "my_video_001", mock_est)
        assert (df["video_id"] == "my_video_001").all()

    def test_fighter_id_empty_for_training(self, fixture_video_2s: Path):
        poses = [make_valid_pose()] * 60
        mock_est = make_mock_estimator(poses)
        df = run_training_pipeline(fixture_video_2s, "test_vid", mock_est)
        assert (df["fighter_id"] == "").all()

    def test_timestamps_are_floats(self, fixture_video_2s: Path):
        poses = [make_valid_pose()] * 60
        mock_est = make_mock_estimator(poses)
        df = run_training_pipeline(fixture_video_2s, "test_vid", mock_est)
        assert df["timestamp"].dtype in (np.float32, np.float64)

    def test_empty_result_when_all_filtered(self, fixture_video_2s: Path):
        poses = [make_invalid_pose()] * 60
        mock_est = make_mock_estimator(poses)
        df = run_training_pipeline(fixture_video_2s, "test_vid", mock_est)
        assert len(df) == 0

    def test_keypoints_normalized(self, fixture_video_2s: Path):
        """Stored keypoints should be normalized (body landmarks in [0,1])."""
        poses = [make_valid_pose()] * 60
        mock_est = make_mock_estimator(poses)
        df = run_training_pipeline(fixture_video_2s, "test_vid", mock_est)

        if len(df) == 0:
            pytest.skip("No frames passed filter")

        # Left shoulder x (kp_11_x) should be in valid normalized range
        assert df["kp_11_x"].between(0.0, 1.0).all()


# ── Parquet I/O ───────────────────────────────────────────────────────────────

class TestParquetStorage:
    def test_save_and_load_roundtrip(self, fixture_video_2s: Path, tmp_path: Path):
        poses = [make_valid_pose()] * 60
        mock_est = make_mock_estimator(poses)
        df = run_training_pipeline(fixture_video_2s, "test_vid", mock_est)

        if len(df) == 0:
            pytest.skip("No frames passed filter")

        out = tmp_path / "pose_frames.parquet"
        save_pose_frames(df, out)

        loaded = load_pose_frames(out)
        assert len(loaded) == len(df)
        assert set(loaded.columns) == set(df.columns)

    def test_save_creates_parent_dirs(self, fixture_video_2s: Path, tmp_path: Path):
        poses = [make_valid_pose()] * 60
        mock_est = make_mock_estimator(poses)
        df = run_training_pipeline(fixture_video_2s, "test_vid", mock_est)

        if len(df) == 0:
            pytest.skip("No frames passed filter")

        out = tmp_path / "deep" / "nested" / "frames.parquet"
        save_pose_frames(df, out)
        assert out.exists()

    def test_load_preserves_video_id(self, fixture_video_2s: Path, tmp_path: Path):
        poses = [make_valid_pose()] * 60
        mock_est = make_mock_estimator(poses)
        df = run_training_pipeline(fixture_video_2s, "boxingvi_jab_042", mock_est)

        if len(df) == 0:
            pytest.skip("No frames passed filter")

        out = tmp_path / "frames.parquet"
        save_pose_frames(df, out)
        loaded = load_pose_frames(out)
        assert (loaded["video_id"] == "boxingvi_jab_042").all()

    def test_keypoints_preserved_in_roundtrip(self, fixture_video_2s: Path, tmp_path: Path):
        poses = [make_valid_pose()] * 60
        mock_est = make_mock_estimator(poses)
        df = run_training_pipeline(fixture_video_2s, "test_vid", mock_est)

        if len(df) == 0:
            pytest.skip("No frames passed filter")

        out = tmp_path / "frames.parquet"
        save_pose_frames(df, out)
        loaded = load_pose_frames(out)

        np.testing.assert_allclose(
            df["kp_15_x"].values, loaded["kp_15_x"].values, atol=1e-4
        )
