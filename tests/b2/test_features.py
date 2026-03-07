"""Tests for B2 Task 6: Feature engineering from pose keypoints."""

import numpy as np
import pandas as pd
import pytest

from track_b.b2.features import (
    FEATURE_DIM,
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    _summarize,
    build_feature_dataframe,
    extract_angles,
    extract_distances,
    extract_features,
    extract_wrist_accelerations,
    feature_names,
    windows_from_pose_df,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_window(n_frames: int = 30, seed: int = 0) -> np.ndarray:
    """Create a random keypoint window (n_frames, 33, 4) with values in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.random((n_frames, 33, 4)).astype(np.float32)


def make_straight_arm_window(n_frames: int = 30) -> np.ndarray:
    """Window where left arm is fully extended (180 degrees at elbow)."""
    window = np.zeros((n_frames, 33, 4), dtype=np.float32)
    # Place shoulder, elbow, wrist in a straight horizontal line
    window[:, LEFT_SHOULDER, :2] = [0.4, 0.3]
    window[:, LEFT_ELBOW, :2] = [0.5, 0.3]
    window[:, LEFT_WRIST, :2] = [0.6, 0.3]
    return window


def make_bent_arm_window(n_frames: int = 30) -> np.ndarray:
    """Window where left arm is bent 90 degrees at elbow."""
    window = np.zeros((n_frames, 33, 4), dtype=np.float32)
    # shoulder at top, elbow directly below, wrist to the right
    window[:, LEFT_SHOULDER, :2] = [0.5, 0.2]
    window[:, LEFT_ELBOW, :2] = [0.5, 0.4]    # below shoulder
    window[:, LEFT_WRIST, :2] = [0.7, 0.4]    # to the right of elbow
    return window


def make_moving_wrist_window(n_frames: int = 30) -> np.ndarray:
    """Window where right wrist moves in a straight line (constant velocity)."""
    window = np.zeros((n_frames, 33, 4), dtype=np.float32)
    for i in range(n_frames):
        window[i, RIGHT_WRIST, 0] = i * 0.01   # x increases linearly
        window[i, RIGHT_WRIST, 1] = 0.5         # y constant
    return window


def make_pose_df(n_frames: int = 60) -> pd.DataFrame:
    """Minimal PoseFrame DataFrame with flat keypoint columns."""
    rng = np.random.default_rng(42)
    rows = {}
    for lm in range(33):
        for coord in ("x", "y", "z", "vis"):
            rows[f"kp_{lm}_{coord}"] = rng.random(n_frames).astype(np.float32)
    rows["frame_number"] = np.arange(n_frames)
    rows["timestamp"] = np.linspace(0.0, n_frames / 30.0, n_frames)
    return pd.DataFrame(rows)


# ── FEATURE_DIM constant ──────────────────────────────────────────────────────

class TestFeatureDim:
    def test_feature_dim_is_120(self):
        assert FEATURE_DIM == 120

    def test_feature_dim_matches_actual_output(self):
        window = make_window()
        vector = extract_features(window)
        assert len(vector) == FEATURE_DIM

    def test_feature_names_count_matches_dim(self):
        assert len(feature_names()) == FEATURE_DIM


# ── extract_angles ────────────────────────────────────────────────────────────

class TestExtractAngles:
    def test_output_shape(self):
        window = make_window(30)
        angles = extract_angles(window)
        assert angles.shape == (30, 6)

    def test_output_dtype(self):
        angles = extract_angles(make_window())
        assert angles.dtype == np.float32

    def test_angles_in_valid_range(self):
        angles = extract_angles(make_window())
        assert np.all(angles >= 0.0)
        assert np.all(angles <= np.pi + 1e-5)

    def test_straight_arm_is_pi(self):
        """Fully extended arm should produce angle ≈ π at elbow."""
        window = make_straight_arm_window()
        angles = extract_angles(window)
        # Column 0 is left elbow angle
        np.testing.assert_allclose(angles[:, 0], np.pi, atol=1e-4)

    def test_bent_arm_is_approximately_half_pi(self):
        """90-degree bent arm should produce angle ≈ π/2 at elbow."""
        window = make_bent_arm_window()
        angles = extract_angles(window)
        np.testing.assert_allclose(angles[:, 0], np.pi / 2, atol=1e-4)

    def test_single_frame_window(self):
        window = make_window(1)
        angles = extract_angles(window)
        assert angles.shape == (1, 6)

    def test_zero_length_ray_returns_zero(self):
        """Degenerate keypoints (all at origin) should not raise."""
        window = np.zeros((5, 33, 4), dtype=np.float32)
        angles = extract_angles(window)
        assert np.all(angles == 0.0)


# ── extract_distances ─────────────────────────────────────────────────────────

class TestExtractDistances:
    def test_output_shape(self):
        dists = extract_distances(make_window(30))
        assert dists.shape == (30, 8)

    def test_output_dtype(self):
        assert extract_distances(make_window()).dtype == np.float32

    def test_distances_non_negative(self):
        assert np.all(extract_distances(make_window()) >= 0.0)

    def test_known_distance(self):
        """Distance between two known points should match Euclidean formula."""
        window = np.zeros((5, 33, 4), dtype=np.float32)
        # Place left wrist at (0.6, 0.3) and left shoulder at (0.4, 0.3)
        window[:, LEFT_WRIST, :2] = [0.6, 0.3]
        window[:, LEFT_SHOULDER, :2] = [0.4, 0.3]
        dists = extract_distances(window)
        # Column 0 = dist_left_wrist_left_shoulder = 0.2
        np.testing.assert_allclose(dists[:, 0], 0.2, atol=1e-5)

    def test_same_point_gives_zero(self):
        """Same landmark position → zero distance."""
        window = np.zeros((5, 33, 4), dtype=np.float32)
        window[:, LEFT_WRIST, :2] = [0.5, 0.5]
        window[:, LEFT_SHOULDER, :2] = [0.5, 0.5]
        dists = extract_distances(window)
        assert dists[:, 0].sum() == pytest.approx(0.0)


# ── extract_wrist_accelerations ───────────────────────────────────────────────

class TestExtractWristAccelerations:
    def test_output_shape(self):
        accel = extract_wrist_accelerations(make_window(30))
        assert accel.shape == (28, 4)  # n-2 frames

    def test_short_window_returns_zeros(self):
        window = make_window(2)
        accel = extract_wrist_accelerations(window)
        assert accel.shape == (1, 4)
        assert np.all(accel == 0.0)

    def test_constant_velocity_gives_zero_acceleration(self):
        """Constant velocity motion → zero second derivative."""
        window = make_moving_wrist_window(30)
        accel = extract_wrist_accelerations(window)
        # Right wrist x moves at constant rate → zero acceleration
        np.testing.assert_allclose(accel[:, 2], 0.0, atol=1e-5)

    def test_stationary_gives_zero_acceleration(self):
        """Stationary keypoints → zero acceleration."""
        window = np.zeros((10, 33, 4), dtype=np.float32)
        window[:, RIGHT_WRIST, :2] = [0.5, 0.5]
        accel = extract_wrist_accelerations(window)
        np.testing.assert_allclose(accel, 0.0, atol=1e-7)


# ── _summarize ────────────────────────────────────────────────────────────────

class TestSummarize:
    def test_output_shape(self):
        series = np.ones((30, 6), dtype=np.float32)
        result = _summarize(series)
        assert result.shape == (30,)  # 6 features × 5 stats

    def test_constant_series_std_is_zero(self):
        series = np.ones((30, 4), dtype=np.float32) * 3.0
        result = _summarize(series)
        # Columns: [mean×4, std×4, min×4, max×4, range×4]
        stds = result[4:8]
        np.testing.assert_allclose(stds, 0.0, atol=1e-6)

    def test_constant_series_range_is_zero(self):
        series = np.ones((30, 4), dtype=np.float32) * 5.0
        result = _summarize(series)
        ranges = result[16:20]
        np.testing.assert_allclose(ranges, 0.0, atol=1e-6)

    def test_mean_correct(self):
        series = np.arange(20, dtype=np.float32).reshape(4, 5)
        result = _summarize(series)
        expected_means = series.mean(axis=0)
        np.testing.assert_allclose(result[:5], expected_means, atol=1e-5)


# ── extract_features ─────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_output_shape(self):
        vector = extract_features(make_window(30))
        assert vector.shape == (FEATURE_DIM,)

    def test_output_dtype(self):
        assert extract_features(make_window()).dtype == np.float32

    def test_deterministic(self):
        window = make_window(30, seed=7)
        v1 = extract_features(window)
        v2 = extract_features(window)
        np.testing.assert_array_equal(v1, v2)

    def test_different_windows_give_different_vectors(self):
        v1 = extract_features(make_window(30, seed=0))
        v2 = extract_features(make_window(30, seed=1))
        assert not np.allclose(v1, v2)

    def test_no_nan_in_output(self):
        vector = extract_features(make_window())
        assert not np.any(np.isnan(vector))

    def test_no_inf_in_output(self):
        vector = extract_features(make_window())
        assert not np.any(np.isinf(vector))

    def test_works_with_short_window(self):
        """Minimum 2-frame window should not raise."""
        vector = extract_features(make_window(2))
        assert vector.shape == (FEATURE_DIM,)

    def test_raises_on_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            extract_features(np.zeros((30, 17, 4)))  # MoveNet shape, not MediaPipe

    def test_raises_on_single_frame(self):
        with pytest.raises(ValueError, match="2 frames"):
            extract_features(make_window(1))

    def test_straight_arm_produces_high_elbow_angle(self):
        """Straight arm → elbow angle near π → angle stats near π."""
        window = make_straight_arm_window(30)
        vector = extract_features(window)
        names = feature_names()
        # angle_left_elbow_mean should be near π
        idx = names.index("angle_left_elbow_mean")
        assert vector[idx] == pytest.approx(np.pi, abs=1e-3)


# ── feature_names ─────────────────────────────────────────────────────────────

class TestFeatureNames:
    def test_returns_list_of_strings(self):
        names = feature_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_no_duplicate_names(self):
        names = feature_names()
        assert len(names) == len(set(names)), "Duplicate feature names found"

    def test_contains_angle_features(self):
        names = feature_names()
        assert "angle_left_elbow_mean" in names
        assert "angle_right_elbow_range" in names

    def test_contains_velocity_features(self):
        names = feature_names()
        assert any("vel_angle" in n for n in names)

    def test_contains_distance_features(self):
        names = feature_names()
        assert "dist_left_wrist_left_shoulder_mean" in names
        assert "dist_right_wrist_nose_std" in names

    def test_contains_acceleration_features(self):
        names = feature_names()
        assert "accel_left_wrist_x_mean" in names
        assert "accel_right_wrist_y_range" in names

    def test_all_stat_suffixes_present(self):
        names = feature_names()
        for suffix in ("mean", "std", "min", "max", "range"):
            assert any(n.endswith(f"_{suffix}") for n in names)


# ── build_feature_dataframe ───────────────────────────────────────────────────

class TestBuildFeatureDataframe:
    def test_output_columns(self):
        windows = [make_window(30, seed=i) for i in range(5)]
        labels = [0, 1, 2, 3, 4]
        df = build_feature_dataframe(windows, labels)
        assert "label" in df.columns
        assert set(feature_names()).issubset(df.columns)

    def test_row_count(self):
        windows = [make_window() for _ in range(10)]
        df = build_feature_dataframe(windows, list(range(10)))
        assert len(df) == 10

    def test_label_values_correct(self):
        windows = [make_window(seed=i) for i in range(3)]
        labels = [0, 2, 5]
        df = build_feature_dataframe(windows, labels)
        assert list(df["label"]) == labels

    def test_raises_on_length_mismatch(self):
        windows = [make_window() for _ in range(5)]
        with pytest.raises(ValueError, match="same length"):
            build_feature_dataframe(windows, [0, 1, 2])

    def test_no_nan_in_features(self):
        windows = [make_window(seed=i) for i in range(6)]
        df = build_feature_dataframe(windows, list(range(6)))
        feature_cols = feature_names()
        assert not df[feature_cols].isna().any().any()


# ── windows_from_pose_df ──────────────────────────────────────────────────────

class TestWindowsFromPoseDf:
    def test_correct_window_count_stride1(self):
        pose_df = make_pose_df(60)
        windows = windows_from_pose_df(pose_df, window_size=30, stride=1)
        assert len(windows) == 60 - 30 + 1  # 31 windows

    def test_correct_window_count_stride5(self):
        pose_df = make_pose_df(60)
        windows = windows_from_pose_df(pose_df, window_size=30, stride=5)
        # (60 - 30) // 5 + 1 = 7 windows
        assert len(windows) == 7

    def test_window_shape(self):
        pose_df = make_pose_df(60)
        windows = windows_from_pose_df(pose_df, window_size=30)
        assert windows[0].shape == (30, 33, 4)

    def test_empty_when_too_few_frames(self):
        pose_df = make_pose_df(20)
        windows = windows_from_pose_df(pose_df, window_size=30)
        assert windows == []

    def test_single_window_when_exactly_window_size(self):
        pose_df = make_pose_df(30)
        windows = windows_from_pose_df(pose_df, window_size=30)
        assert len(windows) == 1

    def test_windows_are_float32(self):
        pose_df = make_pose_df(60)
        windows = windows_from_pose_df(pose_df, window_size=30)
        assert windows[0].dtype == np.float32

    def test_consecutive_windows_overlap_correctly(self):
        """With stride=1, windows[1] should be windows[0] shifted by 1 frame."""
        pose_df = make_pose_df(60)
        windows = windows_from_pose_df(pose_df, window_size=10, stride=1)
        np.testing.assert_array_equal(windows[0][1:], windows[1][:-1])
