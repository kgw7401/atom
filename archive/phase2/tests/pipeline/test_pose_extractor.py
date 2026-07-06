"""Tests for Pipeline Stage 1: Pose Extraction."""

import numpy as np
import pytest

from ml.configs import BoxingConfig
from ml.graph.boxing_graph import SUBSET_INDICES
from ml.pipeline.pose_extractor import PoseExtractor


@pytest.fixture
def config():
    return BoxingConfig()


# ── Normalization ──

class TestNormalization:
    """Test normalization logic without MediaPipe."""

    def test_normalize_centers_on_hips(self, config):
        """After normalization, hip midpoint should be at origin."""
        # Create extractor without model (normalization doesn't need it)
        extractor = PoseExtractor(config, model_path=None)

        # Create keypoints where hips are at (1, 2, 3) and (3, 4, 5)
        T, V = 10, 15
        keypoints = np.random.randn(T, V, 3).astype(np.float32)
        keypoints[:, 9, :] = [1.0, 2.0, 3.0]  # left_hip
        keypoints[:, 10, :] = [3.0, 4.0, 5.0]  # right_hip

        normalized = extractor._normalize(keypoints)

        # Hip midpoint should be (2, 3, 4)
        # After centering, new hip midpoint should be near (0, 0, 0)
        hip_center = (normalized[:, 9, :] + normalized[:, 10, :]) / 2.0
        np.testing.assert_allclose(hip_center, 0.0, atol=1e-5)

    def test_normalize_scales_by_shoulder_width(self, config):
        """After normalization, shoulder width should be 1.0."""
        extractor = PoseExtractor(config, model_path=None)

        T, V = 10, 15
        keypoints = np.zeros((T, V, 3), dtype=np.float32)

        # Set hips at origin
        keypoints[:, 9, :] = [0.0, 0.0, 0.0]  # left_hip
        keypoints[:, 10, :] = [0.0, 0.0, 0.0]  # right_hip

        # Set shoulders 2 units apart
        keypoints[:, 3, :] = [-1.0, 0.0, 0.0]  # left_shoulder
        keypoints[:, 4, :] = [1.0, 0.0, 0.0]   # right_shoulder

        normalized = extractor._normalize(keypoints)

        # Shoulder width should now be 1.0
        shoulder_width = np.linalg.norm(
            normalized[:, 3, :] - normalized[:, 4, :], axis=1
        )
        np.testing.assert_allclose(shoulder_width, 1.0, atol=1e-5)

    def test_normalize_handles_zero_shoulder_width(self, config):
        """Normalization should not crash with zero shoulder width."""
        extractor = PoseExtractor(config, model_path=None)

        T, V = 5, 15
        keypoints = np.zeros((T, V, 3), dtype=np.float32)

        # Shoulders at same position → width = 0
        keypoints[:, 3, :] = [0.0, 1.0, 0.0]
        keypoints[:, 4, :] = [0.0, 1.0, 0.0]

        # Should not raise
        normalized = extractor._normalize(keypoints)
        assert normalized.shape == keypoints.shape

    def test_normalize_preserves_shape(self, config):
        extractor = PoseExtractor(config, model_path=None)
        keypoints = np.random.randn(20, 15, 3).astype(np.float32)
        normalized = extractor._normalize(keypoints)
        assert normalized.shape == keypoints.shape

    def test_normalize_deterministic(self, config):
        """Same input → same output."""
        extractor = PoseExtractor(config, model_path=None)
        keypoints = np.random.randn(10, 15, 3).astype(np.float32)
        norm1 = extractor._normalize(keypoints)
        norm2 = extractor._normalize(keypoints)
        np.testing.assert_array_equal(norm1, norm2)

    def test_normalize_per_frame(self, config):
        """Normalization is applied per-frame independently."""
        extractor = PoseExtractor(config, model_path=None)

        T, V = 5, 15
        keypoints = np.random.randn(T, V, 3).astype(np.float32)

        # Set different hip centers per frame
        for t in range(T):
            keypoints[t, 9, :] = [t * 1.0, 0.0, 0.0]
            keypoints[t, 10, :] = [t * 1.0, 0.0, 0.0]

        normalized = extractor._normalize(keypoints)

        # Each frame should have hips at origin
        for t in range(T):
            hip_t = (normalized[t, 9, :] + normalized[t, 10, :]) / 2.0
            np.testing.assert_allclose(hip_t, 0.0, atol=1e-5)


# ── Subset extraction ──

class TestSubsetExtraction:
    def test_subset_indices_count(self):
        """Verify we have exactly 15 joints."""
        assert len(SUBSET_INDICES) == 15

    def test_subset_indices_range(self):
        """All indices should be valid MediaPipe indices (0-32)."""
        for idx in SUBSET_INDICES:
            assert 0 <= idx <= 32

    def test_subset_indices_unique(self):
        """No duplicate indices."""
        assert len(SUBSET_INDICES) == len(set(SUBSET_INDICES))

    def test_subset_indices_sorted(self):
        """Indices should be in ascending order."""
        assert SUBSET_INDICES == sorted(SUBSET_INDICES)

    def test_key_joints_present(self):
        """Verify critical joints are in the subset."""
        # These are MediaPipe 33-landmark indices
        assert 0 in SUBSET_INDICES   # nose
        assert 11 in SUBSET_INDICES  # left_shoulder
        assert 12 in SUBSET_INDICES  # right_shoulder
        assert 15 in SUBSET_INDICES  # left_wrist
        assert 16 in SUBSET_INDICES  # right_wrist
        assert 23 in SUBSET_INDICES  # left_hip
        assert 24 in SUBSET_INDICES  # right_hip


# ── Configuration ──

class TestConfiguration:
    def test_default_config(self, config):
        extractor = PoseExtractor(config, model_path=None)
        assert extractor.target_fps == 30
        assert extractor.visibility_threshold == 0.5
        assert extractor.landmarker is None  # no model provided

    def test_custom_config(self):
        raw = BoxingConfig()._raw.copy()
        raw["pipeline"] = dict(raw["pipeline"])
        raw["pipeline"]["fps"] = 60
        raw["pipeline"]["pose_visibility_threshold"] = 0.8
        config = BoxingConfig(raw=raw)

        extractor = PoseExtractor(config, model_path=None)
        assert extractor.target_fps == 60
        assert extractor.visibility_threshold == 0.8

    def test_no_model_path_allows_normalization(self, config):
        """Can use normalization even without model."""
        extractor = PoseExtractor(config, model_path=None)
        keypoints = np.random.randn(10, 15, 3).astype(np.float32)
        normalized = extractor._normalize(keypoints)
        assert normalized.shape == keypoints.shape


# ── Error handling ──

class TestErrorHandling:
    def test_extract_without_model_raises(self, config):
        """Extract should raise if no model provided."""
        extractor = PoseExtractor(config, model_path=None)
        with pytest.raises(RuntimeError, match="PoseLandmarker not initialized"):
            extractor.extract("dummy.mp4")

    def test_extract_nonexistent_file_raises(self, config):
        """Extract should raise for nonexistent file."""
        # Skip actual model loading, just test the file check
        extractor = PoseExtractor(config, model_path=None)
        # Manually set landmarker to non-None to bypass that check
        extractor.landmarker = object()  # dummy

        with pytest.raises(FileNotFoundError):
            extractor.extract("nonexistent_video_xyz123.mp4")


# ── Cleanup ──

class TestCleanup:
    def test_close_method_exists(self, config):
        extractor = PoseExtractor(config, model_path=None)
        # Should not raise
        extractor.close()

    def test_close_sets_landmarker_to_none(self, config):
        extractor = PoseExtractor(config, model_path=None)
        extractor.landmarker = object()  # set to something
        extractor.close()
        assert extractor.landmarker is None

    def test_close_idempotent(self, config):
        """Calling close multiple times should be safe."""
        extractor = PoseExtractor(config, model_path=None)
        extractor.close()
        extractor.close()  # should not raise
        assert extractor.landmarker is None
