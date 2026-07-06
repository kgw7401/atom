"""Tests for BoxingFeeder and BoxingFeederFromArrays."""

import numpy as np
import pytest
import torch

from ml.configs import BoxingConfig
from ml.feeders.boxing_feeder import BoxingFeeder, BoxingFeederFromArrays


@pytest.fixture
def config():
    return BoxingConfig()


def _random_keypoints(T: int = 30, V: int = 15, C: int = 3) -> np.ndarray:
    """Generate random keypoint array (T, V, C)."""
    return np.random.randn(T, V, C).astype(np.float32)


# ── BoxingFeederFromArrays basic ──

class TestFromArraysBasic:
    def test_length(self, config):
        kps = [_random_keypoints() for _ in range(5)]
        labels = [0, 1, 2, 3, 4]
        ds = BoxingFeederFromArrays(kps, labels, config)
        assert len(ds) == 5

    def test_output_shape(self, config):
        kps = [_random_keypoints(T=30)]
        labels = [0]
        ds = BoxingFeederFromArrays(kps, labels, config, augment=False)
        tensor, label = ds[0]
        # (C, T, V, 1) = (3, 30, 15, 1)
        assert tensor.shape == (3, 30, 15, 1)
        assert label == 0

    def test_output_dtype(self, config):
        kps = [_random_keypoints()]
        labels = [0]
        ds = BoxingFeederFromArrays(kps, labels, config, augment=False)
        tensor, _ = ds[0]
        assert tensor.dtype == torch.float32


# ── Temporal windowing ──

class TestTemporalWindow:
    def test_exact_length_no_change(self, config):
        """T == window_size → no crop/pad."""
        kps = [_random_keypoints(T=30)]
        labels = [0]
        ds = BoxingFeederFromArrays(kps, labels, config, augment=False)
        tensor, _ = ds[0]
        assert tensor.shape[1] == 30

    def test_longer_sequence_cropped(self, config):
        """T > window_size → cropped to window_size."""
        kps = [_random_keypoints(T=60)]
        labels = [0]
        ds = BoxingFeederFromArrays(kps, labels, config, augment=False)
        tensor, _ = ds[0]
        assert tensor.shape[1] == 30

    def test_shorter_sequence_padded(self, config):
        """T < window_size → padded to window_size."""
        kps = [_random_keypoints(T=10)]
        labels = [0]
        ds = BoxingFeederFromArrays(kps, labels, config, augment=False)
        tensor, _ = ds[0]
        assert tensor.shape[1] == 30

    def test_very_short_padded(self, config):
        """Even T=1 gets padded."""
        kps = [_random_keypoints(T=1)]
        labels = [0]
        ds = BoxingFeederFromArrays(kps, labels, config, augment=False)
        tensor, _ = ds[0]
        assert tensor.shape[1] == 30

    def test_pad_repeats_content(self, config):
        """Padding by tiling preserves original values."""
        kp = np.ones((5, 15, 3), dtype=np.float32) * 42.0
        ds = BoxingFeederFromArrays([kp], [0], config, augment=False)
        tensor, _ = ds[0]
        # All values should be 42.0 (tiled)
        assert torch.allclose(tensor, torch.tensor(42.0))


# ── Augmentation ──

class TestAugmentation:
    def test_augment_changes_data(self, config):
        """With augment=True, output differs from raw (statistically)."""
        np.random.seed(42)
        kp = np.ones((30, 15, 3), dtype=np.float32)
        ds_aug = BoxingFeederFromArrays([kp], [0], config, augment=True)
        ds_raw = BoxingFeederFromArrays([kp], [0], config, augment=False)

        # Run multiple times - augmentation should produce different results
        raw_tensor, _ = ds_raw[0]
        different = False
        for _ in range(10):
            aug_tensor, _ = ds_aug[0]
            if not torch.allclose(aug_tensor, raw_tensor, atol=1e-6):
                different = True
                break
        assert different, "Augmentation should modify data"

    def test_augment_preserves_shape(self, config):
        kp = _random_keypoints(T=30)
        ds = BoxingFeederFromArrays([kp], [0], config, augment=True)
        tensor, _ = ds[0]
        assert tensor.shape == (3, 30, 15, 1)

    def test_no_augment_deterministic(self, config):
        """Without augmentation, same input → same output."""
        kp = _random_keypoints(T=30)
        ds = BoxingFeederFromArrays([kp], [0], config, augment=False)
        t1, _ = ds[0]
        t2, _ = ds[0]
        assert torch.allclose(t1, t2)


# ── Flip augmentation ──

class TestFlip:
    def test_flip_negates_x(self, config):
        feeder = BoxingFeeder.__new__(BoxingFeeder)
        feeder.num_joints = 15
        kp = np.ones((10, 15, 3), dtype=np.float32)
        flipped = feeder._flip(kp)
        # x-coordinate should be negated
        assert np.all(flipped[:, :, 0] == -1.0)
        # y, z unchanged
        assert np.all(flipped[:, :, 1] == 1.0)
        assert np.all(flipped[:, :, 2] == 1.0)

    def test_flip_swaps_pairs(self, config):
        feeder = BoxingFeeder.__new__(BoxingFeeder)
        feeder.num_joints = 15
        kp = np.zeros((10, 15, 3), dtype=np.float32)
        # Set distinct values for joint pair (1, 2)
        kp[:, 1, :] = 1.0
        kp[:, 2, :] = 2.0
        flipped = feeder._flip(kp)
        # After flip: joint 1 should have joint 2's values (x negated)
        assert np.all(flipped[:, 1, 1] == 2.0)  # y of original joint 2
        assert np.all(flipped[:, 2, 1] == 1.0)  # y of original joint 1

    def test_flip_is_involution(self, config):
        """Flipping twice returns to original (except x-negation cancels)."""
        feeder = BoxingFeeder.__new__(BoxingFeeder)
        feeder.num_joints = 15
        kp = np.random.randn(10, 15, 3).astype(np.float32)
        double_flip = feeder._flip(feeder._flip(kp))
        np.testing.assert_allclose(double_flip, kp, atol=1e-6)


# ── Joint dropout ──

class TestJointDropout:
    def test_dropout_zeros_some_joints(self, config):
        feeder = BoxingFeeder.__new__(BoxingFeeder)
        np.random.seed(0)
        kp = np.ones((10, 15, 3), dtype=np.float32)
        dropped = feeder._joint_dropout(kp, p=0.5)
        # Some joints should be zeroed
        zero_joints = np.all(dropped == 0.0, axis=(0, 2))
        assert zero_joints.any(), "Some joints should be dropped"

    def test_dropout_preserves_shape(self, config):
        feeder = BoxingFeeder.__new__(BoxingFeeder)
        kp = np.random.randn(10, 15, 3).astype(np.float32)
        dropped = feeder._joint_dropout(kp, p=0.1)
        assert dropped.shape == kp.shape

    def test_dropout_p0_keeps_all(self, config):
        feeder = BoxingFeeder.__new__(BoxingFeeder)
        kp = np.ones((10, 15, 3), dtype=np.float32)
        dropped = feeder._joint_dropout(kp, p=0.0)
        np.testing.assert_array_equal(dropped, kp)

    def test_dropout_doesnt_modify_original(self, config):
        feeder = BoxingFeeder.__new__(BoxingFeeder)
        kp = np.ones((10, 15, 3), dtype=np.float32)
        original = kp.copy()
        feeder._joint_dropout(kp, p=0.5)
        np.testing.assert_array_equal(kp, original)


# ── Multiple samples ──

class TestMultipleSamples:
    def test_different_labels(self, config):
        kps = [_random_keypoints() for _ in range(3)]
        labels = [0, 5, 10]
        ds = BoxingFeederFromArrays(kps, labels, config, augment=False)
        for i, expected in enumerate(labels):
            _, label = ds[i]
            assert label == expected

    def test_varying_lengths(self, config):
        """Samples with different T values all produce same output shape."""
        kps = [
            _random_keypoints(T=10),
            _random_keypoints(T=30),
            _random_keypoints(T=60),
        ]
        labels = [0, 1, 2]
        ds = BoxingFeederFromArrays(kps, labels, config, augment=False)
        for i in range(3):
            tensor, _ = ds[i]
            assert tensor.shape == (3, 30, 15, 1)
