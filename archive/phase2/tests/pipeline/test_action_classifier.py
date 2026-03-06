"""Tests for Pipeline Stage 2: Action Classification."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from ml.configs import BoxingConfig
from ml.pipeline.action_classifier import ActionClassifier, classify_video
from ml.pipeline.types import DetectedAction


@pytest.fixture
def config():
    return BoxingConfig()


@pytest.fixture
def mock_model():
    """Create a mock TorchScript model."""
    model = Mock()
    model.eval = Mock(return_value=None)

    # Mock forward pass to return predictable logits (as torch.Tensor)
    def forward(x):
        batch_size = x.shape[0]
        num_classes = 11
        # Return logits with class 0 (jab) having highest score
        logits = torch.zeros(batch_size, num_classes)
        logits[:, 0] = 2.0  # jab
        logits[:, 1] = 1.0  # cross
        return logits

    model.side_effect = forward  # Mock objects called with () use side_effect

    return model


def _random_keypoints(T: int = 100, V: int = 15, C: int = 3) -> np.ndarray:
    """Generate random keypoint sequence."""
    return np.random.randn(T, V, C).astype(np.float32)


# ── Sliding windows ──

class TestSlidingWindows:
    @patch('torch.jit.load')
    def test_window_generation(self, mock_load, config, mock_model):
        """Test sliding window generation."""
        mock_load.return_value = mock_model

        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        # 100 frames, window=30, stride=5
        # Windows: 0-30, 5-35, 10-40, ..., 70-100
        # Count: (100 - 30) // 5 + 1 = 15
        keypoints = _random_keypoints(T=100)
        windows = classifier._sliding_windows(keypoints)

        assert len(windows) == 15
        assert windows[0][1] == 0  # First window starts at frame 0
        assert windows[1][1] == 5  # Second window starts at frame 5
        assert windows[-1][1] == 70  # Last window starts at frame 70

    @patch('torch.jit.load')
    def test_window_shape(self, mock_load, config, mock_model):
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        keypoints = _random_keypoints(T=100)
        windows = classifier._sliding_windows(keypoints)

        # Each window should be (window_size, V, C)
        for window, _ in windows:
            assert window.shape == (30, 15, 3)

    @patch('torch.jit.load')
    def test_short_sequence(self, mock_load, config, mock_model):
        """Sequence shorter than window_size returns empty."""
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        keypoints = _random_keypoints(T=20)  # < 30
        windows = classifier._sliding_windows(keypoints)

        assert len(windows) == 0

    @patch('torch.jit.load')
    def test_exact_window_size(self, mock_load, config, mock_model):
        """Sequence exactly window_size returns one window."""
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        keypoints = _random_keypoints(T=30)
        windows = classifier._sliding_windows(keypoints)

        assert len(windows) == 1


# ── Inference ──

class TestInference:
    @patch('torch.jit.load')
    def test_batch_inference_output(self, mock_load, config, mock_model):
        """Test that batch inference produces DetectedAction objects."""
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        keypoints = _random_keypoints(T=60)
        windows = classifier._sliding_windows(keypoints)
        detections = classifier._batch_inference(windows, fps=30.0)

        assert len(detections) == len(windows)
        assert all(isinstance(d, DetectedAction) for d in detections)

    @patch('torch.jit.load')
    def test_detection_fields(self, mock_load, config, mock_model):
        """Test DetectedAction fields are populated correctly."""
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        keypoints = _random_keypoints(T=60)
        windows = classifier._sliding_windows(keypoints)
        detections = classifier._batch_inference(windows, fps=30.0)

        det = detections[0]
        assert det.action == "jab"  # mock model returns class 0
        assert 0.0 <= det.confidence <= 1.0
        assert det.timestamp >= 0.0
        assert det.window_start >= 0.0
        assert det.window_end > det.window_start

    @patch('torch.jit.load')
    def test_timestamp_calculation(self, mock_load, config, mock_model):
        """Test timestamp is center of window."""
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        # Window at frames 0-30, center = 15
        # FPS = 30 → timestamp = 15/30 = 0.5s
        windows = [(_random_keypoints(T=30), 0)]
        detections = classifier._batch_inference(windows, fps=30.0)

        assert abs(detections[0].timestamp - 0.5) < 1e-6

    @patch('torch.jit.load')
    def test_empty_windows(self, mock_load, config, mock_model):
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        detections = classifier._batch_inference([], fps=30.0)
        assert detections == []


# ── Confidence filtering ──

class TestConfidenceFiltering:
    @patch('torch.jit.load')
    def test_filters_low_confidence(self, mock_load, config):
        """Detections below threshold are filtered."""
        # Mock model that returns low confidence
        def low_conf_forward(x):
            batch_size = x.shape[0]
            logits = torch.zeros(batch_size, 11)
            logits[:, 0] = 0.1  # Low score → low confidence after softmax
            return logits

        model = Mock()
        model.eval = Mock(return_value=None)
        model.side_effect = low_conf_forward
        mock_load.return_value = model

        classifier = ActionClassifier('dummy.pt', config, device='cpu')
        keypoints = _random_keypoints(T=60)
        detections = classifier.classify(keypoints, fps=30.0)

        # All should be filtered (confidence < 0.7)
        assert len(detections) == 0


# ── NMS ──

class TestNMS:
    @patch('torch.jit.load')
    def test_temporal_overlap_full(self, mock_load, config, mock_model):
        """Test overlap calculation for identical windows."""
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        det1 = DetectedAction(
            timestamp=1.0, action="jab", confidence=0.9,
            window_start=0.5, window_end=1.5
        )
        det2 = DetectedAction(
            timestamp=1.0, action="jab", confidence=0.8,
            window_start=0.5, window_end=1.5
        )

        overlap = classifier._temporal_overlap(det1, det2)
        assert abs(overlap - 1.0) < 1e-6  # Full overlap

    @patch('torch.jit.load')
    def test_temporal_overlap_none(self, mock_load, config, mock_model):
        """Test no overlap."""
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        det1 = DetectedAction(
            timestamp=1.0, action="jab", confidence=0.9,
            window_start=0.0, window_end=1.0
        )
        det2 = DetectedAction(
            timestamp=2.0, action="jab", confidence=0.8,
            window_start=2.0, window_end=3.0
        )

        overlap = classifier._temporal_overlap(det1, det2)
        assert overlap == 0.0

    @patch('torch.jit.load')
    def test_temporal_overlap_partial(self, mock_load, config, mock_model):
        """Test partial overlap."""
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        det1 = DetectedAction(
            timestamp=1.0, action="jab", confidence=0.9,
            window_start=0.0, window_end=2.0
        )
        det2 = DetectedAction(
            timestamp=1.5, action="jab", confidence=0.8,
            window_start=1.0, window_end=3.0
        )

        # Intersection: 1.0 to 2.0 = 1.0
        # Union: 2.0 + 2.0 - 1.0 = 3.0
        # IoU: 1.0 / 3.0 ≈ 0.333
        overlap = classifier._temporal_overlap(det1, det2)
        assert abs(overlap - 1.0/3.0) < 1e-6

    @patch('torch.jit.load')
    def test_nms_keeps_highest_confidence(self, mock_load, config, mock_model):
        """NMS should keep highest confidence when detections overlap."""
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        dets = [
            DetectedAction(
                timestamp=1.0, action="jab", confidence=0.8,
                window_start=0.5, window_end=1.5
            ),
            DetectedAction(
                timestamp=1.1, action="jab", confidence=0.95,  # Higher
                window_start=0.6, window_end=1.6
            ),
            DetectedAction(
                timestamp=1.2, action="jab", confidence=0.7,
                window_start=0.7, window_end=1.7
            ),
        ]

        filtered = classifier._nms(dets)

        # Should keep only the one with conf=0.95
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.95

    @patch('torch.jit.load')
    def test_nms_different_actions(self, mock_load, config, mock_model):
        """NMS applies per action class."""
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        dets = [
            DetectedAction(
                timestamp=1.0, action="jab", confidence=0.9,
                window_start=0.5, window_end=1.5
            ),
            DetectedAction(
                timestamp=1.1, action="cross", confidence=0.85,
                window_start=0.6, window_end=1.6
            ),
        ]

        filtered = classifier._nms(dets)

        # Both should be kept (different actions)
        assert len(filtered) == 2


# ── Full classification ──

class TestClassify:
    @patch('torch.jit.load')
    def test_classify_returns_sorted(self, mock_load, config, mock_model):
        """Results should be sorted by timestamp."""
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        keypoints = _random_keypoints(T=100)
        detections = classifier.classify(keypoints, fps=30.0)

        # Check sorted
        timestamps = [d.timestamp for d in detections]
        assert timestamps == sorted(timestamps)

    @patch('torch.jit.load')
    def test_classify_short_sequence(self, mock_load, config, mock_model):
        """Short sequence returns empty."""
        mock_load.return_value = mock_model
        classifier = ActionClassifier('dummy.pt', config, device='cpu')

        keypoints = _random_keypoints(T=20)  # < window_size
        detections = classifier.classify(keypoints, fps=30.0)

        assert detections == []


# ── Convenience function ──

class TestConvenienceFunction:
    @patch('torch.jit.load')
    def test_classify_video(self, mock_load, mock_model):
        mock_load.return_value = mock_model

        keypoints = _random_keypoints(T=100)
        detections = classify_video(
            keypoints, 'dummy.pt', fps=30.0, device='cpu'
        )

        assert isinstance(detections, list)
        assert all(isinstance(d, DetectedAction) for d in detections)
