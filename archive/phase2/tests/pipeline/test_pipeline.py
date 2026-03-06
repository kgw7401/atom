"""Tests for Pipeline Orchestrator."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from ml.configs import BoxingConfig
from ml.pipeline.pipeline import AnalysisPipeline, analyze_session
from ml.pipeline.types import (
    DetectedAction,
    DetectedCombo,
    DrillResult,
    TTSInstruction,
)


@pytest.fixture
def config():
    return BoxingConfig()


@pytest.fixture
def mock_components():
    """Mock all pipeline components."""
    # Mock PoseExtractor
    mock_pose = Mock()
    mock_pose.extract = Mock(return_value=(
        np.random.randn(100, 15, 3).astype(np.float32),
        {
            'original_fps': 30.0,
            'processed_frames': 100,
            'dropped_frames': 10,
        }
    ))
    mock_pose.close = Mock()

    # Mock ActionClassifier
    mock_actions = [
        DetectedAction(
            timestamp=1.0, action='jab', confidence=0.9,
            window_start=0.5, window_end=1.5
        ),
        DetectedAction(
            timestamp=2.0, action='cross', confidence=0.85,
            window_start=1.5, window_end=2.5
        ),
    ]
    mock_classifier = Mock()
    mock_classifier.classify = Mock(return_value=mock_actions)

    # Mock SequenceRecognizer
    mock_combos = [
        DetectedCombo(start_time=0.5, end_time=2.5, actions=['jab', 'cross'])
    ]
    mock_recognizer = Mock()
    mock_recognizer.recognize = Mock(return_value=mock_combos)

    # Mock SessionMatcher
    mock_drill_result = DrillResult(
        matches=[],
        combo_stats={},
        total_instructions=1,
        total_success=1,
        total_partial=0,
        total_miss=0,
    )
    mock_matcher = Mock()
    mock_matcher.match = Mock(return_value=mock_drill_result)

    return {
        'pose': mock_pose,
        'classifier': mock_classifier,
        'recognizer': mock_recognizer,
        'matcher': mock_matcher,
    }


# ── Pipeline initialization ──

class TestInitialization:
    @patch('ml.pipeline.pipeline.SessionMatcher')
    @patch('ml.pipeline.pipeline.SequenceRecognizer')
    @patch('ml.pipeline.pipeline.ActionClassifier')
    @patch('ml.pipeline.pipeline.PoseExtractor')
    def test_creates_all_components(
        self, mock_pose_cls, mock_classifier_cls, mock_recognizer_cls,
        mock_matcher_cls, config
    ):
        """Test that all 4 components are created."""
        pipeline = AnalysisPipeline(
            pose_model_path='pose.task',
            action_model_path='action.pt',
            config=config,
        )

        mock_pose_cls.assert_called_once()
        mock_classifier_cls.assert_called_once()
        mock_recognizer_cls.assert_called_once()
        mock_matcher_cls.assert_called_once()


# ── Full session analysis ──

class TestAnalyzeSession:
    @patch('ml.pipeline.pipeline.SessionMatcher')
    @patch('ml.pipeline.pipeline.SequenceRecognizer')
    @patch('ml.pipeline.pipeline.ActionClassifier')
    @patch('ml.pipeline.pipeline.PoseExtractor')
    def test_analyze_session_flow(
        self, mock_pose_cls, mock_classifier_cls, mock_recognizer_cls,
        mock_matcher_cls, config, mock_components
    ):
        """Test complete analyze_session flow."""
        # Setup mocks
        mock_pose_cls.return_value = mock_components['pose']
        mock_classifier_cls.return_value = mock_components['classifier']
        mock_recognizer_cls.return_value = mock_components['recognizer']
        mock_matcher_cls.return_value = mock_components['matcher']

        pipeline = AnalysisPipeline('pose.task', 'action.pt', config)

        # Run analysis
        instructions = [
            TTSInstruction(
                timestamp=1.0,
                combo_name='원-투',
                expected_actions=['jab', 'cross']
            )
        ]

        result = pipeline.analyze_session('video.mp4', instructions)

        # Verify all stages called
        mock_components['pose'].extract.assert_called_once_with('video.mp4')
        mock_components['classifier'].classify.assert_called_once()
        mock_components['recognizer'].recognize.assert_called_once()
        mock_components['matcher'].match.assert_called_once()

        # Verify result structure
        assert 'drill_result' in result
        assert 'keypoints' in result
        assert 'detected_actions' in result
        assert 'detected_combos' in result
        assert 'metadata' in result

    @patch('ml.pipeline.pipeline.SessionMatcher')
    @patch('ml.pipeline.pipeline.SequenceRecognizer')
    @patch('ml.pipeline.pipeline.ActionClassifier')
    @patch('ml.pipeline.pipeline.PoseExtractor')
    def test_analyze_session_passes_fps(
        self, mock_pose_cls, mock_classifier_cls, mock_recognizer_cls,
        mock_matcher_cls, config, mock_components
    ):
        """Test that FPS is passed from pose extraction to action classifier."""
        mock_pose_cls.return_value = mock_components['pose']
        mock_classifier_cls.return_value = mock_components['classifier']
        mock_recognizer_cls.return_value = mock_components['recognizer']
        mock_matcher_cls.return_value = mock_components['matcher']

        pipeline = AnalysisPipeline('pose.task', 'action.pt', config)

        instructions = [
            TTSInstruction(
                timestamp=1.0,
                combo_name='test',
                expected_actions=['jab']
            )
        ]

        pipeline.analyze_session('video.mp4', instructions)

        # Check that classify was called with fps=30.0 from pose metadata
        call_args = mock_components['classifier'].classify.call_args
        assert call_args[1]['fps'] == 30.0


# ── Video-only analysis ──

class TestAnalyzeVideoOnly:
    @patch('ml.pipeline.pipeline.SessionMatcher')
    @patch('ml.pipeline.pipeline.SequenceRecognizer')
    @patch('ml.pipeline.pipeline.ActionClassifier')
    @patch('ml.pipeline.pipeline.PoseExtractor')
    def test_analyze_video_only(
        self, mock_pose_cls, mock_classifier_cls, mock_recognizer_cls,
        mock_matcher_cls, config, mock_components
    ):
        """Test video-only analysis (no TTS instructions)."""
        mock_pose_cls.return_value = mock_components['pose']
        mock_classifier_cls.return_value = mock_components['classifier']
        mock_recognizer_cls.return_value = mock_components['recognizer']
        mock_matcher_cls.return_value = mock_components['matcher']

        pipeline = AnalysisPipeline('pose.task', 'action.pt', config)
        result = pipeline.analyze_video_only('video.mp4')

        # Stages 1-3 called
        mock_components['pose'].extract.assert_called_once()
        mock_components['classifier'].classify.assert_called_once()
        mock_components['recognizer'].recognize.assert_called_once()

        # Stage 4 NOT called
        mock_components['matcher'].match.assert_not_called()

        # Result structure (no drill_result)
        assert 'keypoints' in result
        assert 'detected_actions' in result
        assert 'detected_combos' in result
        assert 'metadata' in result
        assert 'drill_result' not in result


# ── Resource cleanup ──

class TestCleanup:
    @patch('ml.pipeline.pipeline.SessionMatcher')
    @patch('ml.pipeline.pipeline.SequenceRecognizer')
    @patch('ml.pipeline.pipeline.ActionClassifier')
    @patch('ml.pipeline.pipeline.PoseExtractor')
    def test_close_calls_pose_extractor_close(
        self, mock_pose_cls, mock_classifier_cls, mock_recognizer_cls,
        mock_matcher_cls, config, mock_components
    ):
        """Test that close() calls pose_extractor.close()."""
        mock_pose_cls.return_value = mock_components['pose']
        mock_classifier_cls.return_value = mock_components['classifier']
        mock_recognizer_cls.return_value = mock_components['recognizer']
        mock_matcher_cls.return_value = mock_components['matcher']

        pipeline = AnalysisPipeline('pose.task', 'action.pt', config)
        pipeline.close()

        mock_components['pose'].close.assert_called_once()


# ── Convenience function ──

class TestConvenienceFunction:
    @patch('ml.pipeline.pipeline.AnalysisPipeline')
    def test_analyze_session_convenience(self, mock_pipeline_cls):
        """Test convenience wrapper function."""
        mock_pipeline = Mock()
        mock_pipeline.analyze_session = Mock(return_value={'test': 'result'})
        mock_pipeline.close = Mock()
        mock_pipeline_cls.return_value = mock_pipeline

        instructions = [
            TTSInstruction(
                timestamp=1.0,
                combo_name='test',
                expected_actions=['jab']
            )
        ]

        result = analyze_session(
            video_path='video.mp4',
            tts_instructions=instructions,
            pose_model_path='pose.task',
            action_model_path='action.pt',
        )

        # Verify pipeline created and used
        mock_pipeline_cls.assert_called_once()
        mock_pipeline.analyze_session.assert_called_once_with(
            'video.mp4', instructions
        )
        mock_pipeline.close.assert_called_once()

        assert result == {'test': 'result'}
