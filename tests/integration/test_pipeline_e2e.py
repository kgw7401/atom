"""Integration tests for the Vision → State pipeline.

Covers Phase 2b verification criteria:
  V1: Keypoint extraction — output shape (T, 11, 2), no NaN
  V2: Action classification — segments match expected actions
  V3: Full pipeline shadow 30s — O_t with ≥10 dims observed
  V4: Full pipeline heavy bag 60s — conditioning dims unobserved (< 90s)
  V5: State update from pipeline — S_1 in DB, audit log present
  V6: Two consecutive sessions — S_2 = EMA(S_1, O_2), delta computed
  V7: Pipeline failure recovery — corrupt video → FAILED, state unchanged

Tests requiring real video/model files are skipped when not available.

Reference: spec/roadmap.md Phase 2b
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from server.models.db import Session as SessionModel, StateTransition, UserState
from src.state.constants import NUM_DIMS
from src.state.observation import compute_observation
from src.state.types import ActionSegment
from tests.integration.conftest import (
    make_diverse_segments,
    make_fake_extraction,
    make_keypoint_sequence,
)

# Fixture availability
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SHADOW_30S = FIXTURES_DIR / "shadow_30s.mp4"
HEAVYBAG_60S = FIXTURES_DIR / "heavybag_60s.mp4"
MODEL_PATH = Path("models/lstm_best.pt")

has_fixtures = SHADOW_30S.exists() and HEAVYBAG_60S.exists()
has_model = MODEL_PATH.exists()

requires_fixtures = pytest.mark.skipif(not has_fixtures, reason="Video fixtures not available")
requires_model = pytest.mark.skipif(not has_model, reason="LSTM model not available")


# ===================================================================
# V1: Keypoint Extraction
# ===================================================================

class TestKeypointExtraction:
    """V1: Output shape (T, 11, 2), no NaN values."""

    def test_to_keypoint_frames_shape(self):
        """Conversion from (33, 4) raw → (11, 2) KeypointFrame."""
        from src.vision.keypoint_extractor import _to_keypoint_frames

        N = 50
        raw = np.random.rand(N, 33, 4).astype(np.float32)
        timestamps_s = np.arange(N, dtype=np.float64) / 30.0

        frames = _to_keypoint_frames(raw, timestamps_s)

        assert len(frames) == N
        for i, f in enumerate(frames):
            assert f.keypoints.shape == (11, 2)
            assert not np.any(np.isnan(f.keypoints))
            assert f.timestamp == pytest.approx(timestamps_s[i])

    def test_to_keypoint_frames_correct_indices(self):
        """Selected keypoints match UPPER_BODY_INDICES."""
        from src.vision.keypoint_extractor import UPPER_BODY_INDICES, _to_keypoint_frames

        raw = np.zeros((1, 33, 4), dtype=np.float32)
        for idx in UPPER_BODY_INDICES:
            raw[0, idx, 0] = idx / 100.0
            raw[0, idx, 1] = idx / 200.0
        timestamps_s = np.array([0.0])

        frames = _to_keypoint_frames(raw, timestamps_s)
        kp = frames[0].keypoints

        for local_idx, global_idx in enumerate(UPPER_BODY_INDICES):
            assert kp[local_idx, 0] == pytest.approx(global_idx / 100.0, abs=1e-6)
            assert kp[local_idx, 1] == pytest.approx(global_idx / 200.0, abs=1e-6)

    @requires_fixtures
    @requires_model
    def test_real_video_extraction(self):
        """Full extraction on real video produces valid output."""
        from src.vision.keypoint_extractor import extract_keypoints

        result = extract_keypoints(SHADOW_30S)

        assert result.raw_keypoints.ndim == 3
        assert result.raw_keypoints.shape[1] == 33
        assert result.raw_keypoints.shape[2] == 4
        assert result.fps > 0
        assert result.duration > 0

        for frame in result.keypoint_frames:
            assert frame.keypoints.shape == (11, 2)
            assert not np.any(np.isnan(frame.keypoints))


# ===================================================================
# V2: Action Classification
# ===================================================================

class TestActionClassification:
    """V2: Segments match expected actions."""

    def test_merge_detections_basic(self):
        """Consecutive same-action detections are merged."""
        from src.vision.action_classifier import ActionClassifier

        classifier = ActionClassifier.__new__(ActionClassifier)
        classifier.class_names = [
            "guard", "jab", "cross", "lead_hook", "rear_hook",
            "lead_uppercut", "rear_uppercut", "lead_bodyshot", "rear_bodyshot",
        ]

        detections = [
            {"timestamp_s": 0.0, "action": "guard", "confidence": 0.9},
            {"timestamp_s": 0.1, "action": "guard", "confidence": 0.9},
            {"timestamp_s": 0.2, "action": "jab", "confidence": 0.8},
            {"timestamp_s": 0.3, "action": "jab", "confidence": 0.85},
            {"timestamp_s": 0.4, "action": "guard", "confidence": 0.9},
        ]

        segments = classifier._merge_detections(detections)

        assert len(segments) == 3
        assert segments[0].class_name == "guard"
        assert segments[0].t_start == 0.0
        assert segments[0].t_end == 0.1
        assert segments[1].class_name == "jab"
        assert segments[1].class_id == 1
        assert segments[1].t_start == 0.2
        assert segments[1].t_end == 0.3
        assert segments[2].class_name == "guard"
        assert segments[2].class_id == 0

    def test_merge_detections_single(self):
        """Single detection produces one segment."""
        from src.vision.action_classifier import ActionClassifier

        classifier = ActionClassifier.__new__(ActionClassifier)
        classifier.class_names = ["guard", "jab"]

        detections = [{"timestamp_s": 1.0, "action": "jab", "confidence": 0.9}]
        segments = classifier._merge_detections(detections)

        assert len(segments) == 1
        assert segments[0].class_name == "jab"
        assert segments[0].t_start == 1.0
        assert segments[0].t_end == 1.0

    def test_merge_detections_empty(self):
        """Empty detections → empty segments."""
        from src.vision.action_classifier import ActionClassifier

        classifier = ActionClassifier.__new__(ActionClassifier)
        segments = classifier._merge_detections([])
        assert segments == []

    @requires_fixtures
    @requires_model
    def test_real_video_classification(self):
        """Real video → valid ActionSegments."""
        from src.vision.action_classifier import ActionClassifier
        from src.vision.keypoint_extractor import extract_keypoints

        extraction = extract_keypoints(SHADOW_30S)
        classifier = ActionClassifier.load()
        segments = classifier.classify(extraction.raw_keypoints, extraction.timestamps_s)

        assert len(segments) > 0
        for seg in segments:
            assert isinstance(seg, ActionSegment)
            assert 0 <= seg.class_id <= 8
            assert seg.t_start <= seg.t_end


# ===================================================================
# V3: Full Pipeline — Shadow 30s
# ===================================================================

class TestFullPipelineShadow:
    """V3: O_t computed with at least 10 dims observed."""

    def test_observation_from_synthetic(self):
        """Synthetic diverse segments → O_t with ≥10 dims observed."""
        segments = make_diverse_segments(duration=30.0)
        keypoints = make_keypoint_sequence(duration=30.0)

        obs = compute_observation(
            segments=segments, keypoints=keypoints, duration=30.0, mode="shadow",
        )

        assert obs.num_observed >= 10
        observed = obs.values[obs.mask]
        assert np.all(observed >= 0.0)
        assert np.all(observed <= 1.0)

    @requires_fixtures
    @requires_model
    def test_real_shadow_30s(self):
        """Real shadow 30s video → O_t with at least 10 dims observed."""
        from src.vision.action_classifier import ActionClassifier
        from src.vision.keypoint_extractor import extract_keypoints

        extraction = extract_keypoints(SHADOW_30S)
        classifier = ActionClassifier.load()
        segments = classifier.classify(extraction.raw_keypoints, extraction.timestamps_s)

        obs = compute_observation(
            segments=segments, keypoints=extraction.keypoint_frames,
            duration=extraction.duration, mode="shadow",
        )
        assert obs.num_observed >= 10


# ===================================================================
# V4: Full Pipeline — Heavy Bag 60s
# ===================================================================

class TestFullPipelineHeavyBag:
    """V4: Conditioning dims NOT observed for < 90s video."""

    def test_short_session_no_conditioning(self):
        """60s session → conditioning dims (15-17) unobserved."""
        segments = make_diverse_segments(duration=60.0)
        keypoints = make_keypoint_sequence(duration=60.0)

        obs = compute_observation(
            segments=segments, keypoints=keypoints, duration=60.0, mode="heavy_bag",
        )

        assert not obs.mask[15], "volume_endurance should not be observed for 60s"
        assert not obs.mask[16], "technique_endurance should not be observed for 60s"
        assert not obs.mask[17], "rhythm_stability should not be observed for 60s"

    @requires_fixtures
    @requires_model
    def test_real_heavybag_60s(self):
        """Real heavy bag 60s → conditioning dims NOT observed."""
        from src.vision.action_classifier import ActionClassifier
        from src.vision.keypoint_extractor import extract_keypoints

        extraction = extract_keypoints(HEAVYBAG_60S)
        classifier = ActionClassifier.load()
        segments = classifier.classify(extraction.raw_keypoints, extraction.timestamps_s)

        obs = compute_observation(
            segments=segments, keypoints=extraction.keypoint_frames,
            duration=extraction.duration, mode="heavy_bag",
        )
        assert not obs.mask[15]
        assert not obs.mask[16]
        assert not obs.mask[17]


# ===================================================================
# V5: State Update from Pipeline — DB Persistence
# ===================================================================

class TestStateUpdatePersistence:
    """V5: S_1 written to DB, audit log present."""

    @pytest.mark.asyncio
    async def test_state_written_to_db(self, db_session_factory, session_id, user_id):
        """Pipeline writes S_1 to user_state and creates state_transition."""
        # Create session record
        async with db_session_factory() as db:
            db.add(SessionModel(
                id=session_id, user_id=user_id, mode="shadow",
                status="CREATED", video_path="/tmp/fake.mp4",
            ))
            await db.commit()

        segments = make_diverse_segments(duration=30.0)
        fake_ext = make_fake_extraction(30.0)

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = segments

        with (
            patch("server.services.analysis_pipeline.async_session", db_session_factory),
            patch(
                "src.vision.keypoint_extractor.extract_keypoints",
                return_value=fake_ext,
            ),
            patch(
                "src.vision.action_classifier.ActionClassifier.load",
                return_value=mock_classifier,
            ),
        ):
            from server.services.analysis_pipeline import run_pipeline

            await run_pipeline(session_id)

        # Verify state written
        async with db_session_factory() as db:
            user_state = await db.get(UserState, user_id)
            assert user_state is not None
            values = json.loads(user_state.vector_json)
            assert len(values) == NUM_DIMS
            assert user_state.row_version == 1

            # Verify audit log
            from sqlalchemy import select

            result = await db.execute(
                select(StateTransition).where(StateTransition.session_id == session_id)
            )
            transition = result.scalar_one()
            assert transition is not None
            assert transition.version_before == 0
            assert transition.version_after == 1
            assert json.loads(transition.delta_json) is not None

            # Verify session completed
            session = await db.get(SessionModel, session_id)
            assert session.status == "COMPLETED"
            assert session.state_update_applied is True
            assert session.pipeline_progress == 1.0


# ===================================================================
# V6: Two Consecutive Sessions
# ===================================================================

class TestConsecutiveSessions:
    """V6: S_2 = EMA(S_1, O_2), delta computed."""

    @pytest.mark.asyncio
    async def test_two_sessions_ema(self, db_session_factory, user_id):
        """Two sessions for same user → EMA update with correct delta."""
        session_id_1 = str(uuid.uuid4())
        session_id_2 = str(uuid.uuid4())

        async with db_session_factory() as db:
            for sid in [session_id_1, session_id_2]:
                db.add(SessionModel(
                    id=sid, user_id=user_id, mode="shadow",
                    status="CREATED", video_path="/tmp/fake.mp4",
                ))
            await db.commit()

        segments_1 = make_diverse_segments(duration=30.0)
        segments_2 = make_diverse_segments(duration=30.0)
        fake_ext = make_fake_extraction(30.0)

        # Run session 1
        mock_clf_1 = MagicMock()
        mock_clf_1.classify.return_value = segments_1

        with (
            patch("server.services.analysis_pipeline.async_session", db_session_factory),
            patch("src.vision.keypoint_extractor.extract_keypoints", return_value=fake_ext),
            patch("src.vision.action_classifier.ActionClassifier.load", return_value=mock_clf_1),
        ):
            from server.services.analysis_pipeline import run_pipeline

            await run_pipeline(session_id_1)

        # Capture S_1
        async with db_session_factory() as db:
            us = await db.get(UserState, user_id)
            s1_values = np.array(json.loads(us.vector_json))
            assert us.row_version == 1

        # Run session 2
        mock_clf_2 = MagicMock()
        mock_clf_2.classify.return_value = segments_2

        with (
            patch("server.services.analysis_pipeline.async_session", db_session_factory),
            patch("src.vision.keypoint_extractor.extract_keypoints", return_value=fake_ext),
            patch("src.vision.action_classifier.ActionClassifier.load", return_value=mock_clf_2),
        ):
            await run_pipeline(session_id_2)

        # Verify S_2
        async with db_session_factory() as db:
            us = await db.get(UserState, user_id)
            s2_values = np.array(json.loads(us.vector_json))
            assert us.row_version == 2

            # Verify delta in audit log
            from sqlalchemy import select

            result = await db.execute(
                select(StateTransition).where(StateTransition.session_id == session_id_2)
            )
            transition = result.scalar_one()
            delta = np.array(json.loads(transition.delta_json))
            s1_before = np.array(json.loads(transition.vector_before_json))
            s2_after = np.array(json.loads(transition.vector_after_json))

            np.testing.assert_array_almost_equal(s1_before, s1_values)
            np.testing.assert_array_almost_equal(s2_after, s2_values)
            np.testing.assert_array_almost_equal(delta, s2_values - s1_values)


# ===================================================================
# V7: Pipeline Failure Recovery
# ===================================================================

class TestPipelineFailureRecovery:
    """V7: Corrupt video → FAILED status, state unchanged."""

    @pytest.mark.asyncio
    async def test_extraction_failure_sets_failed(self, db_session_factory, session_id, user_id):
        """Keypoint extraction failure → session FAILED, no state change."""
        async with db_session_factory() as db:
            db.add(SessionModel(
                id=session_id, user_id=user_id, mode="shadow",
                status="CREATED", video_path="/tmp/corrupt.mp4",
            ))
            await db.commit()

        with (
            patch("server.services.analysis_pipeline.async_session", db_session_factory),
            patch(
                "src.vision.keypoint_extractor.extract_keypoints",
                side_effect=ValueError("No poses detected"),
            ),
        ):
            from server.services.analysis_pipeline import run_pipeline

            with pytest.raises(ValueError):
                await run_pipeline(session_id)

        async with db_session_factory() as db:
            session = await db.get(SessionModel, session_id)
            assert session.status == "FAILED"
            assert session.error_detail is not None

            us = await db.get(UserState, user_id)
            assert us is None

    @pytest.mark.asyncio
    async def test_idempotency_guard(self, db_session_factory, session_id, user_id):
        """Already-processed session is skipped."""
        async with db_session_factory() as db:
            db.add(SessionModel(
                id=session_id, user_id=user_id, mode="shadow",
                status="COMPLETED", video_path="/tmp/fake.mp4",
                state_update_applied=True, pipeline_progress=1.0,
            ))
            await db.commit()

        with patch("server.services.analysis_pipeline.async_session", db_session_factory):
            from server.services.analysis_pipeline import run_pipeline

            await run_pipeline(session_id)

        async with db_session_factory() as db:
            session = await db.get(SessionModel, session_id)
            assert session.status == "COMPLETED"

    @pytest.mark.asyncio
    async def test_classification_failure_no_state_change(
        self, db_session_factory, session_id, user_id
    ):
        """Classification failure after extraction → FAILED, no state change."""
        async with db_session_factory() as db:
            db.add(SessionModel(
                id=session_id, user_id=user_id, mode="shadow",
                status="CREATED", video_path="/tmp/fake.mp4",
            ))
            await db.commit()

        fake_ext = make_fake_extraction(30.0)

        with (
            patch("server.services.analysis_pipeline.async_session", db_session_factory),
            patch("src.vision.keypoint_extractor.extract_keypoints", return_value=fake_ext),
            patch(
                "src.vision.action_classifier.ActionClassifier.load",
                side_effect=RuntimeError("Model file corrupt"),
            ),
        ):
            from server.services.analysis_pipeline import run_pipeline

            with pytest.raises(RuntimeError):
                await run_pipeline(session_id)

        async with db_session_factory() as db:
            session = await db.get(SessionModel, session_id)
            assert session.status == "FAILED"

            us = await db.get(UserState, user_id)
            assert us is None
