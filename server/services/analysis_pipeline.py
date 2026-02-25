"""Video → keypoints → segments → O_t → S_t update pipeline.

Orchestrates the full analysis pipeline for a session video.
Pipeline is synchronous (MVP). Async task queue deferred to Phase 2d.

Reference: spec/roadmap.md Phase 2b, spec/runtime.md
"""

from __future__ import annotations

import json
import logging

import numpy as np
from sqlalchemy import select

from server.models.db import Session, StateTransition, UserState, async_session
from src.state.observation import compute_observation
from src.state.types import ObservationVector, StateVector
from src.state.update import apply_observation, compute_delta

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Error during pipeline execution."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


async def run_pipeline(session_id: str) -> None:
    """Run the full video → state update pipeline for a session.

    Pipeline stages:
        1. keypoint_extraction — MediaPipe pose extraction
        2. action_classification — LSTM action segmentation
        3. observation — Compute ObservationVector
        4. state_update — EMA update and DB persistence

    On failure, sets session status to FAILED with error details.
    State is never modified on failure (transactional safety).

    Args:
        session_id: The session to process.
    """
    async with async_session() as db:
        session = await db.get(Session, session_id)
        if session is None:
            raise PipelineError("init", f"Session not found: {session_id}")
        if not session.video_path:
            raise PipelineError("init", "No video path set on session")

        # Idempotency guard (runtime.md §3)
        if session.state_update_applied:
            logger.info("Session %s already processed, skipping", session_id)
            return

        session.status = "PROCESSING"
        session.pipeline_stage = "keypoint_extraction"
        session.pipeline_progress = 0.0
        await db.commit()

        try:
            # --- Stage 1: Keypoint Extraction ---
            session.pipeline_progress = 0.1
            await db.commit()

            from src.vision.keypoint_extractor import extract_keypoints

            extraction = extract_keypoints(session.video_path)

            session.duration_seconds = extraction.duration
            session.pipeline_progress = 0.3
            session.pipeline_stage = "action_classification"
            await db.commit()

            # --- Stage 2: Action Classification ---
            from src.vision.action_classifier import ActionClassifier

            classifier = ActionClassifier.load()
            segments = classifier.classify(
                extraction.raw_keypoints, extraction.timestamps_s
            )

            session.pipeline_progress = 0.6
            session.pipeline_stage = "observation"
            await db.commit()

            # --- Stage 3: Compute Observation ---
            obs = compute_observation(
                segments=segments,
                keypoints=extraction.keypoint_frames,
                duration=extraction.duration,
                mode=session.mode,
            )

            session.pipeline_progress = 0.8
            session.pipeline_stage = "state_update"
            await db.commit()

            # --- Stage 4: State Update ---
            if not obs.is_empty:
                await _update_state(db, session, obs)

            session.status = "COMPLETED"
            session.pipeline_stage = None
            session.pipeline_progress = 1.0
            session.state_update_applied = True
            await db.commit()

            logger.info("Pipeline completed for session %s", session_id)

        except Exception as e:
            await db.rollback()
            # Reload session after rollback
            session = await db.get(Session, session_id)
            if session is not None:
                session.status = "FAILED"
                stage = e.stage if isinstance(e, PipelineError) else "unknown"
                session.error_code = f"PIPELINE_{stage.upper()}"
                session.error_detail = str(e)[:500]
                await db.commit()

            logger.exception("Pipeline failed for session %s", session_id)
            raise


async def _update_state(
    db, session: Session, obs: ObservationVector
) -> None:
    """Load current state, apply observation, write back with audit log.

    All writes happen within the caller's DB transaction.
    """
    # Load current state
    user_state = await db.get(UserState, session.user_id)

    if user_state is not None:
        current = StateVector.from_json({
            "values": json.loads(user_state.vector_json),
            "confidence": json.loads(user_state.confidence_json),
            "obs_counts": json.loads(user_state.obs_counts_json),
            "version": user_state.row_version,
            "schema_version": user_state.schema_version,
        })
    else:
        current = None

    # Apply observation → new state
    new_state = apply_observation(current, obs)

    # Compute delta
    delta = compute_delta(new_state, current) if current is not None else new_state.values.copy()

    # Persist state
    if user_state is None:
        user_state = UserState(
            user_id=session.user_id,
            device_id=session.user_id,  # placeholder — real device_id set via API
            vector_json=json.dumps(new_state.values.tolist()),
            confidence_json=json.dumps(new_state.confidence.tolist()),
            obs_counts_json=json.dumps(new_state.obs_counts.astype(int).tolist()),
            row_version=new_state.version,
            schema_version=new_state.schema_version,
        )
        db.add(user_state)
    else:
        user_state.vector_json = json.dumps(new_state.values.tolist())
        user_state.confidence_json = json.dumps(new_state.confidence.tolist())
        user_state.obs_counts_json = json.dumps(new_state.obs_counts.astype(int).tolist())
        user_state.row_version = new_state.version
        user_state.schema_version = new_state.schema_version

    # Audit log (append-only)
    transition = StateTransition(
        user_id=session.user_id,
        session_id=session.id,
        version_before=current.version if current else 0,
        version_after=new_state.version,
        vector_before_json=json.dumps(current.values.tolist() if current else []),
        vector_after_json=json.dumps(new_state.values.tolist()),
        observation_json=json.dumps(_nan_safe_list(obs.values)),
        observation_mask_json=json.dumps(obs.mask.tolist()),
        delta_json=json.dumps(delta.tolist()),
    )
    db.add(transition)

    await db.flush()


def _nan_safe_list(arr: np.ndarray) -> list:
    """Convert numpy array to JSON-safe list. NaN → null."""
    result = []
    for v in arr.flat:
        if isinstance(v, (np.floating, float)) and np.isnan(v):
            result.append(None)
        else:
            result.append(float(v))
    return result
