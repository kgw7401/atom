"""Session endpoints: create, upload, status, report."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from sqlalchemy import select

import numpy as np

from server.models.db import Session, StateTransition, UserState, async_session
from server.schemas import (
    CreateSessionRequest,
    EnrichedReportResponse,
    SessionReportResponse,
    SessionResponse,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _session_response(s: Session) -> SessionResponse:
    return SessionResponse(
        session_id=s.id,
        user_id=s.user_id,
        mode=s.mode,
        status=s.status,
        pipeline_stage=s.pipeline_stage,
        pipeline_progress=s.pipeline_progress,
        duration_seconds=s.duration_seconds,
        error_code=s.error_code,
        created_at=s.created_at,
    )


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(req: CreateSessionRequest):
    """Create a new session."""
    async with async_session() as db:
        session = Session(user_id=req.user_id, mode=req.mode)
        db.add(session)
        await db.commit()
        await db.refresh(session)
        return _session_response(session)


@router.post("/{session_id}/upload", response_model=SessionResponse)
async def upload_video(
    session_id: str,
    file: UploadFile,
    background_tasks: BackgroundTasks,
):
    """Upload video and trigger analysis pipeline."""
    async with async_session() as db:
        session = await db.get(Session, session_id)
        if session is None:
            raise HTTPException(404, "Session not found")

        if session.status not in ("CREATED", "FAILED"):
            raise HTTPException(400, f"Cannot upload in status {session.status}")

        # Save file
        upload_dir = Path("uploads") / session.user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        video_path = upload_dir / f"{session_id}.mp4"

        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        session.video_path = str(video_path)
        session.status = "UPLOADING"
        await db.commit()
        await db.refresh(session)

    # Trigger pipeline in background
    background_tasks.add_task(_run_pipeline_safe, session_id)

    async with async_session() as db:
        session = await db.get(Session, session_id)
        return _session_response(session)


async def _run_pipeline_safe(session_id: str) -> None:
    """Run pipeline with error handling (for BackgroundTasks)."""
    try:
        from server.services.analysis_pipeline import run_pipeline
        await run_pipeline(session_id)
    except Exception:
        pass  # Pipeline already handles errors and sets FAILED status


@router.get("/{session_id}/status", response_model=SessionResponse)
async def get_status(session_id: str):
    """Get session status and pipeline progress."""
    async with async_session() as db:
        session = await db.get(Session, session_id)
        if session is None:
            raise HTTPException(404, "Session not found")
        return _session_response(session)


@router.get("/{session_id}/report", response_model=SessionReportResponse)
async def get_report(session_id: str):
    """Get analysis report for a completed session."""
    async with async_session() as db:
        session = await db.get(Session, session_id)
        if session is None:
            raise HTTPException(404, "Session not found")

        if session.status != "COMPLETED":
            raise HTTPException(400, f"Session not completed (status: {session.status})")

        result = await db.execute(
            select(StateTransition).where(StateTransition.session_id == session_id)
        )
        transition = result.scalar_one_or_none()
        if transition is None:
            raise HTTPException(404, "No state transition found for this session")

        observation = json.loads(transition.observation_json) if transition.observation_json else [None] * 18
        mask = json.loads(transition.observation_mask_json)
        delta = json.loads(transition.delta_json)

        return SessionReportResponse(
            session_id=session_id,
            observation=observation,
            observation_mask=mask,
            delta=delta,
        )


@router.get("/{session_id}/coaching", response_model=EnrichedReportResponse)
async def get_coaching(session_id: str):
    """Get enriched coaching report for a completed session."""
    async with async_session() as db:
        session = await db.get(Session, session_id)
        if session is None:
            raise HTTPException(404, "Session not found")

        if session.status != "COMPLETED":
            raise HTTPException(400, f"Session not completed (status: {session.status})")

        result = await db.execute(
            select(StateTransition).where(StateTransition.session_id == session_id)
        )
        transition = result.scalar_one_or_none()
        if transition is None:
            raise HTTPException(404, "No state transition found for this session")

        # Load current user state
        user = await db.get(UserState, session.user_id)
        if user is None:
            raise HTTPException(404, "User not found")

        from server.services.report_generator import generate_session_report
        from src.state.types import StateVector

        current_state = StateVector.from_json({
            "values": json.loads(user.vector_json),
            "confidence": json.loads(user.confidence_json),
            "obs_counts": json.loads(user.obs_counts_json),
            "version": user.row_version,
            "schema_version": user.schema_version,
        })

        delta_arr = np.array(json.loads(transition.delta_json), dtype=np.float64)
        mask_arr = np.array(json.loads(transition.observation_mask_json), dtype=bool)
        observation = json.loads(transition.observation_json) if transition.observation_json else [None] * 18

        report = generate_session_report(
            session_id=session_id,
            delta=delta_arr,
            mask=mask_arr,
            current_state=current_state,
            mode=session.mode,
            duration_seconds=session.duration_seconds or 0.0,
        )

        raw = SessionReportResponse(
            session_id=session_id,
            observation=observation,
            observation_mask=mask_arr.tolist(),
            delta=delta_arr.tolist(),
        )

        return EnrichedReportResponse(
            session_id=session_id,
            session_summary=report.session_summary,
            state_delta=report.state_delta,
            weaknesses=report.weaknesses,
            coaching=report.coaching,
            raw=raw,
        )
