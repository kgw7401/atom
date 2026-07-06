"""Session lifecycle endpoints."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile

from server.models.db import Session, SessionReport, Script, async_session
from server.models.schemas import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionReportResponse,
    SessionStatusResponse,
    UploadCompleteRequest,
    InstructionResult,
    ReportSummary,
    CoachingFeedback,
)
from server.services.analysis_pipeline import run_analysis

router = APIRouter(tags=["sessions"])

# Video upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/sessions", response_model=SessionCreateResponse)
async def create_session(req: SessionCreateRequest) -> SessionCreateResponse:
    session_id = str(uuid.uuid4())

    async with async_session() as db:
        session = Session(
            id=session_id,
            user_id=str(req.user_id),
            script_id=str(req.script_id),
            started_at=req.started_at,
            status="created",
        )
        db.add(session)
        await db.commit()

    # For MVP: upload URL is just a local path placeholder.
    # Production: generate GCS signed URL here.
    upload_url = f"/uploads/{session_id}/video.mp4"

    return SessionCreateResponse(
        session_id=uuid.UUID(session_id),
        upload_url=upload_url,
        status="created",
    )


@router.post("/sessions/{session_id}/upload-complete")
async def upload_complete(
    session_id: str,
    req: UploadCompleteRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> SessionStatusResponse:
    async with async_session() as db:
        session = await db.get(Session, session_id)
        if not session:
            raise HTTPException(404, "Session not found")

        session.video_duration_seconds = req.video_duration_seconds
        session.status = "analyzing"

        # For MVP: video_path points to local file.
        # Production: GCS path from upload URL.
        if not session.video_path:
            session.video_path = f"uploads/{session_id}/video.mp4"

        await db.commit()

    # Trigger analysis in background
    inference = request.app.state.inference
    background_tasks.add_task(run_analysis, session_id, inference)

    return SessionStatusResponse(status="analyzing", progress=0.0)


@router.get("/sessions/{session_id}/status", response_model=SessionStatusResponse)
async def get_status(session_id: str) -> SessionStatusResponse:
    async with async_session() as db:
        session = await db.get(Session, session_id)
        if not session:
            raise HTTPException(404, "Session not found")

    return SessionStatusResponse(
        status=session.status,
        progress=session.analysis_progress,
    )


@router.get("/sessions/{session_id}/report", response_model=SessionReportResponse)
async def get_report(session_id: str) -> SessionReportResponse:
    async with async_session() as db:
        session = await db.get(Session, session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        if session.status != "completed":
            raise HTTPException(409, f"Session status is '{session.status}', not 'completed'")

        report = await db.get(SessionReport, None)
        # Query by session_id
        from sqlalchemy import select
        stmt = select(SessionReport).where(SessionReport.session_id == session_id)
        result = await db.execute(stmt)
        report = result.scalar_one_or_none()

        if not report:
            raise HTTPException(404, "Report not found")

    return SessionReportResponse(
        session_id=uuid.UUID(session_id),
        overall_score=report.overall_score,
        summary=ReportSummary.model_validate_json(report.summary_json),
        instructions=[
            InstructionResult.model_validate(r)
            for r in json.loads(report.instruction_results_json)
        ],
        coaching=CoachingFeedback.model_validate_json(report.coaching_json),
    )


@router.post("/sessions/{session_id}/upload-video")
async def upload_video(
    session_id: str,
    video: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    request: Request = None,
) -> SessionStatusResponse:
    """Upload video file for a session and trigger analysis.

    Accepts multipart/form-data with video file.
    Validates file type and saves to uploads/{session_id}/video.mp4.
    """
    # Validate session exists
    async with async_session() as db:
        session = await db.get(Session, session_id)
        if not session:
            raise HTTPException(404, "Session not found")

        # Validate video file type
        if not video.content_type or not video.content_type.startswith("video/"):
            raise HTTPException(400, f"Invalid file type: {video.content_type}. Must be video/*")

        # Create session directory
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save video file
        video_path = session_dir / "video.mp4"
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)

        # Update session with video path and status
        session.video_path = str(video_path)
        session.video_duration_seconds = len(content) / (1024 * 1024)  # rough estimate
        session.status = "analyzing"
        await db.commit()

    # Trigger analysis in background
    if background_tasks and request:
        inference = request.app.state.inference
        background_tasks.add_task(run_analysis, session_id, inference)

    return SessionStatusResponse(status="analyzing", progress=0.0)
