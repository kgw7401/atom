"""Session plan generation and logging API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from atom.api.schemas import PlanRequest, PlanResponse, SessionLogRequest, SessionLogResponse
from atom.models.base import async_session
from atom.models.tables import SessionLog
from atom.services.session_service import SessionService


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("/plan", response_model=PlanResponse, status_code=201)
async def generate_plan(body: PlanRequest):
    try:
        async with async_session() as session:
            svc = SessionService(session)
            result = await svc.generate_plan(
                level=body.level,
                rounds=body.rounds,
                round_duration_sec=body.round_duration_sec,
                rest_sec=body.rest_sec,
                program_day_id=body.program_day_id,
            )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/log", response_model=SessionLogResponse, status_code=201)
async def log_session(body: SessionLogRequest):
    """Save a completed session, update streak, and advance program."""
    async with async_session() as db:
        log = SessionLog(
            drill_plan_id=body.drill_plan_id,
            started_at=body.started_at,
            completed_at=body.completed_at,
            total_duration_sec=body.total_duration_sec,
            rounds_completed=body.rounds_completed,
            rounds_total=body.rounds_total,
            segments_delivered=body.segments_delivered,
            status=body.status,
        )
        db.add(log)
        await db.commit()
        await db.refresh(log)

        # Update profile aggregates + streak + advance program
        from atom.services.profile_service import ProfileService
        profile_svc = ProfileService(db)
        await profile_svc.aggregate()

        if body.status == "completed":
            await profile_svc.update_streak()
            await profile_svc.advance_program()

        return log
