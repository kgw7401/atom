"""Session plan generation API endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from atom.api.schemas import PlanRequest, PlanResponse, SessionLogRequest, SessionSummary
from atom.models.base import async_session
from atom.models.tables import SessionLog
from atom.services.session_service import PlanValidationError, SessionService

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("/plan", response_model=PlanResponse, status_code=201)
async def generate_plan(body: PlanRequest):
    try:
        async with async_session() as session:
            svc = SessionService(session)
            result = await svc.generate_plan(
                body.template,
                user_prompt=body.prompt,
            )
        return result
    except PlanValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/log", response_model=SessionSummary, status_code=201)
async def log_session(body: SessionLogRequest):
    """Save a session result completed on the client side."""
    async with async_session() as db:
        log = SessionLog(
            drill_plan_id=body.drill_plan_id,
            template_name=body.template_name,
            started_at=body.started_at,
            completed_at=body.completed_at,
            total_duration_sec=body.total_duration_sec,
            rounds_completed=body.rounds_completed,
            rounds_total=body.rounds_total,
            combos_delivered=body.combos_delivered,
            delivery_log_json={},
            status=body.status,
        )
        db.add(log)
        await db.commit()
        await db.refresh(log)
        return log
