"""Session plan generation API endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from atom.api.schemas import PlanRequest, PlanResponse
from atom.models.base import async_session
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
