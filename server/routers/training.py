"""Training endpoints: generate plan, get plan."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException

from server.models.db import UserState, async_session
from server.schemas import GeneratePlanRequest, PlanResponse
from src.policy.session_planner import plan_session
from src.state.types import StateVector

router = APIRouter(prefix="/training", tags=["training"])

# In-memory plan cache (MVP — replace with DB storage later)
_plan_cache: dict[str, dict] = {}


@router.post("/generate", response_model=PlanResponse, status_code=201)
async def generate_plan(req: GeneratePlanRequest):
    """Generate a training plan based on current state."""
    async with async_session() as db:
        user = await db.get(UserState, req.user_id)
        if user is None:
            raise HTTPException(404, "User not found")

        state = StateVector.from_json({
            "values": json.loads(user.vector_json),
            "confidence": json.loads(user.confidence_json),
            "obs_counts": json.loads(user.obs_counts_json),
            "version": user.row_version,
            "schema_version": user.schema_version,
        })

    plan = plan_session(state)
    plan_id = str(uuid.uuid4())
    plan_dict = plan.to_dict()
    _plan_cache[plan_id] = plan_dict

    return PlanResponse(plan_id=plan_id, plan=plan_dict)


@router.get("/plans/{plan_id}", response_model=PlanResponse)
async def get_plan(plan_id: str):
    """Get a previously generated plan."""
    if plan_id not in _plan_cache:
        raise HTTPException(404, "Plan not found")
    return PlanResponse(plan_id=plan_id, plan=_plan_cache[plan_id])
