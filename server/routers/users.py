"""User endpoints: create, get state, get history."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from server.models.db import StateTransition, UserState, async_session
import numpy as np

from server.schemas import (
    CreateUserRequest,
    HistoryResponse,
    ProgressReportResponse,
    StateResponse,
    TransitionItem,
    UserResponse,
)
from src.state.constants import DIM_NAMES

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserResponse, status_code=201)
async def create_user(req: CreateUserRequest):
    """Create user or return existing by device_id."""
    async with async_session() as db:
        result = await db.execute(
            select(UserState).where(UserState.device_id == req.device_id)
        )
        existing = result.scalar_one_or_none()
        if existing:
            return UserResponse(
                user_id=existing.user_id,
                device_id=existing.device_id,
                created_at=existing.created_at,
            )

        # Create new user with neutral state (no sessions yet)
        import numpy as np
        from src.state.types import StateVector

        initial = StateVector.zeros()
        user = UserState(
            device_id=req.device_id,
            vector_json=json.dumps(initial.values.tolist()),
            confidence_json=json.dumps(initial.confidence.tolist()),
            obs_counts_json=json.dumps(initial.obs_counts.astype(int).tolist()),
            row_version=0,
            schema_version=initial.schema_version,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

        return UserResponse(
            user_id=user.user_id,
            device_id=user.device_id,
            created_at=user.created_at,
        )


@router.get("/{user_id}/state", response_model=StateResponse)
async def get_state(user_id: str):
    """Get current state vector for a user."""
    async with async_session() as db:
        user = await db.get(UserState, user_id)
        if user is None:
            raise HTTPException(404, "User not found")

        return StateResponse(
            user_id=user.user_id,
            values=json.loads(user.vector_json),
            confidence=json.loads(user.confidence_json),
            obs_counts=json.loads(user.obs_counts_json),
            version=user.row_version,
            schema_version=user.schema_version,
        )


@router.get("/{user_id}/history", response_model=HistoryResponse)
async def get_history(user_id: str, limit: int = 20):
    """Get state transition history for a user."""
    async with async_session() as db:
        user = await db.get(UserState, user_id)
        if user is None:
            raise HTTPException(404, "User not found")

        result = await db.execute(
            select(StateTransition)
            .where(StateTransition.user_id == user_id)
            .order_by(StateTransition.created_at.desc())
            .limit(limit)
        )
        transitions = result.scalars().all()

        items = []
        for t in reversed(transitions):  # chronological order
            delta = json.loads(t.delta_json)
            mask = json.loads(t.observation_mask_json)
            observed = [DIM_NAMES[i] for i, m in enumerate(mask) if m]
            items.append(
                TransitionItem(
                    session_id=t.session_id,
                    version_before=t.version_before,
                    version_after=t.version_after,
                    delta=delta,
                    observed_dims=observed,
                    created_at=t.created_at,
                )
            )

        return HistoryResponse(user_id=user_id, transitions=items)


@router.get("/{user_id}/progress", response_model=ProgressReportResponse)
async def get_progress(user_id: str, sessions: int = 5):
    """Get multi-session progress report with trend analysis."""
    async with async_session() as db:
        user = await db.get(UserState, user_id)
        if user is None:
            raise HTTPException(404, "User not found")

        result = await db.execute(
            select(StateTransition)
            .where(StateTransition.user_id == user_id)
            .order_by(StateTransition.created_at.desc())
            .limit(sessions)
        )
        transitions = list(reversed(result.scalars().all()))  # chronological

        from server.services.report_generator import generate_progress_report
        from src.state.types import StateVector

        current_state = StateVector.from_json({
            "values": json.loads(user.vector_json),
            "confidence": json.loads(user.confidence_json),
            "obs_counts": json.loads(user.obs_counts_json),
            "version": user.row_version,
            "schema_version": user.schema_version,
        })

        deltas = [np.array(json.loads(t.delta_json), dtype=np.float64) for t in transitions]
        masks = [np.array(json.loads(t.observation_mask_json), dtype=bool) for t in transitions]

        report = generate_progress_report(
            user_id=user_id,
            deltas=deltas,
            masks=masks,
            current_state=current_state,
        )

        return ProgressReportResponse(
            user_id=user_id,
            num_sessions=report.num_sessions,
            trending_up=report.trending_up,
            trending_down=report.trending_down,
            plateau=report.plateau,
            coaching=report.coaching,
        )
