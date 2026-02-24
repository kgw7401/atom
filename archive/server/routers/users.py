"""User and digital twin endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select

from server.models.db import ActionStat, Session, User, async_session
from server.models.schemas import (
    ActionStat as ActionStatSchema,
    TwinResponse,
    UserCreateRequest,
    UserCreateResponse,
    Weakness,
)

router = APIRouter(tags=["users"])


@router.post("/users", response_model=UserCreateResponse)
async def create_user(req: UserCreateRequest) -> UserCreateResponse:
    async with async_session() as db:
        # Check if device already registered
        stmt = select(User).where(User.device_id == req.device_id)
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            return UserCreateResponse(user_id=uuid.UUID(existing.id))

        user = User(device_id=req.device_id)
        db.add(user)
        await db.commit()
        await db.refresh(user)

    return UserCreateResponse(user_id=uuid.UUID(user.id))


@router.get("/users/{user_id}/twin", response_model=TwinResponse)
async def get_twin(user_id: str) -> TwinResponse:
    async with async_session() as db:
        user = await db.get(User, user_id)
        if not user:
            raise HTTPException(404, "User not found")

        # Count sessions
        stmt = select(func.count()).select_from(Session).where(
            Session.user_id == user_id,
            Session.status == "completed",
        )
        result = await db.execute(stmt)
        total_sessions = result.scalar() or 0

        # Aggregate action stats
        stmt = select(ActionStat).where(ActionStat.user_id == user_id).order_by(
            ActionStat.recorded_at,
        )
        result = await db.execute(stmt)
        stats = result.scalars().all()

    # Build per-action aggregates
    action_agg: dict[str, dict] = {}
    for s in stats:
        if s.action not in action_agg:
            action_agg[s.action] = {
                "attempts": 0, "successes": 0,
                "reactions": [], "scores": [],
            }
        agg = action_agg[s.action]
        agg["attempts"] += s.attempts
        agg["successes"] += s.successes
        if s.avg_reaction_time is not None:
            agg["reactions"].append(s.avg_reaction_time)
        if s.avg_score is not None:
            agg["scores"].append(s.avg_score)

    per_action_stats = {}
    weaknesses = []

    for action, agg in action_agg.items():
        acc = agg["successes"] / agg["attempts"] if agg["attempts"] > 0 else 0
        reactions = agg["reactions"]
        avg_r = sum(reactions) / len(reactions) if reactions else 0
        scores = agg["scores"]

        # Trend: compare last 3 scores to first 3
        trend = "stable"
        if len(scores) >= 6:
            early = sum(scores[:3]) / 3
            recent = sum(scores[-3:]) / 3
            if recent > early + 5:
                trend = "improving"
            elif recent < early - 5:
                trend = "declining"

        per_action_stats[action] = ActionStatSchema(
            accuracy=round(acc, 2),
            avg_reaction=round(avg_r, 2),
            trend=trend,
            total_attempts=agg["attempts"],
        )

        # Detect weaknesses
        if acc < 0.7:
            weaknesses.append(Weakness(
                action=action, metric="accuracy",
                value=round(acc, 2), threshold=0.7, severity="warning",
            ))
        if avg_r > 0.8:
            weaknesses.append(Weakness(
                action=action, metric="reaction_time",
                value=round(avg_r, 2), threshold=0.8, severity="warning",
            ))

    # Growth curves (weekly avg scores)
    weekly_scores: list[float] = []
    if stats:
        scores_by_order = [s.avg_score for s in stats if s.avg_score is not None]
        # Simple: split into chunks of ~N for curve (not real weekly yet)
        chunk_size = max(1, len(scores_by_order) // 10)
        for i in range(0, len(scores_by_order), chunk_size):
            chunk = scores_by_order[i:i + chunk_size]
            weekly_scores.append(round(sum(chunk) / len(chunk), 1))

    return TwinResponse(
        total_sessions=total_sessions,
        per_action_stats=per_action_stats,
        weaknesses=weaknesses,
        growth_curves={"scores": weekly_scores},
    )
