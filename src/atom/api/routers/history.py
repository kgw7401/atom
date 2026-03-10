"""Session history and user profile API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from atom.api.schemas import (
    ProfileResponse,
    ProfileUpdate,
    SessionDetail,
    SessionSummary,
    TemplateResponse,
)
from atom.models.base import async_session
from atom.models.tables import SessionTemplate
from atom.services.profile_service import ProfileService

router = APIRouter(tags=["history"])


# ── Templates ─────────────────────────────────────────────────────────

@router.get("/api/templates", response_model=list[TemplateResponse])
async def list_templates():
    async with async_session() as session:
        result = await session.execute(select(SessionTemplate))
        return list(result.scalars().all())


# ── Session history ───────────────────────────────────────────────────

@router.get("/api/sessions", response_model=list[SessionSummary])
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    async with async_session() as session:
        svc = ProfileService(session)
        return await svc.list_sessions(limit=limit, offset=offset)


@router.get("/api/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    async with async_session() as session:
        svc = ProfileService(session)
        log = await svc.get_session(session_id)
        if log is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return log


# ── Profile ───────────────────────────────────────────────────────────

@router.get("/api/profile", response_model=ProfileResponse)
async def get_profile():
    async with async_session() as session:
        svc = ProfileService(session)
        profile = await svc.get_profile()
        if profile is None:
            raise HTTPException(status_code=404, detail="No profile found. Run `atom init` first.")
        return profile


@router.put("/api/profile", response_model=ProfileResponse)
async def update_profile(body: ProfileUpdate):
    kwargs = body.model_dump(exclude_none=True)
    if not kwargs:
        raise HTTPException(status_code=400, detail="No fields to update")
    async with async_session() as session:
        svc = ProfileService(session)
        return await svc.update_profile(**kwargs)
