"""Today API — single endpoint for home screen data."""

from __future__ import annotations

from fastapi import APIRouter

from atom.api.schemas import TodayResponse
from atom.models.base import async_session
from atom.services.profile_service import ProfileService

router = APIRouter(tags=["today"])


@router.get("/api/today", response_model=TodayResponse)
async def get_today():
    """Returns everything the home screen needs in one call."""
    async with async_session() as session:
        svc = ProfileService(session)
        return await svc.get_today_data()
