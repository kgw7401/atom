"""Seed data for Atom — creates default user profile and templates on first run."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import SessionTemplate, UserProfile


async def seed_all(session: AsyncSession) -> dict[str, int]:
    """Seed reference data. Returns counts of inserted records."""
    counts = {
        "profile": await _seed_profile(session),
        "templates": await _seed_templates(session),
    }
    await session.commit()
    return counts


async def _seed_profile(session: AsyncSession) -> int:
    """Ensure a default user profile exists."""
    result = await session.execute(select(UserProfile))
    profile = result.scalar_one_or_none()
    if profile is None:
        session.add(UserProfile(experience_level="beginner", goal=""))
        return 1
    return 0


async def _seed_templates(session: AsyncSession) -> int:
    """Seed session templates if none exist."""
    result = await session.execute(select(func.count()).select_from(SessionTemplate))
    existing = result.scalar() or 0
    if existing > 0:
        return 0

    from atom.seed_templates import TEMPLATES

    count = 0
    for t in TEMPLATES:
        session.add(
            SessionTemplate(
                name=t["name"],
                level=t["level"],
                topic=t["topic"],
                segments_json=t["segments_json"],
            )
        )
        count += 1
    return count
