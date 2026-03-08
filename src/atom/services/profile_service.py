"""ProfileService — aggregation of session history into user profile."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import SessionLog, UserProfile


class ProfileService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_profile(self) -> UserProfile | None:
        result = await self.session.execute(select(UserProfile))
        return result.scalar_one_or_none()

    async def update_profile(self, **kwargs: Any) -> UserProfile:
        """Update user-editable fields (experience_level, goal)."""
        profile = await self.get_profile()
        if profile is None:
            profile = UserProfile(**kwargs)
            self.session.add(profile)
        else:
            allowed = {"experience_level", "goal"}
            for k, v in kwargs.items():
                if k in allowed and v is not None:
                    setattr(profile, k, v)
        await self.session.commit()
        await self.session.refresh(profile)
        return profile

    async def aggregate(self) -> UserProfile:
        """Re-compute profile stats from all session logs. Saves and returns updated profile."""
        profile = await self.get_profile()
        if profile is None:
            profile = UserProfile()
            self.session.add(profile)

        logs_result = await self.session.execute(select(SessionLog))
        logs = list(logs_result.scalars().all())

        completed = [l for l in logs if l.status == "completed"]

        profile.total_sessions = len(completed)
        profile.total_training_minutes = sum(l.total_duration_sec for l in completed) / 60.0

        if completed:
            profile.last_session_at = max(l.started_at for l in completed)

        # Combo exposure
        exposure: dict[str, int] = {}
        for log in completed:
            events = (log.delivery_log_json or {}).get("events", [])
            for ev in events:
                if ev.get("type") == "combo_called":
                    name = ev.get("combo_display_name", "")
                    if name:
                        exposure[name] = exposure.get(name, 0) + 1
        profile.combo_exposure_json = exposure

        # Template preference
        template_pref: dict[str, int] = {}
        for log in completed:
            t = log.template_name
            template_pref[t] = template_pref.get(t, 0) + 1
        profile.template_preference_json = template_pref

        # Session frequency — rolling 4-week window
        # Strip tz since SQLite returns naive datetimes
        four_weeks_ago = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(weeks=4)
        recent = [
            l for l in completed
            if l.started_at and l.started_at.replace(tzinfo=None) >= four_weeks_ago
        ]
        profile.session_frequency = len(recent) / 4.0  # avg per week

        await self.session.commit()
        await self.session.refresh(profile)
        return profile

    # ── Session history ───────────────────────────────────────────────

    async def list_sessions(self, limit: int = 20, offset: int = 0) -> list[SessionLog]:
        result = await self.session.execute(
            select(SessionLog)
            .order_by(SessionLog.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_session(self, session_id: str) -> SessionLog | None:
        result = await self.session.execute(
            select(SessionLog).where(SessionLog.id == session_id)
        )
        return result.scalar_one_or_none()
