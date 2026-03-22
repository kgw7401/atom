"""ProfileService — aggregation of session history into user profile."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import ProgramDayTemplate, ProgramProgress, SessionLog, UserProfile


class ProfileService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_profile(self) -> UserProfile | None:
        result = await self.session.execute(select(UserProfile))
        return result.scalar_one_or_none()

    async def update_profile(self, **kwargs: Any) -> UserProfile:
        profile = await self.get_profile()
        if profile is None:
            profile = UserProfile(**kwargs)
            self.session.add(profile)
        else:
            allowed = {"experience_level", "goal", "training_preference"}
            for k, v in kwargs.items():
                if k in allowed and v is not None:
                    setattr(profile, k, v)
        await self.session.commit()
        await self.session.refresh(profile)
        return profile

    async def aggregate(self) -> UserProfile:
        """Re-compute profile stats from all session logs."""
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

        await self.session.commit()
        await self.session.refresh(profile)
        return profile

    async def update_streak(self) -> UserProfile:
        """Update streak after a completed session.

        Rules:
        - Complete a session today → streak +1
        - Already trained today → no change
        - Missed 1 day → streak preserved (grace period)
        - Missed 2+ days → streak resets to 1
        """
        profile = await self.get_profile()
        if profile is None:
            profile = UserProfile()
            self.session.add(profile)

        today_str = date.today().isoformat()

        if profile.last_training_date == today_str:
            # Already trained today
            return profile

        if profile.last_training_date:
            last_date = date.fromisoformat(profile.last_training_date)
            days_gap = (date.today() - last_date).days
            if days_gap <= 2:
                # Consecutive or 1-day grace period
                profile.current_streak += 1
            else:
                # Missed 2+ days → reset
                profile.current_streak = 1
        else:
            # First ever session
            profile.current_streak = 1

        profile.last_training_date = today_str
        if profile.current_streak > profile.longest_streak:
            profile.longest_streak = profile.current_streak

        await self.session.commit()
        await self.session.refresh(profile)
        return profile

    async def advance_program(self) -> ProgramProgress | None:
        """Advance program progress to next day after session completion."""
        result = await self.session.execute(
            select(ProgramProgress).where(ProgramProgress.completed_at.is_(None))
        )
        progress = result.scalar_one_or_none()
        if progress is None:
            return None

        if progress.current_day >= 7:
            # Week complete — mark completed, create next week
            from datetime import datetime, timezone
            progress.completed_at = datetime.now(timezone.utc)
            # For now, stay on same level, increment week
            new_progress = ProgramProgress(
                level=progress.level,
                week=progress.week + 1,
                current_day=1,
            )
            self.session.add(new_progress)
        else:
            progress.current_day += 1

        await self.session.commit()
        return progress

    async def get_today_data(self) -> dict:
        """Get all data needed for the home screen."""
        profile = await self.get_profile()
        streak = profile.current_streak if profile else 0
        level = profile.experience_level if profile else "beginner"

        # Get current program progress
        result = await self.session.execute(
            select(ProgramProgress).where(ProgramProgress.completed_at.is_(None))
        )
        progress = result.scalar_one_or_none()
        if progress is None:
            # Create default progress
            progress = ProgramProgress(level=level, week=1, current_day=1)
            self.session.add(progress)
            await self.session.commit()
            await self.session.refresh(progress)

        # Get today's template
        result = await self.session.execute(
            select(ProgramDayTemplate).where(
                ProgramDayTemplate.level == progress.level,
                ProgramDayTemplate.week == progress.week,
                ProgramDayTemplate.day_number == progress.current_day,
            )
        )
        template = result.scalar_one_or_none()

        theme = template.theme if template else "훈련"
        theme_desc = template.theme_description if template else ""
        coach_comment = template.coach_comment if template else "오늘도 화이팅!"

        # Next day preview
        next_day = progress.current_day + 1 if progress.current_day < 7 else None
        next_preview = None
        if next_day:
            result = await self.session.execute(
                select(ProgramDayTemplate).where(
                    ProgramDayTemplate.level == progress.level,
                    ProgramDayTemplate.week == progress.week,
                    ProgramDayTemplate.day_number == next_day,
                )
            )
            next_template = result.scalar_one_or_none()
            if next_template:
                next_preview = {"day_number": next_day, "theme": next_template.theme}

        return {
            "streak": streak,
            "day_number": progress.current_day,
            "day_total": 7,
            "theme": theme,
            "theme_description": theme_desc,
            "coach_comment": coach_comment,
            "level": progress.level,
            "week": progress.week,
            "next_day_preview": next_preview,
        }

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
