"""SessionService — template-based boxing session generation."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import DrillPlan
from atom.services.template_service import TemplateService


class SessionService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def generate_plan(
        self,
        level: str = "beginner",
        rounds: int = 3,
        round_duration_sec: int = 180,
        rest_sec: int = 30,
    ) -> dict:
        """Pick a template, shuffle segments, resolve audio chunks."""
        template_svc = TemplateService(self.session)

        # 1. Pick template (with "don't repeat last 3" logic)
        template = await template_svc.pick_template(level)

        # 2. Shuffle segments, resolve audio chunks
        plan = await template_svc.build_round_plan(template, rounds, round_duration_sec)

        # 3. Save to DrillPlan
        db_plan = DrillPlan(
            template_id=template.id,
            session_config_json={
                "rounds": rounds,
                "round_duration_sec": round_duration_sec,
                "rest_sec": rest_sec,
                "level": level,
            },
            plan_json=plan,
        )
        self.session.add(db_plan)
        await self.session.commit()
        await self.session.refresh(db_plan)

        # 4. Check if any audio chunks are available
        audio_ready = any(
            chunk["clip_url"]
            for round_data in plan["rounds"]
            for seg in round_data["segments"]
            for chunk in seg["chunks"]
        )

        return {
            "id": db_plan.id,
            "template_name": template.name,
            "template_topic": template.topic,
            "rounds": rounds,
            "round_duration_sec": round_duration_sec,
            "rest_sec": rest_sec,
            "plan": plan,
            "audio_ready": audio_ready,
        }
