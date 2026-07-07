"""SessionService — program-based boxing session generation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import DrillPlan, ProgramDayTemplate, ProgramProgress
from atom.services.template_service import TemplateService

# ── Combo call → action sequence lookup (loaded once at import) ──────
_DICT_PATH = Path(__file__).parent.parent / "data" / "combo_dictionary.json"


def _build_call_to_seq() -> dict[str, list[str]]:
    with open(_DICT_PATH) as f:
        data = json.load(f)
    lookup: dict[str, list[str]] = {}
    for level in ("basic", "intermediate", "advanced"):
        for combo in data["combos"][level]:
            lookup[combo["call"]] = combo["seq"]
    return lookup


_CALL_TO_SEQ = _build_call_to_seq()


class SessionService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def generate_plan(
        self,
        level: str = "beginner",
        rounds: int = 3,
        round_duration_sec: int = 120,
        rest_sec: int = 30,
        program_day_id: str | None = None,
    ) -> dict:
        """Generate session plan from program template or fallback to random."""
        template_svc = TemplateService(self.session)

        # Try program-based generation
        day_template = None
        progress = None
        if program_day_id:
            result = await self.session.execute(
                select(ProgramDayTemplate).where(ProgramDayTemplate.id == program_day_id)
            )
            day_template = result.scalar_one_or_none()

        if day_template is None:
            # Auto-detect from current ProgramProgress
            result = await self.session.execute(
                select(ProgramProgress).where(ProgramProgress.completed_at.is_(None))
            )
            progress = result.scalar_one_or_none()
            if progress:
                result = await self.session.execute(
                    select(ProgramDayTemplate).where(
                        ProgramDayTemplate.level == progress.level,
                        ProgramDayTemplate.week == progress.week,
                        ProgramDayTemplate.day_number == progress.current_day,
                    )
                )
                day_template = result.scalar_one_or_none()
                # Fall back to week 1 templates if current week has none
                if day_template is None and progress.week > 1:
                    result = await self.session.execute(
                        select(ProgramDayTemplate).where(
                            ProgramDayTemplate.level == progress.level,
                            ProgramDayTemplate.week == 1,
                            ProgramDayTemplate.day_number == progress.current_day,
                        )
                    )
                    day_template = result.scalar_one_or_none()

        if day_template:
            week_number = progress.week if progress else 1
            return await self._generate_from_program(day_template, rest_sec, template_svc, week_number)

        # Fallback to old random template system
        return await self._generate_random(level, rounds, round_duration_sec, rest_sec, template_svc)

    @staticmethod
    def _pick_coach_comment(template: ProgramDayTemplate, week_number: int) -> str:
        """Select coach comment variant based on week number."""
        if template.coach_comments_json:
            variants = template.coach_comments_json
            return variants[(week_number - 1) % len(variants)]
        return template.coach_comment

    @staticmethod
    def _shuffle_segments(segments: list[dict]) -> list[dict]:
        """Shuffle combo segments (cues are excluded)."""
        combos = [s for s in segments if not s.get("is_cue")]
        random.shuffle(combos)
        return combos

    async def _generate_from_program(
        self,
        day_template: ProgramDayTemplate,
        rest_sec: int,
        template_svc: TemplateService,
        week_number: int = 1,
    ) -> dict:
        """Build session plan from a ProgramDayTemplate with round-specific segments."""
        # Build rounds from predetermined segments
        round_configs = [
            (1, day_template.r1_segments_json),
            (2, day_template.r2_segments_json),
            (3, day_template.r3_segments_json),
        ]

        rounds_list = []
        for round_num, segments_data in round_configs:
            # Shuffle segments within the round for variety
            shuffled = self._shuffle_segments(list(segments_data))

            round_segments = []
            for seg in shuffled:
                text = seg["text"]
                chunks = await template_svc._resolve_chunks(text, {})
                round_segments.append({
                    "text": text,
                    "chunks": chunks,
                    "impact_actions": _CALL_TO_SEQ.get(text, []),
                })

            # Merge finisher segments into R3 (finisher stays in order)
            if round_num == 3:
                finisher_data = day_template.finisher_json
                for seg in finisher_data["segments"]:
                    text = seg["text"]
                    chunks = await template_svc._resolve_chunks(text, {})
                    round_segments.append({
                        "text": text,
                        "chunks": chunks,
                        "impact_actions": _CALL_TO_SEQ.get(text, []),
                    })

            rounds_list.append({"round": round_num, "segments": round_segments})

        plan = {"rounds": rounds_list}

        # Save DrillPlan
        db_plan = DrillPlan(
            template_id=None,
            session_config_json={
                "rounds": 3,
                "round_duration_sec": 120,
                "rest_sec": rest_sec,
                "level": day_template.level,
                "program_day_id": day_template.id,
                "day_number": day_template.day_number,
                "theme": day_template.theme,
            },
            plan_json=plan,
        )
        self.session.add(db_plan)
        await self.session.commit()
        await self.session.refresh(db_plan)

        # Assemble audio per round
        plan = await template_svc.assemble_round_audio(plan, db_plan.id)
        db_plan.plan_json = plan
        await self.session.commit()

        audio_ready = any(
            round_data.get("audio_url")
            for round_data in plan["rounds"]
        )

        return {
            "id": db_plan.id,
            "template_name": f"Day {day_template.day_number}: {day_template.theme}",
            "template_topic": day_template.theme_description,
            "rounds": 3,
            "round_duration_sec": 120,
            "rest_sec": rest_sec,
            "plan": plan,
            "audio_ready": audio_ready,
            "day_number": day_template.day_number,
            "theme": day_template.theme,
            "coach_comment": self._pick_coach_comment(day_template, week_number),
        }

    async def _generate_random(
        self,
        level: str,
        rounds: int,
        round_duration_sec: int,
        rest_sec: int,
        template_svc: TemplateService,
    ) -> dict:
        """Fallback: old random template-based generation."""
        template = await template_svc.pick_template(level)
        plan = await template_svc.build_round_plan(template, rounds, round_duration_sec)

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

        plan = await template_svc.assemble_round_audio(plan, db_plan.id)
        db_plan.plan_json = plan
        await self.session.commit()

        audio_ready = any(
            round_data.get("audio_url")
            for round_data in plan["rounds"]
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
