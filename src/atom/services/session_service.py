"""SessionService — LLM-powered drill plan generation with fallback."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import Action, Combination, SessionLog, SessionTemplate, UserProfile


# ── System prompt for LLM ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert boxing coach running a shadowboxing drill app called Atom.

## About This App
Atom is a solo shadowboxing training tool. The user trains alone — there is no
opponent. You call combos aloud via text-to-speech, and the user executes them
in shadow form. Your job is to design drill sessions that are safe, progressive,
and effective for solo practice.

## Boxing Action Reference

Offense:
- jab           Quick lead-hand punch. Fast, low power. Sets up combinations.
- cross         Rear-hand straight punch. The primary power shot.
- lead_hook     Lead-hand hook to the head. Sharp, short. Best after jab or cross.
- rear_hook     Rear-hand hook to the head. Powerful finishing punch.
- lead_uppercut Lead-hand upward punch. Close-range. Good after slipping.
- rear_uppercut Rear-hand upward punch. Maximum power. Finishing shot.
- lead_bodyshot Lead-hand hook to the body. Tires opponent, opens the head.
- rear_bodyshot Rear-hand punch to the body. High-damage power shot.

Defense / Movement:
- slip          Head movement to avoid jab or cross. Moves offline.
- duck          Lower body to avoid hooks. Creates counter opportunities.
- backstep      Step backward to reset distance and position.

## Coaching Philosophy
1. Progressive overload: Start with simple 1-2 action combos in round 1.
   Increase complexity each round. Never start with the hardest combos.
2. Muscle memory: Repeat foundational combos (jab, cross, jab-cross) across
   rounds. Familiarity builds speed and automaticity.
3. Weak spot targeting: If the user rarely drills certain combos, include them
   deliberately — variety is essential for well-rounded development.
4. Balance: Mix pure offense with defense-counter sequences when the template
   allows it.
5. Accessible warm-up: Early rounds should feel comfortable. Save the hardest
   combos for the final round.

## Output Format
Output must be valid JSON matching this exact structure:
{
  "session_type": "drill",
  "template": "<template_name>",
  "focus": "<short description of session focus in Korean>",
  "total_duration_minutes": <int>,
  "rounds": [
    {
      "round_number": <int>,
      "duration_seconds": <int>,
      "rest_after_seconds": <int>,
      "instructions": [
        {
          "timestamp_offset": <float, seconds from round start>,
          "combo_display_name": "<Korean combo name>",
          "actions": ["action_name1", "action_name2"]
        }
      ]
    }
  ]
}

Rules:
- Reuse combos from "Existing Combos" OR create new ones from "Available Actions".
- New combos: use a short descriptive Korean name; actions must be from Available
  Actions; complexity must be within the template's complexity range.
- Space instructions according to the pace interval.
- Fill rounds from start to near the round duration.
- Respond with JSON only. No markdown, no explanation.
"""


class PlanValidationError(Exception):
    pass


class SessionService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def generate_plan(
        self,
        template_name: str,
        user_prompt: str | None = None,
        llm_client: Any | None = None,
    ) -> dict:
        """Generate a drill plan using LLM or fallback."""
        template = await self._get_template(template_name)
        combos = await self._get_eligible_combos(template)
        all_actions = await self._get_all_actions()
        profile = await self._get_profile()
        recent_sessions = await self._get_recent_sessions(limit=5)

        plan = None
        llm_model = "fallback"

        if llm_client is not None:
            try:
                prompt = self._build_prompt(
                    template, combos, all_actions, profile, recent_sessions, user_prompt
                )
                raw_plan = await llm_client.generate_json(
                    system=SYSTEM_PROMPT,
                    user=prompt,
                )
                new_combos = self._validate_plan(raw_plan, combos, all_actions, template)
                plan = raw_plan
                llm_model = llm_client.model

                # Save LLM-created combos to DB
                for combo_data in new_combos:
                    combo = Combination(
                        display_name=combo_data["display_name"],
                        actions=combo_data["actions"],
                        complexity=len(combo_data["actions"]),
                        is_system=False,
                    )
                    self.session.add(combo)
                if new_combos:
                    await self.session.flush()
                    print(f"Saved {len(new_combos)} new combo(s): "
                          + ", ".join(c["display_name"] for c in new_combos))

            except Exception as e:
                print(f"LLM plan generation failed: {e}, using fallback")

        if plan is None:
            if not combos:
                raise PlanValidationError(
                    f"No combos match template '{template_name}' constraints"
                )
            plan = self._fallback_plan(template, combos)

        # Save plan to DB
        from atom.models.tables import DrillPlan

        db_plan = DrillPlan(
            template_id=template.id,
            user_prompt=user_prompt,
            llm_model=llm_model,
            plan_json=plan,
        )
        self.session.add(db_plan)
        await self.session.commit()
        await self.session.refresh(db_plan)

        return {"id": db_plan.id, "plan": plan, "llm_model": llm_model}

    # ── Prompt building ──────────────────────────────────────────────────

    def _build_prompt(
        self,
        template: SessionTemplate,
        combos: list[Combination],
        all_actions: list[Action],
        profile: UserProfile | None,
        recent_sessions: list[SessionLog],
        user_prompt: str | None,
    ) -> str:
        lines = []

        # ── Template constraints ──
        lines.append(f"# Template: {template.name} ({template.display_name})")
        lines.append(f"Description: {template.description}")
        lines.append(f"Rounds: {template.default_rounds}")
        lines.append(f"Round duration: {template.default_round_duration_sec}s")
        lines.append(f"Rest between rounds: {template.default_rest_sec}s")
        cmin, cmax = template.combo_complexity_range
        lines.append(f"Combo complexity: {cmin}-{cmax} actions per combo")
        pmin, pmax = template.pace_interval_sec
        lines.append(f"Pace: {pmin}-{pmax} seconds between combo calls (base interval)")
        lines.append("")

        # ── Available actions ──
        lines.append("# Available Actions")
        lines.append("(Use these exact names in the 'actions' arrays)")
        for a in all_actions:
            lines.append(f"- {a.name} ({a.display_name}) [{a.category}]")
        lines.append("")

        # ── Existing combos ──
        if combos:
            lines.append("# Existing Combos (reuse when appropriate)")
            for c in combos:
                actions_str = ", ".join(c.actions)
                lines.append(f"- {c.display_name}: [{actions_str}]  (complexity: {c.complexity})")
            lines.append("")

        # ── User profile ──
        if profile:
            lines.append("# User Profile")
            lines.append(f"Experience level: {profile.experience_level}")
            lines.append(f"Training goal: {profile.goal or '(not set)'}")
            lines.append(f"Total sessions completed: {profile.total_sessions}")
            lines.append(f"Training frequency: {profile.session_frequency:.1f} sessions/week")
            if profile.last_session_at:
                lines.append(f"Last session: {profile.last_session_at.strftime('%Y-%m-%d')}")
            lines.append("")

        # ── Recent session history ──
        if recent_sessions:
            lines.append("# Recent Session History")
            lines.append("(Use this to vary content and avoid repetition)")
            for log in recent_sessions:
                date_str = log.started_at.strftime("%Y-%m-%d")
                lines.append(
                    f"- {date_str}: {log.template_name}, "
                    f"{log.combos_delivered} combos, {log.status}"
                )
            lines.append("")

        # ── Combo exposure ──
        exposure = (profile.combo_exposure_json or {}) if profile else {}
        if exposure:
            sorted_exp = sorted(exposure.items(), key=lambda x: x[1], reverse=True)
            lines.append("# Combo Exposure (training history)")

            top = sorted_exp[:5]
            if top:
                lines.append("Most practiced (consider varying these):")
                for name, count in top:
                    lines.append(f"  - {name}: {count}x")

            bottom = [x for x in sorted_exp if x[1] <= 3]
            if bottom:
                lines.append("Under-practiced (prioritize these):")
                for name, count in bottom[:5]:
                    lines.append(f"  - {name}: {count}x")
            lines.append("")

        # ── User request ──
        if user_prompt:
            lines.append(f'# User Request (highest priority — follow this closely)')
            lines.append(f'"{user_prompt}"')
            lines.append("")

        lines.append(
            "Generate a drill session plan tailored to this user. "
            "Reuse Existing Combos or create new ones from Available Actions. "
            "Follow the coaching philosophy: progressive rounds, target weak spots, "
            "honor the user's request if provided."
        )
        return "\n".join(lines)

    # ── Validation ───────────────────────────────────────────────────────

    def _validate_plan(
        self,
        plan: dict,
        combos: list[Combination],
        all_actions: list[Action],
        template: SessionTemplate,
    ) -> list[dict]:
        """Validate LLM-generated plan. Returns list of new combo dicts to save."""
        required = ["session_type", "rounds"]
        for key in required:
            if key not in plan:
                raise PlanValidationError(f"Missing required key: '{key}'")

        if not isinstance(plan["rounds"], list) or len(plan["rounds"]) == 0:
            raise PlanValidationError("Rounds must be a non-empty list")

        existing_lookup = {c.display_name: c for c in combos}
        valid_action_names = {a.name for a in all_actions}
        cmin, cmax = template.combo_complexity_range

        new_combos: list[dict] = []
        seen_new_names: set[str] = set()

        for rnd in plan["rounds"]:
            if "instructions" not in rnd:
                raise PlanValidationError("Round missing 'instructions'")

            for instr in rnd["instructions"]:
                name = instr.get("combo_display_name", "")
                actions = instr.get("actions", [])

                if name in existing_lookup:
                    # Existing combo — trust DB actions
                    instr["actions"] = existing_lookup[name].actions
                else:
                    # New combo — validate actions
                    if not actions:
                        raise PlanValidationError(f"New combo '{name}' has no actions")
                    invalid = [a for a in actions if a not in valid_action_names]
                    if invalid:
                        raise PlanValidationError(
                            f"Unknown actions in combo '{name}': {invalid}. "
                            f"Valid: {sorted(valid_action_names)}"
                        )
                    if not (cmin <= len(actions) <= cmax):
                        raise PlanValidationError(
                            f"Combo '{name}' has {len(actions)} actions, "
                            f"template requires {cmin}-{cmax}"
                        )
                    if name not in seen_new_names:
                        new_combos.append({"display_name": name, "actions": actions})
                        seen_new_names.add(name)

        return new_combos

    # ── Fallback plan ────────────────────────────────────────────────────

    def _fallback_plan(
        self,
        template: SessionTemplate,
        combos: list[Combination],
    ) -> dict:
        """Generate a plan without LLM."""
        rounds = []
        pmin, pmax = template.pace_interval_sec
        sorted_combos = sorted(combos, key=lambda c: c.complexity)

        for r in range(1, template.default_rounds + 1):
            instructions = []
            t = 5.0

            if len(sorted_combos) > 2:
                fraction = (r - 1) / max(1, template.default_rounds - 1)
                pool_end = max(2, int(len(sorted_combos) * (0.5 + 0.5 * fraction)))
                round_pool = sorted_combos[:pool_end]
            else:
                round_pool = sorted_combos

            while t < template.default_round_duration_sec - 3:
                combo = random.choice(round_pool)
                instructions.append({
                    "timestamp_offset": round(t, 1),
                    "combo_display_name": combo.display_name,
                    "actions": combo.actions,
                })
                t += random.uniform(pmin, pmax)

            rounds.append({
                "round_number": r,
                "duration_seconds": template.default_round_duration_sec,
                "rest_after_seconds": template.default_rest_sec,
                "instructions": instructions,
            })

        total_sec = sum(r["duration_seconds"] + r["rest_after_seconds"] for r in rounds)

        return {
            "session_type": "drill",
            "template": template.name,
            "focus": template.display_name,
            "total_duration_minutes": round(total_sec / 60),
            "pace_interval_sec": list(template.pace_interval_sec),
            "rounds": rounds,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _get_template(self, name: str) -> SessionTemplate:
        result = await self.session.execute(
            select(SessionTemplate).where(SessionTemplate.name == name)
        )
        template = result.scalar_one_or_none()
        if template is None:
            raise PlanValidationError(f"Template not found: '{name}'")
        return template

    async def _get_all_actions(self) -> list[Action]:
        result = await self.session.execute(
            select(Action).order_by(Action.sort_order)
        )
        return list(result.scalars().all())

    async def _get_eligible_combos(self, template: SessionTemplate) -> list[Combination]:
        """Get combos matching template's complexity range and defense filter."""
        cmin, cmax = template.combo_complexity_range
        stmt = select(Combination).where(
            Combination.complexity >= cmin,
            Combination.complexity <= cmax,
        )

        if not template.combo_include_defense:
            result = await self.session.execute(stmt)
            all_combos = list(result.scalars().all())

            defense_result = await self.session.execute(
                select(Action.name).where(Action.category.in_(["defense", "movement"]))
            )
            defense_actions = {row[0] for row in defense_result.all()}

            return [
                c for c in all_combos
                if not any(a in defense_actions for a in c.actions)
            ]

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def _get_recent_sessions(self, limit: int = 5) -> list[SessionLog]:
        """Get most recent session logs."""
        result = await self.session.execute(
            select(SessionLog)
            .order_by(desc(SessionLog.started_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def _get_profile(self) -> UserProfile | None:
        result = await self.session.execute(select(UserProfile))
        return result.scalar_one_or_none()
