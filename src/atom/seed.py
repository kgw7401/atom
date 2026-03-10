"""Seed data for Atom — actions, combinations, and session templates.

Idempotent: safe to run multiple times (skips existing records by unique key).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import Action, Combination, SessionTemplate, UserProfile

ACTIONS = [
    {"name": "jab", "display_name": "잽", "category": "offense", "sort_order": 1},
    {"name": "cross", "display_name": "크로스", "category": "offense", "sort_order": 2},
    {"name": "lead_hook", "display_name": "리드훅", "category": "offense", "sort_order": 3},
    {"name": "rear_hook", "display_name": "리어훅", "category": "offense", "sort_order": 4},
    {"name": "lead_uppercut", "display_name": "리드어퍼컷", "category": "offense", "sort_order": 5},
    {"name": "rear_uppercut", "display_name": "리어어퍼컷", "category": "offense", "sort_order": 6},
    {"name": "lead_bodyshot", "display_name": "리드바디", "category": "offense", "sort_order": 7},
    {"name": "rear_bodyshot", "display_name": "리어바디", "category": "offense", "sort_order": 8},
    {"name": "slip", "display_name": "슬립", "category": "defense", "sort_order": 9},
    {"name": "duck", "display_name": "덕킹", "category": "defense", "sort_order": 10},
    {"name": "backstep", "display_name": "백스텝", "category": "movement", "sort_order": 11},
]

COMBINATIONS = [
    {"display_name": "잽", "actions": ["jab"]},
    {"display_name": "크로스", "actions": ["cross"]},
    {"display_name": "원투", "actions": ["jab", "cross"]},
    {"display_name": "더블잽", "actions": ["jab", "jab"]},
    {"display_name": "리드훅바디", "actions": ["lead_hook", "rear_bodyshot"]},
    {"display_name": "원투훅", "actions": ["jab", "cross", "lead_hook"]},
    {"display_name": "원투바디", "actions": ["jab", "cross", "rear_bodyshot"]},
    {"display_name": "잽잽크로스", "actions": ["jab", "jab", "cross"]},
    {"display_name": "슬립원투", "actions": ["slip", "jab", "cross"]},
    {"display_name": "덕킹원투", "actions": ["duck", "jab", "cross"]},
    {"display_name": "원투쓰리투", "actions": ["jab", "cross", "lead_hook", "cross"]},
    {"display_name": "원투바디훅", "actions": ["jab", "cross", "rear_bodyshot", "lead_hook"]},
]

TEMPLATES = [
    {
        "name": "fundamentals",
        "display_name": "기본기",
        "description": "Singles & doubles, slow pace, form focus",
        "default_rounds": 3,
        "default_round_duration_sec": 120,
        "default_rest_sec": 45,
        "combo_complexity_range": [1, 2],
        "combo_include_defense": False,
        "pace_interval_sec": [1, 3],
    },
    {
        "name": "combos",
        "display_name": "콤비네이션",
        "description": "Multi-action sequences, medium pace",
        "default_rounds": 4,
        "default_round_duration_sec": 150,
        "default_rest_sec": 60,
        "combo_complexity_range": [3, 4],
        "combo_include_defense": False,
        "pace_interval_sec": [1, 3],
    },
    {
        "name": "mixed",
        "display_name": "종합",
        "description": "Offense + defense, varied complexity, high volume",
        "default_rounds": 5,
        "default_round_duration_sec": 180,
        "default_rest_sec": 60,
        "combo_complexity_range": [1, 4],
        "combo_include_defense": True,
        "pace_interval_sec": [1, 2],
    },
]


async def seed_all(session: AsyncSession) -> dict[str, int]:
    """Seed all reference data. Returns counts of inserted records."""
    counts = {
        "actions": await _seed_actions(session),
        "combinations": await _seed_combinations(session),
        "templates": await _seed_templates(session),
        "profile": await _seed_profile(session),
    }
    await session.commit()
    return counts


async def _seed_actions(session: AsyncSession) -> int:
    """Seed actions, skip existing by name."""
    result = await session.execute(select(Action.name))
    existing = {row[0] for row in result.all()}

    inserted = 0
    for data in ACTIONS:
        if data["name"] not in existing:
            session.add(Action(**data))
            inserted += 1
    return inserted


async def _seed_combinations(session: AsyncSession) -> int:
    """Seed default combos, skip existing by display_name."""
    result = await session.execute(select(Combination.display_name))
    existing = {row[0] for row in result.all()}

    inserted = 0
    for data in COMBINATIONS:
        if data["display_name"] not in existing:
            combo = Combination(
                display_name=data["display_name"],
                actions=data["actions"],
                complexity=len(data["actions"]),
                is_system=True,
            )
            session.add(combo)
            inserted += 1
    return inserted


async def _seed_templates(session: AsyncSession) -> int:
    """Seed session templates, skip existing by name."""
    result = await session.execute(select(SessionTemplate.name))
    existing = {row[0] for row in result.all()}

    inserted = 0
    for data in TEMPLATES:
        if data["name"] not in existing:
            session.add(SessionTemplate(**data))
            inserted += 1
    return inserted


async def _seed_profile(session: AsyncSession) -> int:
    """Ensure a default user profile exists."""
    result = await session.execute(select(UserProfile))
    if result.first() is not None:
        return 0

    session.add(UserProfile(experience_level="beginner", goal=""))
    return 1
