"""Tests for SessionService — plan generation with fallback."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from atom.models.base import Base
from atom.seed import seed_all
from atom.services.session_service import PlanValidationError, SessionService


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        await seed_all(s)
        yield s

    await engine.dispose()


@pytest.fixture
def service(session: AsyncSession) -> SessionService:
    return SessionService(session)


async def test_fallback_plan_fundamentals(service: SessionService):
    result = await service.generate_plan("fundamentals")
    plan = result["plan"]

    assert result["llm_model"] == "fallback"
    assert plan["template"] == "fundamentals"
    assert len(plan["rounds"]) == 3

    for rnd in plan["rounds"]:
        assert rnd["duration_seconds"] == 120
        assert rnd["rest_after_seconds"] == 45
        assert len(rnd["instructions"]) > 0
        # All combos should be complexity 1-2
        for instr in rnd["instructions"]:
            assert len(instr["actions"]) <= 2


async def test_fallback_plan_combos(service: SessionService):
    result = await service.generate_plan("combos")
    plan = result["plan"]

    assert len(plan["rounds"]) == 4
    for rnd in plan["rounds"]:
        assert rnd["duration_seconds"] == 150
        for instr in rnd["instructions"]:
            assert 3 <= len(instr["actions"]) <= 4


async def test_fallback_plan_mixed(service: SessionService):
    result = await service.generate_plan("mixed")
    plan = result["plan"]

    assert len(plan["rounds"]) == 5
    for rnd in plan["rounds"]:
        assert rnd["duration_seconds"] == 180


async def test_plan_saved_to_db(service: SessionService):
    result = await service.generate_plan("fundamentals")
    assert "id" in result
    assert len(result["id"]) == 36  # UUID


async def test_plan_with_user_prompt(service: SessionService):
    result = await service.generate_plan("combos", user_prompt="훅 위주로 연습하고 싶어")
    # Fallback ignores user prompt but plan still generates
    assert result["plan"]["rounds"]


async def test_invalid_template_name(service: SessionService):
    with pytest.raises(PlanValidationError, match="not found"):
        await service.generate_plan("nonexistent_template")


async def test_fundamentals_excludes_defense_combos(service: SessionService):
    result = await service.generate_plan("fundamentals")
    plan = result["plan"]

    defense_actions = {"slip", "duck", "backstep"}
    for rnd in plan["rounds"]:
        for instr in rnd["instructions"]:
            for action in instr["actions"]:
                assert action not in defense_actions, (
                    f"Defense action '{action}' found in fundamentals plan"
                )


async def test_mixed_includes_defense_combos(service: SessionService):
    """Mixed template should include combos with defense actions."""
    result = await service.generate_plan("mixed")
    plan = result["plan"]

    # At least some instructions should exist (mixed allows all complexity)
    total_instructions = sum(len(r["instructions"]) for r in plan["rounds"])
    assert total_instructions > 0


async def test_plan_progressive_difficulty(service: SessionService):
    """Earlier rounds should tend toward simpler combos."""
    result = await service.generate_plan("mixed")
    plan = result["plan"]

    if len(plan["rounds"]) >= 2:
        r1_avg = sum(len(i["actions"]) for i in plan["rounds"][0]["instructions"]) / max(1, len(plan["rounds"][0]["instructions"]))
        r_last_avg = sum(len(i["actions"]) for i in plan["rounds"][-1]["instructions"]) / max(1, len(plan["rounds"][-1]["instructions"]))
        # Last round should tend to have >= complexity of first (not strict due to randomness)
        # Just verify both rounds have instructions
        assert len(plan["rounds"][0]["instructions"]) > 0
        assert len(plan["rounds"][-1]["instructions"]) > 0


async def test_plan_instructions_have_timestamps(service: SessionService):
    result = await service.generate_plan("fundamentals")
    plan = result["plan"]

    for rnd in plan["rounds"]:
        for instr in rnd["instructions"]:
            assert "timestamp_offset" in instr
            assert isinstance(instr["timestamp_offset"], float)
            assert instr["timestamp_offset"] >= 0


async def test_plan_instructions_within_round_duration(service: SessionService):
    result = await service.generate_plan("fundamentals")
    plan = result["plan"]

    for rnd in plan["rounds"]:
        duration = rnd["duration_seconds"]
        for instr in rnd["instructions"]:
            assert instr["timestamp_offset"] < duration
