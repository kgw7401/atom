"""Tests for ProfileService — aggregation and session history."""

from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from atom.models.base import Base
from atom.models.tables import SessionLog, UserProfile
from atom.seed import seed_all
from atom.services.profile_service import ProfileService
from atom.services.session_service import SessionService
from atom.services.session_engine import SessionEngine


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        await seed_all(s)
        yield s
    await engine.dispose()


@pytest.fixture
def svc(db: AsyncSession) -> ProfileService:
    return ProfileService(db)


async def _make_session_log(db: AsyncSession, template: str = "fundamentals") -> SessionLog:
    """Helper: generate a plan + run it (mocked) + save log."""
    plan_svc = SessionService(db)
    result = await plan_svc.generate_plan(template)

    engine = SessionEngine(
        plan=result["plan"],
        plan_id=result["id"],
        tts_enabled=False,
    )
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await engine.run()
    return await engine.save_log(db)


# ── Profile tests ─────────────────────────────────────────────────────

async def test_get_profile_returns_seeded(svc: ProfileService):
    profile = await svc.get_profile()
    assert profile is not None
    assert profile.experience_level == "beginner"


async def test_update_experience(svc: ProfileService):
    p = await svc.update_profile(experience_level="intermediate")
    assert p.experience_level == "intermediate"


async def test_update_goal(svc: ProfileService):
    p = await svc.update_profile(goal="스파링 준비")
    assert p.goal == "스파링 준비"


async def test_aggregate_no_sessions(svc: ProfileService):
    p = await svc.aggregate()
    assert p.total_sessions == 0
    assert p.total_training_minutes == 0.0
    assert p.combo_exposure_json == {}
    assert p.template_preference_json == {}
    assert p.session_frequency == 0.0


async def test_aggregate_after_sessions(db: AsyncSession):
    await _make_session_log(db, "fundamentals")
    await _make_session_log(db, "fundamentals")
    await _make_session_log(db, "combos")

    svc = ProfileService(db)
    p = await svc.aggregate()

    assert p.total_sessions == 3
    assert p.total_training_minutes > 0
    assert p.template_preference_json["fundamentals"] == 2
    assert p.template_preference_json["combos"] == 1


async def test_combo_exposure_counts(db: AsyncSession):
    await _make_session_log(db, "fundamentals")

    svc = ProfileService(db)
    p = await svc.aggregate()

    assert len(p.combo_exposure_json) > 0
    assert all(isinstance(v, int) and v > 0 for v in p.combo_exposure_json.values())


async def test_session_frequency_recent(db: AsyncSession):
    await _make_session_log(db)
    await _make_session_log(db)

    svc = ProfileService(db)
    p = await svc.aggregate()

    # 2 sessions in 4 weeks = 0.5/week
    assert p.session_frequency == pytest.approx(0.5)


async def test_abandoned_sessions_not_counted(db: AsyncSession):
    """Abandoned sessions should not count toward totals."""
    plan_svc = SessionService(db)
    result = await plan_svc.generate_plan("fundamentals")

    engine = SessionEngine(plan=result["plan"], plan_id=result["id"], tts_enabled=False)
    call_count = 0

    async def abort_early(d):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            engine.abort()

    with patch("asyncio.sleep", side_effect=abort_early):
        await engine.run()
    await engine.save_log(db)

    svc = ProfileService(db)
    p = await svc.aggregate()
    assert p.total_sessions == 0  # abandoned don't count


# ── Session history tests ─────────────────────────────────────────────

async def test_list_sessions_empty(svc: ProfileService):
    logs = await svc.list_sessions()
    assert logs == []


async def test_list_sessions_after_run(db: AsyncSession):
    await _make_session_log(db)
    await _make_session_log(db)

    svc = ProfileService(db)
    logs = await svc.list_sessions()
    assert len(logs) == 2


async def test_list_sessions_limit(db: AsyncSession):
    for _ in range(5):
        await _make_session_log(db)

    svc = ProfileService(db)
    logs = await svc.list_sessions(limit=3)
    assert len(logs) == 3


async def test_list_sessions_ordered_newest_first(db: AsyncSession):
    await _make_session_log(db)
    await _make_session_log(db)
    await _make_session_log(db)

    svc = ProfileService(db)
    logs = await svc.list_sessions()
    dates = [l.started_at for l in logs]
    assert dates == sorted(dates, reverse=True)


async def test_get_session_by_id(db: AsyncSession):
    log = await _make_session_log(db)

    svc = ProfileService(db)
    fetched = await svc.get_session(log.id)
    assert fetched is not None
    assert fetched.id == log.id
    assert fetched.delivery_log_json  # has events


async def test_get_session_not_found(svc: ProfileService):
    result = await svc.get_session("nonexistent-id")
    assert result is None
