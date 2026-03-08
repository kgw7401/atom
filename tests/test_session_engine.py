"""Tests for SessionEngine — timer-based drill execution."""

import asyncio
from unittest.mock import patch, AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from atom.models.base import Base
from atom.seed import seed_all
from atom.services.session_engine import SessionEngine, State
from atom.services.session_service import SessionService


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        await seed_all(s)
        yield s

    await engine.dispose()


@pytest.fixture
async def plan(db_session: AsyncSession) -> dict:
    """Generate a small fundamentals plan for testing."""
    svc = SessionService(db_session)
    result = await svc.generate_plan("fundamentals")
    return result


def _make_engine(plan: dict, tts: bool = False) -> SessionEngine:
    """Create an engine with TTS disabled and output captured."""
    return SessionEngine(
        plan=plan["plan"],
        plan_id=plan["id"],
        tts_enabled=tts,
    )


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_full_session_completes(mock_sleep, plan):
    engine = _make_engine(plan)
    delivery_log = await engine.run()

    assert engine.state == State.SESSION_END
    assert engine.rounds_completed == 3  # fundamentals = 3 rounds
    assert engine.combos_delivered > 0
    assert not engine._aborted

    events = delivery_log["events"]
    assert events[0]["type"] == "round_start"
    assert events[-1]["type"] == "session_end"
    assert events[-1]["reason"] == "completed"


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_events_have_correct_types(mock_sleep, plan):
    engine = _make_engine(plan)
    delivery_log = await engine.run()

    event_types = {e["type"] for e in delivery_log["events"]}
    assert "round_start" in event_types
    assert "combo_called" in event_types
    assert "round_end" in event_types
    assert "rest_start" in event_types
    assert "rest_end" in event_types
    assert "session_end" in event_types


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_combo_events_have_required_fields(mock_sleep, plan):
    engine = _make_engine(plan)
    delivery_log = await engine.run()

    combo_events = [e for e in delivery_log["events"] if e["type"] == "combo_called"]
    assert len(combo_events) > 0

    for e in combo_events:
        assert "ts" in e
        assert "round" in e
        assert "combo_display_name" in e
        assert "actions" in e
        assert isinstance(e["actions"], list)


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_abort_mid_session(mock_sleep, plan):
    call_count = 0

    async def abort_after_few_sleeps(duration):
        nonlocal call_count
        call_count += 1
        if call_count >= 5:
            engine.abort()

    mock_sleep.side_effect = abort_after_few_sleeps

    engine = _make_engine(plan)
    delivery_log = await engine.run()

    assert engine._aborted
    assert engine.state == State.SESSION_END
    assert engine.rounds_completed < 3  # Didn't finish all rounds

    events = delivery_log["events"]
    assert events[-1]["type"] == "session_end"
    assert events[-1]["reason"] == "abandoned"


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_round_count_matches_plan(mock_sleep, plan):
    engine = _make_engine(plan)
    await engine.run()

    round_starts = [e for e in engine.events if e.type == "round_start"]
    round_ends = [e for e in engine.events if e.type == "round_end"]

    assert len(round_starts) == 3
    assert len(round_ends) == 3


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_rest_between_rounds(mock_sleep, plan):
    engine = _make_engine(plan)
    await engine.run()

    rest_starts = [e for e in engine.events if e.type == "rest_start"]
    rest_ends = [e for e in engine.events if e.type == "rest_end"]

    # Rest between rounds (3 rounds = 2 rest periods)
    assert len(rest_starts) == 2
    assert len(rest_ends) == 2


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_output_callback(mock_sleep, plan):
    output_lines = []
    engine = SessionEngine(
        plan=plan["plan"],
        plan_id=plan["id"],
        tts_enabled=False,
        on_output=output_lines.append,
    )
    await engine.run()

    assert len(output_lines) > 0
    # Should contain round announcements
    assert any("Round 1" in line for line in output_lines)
    assert any("Session complete" in line for line in output_lines)


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_save_log_completed(mock_sleep, plan, db_session):
    engine = _make_engine(plan)
    await engine.run()

    log = await engine.save_log(db_session)

    assert log.status == "completed"
    assert log.rounds_completed == 3
    assert log.rounds_total == 3
    assert log.combos_delivered > 0
    assert log.drill_plan_id == plan["id"]
    assert log.template_name == "fundamentals"
    assert log.delivery_log_json["events"]
    assert log.id  # UUID assigned


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_save_log_abandoned(mock_sleep, plan, db_session):
    call_count = 0

    async def abort_early(duration):
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            engine.abort()

    mock_sleep.side_effect = abort_early

    engine = _make_engine(plan)
    await engine.run()

    log = await engine.save_log(db_session)

    assert log.status == "abandoned"
    assert log.rounds_completed < 3


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_events_timestamps_monotonic(mock_sleep, plan):
    engine = _make_engine(plan)
    delivery_log = await engine.run()

    timestamps = [e["ts"] for e in delivery_log["events"]]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], (
            f"Timestamp {timestamps[i]} < {timestamps[i-1]} at index {i}"
        )


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_initial_state_is_idle(mock_sleep, plan):
    engine = _make_engine(plan)
    assert engine.state == State.IDLE
    assert engine.rounds_completed == 0
    assert engine.combos_delivered == 0
    assert len(engine.events) == 0
