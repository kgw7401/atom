"""Tests for ComboService CRUD operations."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from atom.models.base import Base
from atom.models.tables import Action, Combination
from atom.seed import seed_all
from atom.services.combo_service import (
    ComboImmutableError,
    ComboNotFoundError,
    ComboService,
    ComboValidationError,
)


@pytest.fixture
async def session():
    """Create an in-memory SQLite DB with seed data for each test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        await seed_all(s)
        yield s

    await engine.dispose()


@pytest.fixture
def service(session: AsyncSession) -> ComboService:
    return ComboService(session)


async def test_list_all(service: ComboService):
    combos = await service.list()
    assert len(combos) == 12


async def test_list_by_complexity(service: ComboService):
    singles = await service.list(complexity=1)
    assert all(c.complexity == 1 for c in singles)
    assert len(singles) == 2  # 잽, 크로스


async def test_get_by_display_name(service: ComboService):
    combo = await service.get("원투")
    assert combo.display_name == "원투"
    assert combo.actions == ["jab", "cross"]


async def test_get_not_found(service: ComboService):
    with pytest.raises(ComboNotFoundError):
        await service.get("존재하지않는콤보")


async def test_create(service: ComboService):
    combo = await service.create("테스트콤보", ["jab", "cross", "lead_hook"])
    assert combo.display_name == "테스트콤보"
    assert combo.actions == ["jab", "cross", "lead_hook"]
    assert combo.complexity == 3
    assert combo.is_system is False


async def test_create_invalid_action(service: ComboService):
    with pytest.raises(ComboValidationError, match="not found"):
        await service.create("잘못된콤보", ["jab", "nonexistent_action"])


async def test_create_duplicate_name(service: ComboService):
    with pytest.raises(ComboValidationError, match="already exists"):
        await service.create("원투", ["jab", "cross"])


async def test_create_empty_actions(service: ComboService):
    with pytest.raises(ComboValidationError, match="at least one"):
        await service.create("빈콤보", [])


async def test_update_user_combo(service: ComboService):
    combo = await service.create("수정테스트", ["jab"])
    updated = await service.update(combo.id, display_name="수정됨", actions=["jab", "cross"])
    assert updated.display_name == "수정됨"
    assert updated.actions == ["jab", "cross"]
    assert updated.complexity == 2


async def test_update_system_combo_fails(service: ComboService):
    with pytest.raises(ComboImmutableError):
        await service.update("원투", display_name="변경시도")


async def test_delete_user_combo(service: ComboService):
    combo = await service.create("삭제테스트", ["jab"])
    await service.delete(combo.id)
    with pytest.raises(ComboNotFoundError):
        await service.get(combo.id)


async def test_delete_system_combo_fails(service: ComboService):
    with pytest.raises(ComboImmutableError):
        await service.delete("원투")
