"""Shared fixtures for integration tests."""

from __future__ import annotations

import json
import uuid

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from server.models.db import Base, Session as SessionModel, UserState
from src.state.constants import NUM_DIMS, PUNCH_NAMES
from src.state.types import ActionSegment, KeypointFrame


# ---------------------------------------------------------------------------
# In-memory async SQLite for testing (StaticPool shares one connection)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def db_engine():
    """Create a fresh in-memory database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session_factory(db_engine):
    """Session factory bound to the test database."""
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def db(db_session_factory):
    """Single DB session for a test."""
    async with db_session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Fake data generators
# ---------------------------------------------------------------------------

def make_segment(class_id: int, t_start: float, t_end: float) -> ActionSegment:
    names = ["guard"] + list(PUNCH_NAMES)
    return ActionSegment(
        class_id=class_id, class_name=names[class_id], t_start=t_start, t_end=t_end
    )


def make_keypoint_frame(timestamp: float, seed: int = 0) -> KeypointFrame:
    rng = np.random.RandomState(seed + int(timestamp * 100))
    kp = np.array([
        [0.50, 0.20], [0.45, 0.22], [0.55, 0.22],
        [0.35, 0.40], [0.65, 0.40], [0.38, 0.32],
        [0.62, 0.32], [0.45, 0.25], [0.55, 0.25],
        [0.40, 0.65], [0.60, 0.65],
    ], dtype=np.float64)
    kp += rng.randn(11, 2) * 0.02
    return KeypointFrame(timestamp=timestamp, keypoints=kp)


def make_diverse_segments(duration: float = 30.0) -> list[ActionSegment]:
    """Diverse punch segments with guard intervals."""
    from src.state.constants import (
        CLASS_CROSS, CLASS_GUARD, CLASS_JAB, CLASS_LEAD_BODYSHOT,
        CLASS_LEAD_HOOK, CLASS_LEAD_UPPERCUT, CLASS_REAR_BODYSHOT,
        CLASS_REAR_HOOK, CLASS_REAR_UPPERCUT,
    )
    pattern = [
        CLASS_JAB, CLASS_GUARD, CLASS_CROSS, CLASS_GUARD,
        CLASS_LEAD_HOOK, CLASS_GUARD, CLASS_REAR_HOOK, CLASS_GUARD,
        CLASS_LEAD_UPPERCUT, CLASS_GUARD, CLASS_REAR_UPPERCUT, CLASS_GUARD,
        CLASS_LEAD_BODYSHOT, CLASS_GUARD, CLASS_REAR_BODYSHOT, CLASS_GUARD,
        CLASS_JAB, CLASS_CROSS, CLASS_GUARD,
        CLASS_JAB, CLASS_LEAD_HOOK, CLASS_GUARD,
        CLASS_CROSS, CLASS_REAR_HOOK, CLASS_GUARD,
    ]
    segments = []
    t = 0.0
    for cls_id in pattern:
        if t >= duration:
            break
        dur = 0.4 if cls_id != CLASS_GUARD else 1.0
        segments.append(make_segment(cls_id, t, t + dur))
        t += dur + 0.1
    return segments


def make_keypoint_sequence(duration: float = 30.0, fps: float = 30.0) -> list[KeypointFrame]:
    """Keypoint frames at given fps for given duration."""
    return [
        make_keypoint_frame(i / fps, seed=i) for i in range(int(duration * fps))
    ]


def make_fake_extraction(duration: float = 30.0, fps: float = 30.0):
    """Create a fake ExtractionResult (no cv2/mediapipe needed)."""
    from src.vision.keypoint_extractor import ExtractionResult

    N = int(duration * fps)
    raw_keypoints = np.random.rand(N, 33, 4).astype(np.float32)
    raw_keypoints[:, :, 3] = 0.9  # high visibility
    timestamps_s = np.arange(N, dtype=np.float64) / fps
    keypoint_frames = make_keypoint_sequence(duration, fps)

    return ExtractionResult(
        raw_keypoints=raw_keypoints,
        timestamps_s=timestamps_s,
        fps=fps,
        duration=duration,
        keypoint_frames=keypoint_frames,
    )


@pytest.fixture
def session_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def user_id() -> str:
    return str(uuid.uuid4())
