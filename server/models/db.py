"""SQLAlchemy async models for the state vector system.

Tables follow spec/runtime.md §7:
  - user_state: current state vector per user
  - sessions: session lifecycle tracking
  - state_transitions: audit log of every state update

Uses SQLite for local dev (aiosqlite), PostgreSQL for production (asyncpg).
Switch by setting ATOM_DATABASE_URL env var.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, Float, Index, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from server.config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# UserState — single row per user, stores current S_t, C_t
# ---------------------------------------------------------------------------

class UserState(Base):
    __tablename__ = "user_state"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    device_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # State vector as JSON string: "[0.5, 0.7, ...]"
    vector_json: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_json: Mapped[str] = mapped_column(Text, nullable=False)
    obs_counts_json: Mapped[str] = mapped_column(Text, nullable=False)

    row_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    schema_version: Mapped[str] = mapped_column(String(10), nullable=False, default="v1")

    created_at: Mapped[datetime] = mapped_column(default=_now)
    updated_at: Mapped[datetime] = mapped_column(default=_now, onupdate=_now)


# ---------------------------------------------------------------------------
# Session — session lifecycle tracking
# ---------------------------------------------------------------------------

class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), nullable=False)

    mode: Mapped[str] = mapped_column(String(20), nullable=False)  # shadow, heavy_bag, ai_session
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="CREATED")

    # Pipeline tracking
    pipeline_stage: Mapped[Optional[str]] = mapped_column(String(30), default=None)
    pipeline_progress: Mapped[float] = mapped_column(Float, default=0.0)

    # Idempotency guard (runtime.md §3)
    state_update_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Session data
    video_path: Mapped[Optional[str]] = mapped_column(Text, default=None)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, default=None)
    script_id: Mapped[Optional[str]] = mapped_column(String(36), default=None)

    # Error tracking
    error_code: Mapped[Optional[str]] = mapped_column(String(50), default=None)
    error_detail: Mapped[Optional[str]] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(default=_now)
    updated_at: Mapped[datetime] = mapped_column(default=_now, onupdate=_now)

    __table_args__ = (
        Index("idx_sessions_user", "user_id", "created_at"),
    )


# ---------------------------------------------------------------------------
# StateTransition — audit log (append-only)
# ---------------------------------------------------------------------------

class StateTransition(Base):
    __tablename__ = "state_transitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), nullable=False)
    session_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)

    version_before: Mapped[int] = mapped_column(Integer, nullable=False)
    version_after: Mapped[int] = mapped_column(Integer, nullable=False)

    # State snapshots as JSON
    vector_before_json: Mapped[str] = mapped_column(Text, nullable=False)
    vector_after_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Observation that caused this transition
    observation_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    observation_mask_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Delta
    delta_json: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(default=_now)

    __table_args__ = (
        Index("idx_transitions_user", "user_id", "created_at"),
    )


# ---------------------------------------------------------------------------
# DB Initialization
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Create all tables (dev only — use Alembic for production)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
