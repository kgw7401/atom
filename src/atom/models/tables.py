"""SQLAlchemy table definitions for Atom data models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from atom.models.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionTemplate(Base):
    """Pre-generated session template with segment structure."""

    __tablename__ = "session_templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    level: Mapped[str] = mapped_column(String(20), nullable=False)  # beginner|intermediate|advanced
    topic: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    segments_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AudioChunk(Base):
    """Pre-recorded audio chunk for combo assembly."""

    __tablename__ = "audio_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    text: Mapped[str] = mapped_column(String(100), nullable=False)
    variant: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    audio_path: Mapped[str] = mapped_column(String(500), nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("text", "variant", name="uq_audio_chunk_text_variant"),
    )


class DrillPlan(Base):
    """Instantiated session plan (from template + shuffle)."""

    __tablename__ = "drill_plans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    template_id: Mapped[str | None] = mapped_column(String(36), default=None)
    llm_model: Mapped[str] = mapped_column(String(100), default="")
    session_config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    plan_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SessionLog(Base):
    """Append-only record of a completed (or abandoned) session."""

    __tablename__ = "session_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    drill_plan_id: Mapped[str] = mapped_column(String(36), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    total_duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    rounds_completed: Mapped[int] = mapped_column(Integer, default=0)
    rounds_total: Mapped[int] = mapped_column(Integer, default=0)
    segments_delivered: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # completed/abandoned
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class UserProfile(Base):
    """Aggregated user profile."""

    __tablename__ = "user_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    experience_level: Mapped[str] = mapped_column(String(20), default="beginner")
    goal: Mapped[str] = mapped_column(Text, default="")
    total_sessions: Mapped[int] = mapped_column(Integer, default=0)
    total_training_minutes: Mapped[float] = mapped_column(Float, default=0.0)
    last_session_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    combo_exposure_json: Mapped[dict] = mapped_column(JSON, default=dict)
    template_preference_json: Mapped[dict] = mapped_column(JSON, default=dict)
    session_frequency: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
