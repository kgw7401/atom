"""SQLAlchemy table definitions for all 6 Atom data models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from atom.models.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Action(Base):
    """Reference table of boxing actions (system-seeded, immutable)."""

    __tablename__ = "actions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[str] = mapped_column(String(20), nullable=False)  # offense/defense/movement
    description: Mapped[str] = mapped_column(Text, default="")
    sort_order: Mapped[int] = mapped_column(Integer, default=0)


class Combination(Base):
    """User-defined or system-seeded combo sequences."""

    __tablename__ = "combinations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    display_name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    actions: Mapped[dict] = mapped_column(JSON, nullable=False)  # list[str]
    complexity: Mapped[int] = mapped_column(Integer, nullable=False)  # len(actions)
    is_system: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class SessionTemplate(Base):
    """Preset session templates (system-defined)."""

    __tablename__ = "session_templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    default_rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    default_round_duration_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    default_rest_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    combo_complexity_range: Mapped[dict] = mapped_column(JSON, nullable=False)  # [min, max]
    combo_include_defense: Mapped[bool] = mapped_column(Boolean, default=True)
    pace_interval_sec: Mapped[dict] = mapped_column(JSON, nullable=False)  # [min, max]


class DrillPlan(Base):
    """LLM-generated session plan."""

    __tablename__ = "drill_plans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    template_id: Mapped[str] = mapped_column(String(36), nullable=False)
    user_prompt: Mapped[str | None] = mapped_column(Text, default=None)
    llm_model: Mapped[str] = mapped_column(String(100), default="")
    plan_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SessionLog(Base):
    """Append-only, immutable record of a completed (or abandoned) session."""

    __tablename__ = "session_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    drill_plan_id: Mapped[str] = mapped_column(String(36), nullable=False)
    template_name: Mapped[str] = mapped_column(String(50), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    total_duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    rounds_completed: Mapped[int] = mapped_column(Integer, default=0)
    rounds_total: Mapped[int] = mapped_column(Integer, default=0)
    combos_delivered: Mapped[int] = mapped_column(Integer, default=0)
    delivery_log_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # completed/abandoned/error
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class UserProfile(Base):
    """Aggregated user profile, re-computable from SessionLogs."""

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
