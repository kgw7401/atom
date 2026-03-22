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
    performance_summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    current_streak: Mapped[int] = mapped_column(Integer, default=0)
    longest_streak: Mapped[int] = mapped_column(Integer, default=0)
    last_training_date: Mapped[str | None] = mapped_column(String(10), default=None)  # YYYY-MM-DD
    tier: Mapped[str] = mapped_column(String(20), default="rookie")
    training_preference: Mapped[str] = mapped_column(String(20), default="all")  # basics|cardio|speed|all
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class ProgramDayTemplate(Base):
    """Predetermined daily training template within a program."""

    __tablename__ = "program_day_templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    level: Mapped[str] = mapped_column(String(20), nullable=False)  # beginner|intermediate|advanced
    week: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    day_number: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-7
    theme: Mapped[str] = mapped_column(String(100), nullable=False)
    theme_description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    coach_comment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    r1_segments_json: Mapped[dict] = mapped_column(JSON, nullable=False)  # [{combo, is_cue}, ...]
    r2_segments_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    r3_segments_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    finisher_json: Mapped[dict] = mapped_column(JSON, nullable=False)  # {type, segments: [...]}
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("level", "week", "day_number", name="uq_program_day"),
    )


class ProgramProgress(Base):
    """Tracks user's current position in a program."""

    __tablename__ = "program_progress"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    level: Mapped[str] = mapped_column(String(20), nullable=False, default="beginner")
    week: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    current_day: Mapped[int] = mapped_column(Integer, nullable=False, default=1)  # 1-7
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class VideoUpload(Base):
    """Uploaded video for analysis."""

    __tablename__ = "video_uploads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_log_id: Mapped[str | None] = mapped_column(String(36), default=None)
    drill_plan_id: Mapped[str] = mapped_column(String(36), nullable=False)
    video_path: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="uploaded")
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AnalysisResult(Base):
    """Results of video analysis pipeline."""

    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    video_upload_id: Mapped[str] = mapped_column(String(36), nullable=False)
    predicted_events: Mapped[dict] = mapped_column(JSON, default=list)
    expected_events: Mapped[dict] = mapped_column(JSON, default=list)
    comparison_json: Mapped[dict] = mapped_column(JSON, default=dict)
    feedback_text: Mapped[str] = mapped_column(Text, default="")
    accuracy_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SessionRecommendation(Base):
    """Next session recommendation based on analysis."""

    __tablename__ = "session_recommendations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    analysis_result_id: Mapped[str] = mapped_column(String(36), nullable=False)
    recommended_level: Mapped[str] = mapped_column(String(20), nullable=False)
    recommended_topic: Mapped[str] = mapped_column(String(200), default="")
    narrative_text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
