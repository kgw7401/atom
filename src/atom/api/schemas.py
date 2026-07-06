"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


# ── Session schemas ───────────────────────────────────────────────────

class PlanRequest(BaseModel):
    level: str = Field(default="beginner", pattern=r"^(beginner|intermediate|advanced)$")
    rounds: int = Field(default=3, ge=1, le=12)
    round_duration_sec: int = Field(default=120, ge=30, le=600)
    rest_sec: int = Field(default=30, ge=10, le=120)
    program_day_id: str | None = Field(default=None)  # auto-detect from ProgramProgress if None


class ChunkResponse(BaseModel):
    text: str              # chunk text e.g. "원투"
    clip_url: str          # /audio/chunks/원투_1.mp3
    duration_ms: int


class SegmentResponse(BaseModel):
    text: str              # full combo text e.g. "원투 슥 투훅투"
    chunks: list[ChunkResponse]
    impact_actions: list[str] = []   # e.g. ["J", "C", "LH"]


class TimestampResponse(BaseModel):
    start_ms: int
    end_ms: int
    text: str


class RoundResponse(BaseModel):
    round: int
    segments: list[SegmentResponse]
    audio_url: str = ""
    timestamps: list[TimestampResponse] = []


class PlanDetail(BaseModel):
    rounds: list[RoundResponse]


class PlanResponse(BaseModel):
    id: str
    template_name: str
    template_topic: str
    rounds: int
    round_duration_sec: int
    rest_sec: int
    plan: PlanDetail
    audio_ready: bool = False
    day_number: int = 0
    theme: str = ""
    coach_comment: str = ""


# ── Session log schemas ───────────────────────────────────────────────

class SessionLogRequest(BaseModel):
    drill_plan_id: str
    started_at: datetime
    completed_at: datetime
    total_duration_sec: float
    rounds_completed: int
    rounds_total: int
    segments_delivered: int
    status: str   # "completed" | "abandoned"


class SessionLogResponse(BaseModel):
    id: str
    started_at: datetime
    completed_at: datetime | None
    total_duration_sec: float
    rounds_completed: int
    rounds_total: int
    segments_delivered: int
    status: str

    model_config = {"from_attributes": True}


class SessionSummary(BaseModel):
    id: str
    drill_plan_id: str
    started_at: datetime
    completed_at: datetime | None
    total_duration_sec: float
    rounds_completed: int
    rounds_total: int
    segments_delivered: int
    status: str

    model_config = {"from_attributes": True}


# ── Profile schemas ───────────────────────────────────────────────────

class ProfileResponse(BaseModel):
    id: str
    experience_level: str
    goal: str
    total_sessions: int
    total_training_minutes: float
    last_session_at: datetime | None
    current_streak: int = 0
    longest_streak: int = 0
    tier: str = "rookie"
    training_preference: str = "all"

    model_config = {"from_attributes": True}


class ProfileUpdate(BaseModel):
    experience_level: str | None = Field(
        default=None,
        pattern=r"^(beginner|novice|intermediate|advanced)$",
    )
    goal: str | None = Field(default=None, max_length=500)
    training_preference: str | None = Field(
        default=None,
        pattern=r"^(basics|cardio|speed|all)$",
    )


# ── Today (home screen) ─────────────────────────────────────────────

class ProgramDaySummary(BaseModel):
    id: str
    day_number: int
    theme: str
    theme_description: str
    coach_comment: str


class TodayResponse(BaseModel):
    streak: int
    day_number: int
    day_total: int = 7
    theme: str
    theme_description: str
    coach_comment: str
    level: str
    week: int
    next_day_preview: dict | None = None  # {day_number, theme}
    program_days: list[ProgramDaySummary] = []
