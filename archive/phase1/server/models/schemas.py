"""Pydantic request/response schemas for the API."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# --- Script ---

class ScriptGenerateRequest(BaseModel):
    level: int = Field(ge=1, le=3, default=1)
    duration_seconds: int = Field(ge=30, le=600, default=120)


class Instruction(BaseModel):
    t: float
    type: str  # "attack" or "defend"
    action: str  # e.g. "jab", "jab,cross", "slip"
    audio_key: str
    display: str


class ScriptResponse(BaseModel):
    script_id: uuid.UUID
    instructions: list[Instruction]
    level: int
    total_instructions: int
    estimated_duration_seconds: float


# --- Session ---

class SessionCreateRequest(BaseModel):
    script_id: uuid.UUID
    user_id: uuid.UUID
    started_at: datetime


class SessionCreateResponse(BaseModel):
    session_id: uuid.UUID
    upload_url: str
    status: str = "created"


class UploadCompleteRequest(BaseModel):
    video_duration_seconds: float
    video_fps: int = 30


class SessionStatusResponse(BaseModel):
    status: str
    progress: float = 0.0


# --- Report ---

class InstructionResult(BaseModel):
    index: int
    t: float
    type: str
    command: str
    status: str  # "success", "partial", "missed"
    score: int
    reaction_time: float | None = None
    detected_actions: list[str]
    feedback: str


class ReportSummary(BaseModel):
    total_instructions: int
    completed: int
    success: int
    partial: int
    missed: int
    attack_accuracy: float
    defense_accuracy: float
    avg_reaction_time: float


class CoachingFeedback(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    next_session: str


class SessionReportResponse(BaseModel):
    session_id: uuid.UUID
    overall_score: int
    summary: ReportSummary
    instructions: list[InstructionResult]
    coaching: CoachingFeedback


# --- User / Twin ---

class UserCreateRequest(BaseModel):
    device_id: str


class UserCreateResponse(BaseModel):
    user_id: uuid.UUID


class ActionStat(BaseModel):
    accuracy: float
    avg_reaction: float
    trend: str  # "improving", "stable", "declining"
    total_attempts: int


class Weakness(BaseModel):
    action: str
    metric: str
    value: float
    threshold: float
    severity: str  # "warning", "critical"


class TwinResponse(BaseModel):
    total_sessions: int
    per_action_stats: dict[str, ActionStat]
    weaknesses: list[Weakness]
    growth_curves: dict[str, list]
