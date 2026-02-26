"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from src.state.constants import DIM_NAMES, NUM_DIMS


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class CreateUserRequest(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=255)


class CreateSessionRequest(BaseModel):
    user_id: str
    mode: str = Field(..., pattern="^(shadow|heavy_bag|ai_session)$")


class GeneratePlanRequest(BaseModel):
    user_id: str


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class UserResponse(BaseModel):
    user_id: str
    device_id: str
    created_at: datetime


class StateResponse(BaseModel):
    user_id: str
    values: List[float]
    confidence: List[float]
    obs_counts: List[int]
    dim_names: List[str] = Field(default_factory=lambda: list(DIM_NAMES))
    version: int
    schema_version: str


class TransitionItem(BaseModel):
    session_id: str
    version_before: int
    version_after: int
    delta: List[float]
    observed_dims: List[str]
    created_at: datetime


class HistoryResponse(BaseModel):
    user_id: str
    transitions: List[TransitionItem]


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    mode: str
    status: str
    pipeline_stage: Optional[str] = None
    pipeline_progress: float = 0.0
    duration_seconds: Optional[float] = None
    error_code: Optional[str] = None
    created_at: datetime


class SessionReportResponse(BaseModel):
    session_id: str
    observation: List[Optional[float]]
    observation_mask: List[bool]
    delta: List[float]
    dim_names: List[str] = Field(default_factory=lambda: list(DIM_NAMES))


class PlanResponse(BaseModel):
    plan_id: str
    plan: dict


# ---------------------------------------------------------------------------
# Coaching & Reports (Phase 2e)
# ---------------------------------------------------------------------------

class DeltaItemResponse(BaseModel):
    dim_index: int
    dim_name: str
    dim_label: str
    group: str
    delta: float
    current_value: float
    description: str


class WeaknessItemResponse(BaseModel):
    dim_index: int
    dim_name: str
    dim_label: str
    group: str
    value: float
    threshold: float
    confidence: float
    coaching_hint: str


class CoachingResponse(BaseModel):
    summary: str
    focus_areas: List[str]
    next_session_hint: str


class EnrichedReportResponse(BaseModel):
    session_id: str
    session_summary: dict
    state_delta: dict
    weaknesses: List[WeaknessItemResponse]
    coaching: CoachingResponse
    raw: SessionReportResponse


class ProgressReportResponse(BaseModel):
    user_id: str
    num_sessions: int
    trending_up: List[DeltaItemResponse]
    trending_down: List[DeltaItemResponse]
    plateau: List[str]
    coaching: CoachingResponse


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.2.0"
