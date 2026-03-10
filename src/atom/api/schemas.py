"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ComboCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=200)
    actions: list[str] = Field(min_length=1)


class ComboUpdate(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=200)
    actions: list[str] | None = Field(default=None, min_length=1)


class ComboResponse(BaseModel):
    id: str
    display_name: str
    actions: list[str]
    complexity: int
    is_system: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Session schemas ───────────────────────────────────────────────────

class PlanRequest(BaseModel):
    template: str = Field(min_length=1)
    prompt: str | None = Field(default=None, max_length=500)


class InstructionResponse(BaseModel):
    timestamp_offset: float
    combo_display_name: str
    actions: list[str]


class RoundResponse(BaseModel):
    round_number: int
    duration_seconds: int
    rest_after_seconds: int
    instructions: list[InstructionResponse]


class PlanResponse(BaseModel):
    id: str
    llm_model: str
    plan: PlanDetail


class PlanDetail(BaseModel):
    session_type: str
    template: str
    focus: str
    total_duration_minutes: int
    rounds: list[RoundResponse]
    pace_interval_sec: list[int] = Field(default=[3, 5])


# Fix forward reference
PlanResponse.model_rebuild()


# ── Session history schemas ───────────────────────────────────────────

class SessionSummary(BaseModel):
    id: str
    drill_plan_id: str
    template_name: str
    started_at: datetime
    completed_at: datetime | None
    total_duration_sec: float
    rounds_completed: int
    rounds_total: int
    combos_delivered: int
    status: str

    model_config = {"from_attributes": True}


class SessionDetail(SessionSummary):
    delivery_log_json: dict


# ── Session log schema (client → server after local execution) ────────

class SessionLogRequest(BaseModel):
    drill_plan_id: str
    template_name: str
    started_at: datetime
    completed_at: datetime
    total_duration_sec: float
    rounds_completed: int
    rounds_total: int
    combos_delivered: int
    status: str  # "completed" | "abandoned"


# ── Profile schemas ───────────────────────────────────────────────────

class ProfileResponse(BaseModel):
    id: str
    experience_level: str
    goal: str
    total_sessions: int
    total_training_minutes: float
    last_session_at: datetime | None
    combo_exposure_json: dict
    template_preference_json: dict
    session_frequency: float

    model_config = {"from_attributes": True}


class ProfileUpdate(BaseModel):
    experience_level: str | None = Field(
        default=None,
        pattern=r"^(beginner|intermediate|advanced)$",
    )
    goal: str | None = Field(default=None, max_length=500)


# ── Template schemas ──────────────────────────────────────────────────

class TemplateResponse(BaseModel):
    id: str
    name: str
    display_name: str
    description: str
    default_rounds: int
    default_round_duration_sec: int
    default_rest_sec: int
    combo_complexity_range: list[int]
    combo_include_defense: bool
    pace_interval_sec: list[int]

    model_config = {"from_attributes": True}
