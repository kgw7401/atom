"""Coaching report generator: ΔS_t + weaknesses → structured report.

Template-only MVP (no LLM). Generates Korean coaching text from
state deltas and weakness detection using configs/prompts/templates.yaml.

Reference: spec/roadmap.md Phase 2e
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import yaml

from src.policy.weakness import Weakness, detect_weaknesses
from src.state.constants import DIM_GROUPS, DIM_NAMES, EPSILON, NUM_DIMS
from src.state.types import StateVector

# ---------------------------------------------------------------------------
# Templates (loaded once)
# ---------------------------------------------------------------------------

_TEMPLATES: Optional[dict] = None


def _load_templates() -> dict:
    global _TEMPLATES
    if _TEMPLATES is None:
        path = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts" / "templates.yaml"
        with open(path, encoding="utf-8") as f:
            _TEMPLATES = yaml.safe_load(f)
    return _TEMPLATES


def _dim_label(dim_name: str) -> str:
    """Get Korean label for a dimension name."""
    t = _load_templates()
    return t["dim_labels"].get(dim_name, dim_name)


def _group_label(group_name: str) -> str:
    """Get Korean label for a group name."""
    t = _load_templates()
    return t["group_labels"].get(group_name, group_name)


def _dim_to_group(dim_index: int) -> str:
    """Map a dimension index to its group name."""
    for group_name, indices in DIM_GROUPS.items():
        if dim_index in indices:
            return group_name
    raise ValueError(f"Dimension {dim_index} not found in any group")


# ---------------------------------------------------------------------------
# Delta classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaItem:
    """A classified state change for a single dimension."""

    dim_index: int
    dim_name: str
    dim_label: str
    group: str
    delta: float
    current_value: float
    description: str


@dataclass(frozen=True)
class WeaknessItem:
    """A weakness with coaching hint."""

    dim_index: int
    dim_name: str
    dim_label: str
    group: str
    value: float
    threshold: float
    confidence: float
    coaching_hint: str


@dataclass(frozen=True)
class Coaching:
    """Coaching text bundle."""

    summary: str
    focus_areas: list
    next_session_hint: str


@dataclass
class SessionReport:
    """Full session coaching report."""

    session_summary: dict
    state_delta: dict  # {improved: [], regressed: [], unchanged: []}
    weaknesses: list
    coaching: dict  # {summary, focus_areas, next_session_hint}


@dataclass
class ProgressReport:
    """Multi-session progress report."""

    user_id: str
    num_sessions: int
    trending_up: list
    trending_down: list
    plateau: list
    coaching: dict


def _classify_delta(
    delta: np.ndarray,
    mask: np.ndarray,
    current_values: np.ndarray,
    epsilon: float = EPSILON,
) -> tuple[list[DeltaItem], list[DeltaItem], list[DeltaItem]]:
    """Classify each observed dimension's delta as improved/regressed/unchanged."""
    t = _load_templates()
    improved: list[DeltaItem] = []
    regressed: list[DeltaItem] = []
    unchanged: list[DeltaItem] = []

    for i in range(NUM_DIMS):
        if not mask[i]:
            continue

        d = float(delta[i])
        v = float(current_values[i])
        name = DIM_NAMES[i]
        label = _dim_label(name)
        group = _dim_to_group(i)
        delta_pct = round(abs(d) * 100, 1)
        value_pct = round(v * 100, 1)

        if d > epsilon:
            desc = t["improved_templates"][0].format(
                dim_label=label, delta_pct=delta_pct, value_pct=value_pct,
            )
            improved.append(DeltaItem(i, name, label, group, round(d, 4), round(v, 4), desc))
        elif d < -epsilon:
            desc = t["regressed_templates"][0].format(
                dim_label=label, delta_pct=delta_pct, value_pct=value_pct,
            )
            regressed.append(DeltaItem(i, name, label, group, round(d, 4), round(v, 4), desc))
        else:
            unchanged.append(DeltaItem(i, name, label, group, round(d, 4), round(v, 4), ""))

    return improved, regressed, unchanged


def _build_weakness_items(weaknesses: list[Weakness]) -> list[WeaknessItem]:
    """Convert Weakness objects to WeaknessItems with coaching hints."""
    t = _load_templates()
    items = []
    for w in weaknesses:
        label = _dim_label(w.dim_name)
        template = t["weakness_coaching"].get(w.group, "{dim_label} 개선이 필요합니다.")
        hint = template.format(dim_label=label)
        items.append(WeaknessItem(
            dim_index=w.dim_index,
            dim_name=w.dim_name,
            dim_label=label,
            group=w.group,
            value=round(w.value, 4),
            threshold=w.threshold,
            confidence=round(w.confidence, 4),
            coaching_hint=hint,
        ))
    return items


def _build_coaching(
    improved: list[DeltaItem],
    regressed: list[DeltaItem],
    weakness_items: list[WeaknessItem],
) -> dict:
    """Build coaching text from classified deltas and weaknesses."""
    t = _load_templates()

    # Summary
    n_improved = len(improved)
    n_regressed = len(regressed)

    if n_improved > 0 and n_regressed == 0:
        summary = t["summary"]["all_improved"]
    elif n_improved == 0 and n_regressed > 0:
        summary = t["summary"]["all_regressed"]
    elif n_improved == 0 and n_regressed == 0:
        summary = t["summary"]["no_change"]
    else:
        summary = t["summary"]["mixed"].format(
            n_improved=n_improved, n_regressed=n_regressed,
        )

    # Focus areas: coaching hints from weaknesses
    focus_areas = [w.coaching_hint for w in weakness_items]

    # Next session hint
    if weakness_items:
        top_label = weakness_items[0].dim_label
        next_hint = t["next_session"]["has_weakness"].format(top_weakness=top_label)
    else:
        next_hint = t["next_session"]["no_weakness"]

    return {
        "summary": summary,
        "focus_areas": focus_areas,
        "next_session_hint": next_hint,
    }


def _delta_item_to_dict(item: DeltaItem) -> dict:
    return {
        "dim_index": item.dim_index,
        "dim_name": item.dim_name,
        "dim_label": item.dim_label,
        "group": item.group,
        "delta": item.delta,
        "current_value": item.current_value,
        "description": item.description,
    }


def _weakness_item_to_dict(item: WeaknessItem) -> dict:
    return {
        "dim_index": item.dim_index,
        "dim_name": item.dim_name,
        "dim_label": item.dim_label,
        "group": item.group,
        "value": item.value,
        "threshold": item.threshold,
        "confidence": item.confidence,
        "coaching_hint": item.coaching_hint,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_session_report(
    session_id: str,
    delta: np.ndarray,
    mask: np.ndarray,
    current_state: StateVector,
    mode: str = "shadow",
    duration_seconds: float = 0.0,
) -> SessionReport:
    """Generate a coaching report for a single session.

    Args:
        session_id: Session identifier.
        delta: ΔS array (18 dims).
        mask: Observation mask (18 bools).
        current_state: Current state vector (after update).
        mode: Session mode (shadow/heavy_bag/ai_session).
        duration_seconds: Session duration.

    Returns:
        SessionReport with coaching text.
    """
    # Classify deltas
    improved, regressed, unchanged = _classify_delta(
        delta, mask, current_state.values,
    )

    # Detect weaknesses from current state
    weaknesses = detect_weaknesses(current_state)
    weakness_items = _build_weakness_items(weaknesses)

    # Build coaching
    coaching = _build_coaching(improved, regressed, weakness_items)

    # Count observed punches from mask (rough: observed offensive dims)
    observed_count = int(np.sum(mask))

    return SessionReport(
        session_summary={
            "session_id": session_id,
            "mode": mode,
            "duration_seconds": duration_seconds,
            "observed_dims": observed_count,
        },
        state_delta={
            "improved": [_delta_item_to_dict(x) for x in improved],
            "regressed": [_delta_item_to_dict(x) for x in regressed],
            "unchanged": [_delta_item_to_dict(x) for x in unchanged],
        },
        weaknesses=[_weakness_item_to_dict(x) for x in weakness_items],
        coaching=coaching,
    )


def generate_progress_report(
    user_id: str,
    deltas: list[np.ndarray],
    masks: list[np.ndarray],
    current_state: StateVector,
) -> ProgressReport:
    """Generate a multi-session progress report.

    Analyzes trends across recent sessions to identify consistently
    improving, declining, or plateaued dimensions.

    Args:
        user_id: User identifier.
        deltas: List of delta arrays from recent sessions (chronological).
        masks: List of observation masks from recent sessions.
        current_state: Current state vector.

    Returns:
        ProgressReport with trend analysis and coaching.
    """
    t = _load_templates()
    n_sessions = len(deltas)

    if n_sessions == 0:
        return ProgressReport(
            user_id=user_id,
            num_sessions=0,
            trending_up=[],
            trending_down=[],
            plateau=[],
            coaching={
                "summary": "아직 분석할 세션이 없습니다.",
                "focus_areas": [],
                "next_session_hint": "첫 세션을 시작해보세요!",
            },
        )

    # Accumulate per-dim trend: count how many sessions each dim improved/regressed
    up_count = np.zeros(NUM_DIMS, dtype=int)
    down_count = np.zeros(NUM_DIMS, dtype=int)
    observed_count = np.zeros(NUM_DIMS, dtype=int)

    for delta, mask in zip(deltas, masks):
        for i in range(NUM_DIMS):
            if mask[i]:
                observed_count[i] += 1
                if delta[i] > EPSILON:
                    up_count[i] += 1
                elif delta[i] < -EPSILON:
                    down_count[i] += 1

    # Classify trends: >50% of observed sessions in same direction
    trending_up = []
    trending_down = []
    plateau = []

    for i in range(NUM_DIMS):
        if observed_count[i] < 2:
            continue  # not enough data

        name = DIM_NAMES[i]
        label = _dim_label(name)
        group = _dim_to_group(i)
        v = round(float(current_state.values[i]), 4)

        ratio_up = up_count[i] / observed_count[i]
        ratio_down = down_count[i] / observed_count[i]

        if ratio_up > 0.5:
            desc = t["progress"]["trending_up"].format(
                dim_label=label, n_sessions=int(observed_count[i]),
            )
            trending_up.append(_delta_item_to_dict(
                DeltaItem(i, name, label, group, 0.0, v, desc)
            ))
        elif ratio_down > 0.5:
            desc = t["progress"]["trending_down"].format(
                dim_label=label, n_sessions=int(observed_count[i]),
            )
            trending_down.append(_delta_item_to_dict(
                DeltaItem(i, name, label, group, 0.0, v, desc)
            ))
        else:
            desc = t["progress"]["plateau"].format(dim_label=label)
            plateau.append(name)

    # Overall coaching
    n_up = len(trending_up)
    n_down = len(trending_down)

    if n_up > 0 and n_down == 0:
        summary = t["progress"]["overall_improving"]
    elif n_down > 0 and n_up == 0:
        summary = t["progress"]["overall_declining"]
    else:
        summary = t["progress"]["overall_mixed"].format(n_up=n_up, n_down=n_down)

    # Focus on weaknesses for next session
    weaknesses = detect_weaknesses(current_state)
    weakness_items = _build_weakness_items(weaknesses)
    focus_areas = [w.coaching_hint for w in weakness_items[:3]]

    if weakness_items:
        next_hint = t["next_session"]["has_weakness"].format(
            top_weakness=weakness_items[0].dim_label,
        )
    else:
        next_hint = t["next_session"]["no_weakness"]

    return ProgressReport(
        user_id=user_id,
        num_sessions=n_sessions,
        trending_up=trending_up,
        trending_down=trending_down,
        plateau=plateau,
        coaching={
            "summary": summary,
            "focus_areas": focus_areas,
            "next_session_hint": next_hint,
        },
    )
