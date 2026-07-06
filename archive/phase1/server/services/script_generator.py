"""Drill script generator — produces timed instruction lists from boxing.yaml combos.

Replaces the real-time DrillEngine state machine with a pre-computed
timed script for the Phase 2 async flow.
"""

from __future__ import annotations

import random
import uuid
from pathlib import Path

import yaml

from server.config import settings
from server.models.schemas import Instruction, ScriptResponse


# Display name mapping for Korean TTS audio cues
_COMBO_DISPLAY: dict[str, str] = {}
_COMBOS: list[dict] = []
_LEVEL_TIMING: dict[int, dict] = {}


def _load_config() -> None:
    """Load combo definitions and timing from boxing.yaml (cached)."""
    global _COMBO_DISPLAY, _COMBOS, _LEVEL_TIMING

    if _COMBOS:
        return  # already loaded

    with open(settings.config_path) as f:
        cfg = yaml.safe_load(f)

    drills = cfg.get("drills", {})
    tts = drills.get("tts", {})
    _COMBO_DISPLAY = tts.get("display_names", {})
    _COMBOS = drills.get("combos", [])

    timing = drills.get("timing", {})
    for i in range(1, 4):
        key = f"level_{i}"
        if key in timing:
            _LEVEL_TIMING[i] = timing[key]


def generate_script(level: int = 1, duration_seconds: int = 120) -> ScriptResponse:
    """Generate a timed drill script for the given level and duration.

    Logic:
    1. Filter combos by level (level 1=singles, 2=2-3 action, 3=3+ action)
    2. Walk forward in time placing instructions
    3. Return sorted instruction list with audio keys
    """
    _load_config()

    # Filter combos by level
    if level == 3:
        available = [c for c in _COMBOS if len(c["actions"]) >= 3]
    elif level == 2:
        available = [c for c in _COMBOS if 2 <= len(c["actions"]) <= 3]
    else:
        available = [c for c in _COMBOS if c.get("level", 1) == 1]

    if not available:
        available = [c for c in _COMBOS if c.get("level", 1) <= level]

    # Get timing params for this level
    timing = _LEVEL_TIMING.get(level, {})
    inter_pause = timing.get("inter_drill_pause", 2.5)
    reaction_window = timing.get("reaction_window", 3.0)

    # Estimate per-combo duration: reaction_window + small buffer
    combo_duration = reaction_window + 1.0

    instructions: list[Instruction] = []
    t = 0.0

    while t < duration_seconds:
        combo = random.choice(available)
        actions_str = ",".join(combo["actions"])

        # Build audio key: single actions use action name, combos use combo name
        if len(combo["actions"]) == 1:
            audio_key = combo["actions"][0]
        else:
            audio_key = combo["name"]

        # Build display text
        display_parts = []
        for act in combo["actions"]:
            display_parts.append(_COMBO_DISPLAY.get(act, act))
        display = "-".join(display_parts) + "!"

        instructions.append(Instruction(
            t=round(t, 1),
            type="attack",
            action=actions_str,
            audio_key=audio_key,
            display=display,
        ))

        # Advance time: combo execution + inter-drill pause + jitter
        jitter = random.uniform(-0.3, 0.5)
        t += combo_duration + inter_pause + jitter

    return ScriptResponse(
        script_id=uuid.uuid4(),
        instructions=instructions,
        level=level,
        total_instructions=len(instructions),
        estimated_duration_seconds=round(t, 1),
    )
