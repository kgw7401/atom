"""Drill engine — state machine for combo instruction, tracking, and evaluation."""

from __future__ import annotations

import random
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import yaml

from src.trainer import evaluator


class DrillState(Enum):
    IDLE = auto()
    ANNOUNCING = auto()
    WAITING = auto()
    EXECUTING = auto()
    GUARD_WATCH = auto()  # watching for guard return after combo
    EVALUATING = auto()


@dataclass
class ComboDefinition:
    name: str
    actions: list[str]
    level: int = 1


@dataclass
class DrillAttempt:
    combo: ComboDefinition
    expected_actions: list[str]
    recognized_actions: list[str] = field(default_factory=list)
    action_times: list[float] = field(default_factory=list)
    instruction_time: float = 0.0
    first_action_time: float = 0.0
    guard_returned: bool = False
    guard_return_time: float = 0.0


@dataclass
class DrillConfig:
    # Level-specific timing (switched at session start)
    level_timing: dict[int, dict[str, float]] = field(default_factory=dict)
    feedback_display: float = 2.0
    display_names: dict[str, str] = field(default_factory=dict)
    tts_voice: str = "Samantha"
    tts_rate: int = 200

    # Active timing (set based on current level)
    reaction_window: float = 3.0
    guard_return_window: float = 1.0
    combo_gap_max: float = 1.5
    inter_drill_pause: float = 2.0


class DrillEngine:
    """Manages the drill flow: instruction -> detection -> evaluation loop."""

    def __init__(self, config_path: str | Path = "configs/boxing.yaml"):
        self._load_config(config_path)
        self.state = DrillState.IDLE
        self.state_entered_at: float = time.time()

        self._drill_queue: list[ComboDefinition] = []
        self._current_combo: ComboDefinition | None = None
        self._current_attempt: DrillAttempt | None = None
        self._combo_pointer: int = 0
        self._last_recognized: str = ""  # dedup consecutive same predictions

        self._last_result: evaluator.DrillResult | None = None
        self._tts_thread: threading.Thread | None = None
        self._tts_busy: bool = False
        self._tts_started: bool = False  # track if we started TTS for current drill

        self.attempts: list[DrillAttempt] = []
        self.total_drills: int = 0
        self.completed_drills: int = 0
        self._session_active: bool = False

    def _load_config(self, config_path: str | Path) -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        drills = cfg.get("drills", {})
        timing = drills.get("timing", {})
        tts = drills.get("tts", {})

        # Load level-specific timing
        level_timing = {}
        for i in range(1, 4):  # levels 1, 2, 3
            key = f"level_{i}"
            if key in timing:
                level_timing[i] = timing[key]

        self.drill_config = DrillConfig(
            level_timing=level_timing,
            feedback_display=timing.get("feedback_display", 2.0),
            display_names=tts.get("display_names", {}),
            tts_voice=tts.get("voice", "Samantha"),
            tts_rate=tts.get("rate", 200),
        )

        self._all_combos: list[ComboDefinition] = []
        for c in drills.get("combos", []):
            self._all_combos.append(ComboDefinition(
                name=c["name"],
                actions=c["actions"],
                level=c.get("level", 1),
            ))

    def start_session(self, level: int = 1, count: int = 10) -> None:
        """Build a drill queue from combos at the given level, start session."""
        # Level 3: only 3+ action combos
        if level == 3:
            available = [c for c in self._all_combos if len(c.actions) >= 3]
        # Level 2: 2-3 action combos (exclude singles and very long combos)
        elif level == 2:
            available = [c for c in self._all_combos if 2 <= len(c.actions) <= 3]
        # Level 1: singles only
        else:
            available = [c for c in self._all_combos if c.level == 1]

        if not available:
            print(f"No combos available at level {level}")
            return

        # Apply level-specific timing
        if level in self.drill_config.level_timing:
            lvl_timing = self.drill_config.level_timing[level]
            self.drill_config.reaction_window = lvl_timing.get("reaction_window", 3.0)
            self.drill_config.guard_return_window = lvl_timing.get("guard_return_window", 1.0)
            self.drill_config.combo_gap_max = lvl_timing.get("combo_gap_max", 1.5)
            self.drill_config.inter_drill_pause = lvl_timing.get("inter_drill_pause", 2.0)

        self._drill_queue = random.choices(available, k=count)
        self.total_drills = count
        self.completed_drills = 0
        self.attempts = []
        self._session_active = True
        self.state = DrillState.IDLE
        self.state_entered_at = time.time()
        print(f"Session started: {count} drills at level {level}")
        print(f"  Timing: reaction={self.drill_config.reaction_window}s, "
              f"gap={self.drill_config.combo_gap_max}s, "
              f"pause={self.drill_config.inter_drill_pause}s")

    @property
    def session_active(self) -> bool:
        return self._session_active

    def update(self, prediction: str, confidence: float, timestamp: float) -> None:
        """Called every classification cycle with the latest prediction."""
        if not self._session_active:
            return

        if self.state == DrillState.IDLE:
            self._handle_idle(timestamp)
        elif self.state == DrillState.ANNOUNCING:
            self._handle_announcing(timestamp)
        elif self.state == DrillState.WAITING:
            self._handle_waiting(prediction, confidence, timestamp)
        elif self.state == DrillState.EXECUTING:
            self._handle_executing(prediction, confidence, timestamp)
        elif self.state == DrillState.GUARD_WATCH:
            self._handle_guard_watch(prediction, timestamp)
        elif self.state == DrillState.EVALUATING:
            self._handle_evaluating(timestamp)

    def get_overlay_info(self) -> dict[str, Any]:
        """Return current state for the visual overlay renderer."""
        info: dict[str, Any] = {
            "state": self.state,
            "instruction": None,
            "result": None,
            "combo_progress": (0, 0),
            "session_progress": (self.completed_drills, self.total_drills),
            "countdown": None,
            "session_active": self._session_active,
        }

        if self.state == DrillState.IDLE:
            elapsed = time.time() - self.state_entered_at
            remaining = max(0, self.drill_config.inter_drill_pause - elapsed)
            info["countdown"] = remaining

        elif self.state in (DrillState.ANNOUNCING, DrillState.WAITING,
                            DrillState.EXECUTING, DrillState.GUARD_WATCH):
            if self._current_combo:
                info["instruction"] = self._combo_display_text()
                completed = self._combo_pointer
                total = len(self._current_combo.actions)
                info["combo_progress"] = (completed, total)

        elif self.state == DrillState.EVALUATING:
            if self._last_result:
                info["result"] = {
                    "success": self._last_result.success,
                    "feedback_text": self._last_result.feedback_text,
                    "score": self._last_result.score,
                }

        return info

    # --- State handlers ---

    def _handle_idle(self, timestamp: float) -> None:
        elapsed = timestamp - self.state_entered_at
        if elapsed >= self.drill_config.inter_drill_pause:
            if self._drill_queue:
                self._current_combo = self._drill_queue.pop(0)
                self._transition_to(DrillState.ANNOUNCING, timestamp)
            else:
                self._session_active = False
                print(f"\nSession complete! {self.completed_drills} drills")
                self._print_session_summary()

    def _handle_announcing(self, timestamp: float) -> None:
        # Start TTS if not started yet
        if not self._tts_started:
            text = self._combo_instruction_text()
            self._speak(text)
            self._current_attempt = DrillAttempt(
                combo=self._current_combo,
                expected_actions=list(self._current_combo.actions),
                instruction_time=timestamp,
            )
            self._combo_pointer = 0
            self._last_recognized = ""
            self._tts_started = True
        # Wait for TTS to finish, then transition to WAITING
        elif not self._tts_busy:
            self._tts_started = False  # reset for next drill
            self._transition_to(DrillState.WAITING, timestamp)

    def _handle_waiting(self, prediction: str, confidence: float, timestamp: float) -> None:
        # Timeout
        elapsed = timestamp - self.state_entered_at
        if elapsed > self.drill_config.reaction_window:
            self._current_attempt.recognized_actions = []
            self._evaluate_and_transition(timestamp)
            return

        # Wait for first expected action
        expected = self._current_combo.actions[0]
        if prediction == expected and prediction != "guard":
            self._current_attempt.first_action_time = timestamp
            self._current_attempt.recognized_actions.append(prediction)
            self._current_attempt.action_times.append(timestamp)
            self._last_recognized = prediction
            self._combo_pointer = 1

            if self._combo_pointer >= len(self._current_combo.actions):
                self._transition_to(DrillState.GUARD_WATCH, timestamp)
            else:
                self._transition_to(DrillState.EXECUTING, timestamp)

    def _handle_executing(self, prediction: str, confidence: float, timestamp: float) -> None:
        # Gap timeout
        last_time = self._current_attempt.action_times[-1]
        if timestamp - last_time > self.drill_config.combo_gap_max:
            print(f"  [TIMEOUT] Gap {timestamp - last_time:.2f}s > {self.drill_config.combo_gap_max}s")
            self._transition_to(DrillState.GUARD_WATCH, timestamp)
            return

        expected = self._current_combo.actions[self._combo_pointer]

        # Debug: show what we're looking for vs what we got
        if prediction != "guard":  # reduce noise
            print(f"  [EXEC] pred={prediction} ({confidence:.0%}) | expect={expected} | last={self._last_recognized} | ptr={self._combo_pointer}")

        # Advance when we see the expected action and it's different from
        # the last recognized (prevents same prediction advancing twice).
        # For repeated same-type actions (jab-jab), user must go through
        # guard between them — the prediction must change then come back.
        if prediction == expected and prediction != self._last_recognized:
            self._current_attempt.recognized_actions.append(prediction)
            self._current_attempt.action_times.append(timestamp)
            self._last_recognized = prediction
            self._combo_pointer += 1
            print(f"  [✓] Recognized {prediction} | progress {self._combo_pointer}/{len(self._current_combo.actions)}")

            if self._combo_pointer >= len(self._current_combo.actions):
                self._transition_to(DrillState.GUARD_WATCH, timestamp)
        elif prediction != self._last_recognized:
            self._last_recognized = prediction

    def _handle_guard_watch(self, prediction: str, timestamp: float) -> None:
        """Watch for guard return after combo completion."""
        elapsed = timestamp - self.state_entered_at
        if prediction == "guard":
            self._current_attempt.guard_returned = True
            self._current_attempt.guard_return_time = timestamp
            self._evaluate_and_transition(timestamp)
        elif elapsed > self.drill_config.guard_return_window:
            self._current_attempt.guard_returned = False
            self._evaluate_and_transition(timestamp)

    def _handle_evaluating(self, timestamp: float) -> None:
        elapsed = timestamp - self.state_entered_at
        if elapsed >= self.drill_config.feedback_display:
            self._transition_to(DrillState.IDLE, timestamp)

    # --- Helpers ---

    def _transition_to(self, new_state: DrillState, timestamp: float) -> None:
        self.state = new_state
        self.state_entered_at = timestamp

    def _evaluate_and_transition(self, timestamp: float) -> None:
        self._last_result = evaluator.evaluate(
            self._current_attempt, self.drill_config,
        )
        self.attempts.append(self._current_attempt)
        self.completed_drills += 1

        status = "OK" if self._last_result.success else "X"
        print(f"  [{status}] {self._last_result.feedback_text}  (score: {self._last_result.score})")

        self._transition_to(DrillState.EVALUATING, timestamp)

    def _combo_instruction_text(self) -> str:
        """Build TTS instruction with pauses between actions for clarity."""
        # For single actions, just use the name
        if len(self._current_combo.actions) == 1:
            return self._current_combo.name + "!"

        # For combos, add commas between action names for clearer pronunciation
        # e.g., "원투훅" -> "원, 투, 훅!" (TTS will pause at commas)
        combo_name = self._current_combo.name

        # Split Korean combo names at known boundaries
        # This handles common patterns like 원투, 잽잽, 훅투, etc.
        parts = []
        buffer = ""
        for char in combo_name:
            buffer += char
            # Common action endings that mark boundaries
            if buffer in ['원', '투', '잽', '훅', '어퍼', '바디']:
                parts.append(buffer)
                buffer = ""
        if buffer:  # remaining characters
            parts.append(buffer)

        return ", ".join(parts) + "!"

    def _combo_display_text(self) -> str:
        """Build visual instruction, e.g. 'ONE TWO HOOK!'"""
        # Use combo name for consistency with TTS
        combo_name = self._current_combo.name.replace("-", " ").upper()
        return combo_name + "!"

    def _speak(self, text: str) -> None:
        """Non-blocking TTS using macOS say command."""
        def _run():
            self._tts_busy = True
            try:
                subprocess.run(
                    ["say", "-v", self.drill_config.tts_voice,
                     "-r", str(self.drill_config.tts_rate), text],
                    timeout=5,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            finally:
                self._tts_busy = False

        self._tts_thread = threading.Thread(target=_run, daemon=True)
        self._tts_thread.start()

    def _print_session_summary(self) -> None:
        if not self.attempts:
            return

        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)

        results = [evaluator.evaluate(a, self.drill_config) for a in self.attempts]

        # Detailed results for each drill
        for i, (attempt, result) in enumerate(zip(self.attempts, results), 1):
            status = "✓" if result.success else "✗"
            combo_name = attempt.combo.name

            print(f"\n[{i}] {combo_name} {status}")
            print(f"    Expected: {' → '.join(attempt.expected_actions)}")
            print(f"    Executed: {' → '.join(attempt.recognized_actions) if attempt.recognized_actions else '(none)'}")

            if result.reaction_time:
                print(f"    Reaction: {result.reaction_time:.2f}s")
            if result.total_time:
                print(f"    Total time: {result.total_time:.2f}s")

            print(f"    Score: {result.score}/100 - {result.feedback_text}")

        # Overall statistics
        avg_score = sum(r.score for r in results) / len(results)
        success_count = sum(1 for r in results if r.success)

        print("\n" + "-"*60)
        print(f"Overall: {success_count}/{len(results)} success | Avg score: {avg_score:.0f}/100")
        print("="*60 + "\n")
