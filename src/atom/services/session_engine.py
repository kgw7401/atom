"""Session engine — timer-based drill execution with TTS and delivery logging."""

from __future__ import annotations

import asyncio
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import SessionLog


class State(str, Enum):
    IDLE = "idle"
    ROUND_ACTIVE = "round_active"
    REST_PERIOD = "rest_period"
    SESSION_END = "session_end"


@dataclass
class SessionEvent:
    """A single event in the delivery log."""

    type: str
    ts: float
    round: int | None = None
    combo_display_name: str | None = None
    actions: list[str] | None = None
    reason: str | None = None
    # Protocol fields (sent to mobile client)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        # Map internal type names → WebSocket protocol names
        protocol_type = "combo_call" if self.type == "combo_called" else self.type
        d: dict[str, Any] = {"type": protocol_type, "ts": round(self.ts, 1)}
        if self.round is not None:
            d["round"] = self.round
        if self.combo_display_name is not None:
            d["name"] = self.combo_display_name  # protocol uses "name"
        if self.actions is not None:
            d["actions"] = self.actions
        if self.reason is not None:
            d["reason"] = self.reason
        d.update(self.extra)
        return d


@dataclass
class SessionEngine:
    """Timer-based drill session runner.

    Executes a drill plan in real-time with terminal output and TTS.
    Records all events for the delivery log.
    """

    plan: dict
    plan_id: str
    tts_enabled: bool = True
    voice: str = "Yuna"
    on_output: Callable[[str], None] | None = None  # terminal output callback
    on_event: Callable[[dict], None] | None = None  # structured event callback

    # Internal state
    state: State = field(default=State.IDLE, init=False)
    events: list[SessionEvent] = field(default_factory=list, init=False)
    started_at: datetime | None = field(default=None, init=False)
    rounds_completed: int = field(default=0, init=False)
    combos_delivered: int = field(default=0, init=False)
    _aborted: bool = field(default=False, init=False)
    _session_start_time: float = field(default=0.0, init=False)
    _tts_process: subprocess.Popen | None = field(default=None, init=False)

    def _elapsed(self) -> float:
        return time.monotonic() - self._session_start_time

    def _emit(self, msg: str) -> None:
        if self.on_output:
            self.on_output(msg)

    def _record(self, event: SessionEvent) -> None:
        self.events.append(event)
        if self.on_event:
            self.on_event(event.to_dict())

    @staticmethod
    def _actions_to_speech(actions: list[str]) -> str:
        """Convert snake_case action names to natural English for TTS."""
        return ", ".join(a.replace("_", " ") for a in actions)

    def _speak_combo(self, actions: list[str]) -> None:
        """Speak each action with a built-in 400ms pause between them.

        Uses macOS say's [[slnc N]] command — no asyncio.sleep needed,
        so the timing loop is never disrupted.
        """
        # [[slnc 400]] = 400ms silence (macOS say built-in command)
        pause = " [[slnc 300]] "
        text = pause.join(a.replace("_", " ") for a in actions)
        self._speak(text)

    def _speak(self, text: str) -> None:
        """Non-blocking macOS TTS via `say` command."""
        if not self.tts_enabled:
            return
        try:
            # Kill any ongoing speech
            if self._tts_process and self._tts_process.poll() is None:
                self._tts_process.terminate()
            self._tts_process = subprocess.Popen(
                ["say", "-v", self.voice, "-r", "220", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass  # `say` not available (non-macOS)

    def abort(self) -> None:
        """Signal the engine to stop after the current sleep."""
        self._aborted = True

    async def run(self) -> dict:
        """Execute the full session plan. Returns delivery log dict.

        Can be interrupted by calling abort() (e.g., from a signal handler).
        """
        rounds = self.plan["rounds"]
        total_rounds = len(rounds)

        self.state = State.ROUND_ACTIVE
        self.started_at = datetime.now(timezone.utc)
        self._session_start_time = time.monotonic()

        template = self.plan.get("template", "unknown")
        focus = self.plan.get("focus", "")
        self._emit(f"\n{'='*50}")
        self._emit(f"  Session: {focus} ({template})")
        self._emit(f"  {total_rounds} rounds")
        self._emit(f"{'='*50}\n")

        try:
            for rnd in rounds:
                if self._aborted:
                    break
                await self._run_round(rnd, total_rounds)
                if self._aborted:
                    break
        except asyncio.CancelledError:
            self._aborted = True

        # Session end
        self.state = State.SESSION_END
        elapsed = self._elapsed()
        status = "abandoned" if self._aborted else "completed"
        self._record(SessionEvent(
            type="session_end",
            ts=elapsed,
            reason=status,
            extra={
                "status": status,
                "rounds": self.rounds_completed,
                "combos": self.combos_delivered,
                "duration_sec_end": round(elapsed, 1),
            },
        ))

        if self._aborted:
            self._emit(f"\n  Session aborted. ({self.rounds_completed}/{total_rounds} rounds)")
        else:
            self._emit(f"\n{'='*50}")
            self._emit(f"  Session complete!")
            self._emit(f"  {self.combos_delivered} combos in {self.rounds_completed} rounds")
            self._emit(f"  Duration: {elapsed:.0f}s")
            self._emit(f"{'='*50}")

        # Cleanup TTS
        if self._tts_process and self._tts_process.poll() is None:
            self._tts_process.terminate()

        return {"events": [e.to_dict() for e in self.events]}

    async def _run_round(self, rnd: dict, total_rounds: int) -> None:
        """Execute a single round."""
        round_num = rnd["round_number"]
        duration = rnd["duration_seconds"]
        rest_duration = rnd["rest_after_seconds"]
        instructions = rnd["instructions"]

        # Round start
        self.state = State.ROUND_ACTIVE
        round_start_time = self._elapsed()
        self._record(SessionEvent(
            type="round_start",
            ts=round_start_time,
            round=round_num,
            extra={"total": total_rounds, "duration_sec": duration},
        ))

        self._emit(f"  Round {round_num}/{total_rounds}  ({duration}s)")
        self._emit(f"  {'-'*30}")
        self._speak(f"Round {round_num}")

        # Interval-based delivery: speak → await TTS → throw_time + base_interval
        # throw_time scales with combo length so longer combos get more recovery time
        pmin, pmax = self.plan.get("pace_interval_sec", [2, 4])
        THROW_TIME_PER_ACTION = 0.4  # seconds per action to shadow-box the combo

        round_mono_start = time.monotonic()

        # Initial lead-in before first combo
        try:
            await asyncio.sleep(3.0)
        except asyncio.CancelledError:
            self._aborted = True
            return

        for instr in instructions:
            if self._aborted:
                return

            # Check if round time has expired
            if time.monotonic() - round_mono_start >= duration:
                break

            # Deliver the combo
            name = instr["combo_display_name"]
            actions = instr["actions"]
            actions_str = " → ".join(actions)
            self._record(SessionEvent(
                type="combo_called",
                ts=self._elapsed(),
                round=round_num,
                combo_display_name=name,
                actions=actions,
            ))
            self.combos_delivered += 1

            self._emit(f"    {name}  ({actions_str})")
            self._speak(name)

            # Wait for TTS to finish
            if self._tts_process:
                try:
                    await asyncio.to_thread(self._tts_process.wait)
                except asyncio.CancelledError:
                    self._aborted = True
                    return

            if self._aborted:
                return

            # Wait: time to throw the combo + base rest interval
            throw_time = len(actions) * THROW_TIME_PER_ACTION
            interval = random.uniform(pmin, pmax) + throw_time
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                self._aborted = True
                return

        # Wait for round to end
        elapsed_in_round = time.monotonic() - round_mono_start
        remaining = duration - elapsed_in_round
        if remaining > 0 and not self._aborted:
            try:
                await asyncio.sleep(remaining)
            except asyncio.CancelledError:
                self._aborted = True
                return

        # Round end
        self._record(SessionEvent(type="round_end", ts=self._elapsed(), round=round_num))
        self.rounds_completed += 1
        self._emit(f"  Round {round_num} end\n")

        if self._aborted:
            return

        # Rest period (skip after last round)
        if round_num < total_rounds and rest_duration > 0:
            self.state = State.REST_PERIOD
            self._record(SessionEvent(
                type="rest_start",
                ts=self._elapsed(),
                round=round_num,
                extra={"rest_sec": rest_duration},
            ))
            self._emit(f"  Rest: {rest_duration}s")
            self._speak("Rest")

            try:
                await asyncio.sleep(rest_duration)
            except asyncio.CancelledError:
                self._aborted = True
                return

            self._record(SessionEvent(type="rest_end", ts=self._elapsed(), round=round_num))
            self._emit("")

    async def save_log(self, db_session: AsyncSession) -> SessionLog:
        """Save the session log to the database."""
        status = "abandoned" if self._aborted else "completed"
        elapsed = self._elapsed()

        log = SessionLog(
            drill_plan_id=self.plan_id,
            template_name=self.plan.get("template", "unknown"),
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            total_duration_sec=elapsed,
            rounds_completed=self.rounds_completed,
            rounds_total=len(self.plan["rounds"]),
            combos_delivered=self.combos_delivered,
            delivery_log_json={"events": [e.to_dict() for e in self.events]},
            status=status,
        )
        db_session.add(log)
        await db_session.commit()
        await db_session.refresh(log)
        return log
