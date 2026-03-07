"""B4 Task 14: Gemini 2.5 Pro integration for fight footage analysis.

API client for sending ActionTimelines to Gemini for tactical analysis.
Tests inject a mock_client to avoid real API calls.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from track_b.b2.tad import ActionTimeline


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class SituationalTactic:
    """A recurring tactical pattern extracted from fight footage."""

    video_id: str
    situation: str           # e.g., "Opponent throws jab-cross"
    effective_response: str  # e.g., "Slip outside, counter with cross-hook"
    frequency: int           # times observed
    success_rate: float      # VLM-estimated (approximate)
    evidence: list[dict] = field(default_factory=list)  # [{timestamp, outcome}]


# ── Prompt construction ───────────────────────────────────────────────────────


def _timeline_to_text(timeline: ActionTimeline) -> str:
    """Format ActionTimeline as text for Gemini prompt."""
    actions = timeline.sorted_actions()
    if not actions:
        return f"Fighter {timeline.fighter_id}: No actions detected."

    lines = [f"Fighter {timeline.fighter_id} actions:"]
    for a in actions:
        lines.append(
            f"  t={a.start_time:.1f}s: {a.action_class} (conf={a.confidence:.2f})"
        )
    return "\n".join(lines)


def build_analysis_prompt(
    fighter_a_timeline: ActionTimeline,
    fighter_b_timeline: ActionTimeline,
) -> str:
    """Build the tactical analysis prompt for Gemini.

    Args:
        fighter_a_timeline: ActionTimeline for fighter A.
        fighter_b_timeline: ActionTimeline for fighter B.

    Returns:
        Formatted prompt string.
    """
    timeline_a_text = _timeline_to_text(fighter_a_timeline)
    timeline_b_text = _timeline_to_text(fighter_b_timeline)

    return f"""You are a boxing tactical analyst. Analyze the following fight data and identify recurring tactical patterns.

{timeline_a_text}

{timeline_b_text}

Identify situational tactics: when Fighter A does X, Fighter B responds with Y, and the estimated outcome.
Return ONLY a JSON array of tactical patterns. Each element must have:
- "situation": string describing what triggered the response (e.g., "Opponent throws jab-cross")
- "effective_response": string describing the effective counter (e.g., "Slip outside, counter with cross-hook")
- "frequency": integer — how many times this pattern was observed
- "success_rate": float 0.0-1.0 — estimated effectiveness (approximate)
- "evidence": array of {{"timestamp": float, "outcome": string}}

Focus on actionable patterns that a boxer can train. Return at minimum 1 pattern if any patterns exist.
If no clear patterns exist, return an empty array [].

Return only valid JSON, no other text."""


# ── Response parsing ──────────────────────────────────────────────────────────


def parse_tactics_response(
    response_text: str,
    video_id: str,
) -> list[SituationalTactic]:
    """Parse Gemini's JSON response into SituationalTactic objects.

    Handles markdown code blocks if present. Returns empty list on parse error.

    Args:
        response_text: Raw text response from Gemini.
        video_id: Source video identifier to attach to each tactic.

    Returns:
        List of SituationalTactic objects.
    """
    text = response_text.strip()

    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        end = len(lines) - 1 if lines[-1].strip().startswith("```") else len(lines)
        text = "\n".join(lines[1:end])

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    tactics = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            tactic = SituationalTactic(
                video_id=video_id,
                situation=str(item.get("situation", "")),
                effective_response=str(item.get("effective_response", "")),
                frequency=int(item.get("frequency", 1)),
                success_rate=float(item.get("success_rate", 0.5)),
                evidence=list(item.get("evidence", [])),
            )
            tactics.append(tactic)
        except (KeyError, TypeError, ValueError):
            continue

    return tactics


# ── API client ────────────────────────────────────────────────────────────────


class GeminiAnalysisClient:
    """Client for Gemini 2.5 Pro fight footage analysis.

    Tests should inject a mock_client to avoid real API calls.
    """

    MODEL_ID = "gemini-2.5-pro"

    def __init__(self, api_key: str | None = None, mock_client: Any = None) -> None:
        """
        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
            mock_client: Optional mock for testing (avoids real API calls).
                         Must implement generate(prompt: str) -> str.
        """
        self._mock_client = mock_client
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        """Initialize Gemini client lazily."""
        if self._mock_client is not None:
            return self._mock_client

        if self._client is None:
            try:
                import google.generativeai as genai  # noqa: PLC0415
                key = self._api_key or os.environ.get("GEMINI_API_KEY", "")
                genai.configure(api_key=key)
                self._client = genai.GenerativeModel(self.MODEL_ID)
            except ImportError as exc:
                raise ImportError(
                    "google-generativeai package required for B4 fight analysis. "
                    "Install with: pip install google-generativeai"
                ) from exc

        return self._client

    def analyze_fight(
        self,
        fighter_a_timeline: ActionTimeline,
        fighter_b_timeline: ActionTimeline,
        video_id: str,
    ) -> list[SituationalTactic]:
        """Send timelines to Gemini for tactical analysis.

        Args:
            fighter_a_timeline: ActionTimeline for fighter A.
            fighter_b_timeline: ActionTimeline for fighter B.
            video_id: Source video identifier.

        Returns:
            List of SituationalTactic patterns. May be empty if no patterns found.
        """
        prompt = build_analysis_prompt(fighter_a_timeline, fighter_b_timeline)
        client = self._get_client()

        if self._mock_client is not None:
            response_text = client.generate(prompt)
        else:
            response = client.generate_content(prompt)
            response_text = response.text

        return parse_tactics_response(response_text, video_id)
