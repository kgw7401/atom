"""B4 Task 15: Situational tactic extraction — prompt engineering.

Orchestrates fight analysis requests to Gemini and provides filtering
utilities for actionable tactics.
"""

from __future__ import annotations

from dataclasses import dataclass

from track_b.b2.tad import ActionTimeline
from track_b.b4.gemini_client import GeminiAnalysisClient, SituationalTactic


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class FightAnalysisRequest:
    """Input for fight tactical analysis."""

    video_id: str
    fighter_a_timeline: ActionTimeline
    fighter_b_timeline: ActionTimeline
    fighter_a_id: str = "fighter_a"
    fighter_b_id: str = "fighter_b"


@dataclass
class FightAnalysisResult:
    """Result of fight tactical analysis."""

    video_id: str
    tactics: list[SituationalTactic]

    @property
    def top_tactics(self) -> list[SituationalTactic]:
        """Return tactics sorted by frequency descending."""
        return sorted(self.tactics, key=lambda t: t.frequency, reverse=True)

    @property
    def high_confidence_tactics(self) -> list[SituationalTactic]:
        """Return tactics with success_rate >= 0.6."""
        return [t for t in self.tactics if t.success_rate >= 0.6]


# ── Analysis ──────────────────────────────────────────────────────────────────


def analyze_fight_tactics(
    request: FightAnalysisRequest,
    client: GeminiAnalysisClient,
) -> FightAnalysisResult:
    """Analyze a fight and extract situational tactics.

    Args:
        request: Fight analysis request with timelines.
        client: Gemini client for VLM analysis.

    Returns:
        FightAnalysisResult with extracted situational tactics.
    """
    tactics = client.analyze_fight(
        fighter_a_timeline=request.fighter_a_timeline,
        fighter_b_timeline=request.fighter_b_timeline,
        video_id=request.video_id,
    )
    return FightAnalysisResult(video_id=request.video_id, tactics=tactics)


def filter_actionable_tactics(
    tactics: list[SituationalTactic],
    min_frequency: int = 2,
    min_success_rate: float = 0.5,
) -> list[SituationalTactic]:
    """Filter tactics to only include actionable ones.

    Args:
        tactics: All extracted tactics.
        min_frequency: Minimum times observed (default 2).
        min_success_rate: Minimum estimated success rate (default 0.5).

    Returns:
        Filtered tactics sorted by frequency descending.
    """
    filtered = [
        t for t in tactics
        if t.frequency >= min_frequency and t.success_rate >= min_success_rate
    ]
    return sorted(filtered, key=lambda t: t.frequency, reverse=True)
