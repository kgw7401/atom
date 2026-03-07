"""B3 Task 10: Combo sequence recognition.

Groups ActionTimeline entries into ComboSequences using a time-gap heuristic.
Actions within gap_threshold seconds = same combo.
Gap >= gap_threshold = new combo boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from track_b.b2.tad import ActionTimeline


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class ComboInstance:
    """A single detected combo instance."""

    actions: list[str]    # e.g., ["jab", "cross", "lead_hook"]
    start_time: float     # start time of first action (seconds)
    end_time: float       # end time of last action (seconds)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def length(self) -> int:
        return len(self.actions)


@dataclass
class ComboSequence:
    """Grouped combo instances for one fighter in one video."""

    video_id: str
    fighter_id: str
    combos: list[ComboInstance] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.combos)

    def filter_by_length(self, min_length: int) -> list[ComboInstance]:
        """Return combos with at least min_length actions."""
        return [c for c in self.combos if c.length >= min_length]


# ── Grouping ──────────────────────────────────────────────────────────────────


def group_into_combos(
    timeline: ActionTimeline,
    gap_threshold: float = 0.8,
) -> ComboSequence:
    """Group ActionTimeline actions into ComboSequences.

    Actions are grouped by time gap: if the gap between the end of the last
    action and the start of the next action is < gap_threshold seconds, they
    belong to the same combo. A gap >= gap_threshold starts a new combo.

    Args:
        timeline: ActionTimeline with detected actions.
        gap_threshold: Minimum gap (seconds) to split combos. Default 0.8s.

    Returns:
        ComboSequence with grouped ComboInstances sorted by start_time.
    """
    actions = sorted(timeline.actions, key=lambda a: a.start_time)

    if not actions:
        return ComboSequence(
            video_id=timeline.video_id,
            fighter_id=timeline.fighter_id,
            combos=[],
        )

    combos: list[ComboInstance] = []
    current_group = [actions[0]]

    for action in actions[1:]:
        gap = action.start_time - current_group[-1].end_time
        if gap >= gap_threshold:
            combos.append(ComboInstance(
                actions=[a.action_class for a in current_group],
                start_time=current_group[0].start_time,
                end_time=current_group[-1].end_time,
            ))
            current_group = [action]
        else:
            current_group.append(action)

    # Flush final group
    combos.append(ComboInstance(
        actions=[a.action_class for a in current_group],
        start_time=current_group[0].start_time,
        end_time=current_group[-1].end_time,
    ))

    return ComboSequence(
        video_id=timeline.video_id,
        fighter_id=timeline.fighter_id,
        combos=combos,
    )
