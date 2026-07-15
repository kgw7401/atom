"""Ground-truth-pose temporal punch detection data contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from atom.boxingweb import Split, build_oracle_index
from atom.pose_features import extract_match_pose_features


CHANNELS = ("red_active", "blue_active", "red_start", "blue_start", "red_end", "blue_end")


@dataclass(frozen=True)
class PunchEvent:
    side: str
    start_frame: int
    end_frame: int


@dataclass
class TemporalMatch:
    split: Split
    match_id: str
    fps: float
    video_path: Path
    pose_path: Path
    features: np.ndarray
    targets: np.ndarray
    events: tuple[PunchEvent, ...]


def build_temporal_matches(data_root: Path, split: Split) -> list[TemporalMatch]:
    """Create full-round pose timelines and per-frame event targets.

    The oracle index supplies the same valid-event policy used by attribute
    classification.  Unlike the classifier, the detector receives every frame
    in each match, including background frames.
    """

    root = data_root.expanduser().resolve()
    samples = build_oracle_index(root, split).samples
    grouped: dict[str, list] = {}
    for sample in samples:
        grouped.setdefault(sample.match_id, []).append(sample)

    matches: list[TemporalMatch] = []
    for match_id, match_samples in grouped.items():
        features = extract_match_pose_features(root / match_samples[0].pose_path)
        targets = np.zeros((features.shape[0], len(CHANNELS)), dtype=np.float32)
        events: list[PunchEvent] = []
        for sample in match_samples:
            start, end = sample.event_start_frame, sample.event_end_frame
            if end >= features.shape[0]:
                continue
            side_offset = 0 if sample.labels["side"] == "red" else 1
            targets[start:end + 1, side_offset] = 1.0
            targets[start, side_offset + 2] = 1.0
            targets[end, side_offset + 4] = 1.0
            events.append(PunchEvent(sample.labels["side"], start, end))
        matches.append(TemporalMatch(
            split,
            match_id,
            match_samples[0].fps,
            root / match_samples[0].video_path,
            root / match_samples[0].pose_path,
            features,
            targets,
            tuple(events),
        ))
    return matches
