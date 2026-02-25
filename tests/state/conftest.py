"""Shared test fixtures for state engine tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.state.constants import (
    ALL_PUNCH_CLASSES,
    CLASS_CROSS,
    CLASS_GUARD,
    CLASS_JAB,
    CLASS_LEAD_BODYSHOT,
    CLASS_LEAD_HOOK,
    CLASS_LEAD_UPPERCUT,
    CLASS_REAR_BODYSHOT,
    CLASS_REAR_HOOK,
    CLASS_REAR_UPPERCUT,
    PUNCH_NAMES,
)
from src.state.types import ActionSegment, DefensiveCommand, KeypointFrame


def make_segment(class_id: int, t_start: float, t_end: float) -> ActionSegment:
    """Create an ActionSegment with the correct class_name."""
    names = ["guard"] + list(PUNCH_NAMES)
    return ActionSegment(
        class_id=class_id,
        class_name=names[class_id],
        t_start=t_start,
        t_end=t_end,
    )


def make_keypoint_frame(timestamp: float, seed: int = 0) -> KeypointFrame:
    """Create a KeypointFrame with deterministic pseudo-random keypoints.

    Keypoints are in normalized coordinates [0, 1].
    Layout simulates an upper-body pose with guard position.
    """
    rng = np.random.RandomState(seed + int(timestamp * 100))

    # Base pose: nose at center-top, shoulders below, wrists near chin (guard)
    kp = np.array([
        [0.50, 0.20],  # 0: nose
        [0.45, 0.22],  # 1: left_ear
        [0.55, 0.22],  # 2: right_ear
        [0.35, 0.40],  # 3: left_shoulder
        [0.65, 0.40],  # 4: right_shoulder
        [0.38, 0.32],  # 5: left_elbow
        [0.62, 0.32],  # 6: right_elbow
        [0.45, 0.25],  # 7: left_wrist (near chin = guard)
        [0.55, 0.25],  # 8: right_wrist (near chin = guard)
        [0.40, 0.65],  # 9: left_hip
        [0.60, 0.65],  # 10: right_hip
    ], dtype=np.float64)

    # Add small noise
    kp += rng.randn(11, 2) * 0.02
    return KeypointFrame(timestamp=timestamp, keypoints=kp)


def make_punching_frame(timestamp: float, is_lead: bool = True) -> KeypointFrame:
    """Create a frame where one arm is extended (punching)."""
    kp = np.array([
        [0.50, 0.20],  # nose
        [0.45, 0.22],  # left_ear
        [0.55, 0.22],  # right_ear
        [0.35, 0.40],  # left_shoulder
        [0.65, 0.40],  # right_shoulder
        [0.30, 0.30] if is_lead else [0.62, 0.32],  # left_elbow
        [0.62, 0.32] if is_lead else [0.70, 0.30],  # right_elbow
        [0.20, 0.20] if is_lead else [0.55, 0.25],  # left_wrist (extended if lead)
        [0.55, 0.25] if is_lead else [0.80, 0.20],  # right_wrist (extended if rear)
        [0.40, 0.65],  # left_hip
        [0.60, 0.65],  # right_hip
    ], dtype=np.float64)
    return KeypointFrame(timestamp=timestamp, keypoints=kp)


@pytest.fixture
def diverse_segments() -> list[ActionSegment]:
    """120-second session with diverse punches and guard segments."""
    segs = []
    t = 0.0
    classes = [
        CLASS_JAB, CLASS_GUARD, CLASS_CROSS, CLASS_GUARD,
        CLASS_LEAD_HOOK, CLASS_GUARD, CLASS_REAR_HOOK, CLASS_GUARD,
        CLASS_LEAD_UPPERCUT, CLASS_GUARD, CLASS_REAR_UPPERCUT, CLASS_GUARD,
        CLASS_LEAD_BODYSHOT, CLASS_GUARD, CLASS_REAR_BODYSHOT, CLASS_GUARD,
        CLASS_JAB, CLASS_CROSS, CLASS_GUARD,
        CLASS_JAB, CLASS_LEAD_HOOK, CLASS_GUARD,
        CLASS_CROSS, CLASS_REAR_HOOK, CLASS_GUARD,
        CLASS_JAB, CLASS_CROSS, CLASS_LEAD_HOOK, CLASS_GUARD,
    ]
    for cls_id in classes:
        duration = 0.4 if cls_id != CLASS_GUARD else 1.5
        segs.append(make_segment(cls_id, t, t + duration))
        t += duration + 0.1
    return segs


@pytest.fixture
def diverse_keypoints() -> list[KeypointFrame]:
    """Keypoint frames at 30fps for 120 seconds."""
    fps = 30
    duration = 120.0
    frames = []
    for i in range(int(duration * fps)):
        ts = i / fps
        frames.append(make_keypoint_frame(ts, seed=i))
    return frames


@pytest.fixture
def short_segments() -> list[ActionSegment]:
    """30-second session with only jabs."""
    return [
        make_segment(CLASS_JAB, 1.0, 1.4),
        make_segment(CLASS_GUARD, 1.5, 3.0),
        make_segment(CLASS_JAB, 3.5, 3.9),
        make_segment(CLASS_GUARD, 4.0, 5.5),
        make_segment(CLASS_JAB, 6.0, 6.4),
    ]


@pytest.fixture
def short_keypoints() -> list[KeypointFrame]:
    """Keypoint frames for 30-second session."""
    fps = 30
    return [make_keypoint_frame(i / fps, seed=i) for i in range(30 * fps)]
