"""B2 Task 8: Sliding window Temporal Action Detection (TAD).

Slides a classification window over continuous PoseFrame sequences to produce
an ActionTimeline — a timestamped list of detected boxing actions.

Algorithm:
    1. Slide a 30-frame window over the PoseFrame DataFrame (stride 5 frames).
    2. For each window: extract 120-dim features → XGBoost predict_proba.
    3. Keep detections where max(proba) ≥ min_confidence (default 0.7).
    4. Apply per-class Non-Maximum Suppression (NMS) to merge overlapping windows.
    5. Return ActionTimeline with timestamped detections sorted by start_time.

Output format (ActionTimeline) matches the B3 data contract:
    {video_id, fighter_id, actions: [{start_time, end_time, action_class, confidence}]}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from track_b.b2.features import extract_features, windows_from_pose_df


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class ActionDetection:
    """A single detected action instance."""

    start_time: float    # seconds from video start
    end_time: float      # seconds from video start
    action_class: str    # e.g., "jab", "cross", "lead_hook"
    confidence: float    # classifier confidence in [0, 1]

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class ActionTimeline:
    """Timestamped sequence of detected actions for one fighter in one video."""

    video_id: str
    fighter_id: str                              # "user", "fighter_a", "fighter_b"
    actions: list[ActionDetection] = field(default_factory=list)

    def sorted_actions(self) -> list[ActionDetection]:
        """Return actions sorted by start_time ascending."""
        return sorted(self.actions, key=lambda a: a.start_time)

    def filter_by_class(self, action_class: str) -> list[ActionDetection]:
        return [a for a in self.actions if a.action_class == action_class]

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "fighter_id": self.fighter_id,
            "actions": [asdict(a) for a in self.sorted_actions()],
        }


# ── Non-Maximum Suppression ───────────────────────────────────────────────────


def _time_iou(a1: float, a2: float, b1: float, b2: float) -> float:
    """Intersection over Union for two time intervals [a1, a2] and [b1, b2].

    Returns:
        IoU in [0, 1]. Returns 0.0 if either interval is degenerate.
    """
    intersection = max(0.0, min(a2, b2) - max(a1, b1))
    union = max(a2, b2) - min(a1, b1)
    return intersection / union if union > 1e-9 else 0.0


def non_maximum_suppression(
    detections: list[ActionDetection],
    iou_threshold: float = 0.5,
) -> list[ActionDetection]:
    """Apply per-class NMS to suppress overlapping action detections.

    Within each action class, detections are sorted by confidence descending.
    A detection is suppressed if its temporal IoU with any higher-confidence
    detection of the same class exceeds iou_threshold.

    Args:
        detections: List of ActionDetection instances.
        iou_threshold: IoU threshold above which lower-confidence detection
                       is suppressed (default 0.5).

    Returns:
        Filtered list of ActionDetection, sorted by start_time.
    """
    if not detections:
        return []

    # Group by class
    by_class: dict[str, list[ActionDetection]] = {}
    for det in detections:
        by_class.setdefault(det.action_class, []).append(det)

    kept: list[ActionDetection] = []
    for dets in by_class.values():
        # Sort by confidence descending
        sorted_dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
        suppressed: set[int] = set()
        for i, det_i in enumerate(sorted_dets):
            if i in suppressed:
                continue
            kept.append(det_i)
            # Suppress all lower-confidence detections with high IoU
            for j in range(i + 1, len(sorted_dets)):
                if j in suppressed:
                    continue
                det_j = sorted_dets[j]
                iou = _time_iou(det_i.start_time, det_i.end_time,
                                det_j.start_time, det_j.end_time)
                if iou >= iou_threshold:
                    suppressed.add(j)

    return sorted(kept, key=lambda d: d.start_time)


# ── Sliding window inference ──────────────────────────────────────────────────


def detect_actions(
    pose_df: pd.DataFrame,
    model: object,
    class_names: list[str],
    window_size: int = 30,
    stride: int = 5,
    min_confidence: float = 0.7,
    nms_iou_threshold: float = 0.5,
) -> list[ActionDetection]:
    """Slide a classifier window over a PoseFrame DataFrame to detect actions.

    Args:
        pose_df: PoseFrame DataFrame with columns kp_*_x/y/z/vis and timestamp.
                 Rows must be sorted by frame_number (ascending).
        model: Trained classifier with predict_proba(X) → (n, n_classes) method.
        class_names: List of class name strings matching model's label order.
                     e.g., ["jab", "cross", "lead_hook", "rear_hook", ...].
        window_size: Frames per window (default 30 = 1 second at 30fps).
        stride: Frame step between windows (default 5 frames).
        min_confidence: Minimum max-class probability to keep a detection (0.7).
        nms_iou_threshold: IoU threshold for NMS suppression (default 0.5).

    Returns:
        List of ActionDetection sorted by start_time, after NMS.
        Empty list if pose_df has fewer rows than window_size.
    """
    windows = windows_from_pose_df(pose_df, window_size=window_size, stride=stride)
    if not windows:
        return []

    # Build feature matrix
    X = np.array([extract_features(w) for w in windows], dtype=np.float32)

    # Predict probabilities
    proba = np.array(model.predict_proba(X))  # (n_windows, n_classes)

    # Get timestamps for window boundaries
    timestamps = pose_df["timestamp"].to_numpy(dtype=np.float64) if "timestamp" in pose_df.columns else None

    raw_detections: list[ActionDetection] = []
    for win_idx in range(len(windows)):
        row_start = win_idx * stride
        row_end = row_start + window_size - 1

        if timestamps is not None:
            start_time = float(timestamps[row_start])
            end_time = float(timestamps[min(row_end, len(timestamps) - 1)])
        else:
            start_time = float(row_start) / 30.0
            end_time = float(row_end) / 30.0

        max_conf = float(proba[win_idx].max())
        if max_conf < min_confidence:
            continue

        class_idx = int(proba[win_idx].argmax())
        if class_idx >= len(class_names):
            continue
        action_class = class_names[class_idx]

        raw_detections.append(ActionDetection(
            start_time=start_time,
            end_time=end_time,
            action_class=action_class,
            confidence=max_conf,
        ))

    return non_maximum_suppression(raw_detections, iou_threshold=nms_iou_threshold)


def build_timeline(
    pose_df: pd.DataFrame,
    model: object,
    class_names: list[str],
    video_id: str,
    fighter_id: str = "user",
    window_size: int = 30,
    stride: int = 5,
    min_confidence: float = 0.7,
    nms_iou_threshold: float = 0.5,
) -> ActionTimeline:
    """Build a complete ActionTimeline from a PoseFrame DataFrame.

    Convenience wrapper around detect_actions() that packages results into
    the ActionTimeline data model.

    Args:
        pose_df: PoseFrame DataFrame.
        model: Trained classifier.
        class_names: Class name list matching model label order.
        video_id: Source video identifier.
        fighter_id: Fighter identifier (default "user" for solo analysis).
        window_size: Frames per window (default 30).
        stride: Frame step between windows (default 5).
        min_confidence: Minimum confidence threshold (default 0.7).
        nms_iou_threshold: NMS IoU threshold (default 0.5).

    Returns:
        ActionTimeline with detected actions sorted by start_time.
    """
    actions = detect_actions(
        pose_df=pose_df,
        model=model,
        class_names=class_names,
        window_size=window_size,
        stride=stride,
        min_confidence=min_confidence,
        nms_iou_threshold=nms_iou_threshold,
    )
    return ActionTimeline(video_id=video_id, fighter_id=fighter_id, actions=actions)


# ── Persistence ───────────────────────────────────────────────────────────────


def save_timeline(timeline: ActionTimeline, path: str | Path) -> None:
    """Serialize an ActionTimeline to a JSON file.

    Args:
        timeline: ActionTimeline to save.
        path: Output path (parent directories created automatically).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(timeline.to_dict(), f, indent=2)


def load_timeline(path: str | Path) -> ActionTimeline:
    """Load an ActionTimeline from a JSON file.

    Args:
        path: Path to ActionTimeline JSON file.

    Returns:
        ActionTimeline with deserialized actions.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If required fields are missing from the JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ActionTimeline file not found: {p}")

    with open(p) as f:
        data = json.load(f)

    actions = [
        ActionDetection(
            start_time=a["start_time"],
            end_time=a["end_time"],
            action_class=a["action_class"],
            confidence=a["confidence"],
        )
        for a in data.get("actions", [])
    ]
    return ActionTimeline(
        video_id=data["video_id"],
        fighter_id=data["fighter_id"],
        actions=actions,
    )
