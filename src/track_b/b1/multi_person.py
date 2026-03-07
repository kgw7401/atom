"""B1 Task 4: Multi-person detection + tracking for analysis mode.

Uses YOLOv8 for person detection and a simple IoU-based tracker to assign
consistent fighter_id ("fighter_a", "fighter_b") across frames.

Each detected fighter gets their ROI cropped and passed to MediaPipe individually.
Outputs: separate PoseFrame DataFrames per fighter.

Architecture:
  frame → YOLO detect persons → sort by position → IoU tracker → ROI crop
        → MediaPipe (reuses PoseEstimator) → PoseKeypoints per fighter
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Lazy import — YOLO downloads weights on first use
_yolo_model = None


def _get_yolo() -> object:
    """Lazy-load YOLOv8n model (downloads ~6MB on first call)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


@dataclass
class BoundingBox:
    """Person bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        """Horizontal center."""
        return (self.x1 + self.x2) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def iou(self, other: BoundingBox) -> float:
        """Intersection over Union."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class TrackedFighter:
    """Fighter with persistent identity across frames."""
    fighter_id: str  # "fighter_a" or "fighter_b"
    bbox: BoundingBox
    frames_missing: int = 0  # consecutive frames without detection


class FighterTracker:
    """Simple IoU-based tracker for two fighters.

    Assignment strategy:
    - Frame 0: Sort detected persons by horizontal center (left→right).
                Left = fighter_a, right = fighter_b.
    - Subsequent frames: Match detections to tracked fighters by max IoU.
                         If IoU ≥ threshold, update position. Otherwise mark missing.
    - Max 2 fighters tracked. Extra detections ignored.

    Args:
        iou_threshold: Minimum IoU to match detection to existing track.
        max_missing_frames: Remove a track after this many consecutive missing frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missing_frames: int = 5):
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self._tracks: list[TrackedFighter] = []

    def update(self, detections: list[BoundingBox]) -> list[TrackedFighter]:
        """Update tracker with new detections. Returns current tracks.

        Args:
            detections: Person bounding boxes from YOLO (confidence ≥ 0.3).

        Returns:
            List of TrackedFighter (up to 2) with current positions.
        """
        # Keep at most 2 highest-confidence detections
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)[:2]

        if not self._tracks:
            # First frame: initialize by horizontal position
            detections_sorted = sorted(detections, key=lambda d: d.cx)
            ids = ["fighter_a", "fighter_b"]
            self._tracks = [
                TrackedFighter(fighter_id=ids[i], bbox=det)
                for i, det in enumerate(detections_sorted)
            ]
            return list(self._tracks)

        if not detections:
            # No detections: mark all tracks as missing
            for track in self._tracks:
                track.frames_missing += 1
            self._tracks = [t for t in self._tracks if t.frames_missing <= self.max_missing_frames]
            return list(self._tracks)

        # Match detections to existing tracks by IoU
        used_dets: set[int] = set()
        for track in self._tracks:
            best_iou = 0.0
            best_idx = -1
            for i, det in enumerate(detections):
                if i in used_dets:
                    continue
                iou = track.bbox.iou(det)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0 and best_iou >= self.iou_threshold:
                track.bbox = detections[best_idx]
                track.frames_missing = 0
                used_dets.add(best_idx)
            else:
                track.frames_missing += 1

        # Remove stale tracks
        self._tracks = [t for t in self._tracks if t.frames_missing <= self.max_missing_frames]

        # Initialize new track if a new person appears and we have fewer than 2 tracks
        active_ids = {t.fighter_id for t in self._tracks}
        for i, det in enumerate(detections):
            if i not in used_dets and len(self._tracks) < 2:
                # Assign the missing fighter_id
                new_id = "fighter_a" if "fighter_a" not in active_ids else "fighter_b"
                self._tracks.append(TrackedFighter(fighter_id=new_id, bbox=det))
                active_ids.add(new_id)

        return list(self._tracks)

    def reset(self) -> None:
        """Clear all tracks (start fresh)."""
        self._tracks = []


def detect_persons(
    image_bgr: np.ndarray,
    min_confidence: float = 0.3,
) -> list[BoundingBox]:
    """Detect person bounding boxes in a frame using YOLOv8n.

    Args:
        image_bgr: BGR image (H, W, 3) uint8.
        min_confidence: Minimum detection confidence.

    Returns:
        List of BoundingBox for each detected person.
    """
    model = _get_yolo()
    results = model(
        image_bgr,
        classes=[0],  # class 0 = person
        conf=min_confidence,
        verbose=False,
    )

    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf))
    return boxes


def crop_roi(
    image_bgr: np.ndarray,
    bbox: BoundingBox,
    padding: float = 0.1,
) -> np.ndarray:
    """Crop fighter ROI from full frame with optional padding.

    Args:
        image_bgr: Full frame (H, W, 3).
        bbox: Fighter bounding box.
        padding: Fractional padding around bounding box (default 10%).

    Returns:
        Cropped BGR image. Returns full frame if crop would be empty.
    """
    h, w = image_bgr.shape[:2]
    pad_x = bbox.width * padding
    pad_y = bbox.height * padding

    x1 = max(0, int(bbox.x1 - pad_x))
    y1 = max(0, int(bbox.y1 - pad_y))
    x2 = min(w, int(bbox.x2 + pad_x))
    y2 = min(h, int(bbox.y2 + pad_y))

    if x2 <= x1 or y2 <= y1:
        return image_bgr  # degenerate bbox

    return image_bgr[y1:y2, x1:x2]
