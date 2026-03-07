"""Tests for B1 Task 4: Multi-person detection + tracking."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from track_b.b1.multi_person import (
    BoundingBox,
    FighterTracker,
    crop_roi,
    detect_persons,
)


def make_bbox(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)


def blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── BoundingBox ───────────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_width_height(self):
        b = make_bbox(10, 20, 110, 220)
        assert b.width == pytest.approx(100.0)
        assert b.height == pytest.approx(200.0)

    def test_cx(self):
        b = make_bbox(0, 0, 100, 100)
        assert b.cx == pytest.approx(50.0)

    def test_area(self):
        b = make_bbox(0, 0, 10, 20)
        assert b.area == pytest.approx(200.0)

    def test_iou_identical(self):
        b = make_bbox(0, 0, 100, 100)
        assert b.iou(b) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        b1 = make_bbox(0, 0, 50, 50)
        b2 = make_bbox(100, 100, 200, 200)
        assert b1.iou(b2) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        b1 = make_bbox(0, 0, 100, 100)
        b2 = make_bbox(50, 50, 150, 150)
        # Intersection: (50,50)-(100,100) = 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        expected = 2500 / 17500
        assert b1.iou(b2) == pytest.approx(expected)

    def test_iou_symmetric(self):
        b1 = make_bbox(0, 0, 80, 80)
        b2 = make_bbox(40, 40, 120, 120)
        assert b1.iou(b2) == pytest.approx(b2.iou(b1))


# ── FighterTracker ────────────────────────────────────────────────────────────

class TestFighterTrackerInit:
    def test_first_frame_two_detections(self):
        tracker = FighterTracker()
        dets = [make_bbox(300, 0, 500, 400), make_bbox(10, 0, 200, 400)]
        tracks = tracker.update(dets)
        assert len(tracks) == 2

    def test_first_frame_assigns_left_as_a(self):
        """Left boxer (lower cx) should be fighter_a."""
        tracker = FighterTracker()
        left = make_bbox(0, 0, 100, 300)    # cx=50
        right = make_bbox(300, 0, 400, 300) # cx=350
        tracks = tracker.update([left, right])

        ids_by_cx = {t.fighter_id: t.bbox.cx for t in tracks}
        assert ids_by_cx["fighter_a"] < ids_by_cx["fighter_b"]

    def test_first_frame_one_detection(self):
        tracker = FighterTracker()
        dets = [make_bbox(0, 0, 100, 300)]
        tracks = tracker.update(dets)
        assert len(tracks) == 1

    def test_first_frame_no_detections(self):
        tracker = FighterTracker()
        tracks = tracker.update([])
        assert len(tracks) == 0


class TestFighterTrackerContinuity:
    def test_identity_preserved_across_frames(self):
        """Same person should keep same fighter_id across frames."""
        tracker = FighterTracker()
        # Frame 1: initialize
        t1 = tracker.update([make_bbox(10, 0, 100, 300), make_bbox(300, 0, 400, 300)])
        id_map_1 = {t.fighter_id: t.bbox.cx for t in t1}

        # Frame 2: slight movement
        t2 = tracker.update([make_bbox(15, 0, 105, 300), make_bbox(305, 0, 405, 300)])
        id_map_2 = {t.fighter_id: t.bbox.cx for t in t2}

        # fighter_a should still be on the left
        assert id_map_2["fighter_a"] < id_map_2["fighter_b"]
        # fighter_a x should be close to frame 1 (not swapped)
        assert abs(id_map_2["fighter_a"] - id_map_1["fighter_a"]) < 20

    def test_missing_frame_increments_counter(self):
        tracker = FighterTracker(max_missing_frames=3)
        tracker.update([make_bbox(10, 0, 100, 300), make_bbox(300, 0, 400, 300)])
        # No detections
        tracker.update([])
        tracks = tracker.update([])
        # Tracks still alive (missing=2, threshold=3)
        assert len(tracks) == 2

    def test_track_removed_after_max_missing(self):
        tracker = FighterTracker(max_missing_frames=2)
        tracker.update([make_bbox(10, 0, 100, 300), make_bbox(300, 0, 400, 300)])
        tracker.update([])  # missing=1
        tracker.update([])  # missing=2
        tracks = tracker.update([])  # missing=3 → removed
        assert len(tracks) == 0

    def test_at_most_two_tracks(self):
        """Even with 3 detections, only 2 tracks maintained."""
        tracker = FighterTracker()
        dets = [
            make_bbox(0, 0, 100, 300),
            make_bbox(200, 0, 300, 300),
            make_bbox(400, 0, 500, 300),
        ]
        tracks = tracker.update(dets)
        assert len(tracks) <= 2

    def test_reset_clears_tracks(self):
        tracker = FighterTracker()
        tracker.update([make_bbox(0, 0, 100, 300), make_bbox(300, 0, 400, 300)])
        tracker.reset()
        tracks = tracker.update([])
        assert len(tracks) == 0


# ── crop_roi ──────────────────────────────────────────────────────────────────

class TestCropRoi:
    def test_crop_shape(self):
        frame = blank_frame(480, 640)
        bbox = make_bbox(100, 50, 300, 400)
        crop = crop_roi(frame, bbox, padding=0.0)
        assert crop.shape[0] == 350  # height = 400-50
        assert crop.shape[1] == 200  # width = 300-100

    def test_crop_with_padding(self):
        frame = blank_frame(480, 640)
        bbox = make_bbox(100, 50, 300, 400)  # w=200, h=350
        crop = crop_roi(frame, bbox, padding=0.1)
        # Padded: x1=80, y1=15, x2=320, y2=435 → w=240, h=420
        assert crop.shape[0] > 350  # taller than no-padding
        assert crop.shape[1] > 200  # wider than no-padding

    def test_crop_clipped_to_frame(self):
        frame = blank_frame(100, 100)
        bbox = make_bbox(80, 80, 120, 120)  # extends outside frame
        crop = crop_roi(frame, bbox, padding=0.0)
        assert crop.shape[0] <= 100
        assert crop.shape[1] <= 100

    def test_degenerate_bbox_returns_full_frame(self):
        frame = blank_frame(100, 100)
        bbox = make_bbox(50, 50, 50, 50)  # zero area
        crop = crop_roi(frame, bbox, padding=0.0)
        assert crop.shape == frame.shape

    def test_crop_is_view_or_copy(self):
        frame = blank_frame(480, 640)
        frame[100, 200] = [255, 0, 0]  # marker pixel
        bbox = make_bbox(150, 50, 300, 200)
        crop = crop_roi(frame, bbox, padding=0.0)
        assert crop.shape[2] == 3  # 3 channels


# ── detect_persons (mocked) ───────────────────────────────────────────────────

class TestDetectPersons:
    def _make_mock_yolo_result(self, boxes_xyxyconf: list[tuple]) -> list:
        """Build mock YOLO results."""
        mock_box_list = []
        for x1, y1, x2, y2, conf in boxes_xyxyconf:
            b = MagicMock()
            b.xyxy = [np.array([x1, y1, x2, y2])]
            b.conf = [np.array(conf)]
            mock_box_list.append(b)

        mock_result = MagicMock()
        mock_result.boxes = mock_box_list
        return [mock_result]

    def test_returns_bounding_boxes(self):
        frame = blank_frame()
        mock_results = self._make_mock_yolo_result([(10, 20, 200, 400, 0.9)])

        with patch("track_b.b1.multi_person._get_yolo") as mock_yolo_fn:
            mock_model = MagicMock()
            mock_model.return_value = mock_results
            mock_yolo_fn.return_value = mock_model

            boxes = detect_persons(frame)

        assert len(boxes) == 1
        assert boxes[0].x1 == pytest.approx(10.0)
        assert boxes[0].confidence == pytest.approx(0.9)

    def test_returns_empty_for_no_persons(self):
        frame = blank_frame()
        mock_result = MagicMock()
        mock_result.boxes = []

        with patch("track_b.b1.multi_person._get_yolo") as mock_yolo_fn:
            mock_model = MagicMock()
            mock_model.return_value = [mock_result]
            mock_yolo_fn.return_value = mock_model

            boxes = detect_persons(frame)

        assert boxes == []

    def test_multiple_persons(self):
        frame = blank_frame()
        mock_results = self._make_mock_yolo_result([
            (10, 20, 200, 400, 0.9),
            (300, 20, 500, 400, 0.85),
        ])

        with patch("track_b.b1.multi_person._get_yolo") as mock_yolo_fn:
            mock_model = MagicMock()
            mock_model.return_value = mock_results
            mock_yolo_fn.return_value = mock_model

            boxes = detect_persons(frame)

        assert len(boxes) == 2
