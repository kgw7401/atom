"""Tests for B2 Task 8: Sliding window Temporal Action Detection."""

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from track_b.b2.tad import (
    ActionDetection,
    ActionTimeline,
    _time_iou,
    build_timeline,
    detect_actions,
    load_timeline,
    non_maximum_suppression,
    save_timeline,
)

CLASS_NAMES = ["jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut"]
N_CLASSES = len(CLASS_NAMES)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_pose_df(n_frames: int = 90, fps: float = 30.0) -> pd.DataFrame:
    """PoseFrame DataFrame with flat keypoint columns and timestamps."""
    rng = np.random.default_rng(42)
    rows: dict[str, np.ndarray] = {}
    for lm in range(33):
        for coord in ("x", "y", "z", "vis"):
            rows[f"kp_{lm}_{coord}"] = rng.random(n_frames).astype(np.float32)
    rows["frame_number"] = np.arange(n_frames)
    rows["timestamp"] = np.arange(n_frames) / fps
    return pd.DataFrame(rows)


def make_mock_model(n_classes: int = N_CLASSES, always_class: int = 0, confidence: float = 0.9) -> MagicMock:
    """Mock classifier that always predicts one class with fixed confidence."""
    model = MagicMock()

    def predict_proba(X):
        proba = np.zeros((len(X), n_classes), dtype=np.float32)
        proba[:, always_class] = confidence
        # Distribute remaining probability
        remaining = (1.0 - confidence) / max(1, n_classes - 1)
        for c in range(n_classes):
            if c != always_class:
                proba[:, c] = remaining
        return proba

    model.predict_proba = predict_proba
    return model


def make_low_confidence_model(n_classes: int = N_CLASSES) -> MagicMock:
    """Mock classifier that always predicts below threshold."""
    model = MagicMock()

    def predict_proba(X):
        proba = np.ones((len(X), n_classes), dtype=np.float32) / n_classes
        return proba  # uniform = 1/6 ≈ 0.167, below 0.7 threshold

    model.predict_proba = predict_proba
    return model


def make_detection(start: float, end: float, cls: str = "jab", conf: float = 0.9) -> ActionDetection:
    return ActionDetection(start_time=start, end_time=end, action_class=cls, confidence=conf)


# ── ActionDetection ───────────────────────────────────────────────────────────

class TestActionDetection:
    def test_duration(self):
        det = make_detection(1.0, 2.0)
        assert det.duration == pytest.approx(1.0)

    def test_zero_duration_for_degenerate(self):
        det = make_detection(1.0, 0.5)  # end < start
        assert det.duration == pytest.approx(0.0)

    def test_fields_accessible(self):
        det = ActionDetection(start_time=0.5, end_time=1.5, action_class="cross", confidence=0.85)
        assert det.start_time == 0.5
        assert det.end_time == 1.5
        assert det.action_class == "cross"
        assert det.confidence == 0.85


# ── ActionTimeline ────────────────────────────────────────────────────────────

class TestActionTimeline:
    def test_sorted_actions(self):
        timeline = ActionTimeline(
            video_id="v1", fighter_id="user",
            actions=[make_detection(2.0, 3.0), make_detection(0.0, 1.0), make_detection(1.0, 2.0)]
        )
        sorted_a = timeline.sorted_actions()
        starts = [a.start_time for a in sorted_a]
        assert starts == sorted(starts)

    def test_filter_by_class(self):
        timeline = ActionTimeline(
            video_id="v1", fighter_id="user",
            actions=[make_detection(0.0, 1.0, "jab"), make_detection(1.0, 2.0, "cross"),
                     make_detection(2.0, 3.0, "jab")]
        )
        jabs = timeline.filter_by_class("jab")
        assert len(jabs) == 2
        assert all(a.action_class == "jab" for a in jabs)

    def test_to_dict_keys(self):
        timeline = ActionTimeline(video_id="v1", fighter_id="user")
        d = timeline.to_dict()
        assert {"video_id", "fighter_id", "actions"}.issubset(d.keys())

    def test_to_dict_actions_sorted(self):
        timeline = ActionTimeline(
            video_id="v1", fighter_id="user",
            actions=[make_detection(2.0, 3.0), make_detection(0.0, 1.0)]
        )
        d = timeline.to_dict()
        starts = [a["start_time"] for a in d["actions"]]
        assert starts == sorted(starts)

    def test_empty_actions_default(self):
        timeline = ActionTimeline(video_id="v1", fighter_id="user")
        assert timeline.actions == []


# ── _time_iou ─────────────────────────────────────────────────────────────────

class TestTimeIou:
    def test_identical_intervals(self):
        assert _time_iou(0.0, 1.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _time_iou(0.0, 1.0, 2.0, 3.0) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # [0, 2] and [1, 3]: intersection=1, union=3
        iou = _time_iou(0.0, 2.0, 1.0, 3.0)
        assert iou == pytest.approx(1.0 / 3.0)

    def test_contained_interval(self):
        # [0, 4] contains [1, 3]: intersection=2, union=4
        iou = _time_iou(0.0, 4.0, 1.0, 3.0)
        assert iou == pytest.approx(2.0 / 4.0)

    def test_degenerate_zero_length(self):
        assert _time_iou(1.0, 1.0, 0.0, 2.0) == pytest.approx(0.0)

    def test_symmetric(self):
        iou_ab = _time_iou(0.0, 2.0, 1.0, 3.0)
        iou_ba = _time_iou(1.0, 3.0, 0.0, 2.0)
        assert iou_ab == pytest.approx(iou_ba)


# ── non_maximum_suppression ───────────────────────────────────────────────────

class TestNonMaximumSuppression:
    def test_empty_returns_empty(self):
        assert non_maximum_suppression([]) == []

    def test_single_detection_kept(self):
        dets = [make_detection(0.0, 1.0, "jab", 0.9)]
        result = non_maximum_suppression(dets)
        assert len(result) == 1

    def test_non_overlapping_both_kept(self):
        dets = [make_detection(0.0, 1.0, "jab", 0.9), make_detection(2.0, 3.0, "jab", 0.8)]
        result = non_maximum_suppression(dets, iou_threshold=0.5)
        assert len(result) == 2

    def test_identical_detections_one_suppressed(self):
        dets = [make_detection(0.0, 1.0, "jab", 0.9), make_detection(0.0, 1.0, "jab", 0.8)]
        result = non_maximum_suppression(dets, iou_threshold=0.5)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)  # higher kept

    def test_high_iou_lower_confidence_suppressed(self):
        # Strongly overlapping — should suppress the lower one
        dets = [make_detection(0.0, 1.0, "jab", 0.9), make_detection(0.2, 1.2, "jab", 0.7)]
        result = non_maximum_suppression(dets, iou_threshold=0.5)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)

    def test_different_classes_not_suppressed(self):
        # Same time window but different classes → both kept
        dets = [make_detection(0.0, 1.0, "jab", 0.9), make_detection(0.0, 1.0, "cross", 0.85)]
        result = non_maximum_suppression(dets, iou_threshold=0.5)
        assert len(result) == 2

    def test_output_sorted_by_start_time(self):
        dets = [make_detection(3.0, 4.0, "jab", 0.9), make_detection(0.0, 1.0, "jab", 0.8)]
        result = non_maximum_suppression(dets)
        starts = [d.start_time for d in result]
        assert starts == sorted(starts)

    def test_low_iou_threshold_suppresses_more(self):
        dets = [make_detection(0.0, 1.0, "jab", 0.9), make_detection(0.8, 1.8, "jab", 0.8)]
        # With low threshold, the partially overlapping pair is suppressed
        result_strict = non_maximum_suppression(dets, iou_threshold=0.1)
        result_loose = non_maximum_suppression(dets, iou_threshold=0.9)
        assert len(result_strict) <= len(result_loose)


# ── detect_actions ────────────────────────────────────────────────────────────

class TestDetectActions:
    def test_returns_list(self):
        pose_df = make_pose_df(90)
        model = make_mock_model(always_class=0, confidence=0.95)
        result = detect_actions(pose_df, model, CLASS_NAMES)
        assert isinstance(result, list)

    def test_all_results_are_action_detections(self):
        pose_df = make_pose_df(90)
        model = make_mock_model(always_class=0, confidence=0.95)
        result = detect_actions(pose_df, model, CLASS_NAMES)
        assert all(isinstance(d, ActionDetection) for d in result)

    def test_high_confidence_model_produces_detections(self):
        pose_df = make_pose_df(90)
        model = make_mock_model(always_class=0, confidence=0.95)
        result = detect_actions(pose_df, model, CLASS_NAMES, min_confidence=0.7)
        assert len(result) > 0

    def test_low_confidence_model_produces_no_detections(self):
        pose_df = make_pose_df(90)
        model = make_low_confidence_model()
        result = detect_actions(pose_df, model, CLASS_NAMES, min_confidence=0.7)
        assert len(result) == 0

    def test_detected_class_matches_model_prediction(self):
        pose_df = make_pose_df(90)
        model = make_mock_model(always_class=1, confidence=0.95)  # always "cross"
        result = detect_actions(pose_df, model, CLASS_NAMES, min_confidence=0.7)
        assert len(result) > 0
        assert all(d.action_class == "cross" for d in result)

    def test_timestamps_are_non_negative(self):
        pose_df = make_pose_df(90)
        model = make_mock_model(always_class=0, confidence=0.95)
        result = detect_actions(pose_df, model, CLASS_NAMES)
        for d in result:
            assert d.start_time >= 0.0
            assert d.end_time >= 0.0

    def test_end_time_after_start_time(self):
        pose_df = make_pose_df(90)
        model = make_mock_model(always_class=0, confidence=0.95)
        result = detect_actions(pose_df, model, CLASS_NAMES)
        for d in result:
            assert d.end_time >= d.start_time

    def test_confidence_at_or_above_threshold(self):
        pose_df = make_pose_df(90)
        model = make_mock_model(always_class=0, confidence=0.9)
        result = detect_actions(pose_df, model, CLASS_NAMES, min_confidence=0.7)
        for d in result:
            assert d.confidence >= 0.7

    def test_empty_when_pose_df_too_short(self):
        pose_df = make_pose_df(10)  # fewer than 30 frames
        model = make_mock_model(always_class=0, confidence=0.95)
        result = detect_actions(pose_df, model, CLASS_NAMES, window_size=30)
        assert result == []

    def test_results_sorted_by_start_time(self):
        pose_df = make_pose_df(120)
        model = make_mock_model(always_class=0, confidence=0.95)
        result = detect_actions(pose_df, model, CLASS_NAMES)
        starts = [d.start_time for d in result]
        assert starts == sorted(starts)

    def test_timestamps_from_pose_df(self):
        """Detections should use the timestamp column, not frame index."""
        pose_df = make_pose_df(90, fps=30.0)
        model = make_mock_model(always_class=0, confidence=0.95)
        result = detect_actions(pose_df, model, CLASS_NAMES)
        # First detection should start at t=0 (frame 0 at 30fps)
        assert result[0].start_time == pytest.approx(0.0)

    def test_custom_min_confidence(self):
        pose_df = make_pose_df(90)
        model = make_mock_model(always_class=0, confidence=0.75)
        result_strict = detect_actions(pose_df, model, CLASS_NAMES, min_confidence=0.8)
        result_loose = detect_actions(pose_df, model, CLASS_NAMES, min_confidence=0.7)
        # With higher threshold, fewer (or equal) detections
        assert len(result_strict) <= len(result_loose)


# ── build_timeline ────────────────────────────────────────────────────────────

class TestBuildTimeline:
    def test_returns_action_timeline(self):
        pose_df = make_pose_df(90)
        model = make_mock_model(always_class=0, confidence=0.95)
        timeline = build_timeline(pose_df, model, CLASS_NAMES, "vid_001", "user")
        assert isinstance(timeline, ActionTimeline)

    def test_video_id_set(self):
        pose_df = make_pose_df(90)
        model = make_mock_model(always_class=0, confidence=0.95)
        timeline = build_timeline(pose_df, model, CLASS_NAMES, "fight_042", "fighter_a")
        assert timeline.video_id == "fight_042"

    def test_fighter_id_set(self):
        pose_df = make_pose_df(90)
        model = make_mock_model(always_class=0, confidence=0.95)
        timeline = build_timeline(pose_df, model, CLASS_NAMES, "v1", "fighter_b")
        assert timeline.fighter_id == "fighter_b"

    def test_empty_timeline_for_short_df(self):
        pose_df = make_pose_df(10)
        model = make_mock_model(always_class=0, confidence=0.95)
        timeline = build_timeline(pose_df, model, CLASS_NAMES, "v1")
        assert len(timeline.actions) == 0


# ── save_timeline / load_timeline ─────────────────────────────────────────────

class TestTimelinePersistence:
    def _make_timeline(self) -> ActionTimeline:
        return ActionTimeline(
            video_id="fight_007",
            fighter_id="fighter_a",
            actions=[
                ActionDetection(0.5, 1.5, "jab", 0.92),
                ActionDetection(2.0, 3.0, "cross", 0.87),
                ActionDetection(4.5, 5.5, "lead_hook", 0.75),
            ]
        )

    def test_save_creates_file(self, tmp_path):
        timeline = self._make_timeline()
        path = tmp_path / "timeline.json"
        save_timeline(timeline, path)
        assert path.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        timeline = self._make_timeline()
        path = tmp_path / "deep" / "nested" / "timeline.json"
        save_timeline(timeline, path)
        assert path.exists()

    def test_save_produces_valid_json(self, tmp_path):
        timeline = self._make_timeline()
        path = tmp_path / "timeline.json"
        save_timeline(timeline, path)
        with open(path) as f:
            data = json.load(f)
        assert data["video_id"] == "fight_007"

    def test_roundtrip_video_id(self, tmp_path):
        timeline = self._make_timeline()
        path = tmp_path / "t.json"
        save_timeline(timeline, path)
        loaded = load_timeline(path)
        assert loaded.video_id == timeline.video_id

    def test_roundtrip_fighter_id(self, tmp_path):
        timeline = self._make_timeline()
        path = tmp_path / "t.json"
        save_timeline(timeline, path)
        loaded = load_timeline(path)
        assert loaded.fighter_id == timeline.fighter_id

    def test_roundtrip_action_count(self, tmp_path):
        timeline = self._make_timeline()
        path = tmp_path / "t.json"
        save_timeline(timeline, path)
        loaded = load_timeline(path)
        assert len(loaded.actions) == len(timeline.actions)

    def test_roundtrip_action_fields(self, tmp_path):
        timeline = self._make_timeline()
        path = tmp_path / "t.json"
        save_timeline(timeline, path)
        loaded = load_timeline(path)
        orig = timeline.sorted_actions()[0]
        loaded_a = loaded.sorted_actions()[0]
        assert loaded_a.start_time == pytest.approx(orig.start_time)
        assert loaded_a.end_time == pytest.approx(orig.end_time)
        assert loaded_a.action_class == orig.action_class
        assert loaded_a.confidence == pytest.approx(orig.confidence)

    def test_load_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_timeline(tmp_path / "nonexistent.json")

    def test_empty_timeline_roundtrip(self, tmp_path):
        timeline = ActionTimeline(video_id="v", fighter_id="user")
        path = tmp_path / "empty.json"
        save_timeline(timeline, path)
        loaded = load_timeline(path)
        assert loaded.actions == []

    def test_accepts_string_path(self, tmp_path):
        timeline = self._make_timeline()
        path = str(tmp_path / "t.json")
        save_timeline(timeline, path)
        loaded = load_timeline(path)
        assert loaded.video_id == "fight_007"
