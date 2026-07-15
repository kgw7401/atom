"""Pose-guided RGB motion features for full-round punch detection."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from atom.pose_features import JOINT_INDEX_GT


def _crop(frame: np.ndarray, joints: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    max_dimension = max(width, height)
    padding = (max_dimension - np.array([width, height])) // 2
    points = joints[JOINT_INDEX_GT] * max_dimension - padding
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    center = 0.5 * np.array([x_min + x_max, y_min + y_max])
    side = max(x_max - x_min, y_max - y_min, 32.0) * 1.4
    start = np.maximum(np.floor(center - side / 2).astype(int), 0)
    stop = np.minimum(np.ceil(center + side / 2).astype(int), np.array([width, height]))
    if stop[0] <= start[0] or stop[1] <= start[1]:
        return np.zeros((8, 8), dtype=np.float32)
    # Sampling the crop directly avoids resizing two full images on every
    # frame. The detector only needs coarse glove/arm motion, not appearance.
    x = np.linspace(start[0], stop[0] - 1, num=8).astype(int)
    y = np.linspace(start[1], stop[1] - 1, num=8).astype(int)
    pixels = frame[y[:, None], x[None, :]].astype(np.float32)
    return (0.114 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.299 * pixels[:, :, 2]) / 255.0


def extract_match_rgb_motion_features(video_path: Path, pose_path: Path) -> np.ndarray:
    """Return 8×8 signed RGB-frame differences for red and blue fighters.

    Crops follow the supplied pose tracks. This keeps the RGB input focused on
    arm and glove motion rather than background or camera movement.
    """

    with pose_path.open("rb") as file:
        pose: dict[str, Any] = pickle.load(file)
    red = np.asarray(pose["pose_red_2d"])
    blue = np.asarray(pose["pose_blue_2d"])
    frame_count = min(red.shape[0], blue.shape[0])
    features = np.zeros((frame_count, 128), dtype=np.float32)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    previous_red = previous_blue = None
    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            red_crop, blue_crop = _crop(frame, red[frame_index]), _crop(frame, blue[frame_index])
            if previous_red is not None:
                features[frame_index, :64] = (red_crop - previous_red).reshape(-1)
                features[frame_index, 64:] = (blue_crop - previous_blue).reshape(-1)
            previous_red, previous_blue = red_crop, blue_crop
    finally:
        capture.release()
    return features
