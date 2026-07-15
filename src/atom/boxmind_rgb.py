"""BoxMind-compatible RGB crops for annotated or proposed punch intervals."""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from atom.boxingweb import OracleWindowSample
from atom.pose_features import JOINT_INDEX, _reindex


@lru_cache(maxsize=2)
def _load_pose(pose_path: str) -> dict[str, np.ndarray]:
    with Path(pose_path).open("rb") as file:
        return pickle.load(file)


def _region(person_2d: np.ndarray, height: int, width: int, expansion_rate: float = 0.4) -> tuple[int, int, int, int]:
    """Return the official actor-centric crop, including glove/head extensions."""

    points = person_2d.copy()
    points[:, 20] = points[:, 20] * 2 - points[:, 18]
    points[:, 21] = points[:, 21] * 2 - points[:, 19]
    points[:, 24] = points[:, 24] * 2 - points[:, 12]
    selected = points[:, [24, 12, 17, 19, 21, 16, 18, 20, 2, 1]]
    x_min, y_min = selected.min(axis=(0, 1))
    x_max, y_max = selected.max(axis=(0, 1))
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    crop_width, crop_height = (x_max - x_min) * (1 + expansion_rate), (y_max - y_min) * (1 + expansion_rate)
    return (
        int(max(center_x - crop_width / 2, 0)),
        int(max(center_y - crop_height / 2, 0)),
        int(min(center_x + crop_width / 2, width)),
        int(min(center_y + crop_height / 2, height)),
    )


def _resize_square(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = size / float(max(width, height))
    new_width, new_height = int(width * scale), int(height * scale)
    resized = cv2.resize(frame, (new_width, new_height))
    result = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - new_height) // 2, (size - new_width) // 2
    result[top:top + new_height, left:left + new_width] = resized
    return result


def extract_boxmind_rgb_clip(
    video_path: Path,
    pose_path: Path,
    side: str,
    start_frame: int,
    end_frame: int,
    frames: int = 16,
    size: int = 224,
) -> np.ndarray:
    """Return an official-style actor crop as normalized RGB ``[3, frames, H, W]``.

    The crop bounds use the entire supplied interval.  This is valid for the
    oracle attribute-classification experiment; a detector must provide a
    proposed interval before this extractor can be used at inference time.
    """

    if side not in {"red", "blue"}:
        raise ValueError(f"Expected red or blue side, got {side!r}")
    if end_frame < start_frame:
        raise ValueError("end_frame must not precede start_frame")
    if frames < 2:
        raise ValueError("frames must be at least 2")
    pose = _load_pose(str(pose_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        pad = (max(width, height) - np.array([width, height])) // 2
        red = _reindex(np.asarray(pose["pose_red_2d"])) * max(width, height) - pad
        blue = _reindex(np.asarray(pose["pose_blue_2d"])) * max(width, height) - pad
        actor = red if side == "red" else blue
        x1, y1, x2, y2 = _region(actor[start_frame:end_frame + 1], height, width)
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Empty BoxMind crop for {video_path.name} at {start_frame}:{end_frame}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        decoded: list[np.ndarray] = []
        for frame_index in range(start_frame, end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Unable to decode frame {frame_index} from {video_path.name}")
            decoded.append(_resize_square(frame[y1:y2, x1:x2], size))
        source_frames = np.rint(np.linspace(0, len(decoded) - 1, num=frames)).astype(int)
        output = [decoded[frame_index] for frame_index in source_frames]
    finally:
        capture.release()
    rgb = np.stack(output, axis=0)[..., ::-1].copy().transpose(3, 0, 1, 2)
    return (rgb.astype(np.float32) / 255.0 * 2.0 - 1.0)


def extract_boxmind_rgb_sample(sample: OracleWindowSample, data_root: Path, frames: int = 16, size: int = 224) -> np.ndarray:
    """Extract an RGB clip from a BoxingWeb oracle sample."""

    return extract_boxmind_rgb_clip(
        data_root / sample.video_path,
        data_root / sample.pose_path,
        sample.labels["side"],
        sample.event_start_frame,
        sample.event_end_frame,
        frames=frames,
        size=size,
    )
