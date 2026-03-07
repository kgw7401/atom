"""Shared test fixtures."""

from pathlib import Path

import cv2
import numpy as np
import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_video_2s(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a 2-second test video at 30fps (60 frames total).

    Each frame has a different color to verify frame ordering.
    """
    out_path = tmp_path_factory.mktemp("videos") / "test_2s_30fps.mp4"
    fps = 30
    width, height = 320, 240
    total_frames = 60  # 2 seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    for i in range(total_frames):
        # Gradient color per frame for easy verification
        hue = int(i / total_frames * 180)
        frame = np.full((height, width, 3), (hue, 200, 200), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        writer.write(frame)

    writer.release()
    return out_path


@pytest.fixture(scope="session")
def fixture_video_10s_60fps(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a 10-second test video at 60fps (600 frames total).

    Used to test FPS downsampling in analysis mode.
    """
    out_path = tmp_path_factory.mktemp("videos") / "test_10s_60fps.mp4"
    fps = 60
    width, height = 320, 240
    total_frames = 600  # 10 seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    for i in range(total_frames):
        hue = int(i / total_frames * 180)
        frame = np.full((height, width, 3), (hue, 200, 200), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        writer.write(frame)

    writer.release()
    return out_path
