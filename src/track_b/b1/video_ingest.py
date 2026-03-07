"""B1 Task 1: Video ingestion module.

Unified video loader with two modes:
- Training mode: Local video file (short trimmed clips). Extract all frames.
- Analysis mode: Local file or YouTube URL (via yt-dlp). Configurable FPS. Time-range trimming.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoFrame:
    """A single extracted video frame with metadata."""

    frame_number: int
    timestamp: float  # seconds from video start
    image: np.ndarray  # BGR image (H, W, 3)


@dataclass
class VideoMetadata:
    """Metadata about the source video."""

    source: str  # file path or URL
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float  # seconds


def _is_youtube_url(source: str) -> bool:
    """Check if source is a YouTube URL."""
    return any(
        domain in source
        for domain in ("youtube.com", "youtu.be", "youtube-nocookie.com")
    )


def _download_youtube(url: str, output_dir: Path) -> Path:
    """Download YouTube video using yt-dlp. Returns path to downloaded file."""
    output_path = output_dir / "%(id)s.%(ext)s"
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(output_path),
        "--no-playlist",
        "--quiet",
        url,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Find the downloaded file
    mp4_files = list(output_dir.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"yt-dlp did not produce an mp4 file in {output_dir}")
    return mp4_files[0]


def extract_frames_training(video_path: str | Path) -> tuple[VideoMetadata, list[VideoFrame]]:
    """Training mode: extract ALL frames from a short trimmed clip.

    Used for BoxingVI clips and other labeled training data.
    No FPS sampling, no detection — just raw frame extraction.

    Args:
        video_path: Path to local video file.

    Returns:
        Tuple of (metadata, list of VideoFrame).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    metadata = VideoMetadata(
        source=str(video_path),
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration=duration,
    )

    frames: list[VideoFrame] = []
    frame_number = 0

    while True:
        ret, image = cap.read()
        if not ret:
            break
        timestamp = frame_number / fps if fps > 0 else 0.0
        frames.append(VideoFrame(
            frame_number=frame_number,
            timestamp=timestamp,
            image=image,
        ))
        frame_number += 1

    cap.release()
    return metadata, frames


def extract_frames_analysis(
    source: str | Path,
    target_fps: int = 30,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[VideoMetadata, list[VideoFrame]]:
    """Analysis mode: extract frames from local file or YouTube URL.

    Supports configurable FPS sampling and time-range trimming.

    Args:
        source: Local file path or YouTube URL.
        target_fps: Target frames per second (default 30). If video FPS <= target_fps,
                     all frames are extracted.
        start_time: Start time in seconds (optional).
        end_time: End time in seconds (optional).

    Returns:
        Tuple of (metadata, list of VideoFrame).
    """
    temp_dir = None
    video_path: Path

    if isinstance(source, str) and _is_youtube_url(source):
        temp_dir = tempfile.mkdtemp(prefix="atom_b1_")
        video_path = _download_youtube(source, Path(temp_dir))
    else:
        video_path = Path(source)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / source_fps if source_fps > 0 else 0.0

    metadata = VideoMetadata(
        source=str(source),
        width=width,
        height=height,
        fps=source_fps,
        total_frames=total_frames,
        duration=duration,
    )

    # Calculate frame sampling interval
    # If source_fps <= target_fps, take every frame
    if source_fps <= target_fps:
        frame_interval = 1
    else:
        frame_interval = source_fps / target_fps

    # Calculate start/end frame numbers
    start_frame = int(start_time * source_fps) if start_time is not None else 0
    end_frame = int(end_time * source_fps) if end_time is not None else total_frames

    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[VideoFrame] = []
    current_frame = start_frame
    next_sample_frame = float(start_frame)
    output_frame_number = 0

    while current_frame < end_frame:
        ret, image = cap.read()
        if not ret:
            break

        if current_frame >= next_sample_frame:
            timestamp = current_frame / source_fps if source_fps > 0 else 0.0
            frames.append(VideoFrame(
                frame_number=output_frame_number,
                timestamp=timestamp,
                image=image,
            ))
            output_frame_number += 1
            next_sample_frame += frame_interval

        current_frame += 1

    cap.release()
    return metadata, frames
