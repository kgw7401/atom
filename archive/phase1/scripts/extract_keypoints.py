#!/usr/bin/env python3
"""Batch extract keypoints from recorded videos.

Processes all .mp4 files in data/raw/, extracts 33-keypoint poses via
MediaPipe, and saves per-video .npy arrays + metadata to data/keypoints/.

Usage:
    python scripts/extract_keypoints.py
    python scripts/extract_keypoints.py --raw-dir data/raw --out-dir data/keypoints
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.extraction.pose_extractor import PoseExtractor


def parse_filename(filename: str) -> dict:
    """Extract label metadata from filename convention.

    Expected: {subject}_{action}_{angle}_{speed}.mp4
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) == 4:
        return {
            "subject": parts[0],
            "action": parts[1],
            "angle": parts[2],
            "speed": parts[3],
        }
    elif len(parts) == 5:
        # handle actions with underscore like lead_hook
        return {
            "subject": parts[0],
            "action": f"{parts[1]}_{parts[2]}",
            "angle": parts[3],
            "speed": parts[4],
        }
    elif len(parts) == 6:
        # handle actions like lead_uppercut with angle and speed
        return {
            "subject": parts[0],
            "action": f"{parts[1]}_{parts[2]}",
            "angle": parts[3],
            "speed": parts[4],
        }
    else:
        raise ValueError(f"Cannot parse filename: {filename} (got {len(parts)} parts)")


def extract_video(
    video_path: Path,
    extractor: PoseExtractor,
) -> tuple[np.ndarray, list[float]]:
    """Extract keypoints from a single video.

    Returns:
        keypoints: (N, 33, 4) array
        timestamps: list of N timestamp values (ms)
    """
    keypoints = []
    timestamps = []

    for frame in extractor.process_video(str(video_path)):
        keypoints.append(frame.keypoints)
        timestamps.append(frame.timestamp_ms)

    return np.stack(keypoints), timestamps


def main():
    parser = argparse.ArgumentParser(description="Batch extract keypoints from videos")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="data/keypoints")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(raw_dir.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {raw_dir}")
        return

    print(f"Found {len(videos)} videos in {raw_dir}")

    with PoseExtractor() as extractor:
        for i, video_path in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] {video_path.name}")

            try:
                meta = parse_filename(video_path.name)
            except ValueError as e:
                print(f"  SKIP: {e}")
                continue

            t0 = time.time()
            keypoints, timestamps = extract_video(video_path, extractor)
            elapsed = time.time() - t0

            # Save keypoints as .npy
            npy_path = out_dir / f"{video_path.stem}.npy"
            np.save(npy_path, keypoints)

            # Save metadata as .json
            meta_path = out_dir / f"{video_path.stem}.json"
            meta.update({
                "source_video": video_path.name,
                "num_frames": len(timestamps),
                "shape": list(keypoints.shape),
                "timestamps_ms": [round(t, 1) for t in timestamps],
            })
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            fps = len(timestamps) / elapsed if elapsed > 0 else 0
            print(f"  {keypoints.shape} | {elapsed:.1f}s ({fps:.0f} fps) | {meta['action']}")

    print(f"\nDone. Keypoints saved to {out_dir}/")


if __name__ == "__main__":
    main()
