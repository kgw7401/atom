#!/usr/bin/env python3
"""Trim idle guard frames from the start/end of action segments.

Detects motion boundaries using wrist velocity and trims each segment
to only include the actual punch motion. Guard segments are left untouched.

Usage:
    python scripts/trim_segments.py              # dry run (preview)
    python scripts/trim_segments.py --apply      # overwrite .npy files
"""

import argparse
import json
from pathlib import Path

import numpy as np


# MediaPipe landmark indices for wrists
LEFT_WRIST = 15
RIGHT_WRIST = 16
MARGIN = 1  # keep N extra frames around detected motion


def detect_motion_bounds(
    keypoints: np.ndarray,
    threshold_ratio: float = 0.15,
) -> tuple[int, int]:
    """Find first and last frames with significant wrist motion.

    Args:
        keypoints: (N, 33, 4) raw keypoints
        threshold_ratio: fraction of peak velocity to use as cutoff

    Returns:
        (start, end) frame indices (inclusive)
    """
    # Wrist positions (x, y only — z is noisy)
    lw = keypoints[:, LEFT_WRIST, :2]   # (N, 2)
    rw = keypoints[:, RIGHT_WRIST, :2]  # (N, 2)

    # Frame-to-frame displacement
    lw_vel = np.linalg.norm(np.diff(lw, axis=0), axis=1)  # (N-1,)
    rw_vel = np.linalg.norm(np.diff(rw, axis=0), axis=1)  # (N-1,)
    combined = np.maximum(lw_vel, rw_vel)  # (N-1,)

    # Threshold: fraction of peak velocity
    peak = combined.max()
    if peak < 1e-6:
        return 0, len(keypoints) - 1  # no motion detected, keep all

    threshold = peak * threshold_ratio

    # Find first and last frames above threshold
    above = np.where(combined >= threshold)[0]
    if len(above) == 0:
        return 0, len(keypoints) - 1

    # +1 offset because velocity[i] is between frame i and i+1
    start = max(0, above[0] - MARGIN)
    end = min(len(keypoints) - 1, above[-1] + 1 + MARGIN)

    return start, end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Overwrite .npy files")
    parser.add_argument("--dir", type=Path, default=Path("data/keypoints"))
    args = parser.parse_args()

    total_before = 0
    total_after = 0
    trimmed_count = 0

    for npy_path in sorted(args.dir.glob("*.npy")):
        meta_path = npy_path.with_suffix(".json")
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        action = meta["action"]

        # Skip guard — no trimming needed
        if action == "guard":
            continue

        raw = np.load(npy_path)
        n_before = len(raw)
        total_before += n_before

        start, end = detect_motion_bounds(raw)
        trimmed = raw[start : end + 1]
        n_after = len(trimmed)
        total_after += n_after

        removed = n_before - n_after
        if removed > 0:
            trimmed_count += 1
            print(f"  {npy_path.name:40s}  {n_before:3d} → {n_after:3d}  (-{removed} frames)  [{action}]")

            if args.apply:
                np.save(npy_path, trimmed)

    print(f"\nSummary: {trimmed_count} segments trimmed")
    print(f"  Frames: {total_before} → {total_after} (-{total_before - total_after})")

    if not args.apply:
        print("\nDry run. Use --apply to overwrite files.")


if __name__ == "__main__":
    main()
