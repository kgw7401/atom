#!/usr/bin/env python3
"""Run the complete GT-free MP4-to-punch-event pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Detected punch-event JSON.")
    parser.add_argument("--pose-output", type=Path, required=True,
                        help="Persistent canonical UVE pose pickle used by the detector.")
    parser.add_argument("--tracks-output", type=Path, default=None)
    parser.add_argument("--ensemble", type=Path, default=Path("results/rtmw-punch-detector-ensemble.json"))
    parser.add_argument("--identity-checkpoint", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--pose-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-frames", type=int, default=32)
    parser.add_argument("--cadence", type=int, default=10)
    parser.add_argument("--spatial-bonus", type=float, default=2.0)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--fps", type=float, default=None,
                        help="Optional source FPS for second-based event timestamps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scripts = Path(__file__).resolve().parent
    tracks_output = args.tracks_output or args.pose_output.with_suffix(".tracks.npz")
    extraction = [
        sys.executable, str(scripts / "extract_uve_rtmw_pose.py"),
        "--video", str(args.video), "--output", str(args.pose_output),
        "--tracks-output", str(tracks_output),
        "--identity-checkpoint", str(args.identity_checkpoint),
        "--pose-config", str(args.pose_config), "--pose-checkpoint", str(args.pose_checkpoint),
        "--yolo", str(args.yolo), "--device", args.device,
        "--batch-frames", str(args.batch_frames), "--cadence", str(args.cadence),
        "--spatial-bonus", str(args.spatial_bonus), "--smooth-window", str(args.smooth_window),
        "--interpolate-all-missing",
    ]
    subprocess.run(extraction, check=True)
    detection = [
        sys.executable, str(scripts / "detect_punch_events.py"),
        "--pose", str(args.pose_output), "--ensemble", str(args.ensemble),
        "--output", str(args.output),
    ]
    if args.fps is not None:
        detection.extend(("--fps", str(args.fps)))
    subprocess.run(detection, check=True)


if __name__ == "__main__":
    main()
