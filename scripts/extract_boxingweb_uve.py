#!/usr/bin/env python3
"""Batch GT-free UVE+RTMW pose extraction over BoxingWeb matches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--match-id", action="append", default=[], help="Optional exact match filter; repeat as needed.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--identity-checkpoint", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--pose-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-frames", type=int, default=32)
    parser.add_argument("--cadence", type=int, default=10)
    parser.add_argument("--spatial-bonus", type=float, default=2.0)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def jobs(args: argparse.Namespace):
    splits = ("train", "test") if args.split == "both" else (args.split,)
    selected = set(args.match_id)
    output = []
    for split in splits:
        directory = args.data_root / ("data_train" if split == "train" else "data_test")
        for match_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
            if selected and match_dir.name not in selected:
                continue
            videos = sorted(match_dir.glob("*.mp4"))
            if len(videos) != 1:
                raise ValueError(f"Expected one video in {match_dir}")
            pose = args.output_root / split / f"{match_dir.name}.pkl"
            tracks = args.output_root / "tracks" / split / f"{match_dir.name}.npz"
            output.append((split, match_dir.name, videos[0], pose, tracks))
    return output


def run_job(args: argparse.Namespace, job):
    split, match_id, video, output, tracks = job
    if output.exists() and not args.force:
        return {"split": split, "match_id": match_id, "output": str(output), "status": "cached"}
    output.parent.mkdir(parents=True, exist_ok=True)
    tracks.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python), str(Path(__file__).with_name("extract_uve_rtmw_pose.py")),
        "--video", str(video), "--output", str(output), "--tracks-output", str(tracks),
        "--identity-checkpoint", str(args.identity_checkpoint),
        "--pose-config", str(args.pose_config), "--pose-checkpoint", str(args.pose_checkpoint),
        "--yolo", str(args.yolo), "--device", args.device,
        "--batch-frames", str(args.batch_frames), "--cadence", str(args.cadence),
        "--spatial-bonus", str(args.spatial_bonus), "--smooth-window", str(args.smooth_window),
        "--interpolate-all-missing",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{split}/{match_id} failed:\n{result.stdout}\n{result.stderr}")
    summary = next((line for line in reversed(result.stdout.splitlines()) if line.startswith("pose=")), "complete")
    return {"split": split, "match_id": match_id, "output": str(output), "tracks": str(tracks), "status": summary}


def main() -> None:
    args = parse_args()
    work = jobs(args)
    if not work:
        raise ValueError("No matches selected")
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, args, job): job for job in work}
        for number, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            completed.append(result)
            print(f"matches={number}/{len(work)} {result['split']}/{result['match_id']} {result['status']}", flush=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "GT-free YOLO BoT-SORT + RTMW + periodic RGB UVE",
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "matches": sorted(completed, key=lambda item: (item["split"], item["match_id"])),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
