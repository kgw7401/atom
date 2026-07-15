#!/usr/bin/env python3
"""Extract aligned RTMW poses for every BoxingWeb match in one or both splits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "scripts" / "extract_rtmw3d_pose.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--pose-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-frames", type=int, default=24)
    parser.add_argument("--inference-stride", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--flip-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def match_jobs(args: argparse.Namespace) -> list[tuple[str, str, Path, Path, Path]]:
    splits = ("train", "test") if args.split == "both" else (args.split,)
    jobs = []
    for split in splits:
        for match_dir in sorted((args.data_root / f"data_{split}").iterdir()):
            if not match_dir.is_dir():
                continue
            videos = list(match_dir.glob("*.mp4"))
            poses = list(match_dir.glob("*_pose_gt.pkl"))
            if len(videos) != 1 or len(poses) != 1:
                raise ValueError(f"Expected one video and one GT pose in {match_dir}")
            output = args.output_root / split / f"{match_dir.name}.pkl"
            jobs.append((split, match_dir.name, videos[0], poses[0], output))
    return jobs


def run_job(args: argparse.Namespace, job: tuple[str, str, Path, Path, Path]) -> dict[str, str]:
    split, match_id, video, oracle, output = job
    if output.exists() and not args.force:
        return {"split": split, "match_id": match_id, "output": str(output), "status": "cached"}
    command = [
        str(args.python), str(EXTRACTOR),
        "--video", str(video),
        "--oracle-identity-pose", str(oracle),
        "--output", str(output),
        "--pose-config", str(args.pose_config),
        "--pose-checkpoint", str(args.pose_checkpoint),
        "--yolo", str(args.yolo),
        "--device", args.device,
        "--batch-frames", str(args.batch_frames),
        "--inference-stride", str(args.inference_stride),
        "--smooth-window", str(args.smooth_window),
        "--interpolate-all-missing",
    ]
    if args.flip_test:
        command.append("--flip-test")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{match_id} failed:\n{result.stdout}\n{result.stderr}")
    summary = next((line for line in reversed(result.stdout.splitlines()) if line.startswith("pose=")), "completed")
    return {"split": split, "match_id": match_id, "output": str(output), "status": summary}


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    jobs = match_jobs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, args, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(f"matches={completed}/{len(jobs)} {record['split']}/{record['match_id']} {record['status']}", flush=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root.resolve()),
        "pose_model": str(args.pose_checkpoint.resolve()),
        "inference_stride": args.inference_stride,
        "smooth_window": args.smooth_window,
        "flip_test": args.flip_test,
        "matches": sorted(records, key=lambda item: (item["split"], item["match_id"])),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
