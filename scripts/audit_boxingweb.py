#!/usr/bin/env python3
"""Create a deterministic inventory and validation report for BoxingWeb.

The script reads the dataset in place; it never modifies videos, labels, or pose
files.  It intentionally uses only the Python standard library plus NumPy,
which is required to unpickle the supplied pose arrays.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ATTRIBUTE_FIELDS = ("side", "technique", "distance", "target", "effect")
POSE_KEYS = ("pose_red_2d", "pose_red_3d", "pose_blue_2d", "pose_blue_3d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "boxingweb",
        help="Directory containing data_train and data_test (default: ~/boxingweb).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/boxingweb-audit.json"),
        help="Path for the JSON report.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum examples stored per issue type.",
    )
    return parser.parse_args()


def ffprobe(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,width,height,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError("no video stream")
    stream = streams[0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator) if denominator != "0" else None
    return {
        "fps": fps,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_count": int(stream["nb_frames"]) if stream.get("nb_frames", "N/A") != "N/A" else None,
        "duration_seconds": float(stream["duration"]) if stream.get("duration", "N/A") != "N/A" else None,
    }


def add_example(bucket: dict[str, list[dict[str, Any]]], key: str, value: dict[str, Any], limit: int) -> None:
    examples = bucket.setdefault(key, [])
    if len(examples) < limit:
        examples.append(value)


def integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def inspect_pose(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        pose = pickle.load(file)
    if not isinstance(pose, dict):
        return {"valid": False, "reason": f"expected dict, got {type(pose).__name__}"}

    arrays: dict[str, Any] = {}
    for key in POSE_KEYS:
        value = pose.get(key)
        arrays[key] = {
            "present": value is not None,
            "shape": list(value.shape) if getattr(value, "shape", None) else None,
        }
    frame_lengths = {details["shape"][0] for details in arrays.values() if details["shape"]}
    return {
        "valid": all(details["present"] and details["shape"] for details in arrays.values()),
        "arrays": arrays,
        "frame_count": next(iter(frame_lengths)) if len(frame_lengths) == 1 else None,
        "consistent_frame_count": len(frame_lengths) == 1,
    }


def main() -> None:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {root}")

    issue_counts: Counter[str] = Counter()
    issue_examples: dict[str, list[dict[str, Any]]] = {}
    attribute_counts = {field: Counter() for field in ATTRIBUTE_FIELDS}
    event_name_counts: Counter[str] = Counter()
    split_summaries: dict[str, dict[str, int]] = {}
    video_records: list[dict[str, Any]] = []

    for split in ("data_train", "data_test"):
        split_dir = root / split
        matches = sorted(path for path in split_dir.iterdir() if path.is_dir()) if split_dir.is_dir() else []
        split_summaries[split] = {"matches": len(matches), "punch_events": 0}

        for match_dir in matches:
            record: dict[str, Any] = {"split": split, "match": match_dir.name}
            videos = sorted(match_dir.glob("*.mp4"))
            pose_files = sorted(match_dir.glob("*_pose_gt.pkl"))
            events_path = match_dir / "video_event.json"
            if len(videos) != 1:
                issue_counts["video_file_count"] += 1
                add_example(issue_examples, "video_file_count", record | {"count": len(videos)}, args.max_examples)
                continue
            if len(pose_files) != 1:
                issue_counts["pose_file_count"] += 1
                add_example(issue_examples, "pose_file_count", record | {"count": len(pose_files)}, args.max_examples)

            try:
                video = ffprobe(videos[0])
                record["video"] = video
            except (subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as error:
                issue_counts["unreadable_video"] += 1
                add_example(issue_examples, "unreadable_video", record | {"error": str(error)}, args.max_examples)
                continue

            try:
                pose = inspect_pose(pose_files[0])
                record["pose"] = pose
                if not pose["valid"]:
                    issue_counts["invalid_pose_schema"] += 1
                    add_example(issue_examples, "invalid_pose_schema", record, args.max_examples)
                if not pose.get("consistent_frame_count"):
                    issue_counts["inconsistent_pose_frame_count"] += 1
                    add_example(issue_examples, "inconsistent_pose_frame_count", record, args.max_examples)
                elif video["frame_count"] is not None and pose["frame_count"] != video["frame_count"]:
                    issue_counts["pose_video_frame_count_mismatch"] += 1
                    add_example(issue_examples, "pose_video_frame_count_mismatch", record, args.max_examples)
            except (OSError, pickle.UnpicklingError, EOFError, AttributeError) as error:
                issue_counts["unreadable_pose"] += 1
                add_example(issue_examples, "unreadable_pose", record | {"error": str(error)}, args.max_examples)

            try:
                events = json.loads(events_path.read_text())
                if not isinstance(events, list):
                    raise ValueError(f"expected list, got {type(events).__name__}")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                issue_counts["unreadable_event_json"] += 1
                add_example(issue_examples, "unreadable_event_json", record | {"error": str(error)}, args.max_examples)
                continue

            punch_events = 0
            for event_index, event in enumerate(events):
                if not isinstance(event, dict):
                    issue_counts["non_object_event"] += 1
                    add_example(issue_examples, "non_object_event", record | {"event_index": event_index}, args.max_examples)
                    continue
                name = event.get("name", "")
                event_name_counts[str(name)] += 1
                if name != "punching":
                    continue
                punch_events += 1
                begin, end = integer(event.get("frame_begin")), integer(event.get("frame_end"))
                event_ref = record | {"event_index": event_index, "frame_begin": event.get("frame_begin"), "frame_end": event.get("frame_end")}
                if begin is None or end is None:
                    issue_counts["non_integer_punch_frame"] += 1
                    add_example(issue_examples, "non_integer_punch_frame", event_ref, args.max_examples)
                    continue
                if end < begin:
                    issue_counts["reversed_punch_interval"] += 1
                    add_example(issue_examples, "reversed_punch_interval", event_ref, args.max_examples)
                if end == begin:
                    issue_counts["zero_length_punch_interval"] += 1
                    add_example(issue_examples, "zero_length_punch_interval", event_ref, args.max_examples)
                if video["frame_count"] is not None and (begin >= video["frame_count"] or end >= video["frame_count"]):
                    issue_counts["punch_outside_video"] += 1
                    add_example(issue_examples, "punch_outside_video", event_ref | {"video_frame_count": video["frame_count"]}, args.max_examples)
                for field in ATTRIBUTE_FIELDS:
                    value = event.get(field, "")
                    attribute_counts[field][str(value)] += 1
                    if value in (None, ""):
                        issue_counts[f"missing_punch_{field}"] += 1
                        add_example(issue_examples, f"missing_punch_{field}", event_ref, args.max_examples)

            record["event_count"] = len(events)
            record["punch_event_count"] = punch_events
            split_summaries[split]["punch_events"] += punch_events
            video_records.append(record)

    report = {
        "dataset_root": str(root),
        "scope": "Inventory and structural validation only; zero-length events are reported, not discarded.",
        "splits": split_summaries,
        "event_name_counts": dict(sorted(event_name_counts.items())),
        "punch_attribute_counts": {field: dict(sorted(counts.items())) for field, counts in attribute_counts.items()},
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_examples": issue_examples,
        "videos": video_records,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {output}")
    print(json.dumps({"splits": split_summaries, "issue_counts": report["issue_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
