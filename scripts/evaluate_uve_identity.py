#!/usr/bin/env python3
"""Evaluate UVE red/blue identity assignments against GT pose boxes."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.rtmw_pose import bbox_iou, pose_bbox  # noqa: E402


def identity_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "id_precision": precision,
        "id_recall": recall,
        "idf1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "idtp": tp, "idfp": fp, "idfn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--minimum-iou", type=float, default=.05)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    with args.prediction.open("rb") as file:
        prediction = pickle.load(file)
    with args.ground_truth.open("rb") as file:
        truth = pickle.load(file)
    width, height = int(prediction["width"]), int(prediction["height"])
    source_start = int(prediction.get("source_start_frame", 0))
    frame_count = min(
        len(np.asarray(prediction["pose_red_2d"])),
        len(np.asarray(prediction["pose_blue_2d"])),
        len(np.asarray(truth["pose_red_2d"])) - source_start,
        len(np.asarray(truth["pose_blue_2d"])) - source_start,
    )
    counts = {side: [0, 0, 0] for side in ("red", "blue")}
    for local_frame in range(frame_count):
        source_frame = source_start + local_frame
        gt_boxes = {
            side: pose_bbox(truth[f"pose_{side}_2d"][source_frame], width, height)
            for side in ("red", "blue")
        }
        for side, other in (("red", "blue"), ("blue", "red")):
            valid = bool(np.asarray(prediction.get(f"valid_{side}", np.ones(frame_count, bool)))[local_frame])
            if valid and f"box_{side}" in prediction:
                candidate_box = np.asarray(prediction[f"box_{side}"][local_frame], dtype=np.float32)
                predicted_box = candidate_box if np.any(candidate_box[2:] > candidate_box[:2]) else None
            else:
                predicted_box = pose_bbox(prediction[f"pose_{side}_2d"][local_frame], width, height) if valid else None
            correct_iou = bbox_iou(predicted_box, gt_boxes[side]) if predicted_box is not None and gt_boxes[side] is not None else 0.0
            wrong_iou = bbox_iou(predicted_box, gt_boxes[other]) if predicted_box is not None and gt_boxes[other] is not None else 0.0
            correct = correct_iou >= args.minimum_iou and correct_iou >= wrong_iou
            if correct:
                counts[side][0] += 1
            else:
                if predicted_box is not None:
                    counts[side][1] += 1
                if gt_boxes[side] is not None:
                    counts[side][2] += 1
    totals = np.sum(np.asarray(list(counts.values())), axis=0).tolist()
    payload = {
        "prediction": str(args.prediction.resolve()),
        "ground_truth": str(args.ground_truth.resolve()),
        "frames": frame_count,
        "minimum_iou": args.minimum_iou,
        "overall": identity_metrics(*totals),
        "by_side": {side: identity_metrics(*values) for side, values in counts.items()},
        "coverage": {
            side: float(np.asarray(prediction.get(f"valid_{side}", np.ones(frame_count, bool)))[:frame_count].mean())
            for side in ("red", "blue")
        },
        "unique_track_ids": {
            side: sorted(set(int(value) for value in prediction.get(f"{side}_track_ids", []) if value >= 0))
            for side in ("red", "blue")
        },
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
