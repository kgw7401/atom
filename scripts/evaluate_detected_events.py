#!/usr/bin/env python3
"""Score a detected-event JSON against one BoxingWeb match at temporal IoU 0.5."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.anchor_free_detection import AnchorFreeEvent, temporal_iou  # noqa: E402
from atom.boxingweb import build_oracle_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "boxingweb")
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = json.loads(args.detections.read_text())
    predicted = [
        AnchorFreeEvent(
            side=str(event["side"]),
            hand=str(event["hand"]),
            start_frame=int(event["start_frame"]) + args.frame_offset,
            end_frame=int(event["end_frame"]) + args.frame_offset,
            score=float(event["score"]),
        )
        for event in raw["events"]
    ]
    truth = [
        AnchorFreeEvent(
            side=sample.labels["side"],
            hand="right" if sample.labels["technique"].startswith("r") else "left",
            start_frame=sample.event_start_frame,
            end_frame=sample.event_end_frame,
            score=1.0,
        )
        for sample in build_oracle_index(args.data_root, args.split).samples
        if sample.match_id == args.match_id
    ]
    remaining = set(range(len(truth)))
    matches: list[tuple[AnchorFreeEvent, AnchorFreeEvent, float]] = []
    false_positives: list[AnchorFreeEvent] = []
    for event in sorted(predicted, key=lambda item: item.score, reverse=True):
        candidates = [
            (temporal_iou(event, truth[index]), index)
            for index in remaining
            if event.side == truth[index].side and event.hand == truth[index].hand
        ]
        if candidates and max(candidates)[0] >= 0.5:
            overlap, index = max(candidates)
            remaining.remove(index)
            matches.append((event, truth[index], overlap))
        else:
            false_positives.append(event)
    false_negatives = [truth[index] for index in sorted(remaining)]
    tp, fp, fn = len(matches), len(false_positives), len(false_negatives)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Single-match punch-event detection; matches require the same boxer, hand, and temporal IoU >= 0.5.",
        "match_id": args.match_id,
        "detections": str(args.detections.resolve()),
        "event_detection_iou_0.5": {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
        },
        "matched_mean_iou": sum(item[2] for item in matches) / tp if tp else 0.0,
        "highest_score_false_positives": [asdict(item) for item in sorted(false_positives, key=lambda item: item.score, reverse=True)[:10]],
        "first_false_negatives": [asdict(item) for item in false_negatives[:10]],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["event_detection_iou_0.5"], ensure_ascii=False))


if __name__ == "__main__":
    main()
