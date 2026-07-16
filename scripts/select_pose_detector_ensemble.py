#!/usr/bin/env python3
"""Select a GT-free pose-detector ensemble on held-out training matches."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from atom.anchor_free_detection import build_punch_detector  # noqa: E402
from train_boxmind_anchor_free_detector import build_boxer_rounds, collect_logits, score_logits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--pose-root", type=Path, action="append", required=True,
        help="Pose root for each checkpoint; pass once to share one root or once per checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--validation-matches", type=int, default=4)
    parser.add_argument("--weight-step", type=float, default=0.25)
    parser.add_argument("--threshold-min", type=float, default=0.3)
    parser.add_argument("--threshold-max", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--nms-iou", type=float, action="append",
        help="NMS IoU candidate; repeat as needed. Defaults to 0.2, 0.3, 0.5, and 0.7.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validation_logits(
    args: argparse.Namespace,
    checkpoint_path: Path,
    pose_root: Path,
    validation_ids: set[str],
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    rounds = build_boxer_rounds(
        args.data_root, "train", float(checkpoint["offset_scale"]),
        bool(checkpoint.get("include_opponent", False)), int(checkpoint.get("pose_channels", 5)),
        pose_root, str(checkpoint.get("feature_mode", "absolute")),
    )
    rounds = [item for item in rounds if item[0] in validation_ids]
    model = build_punch_detector(
        str(checkpoint.get("architecture", "tcn")),
        int(np.prod(rounds[0][2].shape[1:])), channels=int(checkpoint.get("channels", 64)),
        batch_norm=bool(checkpoint.get("batch_norm", False)),
        dilations=tuple(checkpoint.get("dilations", (1, 2, 4, 8, 16))),
        dropout=float(checkpoint.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    outputs = collect_logits(
        model, rounds, np.asarray(checkpoint["feature_mean"]), np.asarray(checkpoint["feature_std"]),
        float(checkpoint.get("feature_clip", 10.0)), torch.device("cpu"),
    )
    return checkpoint, outputs


def weight_grid(count: int, step: float):
    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0) or units < 1:
        raise ValueError("weight-step must divide 1.0 exactly")
    for values in itertools.product(range(units + 1), repeat=count):
        if sum(values) == units:
            yield tuple(value / units for value in values)


def main() -> None:
    args = parse_args()
    if len(args.pose_root) == 1:
        pose_roots = args.pose_root * len(args.checkpoint)
    elif len(args.pose_root) == len(args.checkpoint):
        pose_roots = args.pose_root
    else:
        raise ValueError("Pass --pose-root once or exactly once per checkpoint")
    match_ids = sorted(path.stem for path in (pose_roots[0] / "train").glob("*.pkl"))
    if len(match_ids) < args.validation_matches:
        raise ValueError("Not enough extracted training matches")
    validation_ids = set(match_ids[-args.validation_matches:])
    loaded = [
        validation_logits(args, path, pose_root, validation_ids)
        for path, pose_root in zip(args.checkpoint, pose_roots)
    ]
    offset_scales = {float(checkpoint["offset_scale"]) for checkpoint, _ in loaded}
    if len(offset_scales) != 1:
        raise ValueError("All checkpoints must use the same offset scale")
    offset_scale = offset_scales.pop()
    best = None
    for weights in weight_grid(len(loaded), args.weight_step):
        outputs = []
        for items in zip(*(output for _, output in loaded)):
            side, truth = items[0][0], items[0][2]
            if any(item[0] != side or item[2] != truth for item in items[1:]):
                raise ValueError("Checkpoint outputs are not aligned to the same boxer rounds")
            logits = sum(weight * item[1] for weight, item in zip(weights, items))
            outputs.append((side, logits, truth))
        thresholds = np.arange(
            args.threshold_min, args.threshold_max + args.threshold_step * 0.5, args.threshold_step,
        )
        for threshold in thresholds:
            for nms_iou in args.nms_iou or (0.2, 0.3, 0.5, 0.7):
                score = score_logits(outputs, float(threshold), nms_iou, offset_scale)
                candidate = (score["f1"], score["recall"], weights, float(threshold), nms_iou, score)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    assert best is not None
    _, _, weights, threshold, nms_iou, score = best
    payload = {
        "model": "rtmw-punch-detector-ensemble",
        "components": [
            {
                "checkpoint": str(path.resolve()),
                "pose_root": str(pose_root.resolve()),
                "pose_variant": pose_root.name,
                "weight": weight,
            }
            for path, pose_root, weight in zip(args.checkpoint, pose_roots, weights) if weight > 0
        ],
        "threshold": threshold,
        "nms_iou": nms_iou,
        "offset_scale": offset_scale,
        "selection": {
            "split": "held-out training matches",
            "match_ids": sorted(validation_ids),
            "temporal_iou": 0.5,
            **score,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["selection"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
