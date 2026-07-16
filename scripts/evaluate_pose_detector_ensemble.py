#!/usr/bin/env python3
"""Evaluate a frozen detector ensemble without tuning on the requested split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--confirm-test", action="store_true",
                        help="Required for the one-shot held-out test evaluation.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_outputs(data_root: Path, pose_root: Path, split: str, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    rounds = build_boxer_rounds(
        data_root, split, float(checkpoint["offset_scale"]),
        bool(checkpoint.get("include_opponent", False)), int(checkpoint.get("pose_channels", 5)),
        pose_root, str(checkpoint.get("feature_mode", "absolute")),
    )
    model = build_punch_detector(
        str(checkpoint.get("architecture", "tcn")), int(np.prod(rounds[0][2].shape[1:])),
        channels=int(checkpoint.get("channels", 64)), batch_norm=bool(checkpoint.get("batch_norm", False)),
        dilations=tuple(checkpoint.get("dilations", (1, 2, 4, 8, 16))),
        dropout=float(checkpoint.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    outputs = collect_logits(
        model, rounds, np.asarray(checkpoint["feature_mean"]), np.asarray(checkpoint["feature_std"]),
        float(checkpoint.get("feature_clip", 10.0)), torch.device("cpu"),
    )
    return checkpoint, rounds, outputs


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("Pass --confirm-test only after the ensemble and operating point are frozen.")
    specification = json.loads(args.ensemble.read_text())
    loaded = []
    for component in specification["components"]:
        checkpoint_path = Path(component["checkpoint"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = args.ensemble.parent / checkpoint_path
        loaded.append((float(component["weight"]), *load_outputs(
            args.data_root, args.pose_root, args.split, checkpoint_path,
        )))
    offset_scales = {float(checkpoint["offset_scale"]) for _, checkpoint, _, _ in loaded}
    if len(offset_scales) != 1:
        raise ValueError("All checkpoints must use the same offset scale")
    offset_scale = offset_scales.pop()
    if not np.isclose(offset_scale, float(specification.get("offset_scale", offset_scale))):
        raise ValueError("Ensemble and checkpoint offset scales do not match")
    combined = []
    reference_rounds = loaded[0][2]
    for items in zip(*(outputs for _, _, _, outputs in loaded)):
        side, truth = items[0][0], items[0][2]
        if any(item[0] != side or item[2] != truth for item in items[1:]):
            raise ValueError("Checkpoint outputs are not aligned")
        logits = sum(weight * item[1] for (weight, *_), item in zip(loaded, items))
        combined.append((side, logits, truth))
    threshold = float(specification["threshold"])
    nms_iou = float(specification["nms_iou"])
    score = score_logits(combined, threshold, nms_iou, offset_scale)
    match_scores = {}
    for match_id in sorted({item[0] for item in reference_rounds}):
        indices = [index for index, item in enumerate(reference_rounds) if item[0] == match_id]
        match_scores[match_id] = score_logits(
            [combined[index] for index in indices], threshold, nms_iou,
            offset_scale,
        )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": specification.get("model", "rtmw-punch-detector-ensemble"),
        "split": args.split,
        "pose_root": str(args.pose_root.resolve()),
        "temporal_iou": 0.5,
        "threshold": threshold,
        "nms_iou": nms_iou,
        "score": score,
        "per_match": match_scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(score), flush=True)


if __name__ == "__main__":
    main()
