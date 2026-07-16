#!/usr/bin/env python3
"""Detect red/blue left/right punch intervals from a canonical tracked-pose file.

The input pickle must contain the four BoxingWeb-compatible arrays
``pose_{red,blue}_{2d,3d}``, each aligned to the source-video frame index.
The same format is emitted by the video-pose/UVE adapter once its external
tracker has produced two named boxer tracks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.anchor_free_detection import build_punch_detector, decode_events  # noqa: E402
from atom.pose_features import extract_boxer_pose_features, select_pose_feature_channels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", type=Path, required=True, help="Canonical red/blue tracked-pose pickle.")
    parser.add_argument(
        "--pose-variant", action="append", default=[], metavar="NAME=PATH",
        help="Additional canonical pose source for a mixed-rate ensemble; repeat as needed.",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("results/boxmind-anchor-free-gt-pose.pt"))
    parser.add_argument("--ensemble", type=Path, default=None,
                        help="Optional JSON ensemble specification; overrides --checkpoint.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the validation-selected operating threshold in the checkpoint.")
    parser.add_argument("--nms-iou", type=float, default=.5)
    parser.add_argument("--fps", type=float, default=None,
                        help="Optional source-video FPS; adds seconds to every output interval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose_variants = {"default": args.pose}
    for value in args.pose_variant:
        if "=" not in value:
            raise ValueError("--pose-variant must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError("--pose-variant must use non-empty NAME=PATH")
        pose_variants[name] = Path(path)
    if args.ensemble:
        specification = json.loads(args.ensemble.read_text())
        component_specs = specification["components"]
    else:
        specification = None
        component_specs = [{"checkpoint": str(args.checkpoint), "weight": 1.0}]
    components = []
    for component in component_specs:
        checkpoint_path = Path(component["checkpoint"])
        if specification is not None and not checkpoint_path.is_absolute():
            checkpoint_path = args.ensemble.parent / checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        model = build_punch_detector(
            str(checkpoint.get("architecture", "tcn")),
            int(np.prod(mean.shape)),
            channels=int(checkpoint.get("channels", 64)),
            batch_norm=bool(checkpoint.get("batch_norm", False)),
            dilations=tuple(checkpoint.get("dilations", (1, 2, 4, 8, 16))),
            dropout=float(checkpoint.get("dropout", 0.0)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        pose_variant = str(component.get("pose_variant", "default"))
        if pose_variant not in pose_variants:
            raise ValueError(
                f"Ensemble component requires pose variant {pose_variant!r}; "
                f"pass --pose-variant {pose_variant}=PATH"
            )
        components.append((float(component["weight"]), checkpoint, model, mean, pose_variants[pose_variant]))
    total_weight = sum(component[0] for component in components)
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"Ensemble component weights must sum to 1, got {total_weight}")
    first_checkpoint = components[0][1]
    threshold = float(args.threshold if args.threshold is not None else (
        specification["threshold"] if specification else first_checkpoint["threshold"]
    ))
    nms_iou = float(specification["nms_iou"] if specification else first_checkpoint.get("nms_iou", args.nms_iou))
    offset_scale = float(specification.get("offset_scale", first_checkpoint.get("offset_scale", 1.0))
                         if specification else first_checkpoint.get("offset_scale", 1.0))
    events = []
    frame_count = None
    for side in ("red", "blue"):
        component_logits = []
        for weight, checkpoint, model, mean, pose_path in components:
            feature_mode = str(checkpoint.get("feature_mode", "absolute"))
            features = select_pose_feature_channels(extract_boxer_pose_features(
                pose_path, side, bool(checkpoint.get("include_opponent", False)), feature_mode,
            ), int(checkpoint.get("pose_channels", 5)), feature_mode)
            frame_count = len(features) if frame_count is None else min(frame_count, len(features))
            std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
            clip = float(checkpoint.get("feature_clip", 10.0))
            normalized = np.clip((features - mean) / std, -clip, clip).astype(np.float32)
            with torch.no_grad():
                component_logits.append(weight * model(torch.from_numpy(normalized).unsqueeze(0)).squeeze(0).numpy())
        logits = np.sum(component_logits, axis=0)
        events.extend(decode_events(
            logits,
            side,
            threshold=threshold,
            nms_iou=nms_iou,
            offset_scale=offset_scale,
        ))
    events.sort(key=lambda event: (event.start_frame, event.side, event.hand))
    output = {
        "model": specification.get("model", "boxmind-anchor-free-tcn-ensemble") if specification else first_checkpoint.get("model", "boxmind-anchor-free-tcn"),
        "pose_path": str(args.pose.resolve()),
        "pose_variants": {name: str(path.resolve()) for name, path in pose_variants.items()},
        "frame_count": int(frame_count or 0),
        "threshold": threshold,
        "events": [
            {
                "side": event.side,
                "hand": event.hand,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "score": round(event.score, 6),
                **({"start_seconds": event.start_frame / args.fps, "end_seconds": event.end_frame / args.fps} if args.fps else {}),
            }
            for event in events
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Detected {len(events)} punch intervals -> {args.output}")


if __name__ == "__main__":
    main()
