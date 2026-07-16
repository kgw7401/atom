#!/usr/bin/env python3
"""Train the paper-form BoxMind anchor-free detector on full GT-pose rounds."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.anchor_free_detection import AnchorFreeEvent, build_punch_detector, decode_events, encode_targets, focal_loss, temporal_iou  # noqa: E402
from atom.boxingweb import build_oracle_index  # noqa: E402
from atom.pose_features import extract_boxer_pose_features, select_pose_feature_channels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "boxingweb")
    parser.add_argument("--pose-root", type=Path, default=None,
                        help="Optional extracted-pose root containing train/<match>.pkl and test/<match>.pkl.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--positive-windows-per-event", type=int, default=1)
    parser.add_argument("--background-ratio", type=float, default=1.0,
                        help="Number of background windows per positive event.")
    parser.add_argument("--hard-negative-fraction", type=float, default=0.0,
                        help="Fraction of background windows sampled from high-motion non-punch frames.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--scheduler-t-max", type=int, default=None,
                        help="Cosine schedule horizon; defaults to --epochs (set when reproducing an earlier best epoch).")
    parser.add_argument("--focal-alpha", type=float, default=.9)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--offset-scale", type=float, default=32.0)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--iou-regression-weight", type=float, default=0.0,
                        help="Additional positive-frame temporal IoU loss weight.")
    parser.add_argument("--regression-loss", choices=("smooth_l1", "l1"), default="smooth_l1")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--architecture", choices=("tcn", "bigru", "mstcn", "tcngru"), default="tcn")
    parser.add_argument("--temporal-depth", type=int, choices=(5, 6, 7), default=5,
                        help="Number of TCN blocks with dilations 1..2^(depth-1).")
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--feature-clip", type=float, default=10.0)
    parser.add_argument("--normalization", choices=("standard", "raw"), default="standard")
    parser.add_argument("--pose-channels", type=int, choices=(2, 5), default=5,
                        help="Use 2D xy only or the paper's 2D xy + 3D xyz input.")
    parser.add_argument("--feature-mode", choices=("absolute", "local", "absolute-motion", "local-motion", "hybrid-motion", "hybrid-multiscale", "hybrid-kinematic-arm", "hybrid-kinematic-relative", "hybrid-kinematic", "absolute-normalized-3d", "local-normalized-3d", "absolute-depth", "hybrid-depth-motion", "hybrid-motion-3d", "hybrid-multiscale-3d"), default="absolute",
                        help="Use square-normalized image coordinates or actor-centered body-scale coordinates.")
    boxer_input = parser.add_mutually_exclusive_group()
    boxer_input.add_argument("--include-opponent", dest="include_opponent", action="store_true",
                             help="Use the paper-form 28 actor/opponent joints (default).")
    boxer_input.add_argument("--actor-only", dest="include_opponent", action="store_false",
                             help="Ablation using only the focal boxer's 14 joints.")
    parser.set_defaults(include_opponent=True)
    parser.add_argument("--train-all", action="store_true",
                        help="Retrain on all 40 training rounds after selecting hyperparameters.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Do not load or score the held-out test split during model selection.")
    parser.add_argument("--fixed-threshold", type=float, default=.8)
    parser.add_argument("--fixed-nms-iou", type=float, default=.3)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=Path("results/boxmind-anchor-free-gt-pose-report.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/boxmind-anchor-free-gt-pose.pt"))
    return parser.parse_args()


def build_boxer_rounds(root: Path, split: str, offset_scale: float, include_opponent: bool = False,
                       pose_channels: int = 5, pose_root: Path | None = None, feature_mode: str = "absolute"):
    grouped = defaultdict(list)
    for sample in build_oracle_index(root, split).samples:
        grouped[sample.match_id].append(sample)
    rounds = []
    for match_id, samples in sorted(grouped.items()):
        pose_path = pose_root / split / f"{match_id}.pkl" if pose_root else root / samples[0].pose_path
        if not pose_path.is_file():
            raise ValueError(f"Missing pose file: {pose_path}")
        for side in ("red", "blue"):
            features = select_pose_feature_channels(
                extract_boxer_pose_features(pose_path, side, include_opponent, feature_mode),
                pose_channels,
                feature_mode,
            )
            events = [
                AnchorFreeEvent(side, "left" if sample.labels["technique"].startswith("l") else "right", sample.event_start_frame, sample.event_end_frame, 1.0)
                for sample in samples if sample.labels["side"] == side
            ]
            target, mask = encode_targets(events, len(features), offset_scale)
            rounds.append((match_id, side, features, events, target, mask))
    return rounds


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float) -> np.ndarray:
    return np.clip((features - mean) / std, -clip, clip).astype(np.float32)


class Windows(Dataset):
    def __init__(self, rounds, window: int, seed: int, mean: np.ndarray, std: np.ndarray, clip: float,
                 positive_windows_per_event: int = 1, background_ratio: float = 1.0,
                 hard_negative_fraction: float = 0.0):
        self.rounds, self.window, self.items = rounds, window, []
        self.mean, self.std, self.clip = mean, std, clip
        if positive_windows_per_event < 1:
            raise ValueError("positive_windows_per_event must be positive")
        if background_ratio < 0:
            raise ValueError("background_ratio must be non-negative")
        if not 0 <= hard_negative_fraction <= 1:
            raise ValueError("hard_negative_fraction must be between zero and one")
        rng = random.Random(seed)
        for round_index, (_, _, features, events, target, _) in enumerate(rounds):
            for event in events:
                center = (event.start_frame + event.end_frame) // 2
                self.items.append((round_index, center - window // 2))
                for _ in range(positive_windows_per_event - 1):
                    jitter = rng.randint(-window // 4, window // 4)
                    self.items.append((round_index, center + jitter - window // 2))
            background = np.flatnonzero(target[..., 0].sum(axis=1) == 0)
            background_count = round(len(events) * background_ratio)
            hard_count = round(background_count * hard_negative_fraction)
            if hard_count and len(background):
                flattened = features.reshape(len(features), -1)
                motion = np.mean(np.abs(np.diff(flattened, axis=0, prepend=flattened[:1])), axis=1)
                hard_pool_size = min(len(background), max(hard_count * 4, len(background) // 5))
                hard_pool = background[np.argpartition(motion[background], -hard_pool_size)[-hard_pool_size:]]
                for _ in range(hard_count):
                    center = int(hard_pool[rng.randrange(len(hard_pool))])
                    self.items.append((round_index, center - window // 2))
            for _ in range(background_count - hard_count):
                center = int(background[rng.randrange(len(background))])
                self.items.append((round_index, center - window // 2))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        round_index, start = self.items[index]
        _, _, features, _, target, mask = self.rounds[round_index]
        src_start, src_stop = max(start, 0), min(start + self.window, len(features))
        dst_start = src_start - start
        dst_stop = dst_start + src_stop - src_start
        x = np.zeros((self.window, *features.shape[1:]), np.float32)
        y = np.zeros((self.window, 2, 3), np.float32)
        m = np.zeros((self.window, 2), bool)
        x[dst_start:dst_stop] = normalize_features(features[src_start:src_stop], self.mean, self.std, self.clip)
        y[dst_start:dst_stop] = target[src_start:src_stop]
        m[dst_start:dst_stop] = mask[src_start:src_stop]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


def collect_logits(model, rounds, mean, std, clip, device):
    outputs = []
    model.eval()
    with torch.no_grad():
        for _, side, features, truth, _, _ in rounds:
            inputs = torch.from_numpy(normalize_features(features, mean, std, clip)).unsqueeze(0).to(device)
            outputs.append((side, model(inputs).squeeze(0).cpu().numpy(), truth))
    return outputs


def score_logits(outputs, threshold: float, nms_iou: float, offset_scale: float):
    tp = fp = fn = 0
    for side, logits, truth in outputs:
        predicted = decode_events(logits, side, threshold, nms_iou, offset_scale)
        remaining = set(range(len(truth)))
        for event in sorted(predicted, key=lambda candidate: candidate.score, reverse=True):
            candidates = [(temporal_iou(event, truth[index]), index) for index in remaining if event.hand == truth[index].hand]
            if candidates and max(candidates)[0] >= .5:
                _, matched = max(candidates)
                remaining.remove(matched)
                tp += 1
            else:
                fp += 1
        fn += len(remaining)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "tp": tp, "fp": fp, "fn": fn}


def select_operating_point(outputs, offset_scale: float):
    settings = [
        (round(float(threshold), 2), nms_iou)
        for threshold in np.arange(.2, .91, .05)
        for nms_iou in (.3, .5, .7)
    ]
    scored = [(score_logits(outputs, threshold, nms_iou, offset_scale), threshold, nms_iou) for threshold, nms_iou in settings]
    return max(scored, key=lambda item: (item[0]["f1"], item[0]["recall"]))


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.pose_root = args.pose_root.expanduser().resolve() if args.pose_root else None
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable.")
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device)
    train_all = build_boxer_rounds(args.data_root, "train", args.offset_scale, args.include_opponent, args.pose_channels, args.pose_root, args.feature_mode)
    test = [] if args.skip_test else build_boxer_rounds(
        args.data_root, "test", args.offset_scale, args.include_opponent,
        args.pose_channels, args.pose_root, args.feature_mode,
    )
    match_ids = sorted({item[0] for item in train_all})
    validation_ids = set() if args.train_all else set(match_ids[-4:])
    train = [item for item in train_all if item[0] not in validation_ids]
    validation = [item for item in train_all if item[0] in validation_ids]
    stacked = np.concatenate([features for _, _, features, *_ in train])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    if args.normalization == "raw":
        mean = np.zeros_like(mean)
        std = np.ones_like(std)
    dilations = tuple(2 ** index for index in range(args.temporal_depth))
    model = build_punch_detector(
        args.architecture, int(np.prod(train[0][2].shape[1:])), channels=args.channels,
        batch_norm=args.batch_norm, dilations=dilations, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.scheduler_t_max or args.epochs, eta_min=args.learning_rate * 0.05,
        )
        if args.scheduler == "cosine" else None
    )
    loader = DataLoader(Windows(
        train, args.window, args.seed, mean, std, args.feature_clip,
        args.positive_windows_per_event, args.background_ratio, args.hard_negative_fraction,
    ), batch_size=args.batch_size, shuffle=True)
    best = best_state = best_threshold = best_nms = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for features, target, mask in loader:
            output = model(features.to(device)); target, mask = target.to(device), mask.to(device)
            classification = focal_loss(output[..., 0], target[..., 0], args.focal_alpha, args.focal_gamma)
            predicted_offsets = torch.relu(output[..., 1:])[mask]
            expected_offsets = target[..., 1:][mask]
            if not len(predicted_offsets):
                regression = output[..., 1:].sum() * 0.0
                iou_regression = regression
            elif args.regression_loss == "l1":
                regression = torch.nn.functional.l1_loss(predicted_offsets, expected_offsets)
                intersection = torch.minimum(predicted_offsets, expected_offsets).sum(dim=-1)
                union = torch.maximum(predicted_offsets, expected_offsets).sum(dim=-1).clamp_min(1e-6)
                iou_regression = (1.0 - intersection / union).mean()
            else:
                regression = torch.nn.functional.smooth_l1_loss(predicted_offsets, expected_offsets)
                intersection = torch.minimum(predicted_offsets, expected_offsets).sum(dim=-1)
                union = torch.maximum(predicted_offsets, expected_offsets).sum(dim=-1).clamp_min(1e-6)
                iou_regression = (1.0 - intersection / union).mean()
            loss = (
                classification
                + args.regression_weight * regression
                + args.iou_regression_weight * iou_regression
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; use --device cpu or reduce --learning-rate.")
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach()))
        if validation:
            score, threshold, nms_iou = select_operating_point(collect_logits(model, validation, mean, std, args.feature_clip, device), args.offset_scale)
        else:
            score, threshold, nms_iou = None, args.fixed_threshold, args.fixed_nms_iou
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation": score, "threshold": threshold, "nms_iou": nms_iou})
        validation_text = f" val_f1={score['f1']:.4f}" if score else ""
        print(f"epoch={epoch:02d} loss={np.mean(losses):.4f}{validation_text} threshold={threshold:.2f} nms={nms_iou:.1f}", flush=True)
        if score is None or best is None or score["f1"] > best["f1"]:
            best, best_threshold, best_nms = score, threshold, nms_iou
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if scheduler is not None:
            scheduler.step()
    model.load_state_dict(best_state)
    test_score = None if args.skip_test else score_logits(
        collect_logits(model, test, mean, std, args.feature_clip, device),
        best_threshold, best_nms, args.offset_scale,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_mean": mean, "feature_std": std, "feature_clip": args.feature_clip, "threshold": best_threshold, "nms_iou": best_nms, "offset_scale": args.offset_scale, "include_opponent": args.include_opponent, "pose_channels": args.pose_channels, "feature_mode": args.feature_mode, "channels": args.channels, "batch_norm": args.batch_norm, "dilations": dilations, "dropout": args.dropout, "architecture": args.architecture, "model": "boxmind-paper-form-anchor-free-detector-v3"}, args.checkpoint)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "boxmind-paper-form-anchor-free-tcn-v2",
        "scope": f"Independent per-boxer full-round {'extracted RTMW' if args.pose_root else 'GT'} "
                 f"{'2D' if args.pose_channels == 2 else '2D+3D'} pose detection.",
        "configuration": vars(args) | {"data_root": str(args.data_root), "pose_root": str(args.pose_root) if args.pose_root else None, "report": str(args.report), "checkpoint": str(args.checkpoint), "device": str(device)},
        "samples": {"train_matches": len({item[0] for item in train}), "validation_matches": len(validation_ids), "test_matches": len({item[0] for item in test})},
        "validation_iou_0.5": best,
        "selected_threshold": best_threshold,
        "selected_nms_iou": best_nms,
        "test_iou_0.5": test_score,
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(test_score if test_score is not None else best), flush=True)


if __name__ == "__main__":
    main()
