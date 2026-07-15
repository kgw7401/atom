#!/usr/bin/env python3
"""Test whether ground-truth pose can localize boxing punches over full rounds."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.pose_detection import CHANNELS, PunchEvent, TemporalMatch, build_temporal_matches  # noqa: E402
from atom.rgb_features import extract_match_rgb_motion_features  # noqa: E402


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(inputs + self.layers(inputs))


class PoseTemporalDetector(nn.Module):
    """A deliberately small TCN that predicts activity and event boundaries."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.input = nn.Conv1d(feature_count, 32, kernel_size=1)
        self.blocks = nn.Sequential(
            ResidualBlock(32, 1), ResidualBlock(32, 2), ResidualBlock(32, 4),
            ResidualBlock(32, 8), ResidualBlock(32, 16),
        )
        self.output = nn.Conv1d(32, len(CHANNELS), kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(features.transpose(1, 2)))).transpose(1, 2)


class WindowDataset(Dataset):
    def __init__(self, matches: list[TemporalMatch], window: int, seed: int) -> None:
        self.matches = matches
        self.window = window
        self.indices: list[tuple[int, int]] = []
        randomizer = random.Random(seed)
        for match_index, match in enumerate(matches):
            for event in match.events:
                self.indices.append((match_index, event.start_frame - window // 4))
            background = np.where(match.targets[:, :2].sum(axis=1) == 0)[0]
            for _ in match.events:
                center = int(background[randomizer.randrange(len(background))])
                self.indices.append((match_index, center - window // 2))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        match_index, start = self.indices[index]
        match = self.matches[match_index]
        stop = start + self.window
        feature = np.zeros((self.window, match.features.shape[1]), dtype=np.float32)
        target = np.zeros((self.window, len(CHANNELS)), dtype=np.float32)
        source_start, source_stop = max(0, start), min(stop, match.features.shape[0])
        destination_start = source_start - start
        destination_stop = destination_start + source_stop - source_start
        feature[destination_start:destination_stop] = match.features[source_start:source_stop]
        target[destination_start:destination_stop] = match.targets[source_start:source_stop]
        return torch.from_numpy(feature), torch.from_numpy(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "boxingweb")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--use-rgb", action="store_true", help="Fuse pose-guided RGB motion features.")
    parser.add_argument("--rgb-cache-dir", type=Path, default=Path("/tmp/atom-rgb-motion"))
    parser.add_argument("--prepare-rgb-only", action="store_true", help="Populate the RGB cache without training.")
    parser.add_argument("--rgb-limit", type=int, default=0, help="Maximum uncached matches to prepare; 0 means all.")
    parser.add_argument("--report", type=Path, help="Optional path for the evaluation report.")
    parser.add_argument("--checkpoint", type=Path, help="Optional path for model and feature-normalization weights.")
    return parser.parse_args()


def device_for(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device("mps" if requested == "auto" and torch.backends.mps.is_available() else requested)


def binary_f1(predicted: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    true_positive = int(np.logical_and(predicted, expected).sum())
    false_positive = int(np.logical_and(predicted, ~expected).sum())
    false_negative = int(np.logical_and(~predicted, expected).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": true_positive, "fp": false_positive, "fn": false_negative}


def peak_frames(scores: np.ndarray, threshold: float) -> list[int]:
    peaks = []
    for frame, score in enumerate(scores):
        left = scores[max(0, frame - 2):frame]
        right = scores[frame + 1:frame + 3]
        if score >= threshold and (not len(left) or score >= left.max()) and (not len(right) or score > right.max()):
            peaks.append(frame)
    return peaks


def decode_events(scores: np.ndarray, threshold: float = 0.5, activity_threshold: float = 0.5) -> list[PunchEvent]:
    """Pair local start peaks with the strongest plausible future end peak."""

    events = []
    for side_index, side in enumerate(("red", "blue")):
        starts = peak_frames(scores[:, side_index + 2], threshold)
        end_scores = scores[:, side_index + 4]
        for start in starts:
            candidate_end = min(len(end_scores), start + 31)
            if start + 4 >= candidate_end:
                continue
            end = start + 4 + int(end_scores[start + 4:candidate_end].argmax())
            mean_activity = scores[start:end + 1, side_index].mean()
            if end_scores[end] >= threshold and mean_activity >= activity_threshold:
                events.append(PunchEvent(side, start, end))
    return events


def temporal_iou(first: PunchEvent, second: PunchEvent) -> float:
    intersection = max(0, min(first.end_frame, second.end_frame) - max(first.start_frame, second.start_frame) + 1)
    union = max(first.end_frame, second.end_frame) - min(first.start_frame, second.start_frame) + 1
    return intersection / union


def match_events(predicted: list[PunchEvent], expected: list[PunchEvent], threshold: float) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    unmatched = set(range(len(expected)))
    unmatched_predictions = set(range(len(predicted)))
    matches = []
    for prediction_index, prediction in enumerate(predicted):
        candidates = [(temporal_iou(prediction, expected[index]), index) for index in unmatched if expected[index].side == prediction.side]
        if candidates:
            iou, index = max(candidates)
            if iou >= threshold:
                unmatched.remove(index)
                unmatched_predictions.remove(prediction_index)
                matches.append((prediction_index, index))
    return matches, unmatched_predictions, unmatched


def duration_bucket(event: PunchEvent) -> str:
    duration = event.end_frame - event.start_frame
    if duration <= 7:
        return "4-7"
    if duration <= 15:
        return "8-15"
    return "16-30"


def outcome_summary(outcomes: list[tuple[list[PunchEvent], list[PunchEvent], list[tuple[int, int]], set[int], set[int]]]) -> dict[str, object]:
    by_side = {side: {"expected": 0, "predicted": 0, "matched": 0} for side in ("red", "blue")}
    by_duration = {bucket: {"expected": 0, "matched": 0} for bucket in ("4-7", "8-15", "16-30")}
    start_errors, end_errors = [], []
    for predicted, expected, matches, unmatched_prediction, unmatched_expected in outcomes:
        for event in expected:
            by_side[event.side]["expected"] += 1
            by_duration[duration_bucket(event)]["expected"] += 1
        for event in predicted:
            by_side[event.side]["predicted"] += 1
        for prediction_index, expected_index in matches:
            prediction, truth = predicted[prediction_index], expected[expected_index]
            by_side[truth.side]["matched"] += 1
            by_duration[duration_bucket(truth)]["matched"] += 1
            start_errors.append(prediction.start_frame - truth.start_frame)
            end_errors.append(prediction.end_frame - truth.end_frame)
    return {
        "by_side": by_side,
        "gt_recall_by_duration": {
            bucket: values["matched"] / values["expected"] if values["expected"] else 0.0
            for bucket, values in by_duration.items()
        },
        "boundary_error_frames_for_matched_events": {
            "mean_absolute_start_error": float(np.mean(np.abs(start_errors))) if start_errors else None,
            "mean_absolute_end_error": float(np.mean(np.abs(end_errors))) if end_errors else None,
        },
    }


def evaluate(model: nn.Module, matches: list[TemporalMatch], device: torch.device, threshold: float, activity_threshold: float) -> dict[str, object]:
    frame_predictions, frame_targets = [], []
    event_totals = {iou: [0, 0, 0] for iou in (0.1, 0.3, 0.5)}
    iou_half_outcomes = []
    predicted_event_count = 0
    expected_event_count = 0
    model.eval()
    with torch.no_grad():
        for match in matches:
            logits = model(torch.from_numpy(match.features).unsqueeze(0).to(device))
            scores = logits.sigmoid().squeeze(0).cpu().numpy()
            frame_predictions.append(scores >= 0.5)
            frame_targets.append(match.targets.astype(bool))
            decoded = decode_events(scores, threshold, activity_threshold)
            predicted_event_count += len(decoded)
            expected_event_count += len(match.events)
            for iou, totals in event_totals.items():
                matches, unmatched_prediction, unmatched_expected = match_events(decoded, list(match.events), iou)
                totals[0] += len(matches)
                totals[1] += len(unmatched_prediction)
                totals[2] += len(unmatched_expected)
                if iou == 0.5:
                    iou_half_outcomes.append((decoded, list(match.events), matches, unmatched_prediction, unmatched_expected))
    prediction = np.concatenate(frame_predictions)
    target = np.concatenate(frame_targets)
    event_metrics = {}
    for iou, (true_positive, false_positive, false_negative) in event_totals.items():
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        event_metrics[f"iou_{iou}"] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
        }
    return {
        "frame_f1": {channel: binary_f1(prediction[:, index], target[:, index]) for index, channel in enumerate(CHANNELS)},
        "event_f1": event_metrics,
        "predicted_events": predicted_event_count,
        "expected_events": expected_event_count,
        "iou_0.5_error_analysis": outcome_summary(iou_half_outcomes),
    }


def main() -> None:
    args = parse_args()
    report_path = args.report or Path(
        "results/pose-event-detector-rgb-report.json" if args.use_rgb else "results/pose-event-detector-pose-report.json"
    )
    checkpoint_path = args.checkpoint or Path(
        "results/pose-event-detector-rgb.pt" if args.use_rgb else "results/pose-event-detector-pose.pt"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = device_for(args.device)
    all_train_matches = build_temporal_matches(args.data_root, "train")
    train_matches, validation_matches = all_train_matches[:-4], all_train_matches[-4:]
    test_matches = build_temporal_matches(args.data_root, "test")
    if args.use_rgb:
        args.rgb_cache_dir.mkdir(parents=True, exist_ok=True)
        prepared = 0
        for number, match in enumerate(train_matches + validation_matches + test_matches, start=1):
            cache_path = args.rgb_cache_dir / f"{match.split}_{match.match_id}.npy"
            if cache_path.exists():
                rgb_features = np.load(cache_path)
            else:
                rgb_features = extract_match_rgb_motion_features(match.video_path, match.pose_path)
                np.save(cache_path, rgb_features)
                prepared += 1
            match.features = np.concatenate((match.features, rgb_features[:match.features.shape[0]]), axis=1)
            print(f"Prepared RGB motion for {number}/50 matches")
            if args.prepare_rgb_only and args.rgb_limit and prepared >= args.rgb_limit:
                break
        if args.prepare_rgb_only:
            return
    feature_mean = np.concatenate([match.features for match in train_matches]).mean(axis=0)
    feature_std = np.concatenate([match.features for match in train_matches]).std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    for match in train_matches + validation_matches + test_matches:
        match.features[:] = (match.features - feature_mean) / feature_std
    dataset = WindowDataset(train_matches, args.window, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = PoseTemporalDetector(train_matches[0].features.shape[1]).to(device)
    full_targets = np.concatenate([match.targets for match in train_matches])
    positives = full_targets.sum(axis=0)
    pos_weight = torch.tensor((len(full_targets) - positives) / np.maximum(positives, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for features, targets in loader:
            logits = model(features.to(device))
            per_channel = criterion(logits, targets.to(device)).mean(dim=(0, 1))
            loss = per_channel[:2].mean() + 2 * per_channel[2:].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.shape[0]
        print(f"epoch={epoch:02d} loss={total_loss / len(dataset):.4f}")
    threshold_candidates = np.arange(0.2, 0.91, 0.05)
    activity_candidates = (0.3, 0.5, 0.7)
    validation_results = {
        (round(float(threshold), 2), activity): evaluate(model, validation_matches, device, float(threshold), activity)
        for threshold in threshold_candidates for activity in activity_candidates
    }
    selected_threshold, selected_activity_threshold = max(
        validation_results,
        key=lambda settings: validation_results[settings]["event_f1"]["iou_0.5"]["f1"],
    )
    metrics = evaluate(model, test_matches, device, selected_threshold, selected_activity_threshold)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "gt-pose-rgb-temporal-conv-detector-v1" if args.use_rgb else "gt-pose-temporal-conv-detector-v2",
        "scope": "Ground-truth-pose temporal punch localization; excludes video pose-estimation and tracking errors.",
        "configuration": {"epochs": args.epochs, "batch_size": args.batch_size, "window": args.window, "seed": args.seed, "device": str(device), "event_threshold": selected_threshold, "activity_threshold": selected_activity_threshold, "use_rgb_motion": args.use_rgb},
        "samples": {"train_matches": len(train_matches), "validation_matches": len(validation_matches), "test_matches": len(test_matches), "train_events": sum(len(match.events) for match in train_matches), "validation_events": sum(len(match.events) for match in validation_matches), "test_events": sum(len(match.events) for match in test_matches)},
        "validation_event_f1": {f"event={threshold},active={activity}": result["event_f1"]["iou_0.5"] for (threshold, activity), result in validation_results.items()},
        "metrics": metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    torch.save(
        {
            "model": report["model"],
            "feature_count": train_matches[0].features.shape[1],
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "event_threshold": selected_threshold,
            "activity_threshold": selected_activity_threshold,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    print(json.dumps(metrics["event_f1"], indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
