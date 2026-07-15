#!/usr/bin/env python3
"""Evaluate GT-pose event detection followed by BoxMind-style attributes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from atom.boxingweb import build_oracle_index  # noqa: E402
from atom.boxmind_rgb import extract_boxmind_rgb_clip  # noqa: E402
from atom.pose_detection import PunchEvent, build_temporal_matches  # noqa: E402
from atom.pose_features import extract_pose_interval, task_targets  # noqa: E402
from train_boxmind_frozen_classifier import (  # noqa: E402
    FrozenBoxMindClassifier,
    TASKS,
    evaluate as attribute_evaluate,
    load_i3d,
)
from train_pose_event_detector import PoseTemporalDetector, match_events  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "boxingweb")
    parser.add_argument("--boxmind-root", type=Path, default=Path("/tmp/boxmind-reference"))
    parser.add_argument("--detector-checkpoint", type=Path, default=Path("results/pose-event-detector-pose.pt"))
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path("results/boxmind-frozen-classifier.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--report", type=Path, default=Path("results/boxmind-end-to-end-report.json"))
    return parser.parse_args()


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def classify_candidates(
    candidates: list[tuple[Path, Path, PunchEvent]],
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    target_device: torch.device,
    frames: int,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    classifier.eval()
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        clips = np.stack([
            extract_boxmind_rgb_clip(video, pose, event.side, event.start_frame, event.end_frame, frames=frames)
            for video, pose, event in batch
        ])
        poses = np.stack([
            extract_pose_interval(pose, event.side, event.start_frame, event.end_frame, frames=frames)
            for _, pose, event in batch
        ])
        with torch.no_grad():
            rgb = encoder(torch.from_numpy(clips).to(target_device))
            output = classifier(rgb, torch.from_numpy(poses).to(target_device))
        values = [output[index].argmax(dim=1).cpu().numpy() for index in range(3)]
        values.append((torch.sigmoid(output[3].squeeze(1)) >= 0.5).cpu().numpy().astype(np.int64))
        predictions.extend(np.stack(values, axis=1))
        print(f"Classified detector candidates: {min(start + len(batch), len(candidates))}/{len(candidates)}", flush=True)
    return np.asarray(predictions, dtype=np.int64)


def macro_f1(truth: np.ndarray, prediction: np.ndarray, classes: int) -> float:
    scores = []
    for label in range(classes):
        tp = int(np.sum((truth == label) & (prediction == label)))
        fp = int(np.sum((truth != label) & (prediction == label)))
        fn = int(np.sum((truth == label) & (prediction != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.boxmind_root = args.boxmind_root.expanduser().resolve()
    target_device = device()
    detector_state = torch.load(args.detector_checkpoint, map_location="cpu", weights_only=False)
    classifier_state = torch.load(args.classifier_checkpoint, map_location="cpu", weights_only=False)
    frames = int(classifier_state["frames"])
    detector = PoseTemporalDetector(int(detector_state["feature_count"])).to(target_device)
    detector.load_state_dict(detector_state["state_dict"])
    detector.eval()
    classifier = FrozenBoxMindClassifier(args.boxmind_root).to(target_device)
    classifier.load_state_dict(classifier_state["state_dict"])
    classifier.eval()
    encoder = load_i3d(args.boxmind_root, args.boxmind_root / "checkpoint" / "rgb_imagenet.pt", target_device)
    truth_by_interval = {}
    for sample in build_oracle_index(args.data_root, "test").samples:
        truth_by_interval[(sample.match_id, sample.labels["side"], sample.event_start_frame, sample.event_end_frame)] = np.asarray(
            [task_targets(sample)[task] for task in TASKS], dtype=np.int64
        )
    outcomes = []
    candidates: list[tuple[Path, Path, PunchEvent]] = []
    candidate_match_indices: list[tuple[int, int]] = []
    for match_index, match in enumerate(build_temporal_matches(args.data_root, "test")):
        features = (match.features - detector_state["feature_mean"]) / detector_state["feature_std"]
        with torch.no_grad():
            scores = detector(torch.from_numpy(features).unsqueeze(0).to(target_device)).sigmoid().squeeze(0).cpu().numpy()
        from train_pose_event_detector import decode_events  # noqa: E402
        predicted = decode_events(scores, detector_state["event_threshold"], detector_state["activity_threshold"])
        matched, unmatched_prediction, unmatched_expected = match_events(predicted, list(match.events), 0.5)
        outcomes.append((match, predicted, matched, unmatched_prediction, unmatched_expected))
        for event_index, event in enumerate(predicted):
            candidates.append((match.video_path, match.pose_path, event))
            candidate_match_indices.append((match_index, event_index))
    attributes = classify_candidates(candidates, encoder, classifier, target_device, frames, args.batch_size)
    prediction_by_match = {}
    for key, attributes_for_event in zip(candidate_match_indices, attributes):
        prediction_by_match[key] = attributes_for_event
    tp = fp = fn = 0
    truth_attributes, predicted_attributes = [], []
    joint_correct = 0
    for match_index, (match, predicted, matched, unmatched_prediction, unmatched_expected) in enumerate(outcomes):
        tp += len(matched)
        fp += len(unmatched_prediction)
        fn += len(unmatched_expected)
        for predicted_index, expected_index in matched:
            event, truth_event = predicted[predicted_index], match.events[expected_index]
            truth = truth_by_interval[(match.match_id, truth_event.side, truth_event.start_frame, truth_event.end_frame)]
            prediction = prediction_by_match[(match_index, predicted_index)]
            truth_attributes.append(truth)
            predicted_attributes.append(prediction)
            joint_correct += int(np.array_equal(truth, prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    event_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    truth_array, prediction_array = np.stack(truth_attributes), np.stack(predicted_attributes)
    attribute_scores = {
        task: macro_f1(truth_array[:, index], prediction_array[:, index], 2 if task == "effect" else 3)
        for index, task in enumerate(TASKS)
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "GT-pose event detector followed by frozen-I3D BoxMind-compatible attribute classifier; excludes video pose estimation and tracking errors.",
        "samples": {"test_events": int(tp + fn), "detector_candidates": len(candidates), "iou_0.5_matched_events": int(tp)},
        "event_detection_iou_0.5": {"precision": precision, "recall": recall, "f1": event_f1, "tp": tp, "fp": fp, "fn": fn},
        "attribute_macro_f1_on_iou_0.5_matched_events": attribute_scores | {"mean": float(np.mean(list(attribute_scores.values())))},
        "joint_event_and_all_attributes_accuracy_over_ground_truth_events": joint_correct / (tp + fn) if tp + fn else 0.0,
        "limitations": "The official BoxMind classifier has no punch/non-punch output. Unmatched detector candidates therefore remain false positives rather than being filtered by this classifier.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
