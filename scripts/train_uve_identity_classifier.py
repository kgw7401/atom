#!/usr/bin/env python3
"""Train the UVE red/blue/non-boxer appearance classifier on BoxingWeb video."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.rtmw_pose import match_oracle_boxers, pose_bbox  # noqa: E402
from atom.uve_identity import (  # noqa: E402
    APPEARANCE_FEATURE_COUNT,
    IDENTITY_CLASSES,
    UVEAppearanceClassifier,
    appearance_descriptor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=30)
    parser.add_argument("--batch-frames", type=int, default=32)
    parser.add_argument("--max-detections", type=int, default=6)
    parser.add_argument("--minimum-iou", type=float, default=.05)
    parser.add_argument("--validation-matches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--max-matches", type=int, default=0, help="Diagnostic limit; 0 uses all 40 matches.")
    return parser.parse_args()


def match_files(match_dir: Path) -> tuple[Path, Path]:
    videos = sorted(match_dir.glob("*.mp4"))
    poses = sorted(match_dir.glob("*_pose_gt.pkl"))
    if len(videos) != 1 or len(poses) != 1:
        raise ValueError(f"Expected one video and one GT pose in {match_dir}")
    return videos[0], poses[0]


def collect_match(detector, match_dir: Path, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    video_path, pose_path = match_files(match_dir)
    with pose_path.open("rb") as file:
        pose = pickle.load(file)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open {video_path}")
    total = min(
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        len(np.asarray(pose["pose_red_2d"])),
        len(np.asarray(pose["pose_blue_2d"])),
    )
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = list(range(0, total, args.sample_stride))
    descriptors: list[np.ndarray] = []
    labels: list[int] = []
    try:
        for start in range(0, len(indices), args.batch_frames):
            frame_indices = indices[start:start + args.batch_frames]
            frames = []
            valid_indices = []
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if ok:
                    frames.append(frame)
                    valid_indices.append(frame_index)
            if not frames:
                continue
            results = detector.predict(
                frames, classes=[0], conf=.2, device=args.device,
                batch=args.batch_frames, verbose=False,
            )
            for frame, frame_index, result in zip(frames, valid_indices, results):
                boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                if len(boxes) == 0:
                    continue
                areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
                keep = np.argsort(areas)[::-1][:args.max_detections]
                boxes = boxes[keep]
                red_box = pose_bbox(pose["pose_red_2d"][frame_index], width, height)
                blue_box = pose_bbox(pose["pose_blue_2d"][frame_index], width, height)
                red_index, blue_index = match_oracle_boxers(boxes, red_box, blue_box, args.minimum_iou)
                for detection_index, box in enumerate(boxes):
                    label = 0 if detection_index == red_index else 1 if detection_index == blue_index else 2
                    descriptors.append(appearance_descriptor(frame, box))
                    labels.append(label)
    finally:
        capture.release()
    return np.asarray(descriptors, np.float32), np.asarray(labels, np.int64)


def build_cache(args: argparse.Namespace) -> dict[str, np.ndarray]:
    from ultralytics import YOLO

    match_dirs = sorted(path for path in (args.data_root / "data_train").iterdir() if path.is_dir())
    if args.max_matches:
        match_dirs = match_dirs[:args.max_matches]
    if len(match_dirs) <= args.validation_matches:
        raise ValueError("Not enough matches for the requested validation split")
    validation_ids = {path.name for path in match_dirs[-args.validation_matches:]}
    detector = YOLO(str(args.yolo))
    buckets = {"train_x": [], "train_y": [], "val_x": [], "val_y": []}
    for number, match_dir in enumerate(match_dirs, start=1):
        features, labels = collect_match(detector, match_dir, args)
        prefix = "val" if match_dir.name in validation_ids else "train"
        buckets[f"{prefix}_x"].append(features)
        buckets[f"{prefix}_y"].append(labels)
        counts = np.bincount(labels, minlength=3).tolist()
        print(f"matches={number}/{len(match_dirs)} {prefix}/{match_dir.name} samples={len(labels)} classes={counts}", flush=True)
    payload = {
        key: np.concatenate(values) if values else np.empty((0, APPEARANCE_FEATURE_COUNT if key.endswith("x") else 0))
        for key, values in buckets.items()
    }
    payload["validation_ids"] = np.asarray(sorted(validation_ids))
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, **payload)
    return payload


def classification_metrics(expected: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for truth, guess in zip(expected, predicted):
        confusion[int(truth), int(guess)] += 1
    by_class = {}
    for index, name in enumerate(IDENTITY_CLASSES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        by_class[name] = {
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": int(confusion[index].sum()),
        }
    return {
        "accuracy": float(np.trace(confusion) / max(confusion.sum(), 1)),
        "macro_f1": float(np.mean([value["f1"] for value in by_class.values()])),
        "confusion": confusion.tolist(),
        "by_class": by_class,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.cache.exists() and not args.force_cache:
        with np.load(args.cache) as archive:
            data = {key: archive[key] for key in archive.files}
    else:
        data = build_cache(args)
    train_x, train_y = np.asarray(data["train_x"], np.float32), np.asarray(data["train_y"], np.int64)
    val_x, val_y = np.asarray(data["val_x"], np.float32), np.asarray(data["val_y"], np.int64)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = (train_x - mean) / std
    class_counts = np.bincount(train_y, minlength=3)
    sample_weights = (len(train_y) / np.maximum(class_counts, 1))[train_y]
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).double(), len(train_y), replacement=True)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(normalized), torch.from_numpy(train_y)),
        batch_size=args.batch_size, sampler=sampler,
    )
    device = torch.device(args.device)
    model = UVEAppearanceClassifier(APPEARANCE_FEATURE_COUNT, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for features, labels in loader:
            loss = criterion(model(features.to(device)), labels.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:02d} loss={np.mean(losses):.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy((val_x - mean) / std).to(device))
        predicted = logits.argmax(dim=-1).cpu().numpy()
    metrics = classification_metrics(val_y, predicted)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": "uve-rgb-appearance-classifier-v1",
        "state_dict": model.state_dict(),
        "feature_count": APPEARANCE_FEATURE_COUNT,
        "hidden": args.hidden,
        "feature_mean": mean,
        "feature_std": std,
        "classes": IDENTITY_CLASSES,
    }, args.checkpoint)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "uve-rgb-appearance-classifier-v1",
        "scope": "RGB replacement for unpublished BoxMind UV-map identity classifier.",
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "samples": {"train": len(train_y), "validation": len(val_y), "train_classes": class_counts.tolist()},
        "validation": metrics,
        "validation_match_ids": [str(value) for value in data.get("validation_ids", [])],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
