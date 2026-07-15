#!/usr/bin/env python3
"""Train a compact pose-only multi-task Oracle-window BoxingWeb baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.boxingweb import OracleWindowSample, build_oracle_index, write_oracle_index  # noqa: E402
from atom.pose_features import TASK_LABELS, extract_pose_feature, hand_value, task_targets  # noqa: E402


TASKS = ("technique", "distance", "target", "effect")


class PoseMultiTaskMLP(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(256, len(TASK_LABELS[task])) for task in TASKS
        })

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(inputs)
        return {task: head(encoded) for task, head in self.heads.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "boxingweb")
    parser.add_argument("--index-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--report", type=Path, default=Path("results/pose-baseline-report.json"))
    return parser.parse_args()


def load_samples(index_path: Path) -> list[OracleWindowSample]:
    payload = json.loads(index_path.read_text())
    return [OracleWindowSample(**sample) for sample in payload["samples"]]


def load_or_build_samples(data_root: Path, index_dir: Path, split: str) -> list[OracleWindowSample]:
    index_path = index_dir / f"boxingweb-oracle-{split}.json"
    if not index_path.exists():
        write_oracle_index(build_oracle_index(data_root, split), index_path)
    return load_samples(index_path)


def feature_matrix(samples: list[OracleWindowSample], data_root: Path, frames: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features: list[np.ndarray] = []
    targets = {task: [] for task in TASKS}
    for number, sample in enumerate(samples, start=1):
        features.append(np.concatenate((extract_pose_feature(sample, data_root, frames).reshape(-1), [hand_value(sample)])))
        values = task_targets(sample)
        for task, value in values.items():
            targets[task].append(value)
        if number % 1000 == 0:
            print(f"Prepared {number}/{len(samples)} {sample.split} samples")
    return np.stack(features).astype(np.float32), {task: np.asarray(values, dtype=np.int64) for task, values in targets.items()}


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def metrics(predicted: np.ndarray, expected: np.ndarray, class_names: list[str]) -> dict[str, object]:
    classes = len(class_names)
    matrix = np.zeros((classes, classes), dtype=int)
    for actual, prediction in zip(expected, predicted):
        matrix[actual, prediction] += 1
    per_class = []
    for label in range(classes):
        true_positive = int(matrix[label, label])
        false_positive = int(matrix[:, label].sum() - true_positive)
        false_negative = int(matrix[label, :].sum() - true_positive)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"label": class_names[label], "support": int(matrix[label, :].sum()), "precision": precision, "recall": recall, "f1": f1})
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def class_names(task: str) -> list[str]:
    return [label for label, _ in sorted(TASK_LABELS[task].items(), key=lambda item: item[1])]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = select_device(args.device)
    data_root = args.data_root.expanduser().resolve()
    train_samples = load_or_build_samples(data_root, args.index_dir, "train")
    test_samples = load_or_build_samples(data_root, args.index_dir, "test")
    train_x, train_y = feature_matrix(train_samples, data_root, args.frames)
    test_x, test_y = feature_matrix(test_samples, data_root, args.frames)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std

    tensors = [torch.from_numpy(train_x)] + [torch.from_numpy(train_y[task]) for task in TASKS]
    loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=True)
    model = PoseMultiTaskMLP(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for batch in loader:
            inputs, *labels = (tensor.to(device) for tensor in batch)
            output = model(inputs)
            loss = sum(criterion(output[task], labels[index]) for index, task in enumerate(TASKS))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * inputs.shape[0]
        print(f"epoch={epoch:02d} loss={loss_total / len(train_samples):.4f}")

    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(test_x).to(device))
    task_metrics = {}
    for task in TASKS:
        predicted = output[task].argmax(dim=1).cpu().numpy()
        task_metrics[task] = metrics(predicted, test_y[task], class_names(task))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "pose-only-multitask-mlp-v1",
        "scope": "Oracle-window attribute classification; not event detection or coaching evaluation.",
        "official_reference": {"repository": "https://github.com/gouba2333/BoxingWeb", "commit": "e06add251e31fe7db8db9fffcb705636484eb264"},
        "configuration": {"frames": args.frames, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": args.seed, "device": str(device)},
        "samples": {"train": len(train_samples), "test": len(test_samples)},
        "features": "attacker-relative 14-joint 2D+3D pose, opponent pose, and annotated striking hand",
        "metrics": task_metrics,
        "mean_macro_f1": float(np.mean([result["macro_f1"] for result in task_metrics.values()])),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"mean_macro_f1": report["mean_macro_f1"], "tasks": {task: values["macro_f1"] for task, values in task_metrics.items()}}, indent=2))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
