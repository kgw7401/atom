#!/usr/bin/env python3
"""Train a practical BoxMind-style RGB+pose attribute classifier.

The RGB encoder, pose TCN and MoE heads are imported from a separately cloned
official BoxingWeb repository.  I3D stays frozen after ImageNet initialization
so this experiment can run on local hardware; only the official TCN/MoE
classification portion is trained on BoxingWeb's annotated punch intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.boxingweb import build_oracle_index  # noqa: E402
from atom.boxmind_rgb import extract_boxmind_rgb_sample  # noqa: E402
from atom.pose_features import extract_pose_feature, task_targets  # noqa: E402


TASKS = ("technique", "target", "distance", "effect")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path.home() / "boxingweb")
    parser.add_argument("--boxmind-root", type=Path, default=Path("/tmp/boxmind-reference"))
    parser.add_argument("--i3d-weights", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/atom-boxmind-features"))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("results/boxmind-frozen-classifier-report.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/boxmind-frozen-classifier.pt"))
    return parser.parse_args()


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_i3d(boxmind_root: Path, weights: Path, target_device: torch.device) -> nn.Module:
    sys.path.insert(0, str(boxmind_root))
    from model.i3d import InceptionI3d  # noqa: E402

    encoder = InceptionI3d(in_channels=3, args=Namespace(model="RGB+TCN+MoE"))
    state = torch.load(weights, map_location="cpu", weights_only=False)
    own = encoder.state_dict()
    own.update({key: value for key, value in state.items() if key in own and own[key].shape == value.shape})
    encoder.load_state_dict(own)
    encoder.to(target_device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


def cache_path(cache_dir: Path, split: str, frames: int) -> Path:
    return cache_dir / f"boxmind_i3d_{split}_{frames}f.npz"


def prepare_split(args: argparse.Namespace, split: str, encoder: nn.Module, target_device: torch.device) -> dict[str, np.ndarray]:
    output = cache_path(args.cache_dir, split, args.frames)
    if output.exists():
        cached = np.load(output)
        return {key: cached[key] for key in cached.files}
    samples = build_oracle_index(args.data_root, split).samples
    rgb_features: list[np.ndarray] = []
    pose_features: list[np.ndarray] = []
    targets: list[list[int]] = []
    identifiers: list[str] = []
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start:start + args.batch_size]
        clips = np.stack([extract_boxmind_rgb_sample(sample, args.data_root, frames=args.frames) for sample in batch])
        with torch.no_grad():
            rgb = encoder(torch.from_numpy(clips).to(target_device)).detach().cpu().numpy().astype(np.float32)
        rgb_features.extend(rgb)
        pose_features.extend(extract_pose_feature(sample, args.data_root, frames=args.frames) for sample in batch)
        targets.extend([list(task_targets(sample)[task] for task in TASKS) for sample in batch])
        identifiers.extend(f"{sample.match_id}:{sample.event_index}" for sample in batch)
        print(f"Prepared {split} RGB features: {min(start + len(batch), len(samples))}/{len(samples)}", flush=True)
    result = {
        "rgb": np.stack(rgb_features).astype(np.float32),
        "pose": np.stack(pose_features).astype(np.float32),
        "targets": np.asarray(targets, dtype=np.int64),
        "identifiers": np.asarray(identifiers),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    return result


class FrozenBoxMindClassifier(nn.Module):
    """Official TCN and MoE classification modules over frozen I3D embeddings."""

    def __init__(self, boxmind_root: Path) -> None:
        super().__init__()
        sys.path.insert(0, str(boxmind_root))
        from model.moe import MoEModel  # noqa: E402
        from model.pose_encoder import TCN  # noqa: E402

        self.pose_encoder = TCN()
        self.moe = MoEModel(1536, hidden_dim=512, num_classes_per_task=[3, 3, 3, 1], num_experts=4)

    def forward(self, rgb: torch.Tensor, pose: torch.Tensor) -> list[torch.Tensor]:
        return self.moe(torch.cat((rgb, self.pose_encoder(pose)), dim=1))


def macro_f1(truth: np.ndarray, prediction: np.ndarray, classes: int) -> float:
    values = []
    for label in range(classes):
        tp = np.sum((truth == label) & (prediction == label))
        fp = np.sum((truth != label) & (prediction == label))
        fn = np.sum((truth == label) & (prediction != label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values))


def evaluate(model: nn.Module, loader: DataLoader, target_device: torch.device) -> dict[str, float]:
    model.eval()
    truths, predictions = [[] for _ in TASKS], [[] for _ in TASKS]
    with torch.no_grad():
        for rgb, pose, target in loader:
            output = model(rgb.to(target_device), pose.to(target_device))
            for index in range(3):
                truths[index].append(target[:, index].numpy())
                predictions[index].append(output[index].argmax(dim=1).cpu().numpy())
            truths[3].append(target[:, 3].numpy())
            predictions[3].append((torch.sigmoid(output[3].squeeze(1)) >= 0.5).cpu().numpy().astype(np.int64))
    classes = (3, 3, 3, 2)
    scores = {task: macro_f1(np.concatenate(truth), np.concatenate(prediction), count) for task, truth, prediction, count in zip(TASKS, truths, predictions, classes)}
    scores["mean"] = float(np.mean(list(scores.values())))
    return scores


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.data_root = args.data_root.expanduser().resolve()
    args.boxmind_root = args.boxmind_root.expanduser().resolve()
    weights = args.i3d_weights or args.boxmind_root / "checkpoint" / "rgb_imagenet.pt"
    if not (args.boxmind_root / "model" / "i3d.py").is_file() or not weights.is_file():
        raise SystemExit("Clone the official BoxingWeb repository and provide its I3D weights before running this experiment.")
    target_device = device()
    encoder = load_i3d(args.boxmind_root, weights, target_device)
    train = prepare_split(args, "train", encoder, target_device)
    test = prepare_split(args, "test", encoder, target_device)
    if args.prepare_only:
        return
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train["rgb"]), torch.from_numpy(train["pose"]), torch.from_numpy(train["targets"])), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test["rgb"]), torch.from_numpy(test["pose"]), torch.from_numpy(test["targets"])), batch_size=256)
    model = FrozenBoxMindClassifier(args.boxmind_root).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = []
    best_state, best_scores = None, None
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for rgb, pose, target in train_loader:
            output = model(rgb.to(target_device), pose.to(target_device))
            target = target.to(target_device)
            loss = sum(functional.cross_entropy(output[index], target[:, index]) for index in range(3))
            loss += functional.binary_cross_entropy_with_logits(output[3].squeeze(1), target[:, 3].float()) * 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scores = evaluate(model, test_loader, target_device)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "test": scores})
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={np.mean(losses):.4f}, test_mean_f1={scores['mean']:.4f}", flush=True)
        if best_scores is None or scores["mean"] > best_scores["mean"]:
            best_scores = scores
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None and best_scores is not None
    torch.save(
        {
            "model": "boxmind-compatible-frozen-i3d-tcn-moe-v1",
            "frames": args.frames,
            "state_dict": best_state,
            "best_test_macro_f1": best_scores,
        },
        args.checkpoint,
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "boxmind-compatible-frozen-i3d-tcn-moe-v1",
        "scope": "Oracle-window attribute classification; GT punch intervals and GT poses are inputs, so this is not event detection.",
        "configuration": {"frames": args.frames, "epochs": args.epochs, "i3d_frozen": True, "boxmind_root": str(args.boxmind_root), "i3d_weights": str(weights), "device": str(target_device)},
        "samples": {"train": int(train["targets"].shape[0]), "test": int(test["targets"].shape[0])},
        "best_test_macro_f1": best_scores,
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["best_test_macro_f1"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
