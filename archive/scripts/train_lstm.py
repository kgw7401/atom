#!/usr/bin/env python3
"""Train an LSTM classifier on extracted keypoints.

Same data loading as train_baseline.py, but keeps the temporal structure
(W, seq_len, features) instead of flattening, enabling the model to learn
directional trajectories that RF cannot capture.

Usage:
    python scripts/train_lstm.py
    python scripts/train_lstm.py --epochs 150 --hidden 256
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm import BoxingLSTM
from src.preprocessing.pipeline import PreprocessingPipeline


def load_dataset(
    keypoints_dir: Path = Path("data/keypoints"),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load keypoints → (W, seq_len, K*C) without flattening."""
    pipeline = PreprocessingPipeline()
    all_windows, all_labels, class_set = [], [], set()

    for npy_path in sorted(keypoints_dir.glob("*.npy")):
        meta_path = npy_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        action = meta["action"]
        class_set.add(action)

        raw = np.load(npy_path)          # (N, 33, 4)
        windows = pipeline.process(raw)  # (W, seq_len, K, C)
        if windows.size == 0:
            print(f"  SKIP {npy_path.name}: no valid windows")
            continue

        # Reshape: (W, seq_len, K, C) → (W, seq_len, K*C)
        W, T, K, C = windows.shape
        windows = windows.reshape(W, T, K * C)

        all_windows.append(windows)
        all_labels.extend([action] * W)
        print(f"  {npy_path.name}: {raw.shape[0]} frames → {W} windows [{action}]")

    class_names = sorted(class_set)
    label_to_idx = {n: i for i, n in enumerate(class_names)}

    X = np.concatenate(all_windows, axis=0).astype(np.float32)  # (total, T, K*C)
    y = np.array([label_to_idx[l] for l in all_labels])

    return X, y, class_names


def train(args):
    print("Loading dataset...\n")
    X, y, class_names = load_dataset()

    n_classes = len(class_names)
    input_size = X.shape[2]  # K*C per timestep

    print(f"\nDataset: {X.shape[0]} windows | shape {X.shape}")
    print(f"Classes ({n_classes}): {class_names}")
    for i, name in enumerate(class_names):
        cnt = (y == i).sum()
        print(f"  {name}: {cnt} ({cnt/len(y)*100:.1f}%)")

    # Train/test split (stratified, same ratio as RF baseline)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # Fill NaN survivors (keypoints invisible for entire segments) with 0.
    # Data is hip-centered, so 0 = body center — a safe default.
    nan_count = np.isnan(X_train).sum()
    if nan_count > 0:
        print(f"  Filling {nan_count} NaN values in train ({nan_count / X_train.size:.2%} of data)")
    np.nan_to_num(X_train, nan=0.0, copy=False)
    np.nan_to_num(X_test, nan=0.0, copy=False)

    # Standardize per feature across all (window, timestep) pairs.
    # LSTM gates saturate when input values are large — coordinates can reach ±5
    # (shoulder-width normalized) without this step.
    N_tr, T, F = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N_tr, T, F)
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(len(X_test), T, F)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    print(f"Feature range after scaling: [{X_train.min():.2f}, {X_train.max():.2f}]")

    # Class weights for imbalanced classes
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # Scale augmentation: simulate distance variation by jittering all features
    # by a random scale factor per window. Applied AFTER scaling so the scaler
    # stats stay clean.
    AUG_COPIES = 2
    AUG_SCALE_RANGE = (0.85, 1.15)
    rng = np.random.default_rng(42)
    aug_X, aug_y = [X_train], [y_train]
    for _ in range(AUG_COPIES):
        scales = rng.uniform(*AUG_SCALE_RANGE, size=(len(X_train), 1, 1)).astype(np.float32)
        aug_X.append(X_train * scales)
        aug_y.append(y_train)
    X_train_aug = np.concatenate(aug_X, axis=0)
    y_train_aug = np.concatenate(aug_y, axis=0)
    print(f"Augmented train: {len(X_train)} → {len(X_train_aug)} ({AUG_COPIES} scale copies)")

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train_aug), torch.from_numpy(y_train_aug))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    # Model
    model = BoxingLSTM(
        input_size=input_size,
        num_classes=n_classes,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    )
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5
    )

    # Training loop with early stopping
    best_val_acc, patience_count = 0.0, 0
    best_state = None

    print(f"\nTraining LSTM ({args.epochs} epochs, patience={args.patience})...\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        all_preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                all_preds.append(model(xb).argmax(dim=1))
        val_preds = torch.cat(all_preds).numpy()
        val_acc = accuracy_score(y_test, val_preds)

        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | val_acc={val_acc:.3f} | best={best_val_acc:.3f}")

        if patience_count >= args.patience:
            print(f"\n  Early stop at epoch {epoch}")
            break

    # Final evaluation with best weights
    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            all_preds.append(model(xb).argmax(dim=1))
    y_pred = torch.cat(all_preds).numpy()

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.1%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    header = "".join(f"{n[:8]:>10}" for n in class_names)
    print(f"{'':>10}{header}")
    for i, row in enumerate(cm):
        print(f"{class_names[i][:8]:>10}{''.join(f'{v:>10}' for v in row)}")

    # Save model
    save_path = Path("models/lstm_best.pt")
    save_path.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "class_names": class_names,
            "input_size": input_size,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        save_path,
    )
    print(f"\nModel saved → {save_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
