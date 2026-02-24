#!/usr/bin/env python3
"""Train a Random Forest baseline on extracted keypoints.

Loads keypoints from data/keypoints/, runs preprocessing pipeline,
trains an RF classifier, and reports accuracy + confusion matrix.

Usage:
    python scripts/train_baseline.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from src.preprocessing.pipeline import PreprocessingPipeline


def load_dataset(
    keypoints_dir: Path = Path("data/keypoints"),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and preprocess all keypoint files into (X, y).

    Returns:
        X: (N_windows, window_size * num_keypoints * 3) flattened features
        y: (N_windows,) integer labels
        class_names: list of action names
    """
    pipeline = PreprocessingPipeline()

    all_windows = []
    all_labels = []
    class_set = set()

    npy_files = sorted(keypoints_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files in {keypoints_dir}")

    for npy_path in npy_files:
        meta_path = npy_path.with_suffix(".json")
        if not meta_path.exists():
            print(f"  SKIP {npy_path.name}: no metadata")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        action = meta["action"]
        class_set.add(action)

        raw = np.load(npy_path)  # (N, 33, 4)
        windows = pipeline.process(raw)  # (W, window_size, K, 3)

        if windows.size == 0:
            print(f"  SKIP {npy_path.name}: no valid windows")
            continue

        all_windows.append(windows)
        all_labels.extend([action] * len(windows))
        print(f"  {npy_path.name}: {raw.shape[0]} frames → {len(windows)} windows [{action}]")

    # Stack and flatten
    class_names = sorted(class_set)
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    X = np.concatenate(all_windows, axis=0)  # (total_W, ws, K, 3)
    X = X.reshape(X.shape[0], -1)            # flatten to (total_W, ws*K*3)
    y = np.array([label_to_idx[l] for l in all_labels])

    return X, y, class_names


def main():
    print("Loading and preprocessing dataset...\n")
    X, y, class_names = load_dataset()

    print(f"\nDataset: {X.shape[0]} windows, {X.shape[1]} features")
    print(f"Classes: {class_names}")
    for i, name in enumerate(class_names):
        count = (y == i).sum()
        print(f"  {name}: {count} windows ({count / len(y) * 100:.1f}%)")

    # Stratified train/test split (80/20)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.1%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    # Pretty print
    header = "".join(f"{n[:8]:>10}" for n in class_names)
    print(f"{'':>10}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>10}" for v in row)
        print(f"{class_names[i][:8]:>10}{row_str}")


if __name__ == "__main__":
    main()
