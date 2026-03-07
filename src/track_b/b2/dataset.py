"""B2 Task 5: Dataset preparation for BoxingVI and Olympic Boxing Kaggle.

Provides:
- scan_boxingvi(): scan dataset directory → labeled DataFrame
- scan_olympic_boxing(): scan dataset directory → labeled DataFrame
- make_splits(): stratified train/val/test split
- class_distribution(): count clips per class
- save_splits() / load_splits(): persist split CSVs

Label maps:
- BOXINGVI_LABEL_MAP: 6-class punch classifier (primary training target)
- MERGED_LABEL_MAP: 10-class map that adds Olympic Boxing's extra classes
- OLYMPIC_TO_BOXINGVI: maps Olympic class names to BoxingVI equivalents (or None for v2-only classes)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Label definitions ─────────────────────────────────────────────────────────

# BoxingVI: 6 punch types (primary training target)
BOXINGVI_CLASSES: list[str] = [
    "jab",
    "cross",
    "lead_hook",
    "rear_hook",
    "lead_uppercut",
    "rear_uppercut",
]
BOXINGVI_LABEL_MAP: dict[str, int] = {cls: i for i, cls in enumerate(BOXINGVI_CLASSES)}

# Olympic Boxing Kaggle: 8 classes.
# NOTE: Directory names below are inferred from Stefanski et al. (MDPI Entropy 2024).
# Run scripts/download_olympic_boxing.py and verify actual directory names;
# update OLYMPIC_CLASSES and pass a class_map to scan_olympic_boxing() if they differ.
OLYMPIC_CLASSES: list[str] = [
    "jab",
    "cross",
    "hook",
    "uppercut",
    "body_jab",
    "body_cross",
    "body_hook",
    "block",
]
OLYMPIC_LABEL_MAP: dict[str, int] = {cls: i for i, cls in enumerate(OLYMPIC_CLASSES)}

# Unified label space: BoxingVI 6 classes + 4 Olympic-only extensions
# "hook" and "uppercut" from Olympic map to BoxingVI lead variants (approximate).
OLYMPIC_TO_BOXINGVI: dict[str, str | None] = {
    "jab": "jab",
    "cross": "cross",
    "hook": "lead_hook",        # Olympic doesn't distinguish lead/rear
    "uppercut": "lead_uppercut",  # Olympic doesn't distinguish lead/rear
    "body_jab": None,           # v2 class — not in BoxingVI
    "body_cross": None,         # v2 class
    "body_hook": None,          # v2 class
    "block": None,              # v2 class
}

# "hook" and "uppercut" are Olympic's undifferentiated variants (no lead/rear distinction).
# They live in MERGED_LABEL_MAP so scan_olympic_boxing() can catalogue them as-is.
# Use OLYMPIC_TO_BOXINGVI to remap them to BoxingVI labels when training jointly.
MERGED_CLASSES: list[str] = BOXINGVI_CLASSES + [
    "hook",
    "uppercut",
    "body_jab",
    "body_cross",
    "body_hook",
    "block",
]
MERGED_LABEL_MAP: dict[str, int] = {cls: i for i, cls in enumerate(MERGED_CLASSES)}

# Video file extensions recognized by both scan functions
_VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".webm"})
_FRAME_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


# ── Dataset scanners ──────────────────────────────────────────────────────────

def scan_boxingvi(root_dir: str | Path) -> pd.DataFrame:
    """Scan BoxingVI dataset directory and return a labeled DataFrame.

    Expected directory layout::

        root_dir/
          jab/
            clip_001.mp4
            ...
          cross/
          lead_hook/
          rear_hook/
          lead_uppercut/
          rear_uppercut/

    Args:
        root_dir: Path to BoxingVI root directory.

    Returns:
        DataFrame with columns [video_path, class_name, label, dataset].
        video_path is the absolute path to each video file.
        label is the integer class index from BOXINGVI_LABEL_MAP.
        dataset is always "boxingvi".

    Raises:
        FileNotFoundError: If root_dir does not exist.
        ValueError: If no valid class directories (with video files) are found.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"BoxingVI root directory not found: {root}")

    rows: list[dict] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in BOXINGVI_LABEL_MAP:
            continue
        label = BOXINGVI_LABEL_MAP[class_name]
        for video_file in sorted(class_dir.iterdir()):
            if video_file.suffix.lower() in _VIDEO_EXTS:
                rows.append({
                    "video_path": str(video_file.resolve()),
                    "class_name": class_name,
                    "label": label,
                    "dataset": "boxingvi",
                })

    if not rows:
        raise ValueError(
            f"No valid BoxingVI class directories with video files found in {root}. "
            f"Expected one of: {BOXINGVI_CLASSES}"
        )

    return pd.DataFrame(rows)


def scan_olympic_boxing(
    root_dir: str | Path,
    class_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Scan Olympic Boxing Kaggle dataset directory and return a labeled DataFrame.

    The dataset may contain video clips or image frames organized by class subdirectory.
    Pass class_map to remap actual directory names to canonical OLYMPIC_CLASSES names
    if the downloaded directory names differ from what's expected.

    Labels are drawn from MERGED_LABEL_MAP so they are compatible with the extended
    10-class label space (BoxingVI 6 + 4 Olympic extensions).

    Args:
        root_dir: Path to Olympic Boxing dataset root directory.
        class_map: Optional {directory_name: canonical_class_name} remapping.
                   Directories not in class_map pass through unchanged.

    Returns:
        DataFrame with columns [video_path, class_name, label, dataset].
        dataset is always "olympic_boxing".

    Raises:
        FileNotFoundError: If root_dir does not exist.
        ValueError: If no valid class directories (with recognized files) are found.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Olympic Boxing root directory not found: {root}")

    recognized_exts = _VIDEO_EXTS | _FRAME_EXTS
    rows: list[dict] = []

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        raw_name = class_dir.name
        class_name = (class_map or {}).get(raw_name, raw_name)
        if class_name not in MERGED_LABEL_MAP:
            continue
        label = MERGED_LABEL_MAP[class_name]
        for media_file in sorted(class_dir.iterdir()):
            if media_file.suffix.lower() in recognized_exts:
                rows.append({
                    "video_path": str(media_file.resolve()),
                    "class_name": class_name,
                    "label": label,
                    "dataset": "olympic_boxing",
                })

    if not rows:
        raise ValueError(
            f"No valid Olympic Boxing class directories with files found in {root}. "
            f"Expected one of: {OLYMPIC_CLASSES}. "
            "Pass class_map if actual directory names differ."
        )

    return pd.DataFrame(rows)


# ── Dataset splitting ─────────────────────────────────────────────────────────

def make_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split a labeled dataset DataFrame into stratified train/val/test sets.

    Stratification is by class_name to preserve class distribution across splits.
    Each class gets at least 1 sample in train and val (if class has ≥3 samples).

    Args:
        df: DataFrame with at least [video_path, class_name, label] columns.
        train_ratio: Fraction for training set (default 0.70).
        val_ratio: Fraction for validation set (default 0.15).
        test_ratio: Fraction for test set (default 0.15).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Dict with keys "train", "val", "test" mapping to DataFrames.
        All rows from df appear in exactly one split.

    Raises:
        ValueError: If ratios don't sum to ~1.0 or df is empty.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.6f}"
        )
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame")

    rng = np.random.default_rng(seed)
    train_rows: list[pd.DataFrame] = []
    val_rows: list[pd.DataFrame] = []
    test_rows: list[pd.DataFrame] = []

    for _class_name, group in df.groupby("class_name", sort=True):
        idx = rng.permutation(len(group))
        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio)) if n >= 3 else 0
        # Test gets the remainder (may be 0 for very small classes)
        arr = group.iloc[idx].reset_index(drop=True)
        train_rows.append(arr.iloc[:n_train])
        val_rows.append(arr.iloc[n_train : n_train + n_val])
        test_rows.append(arr.iloc[n_train + n_val :])

    return {
        "train": pd.concat(train_rows, ignore_index=True),
        "val": pd.concat(val_rows, ignore_index=True),
        "test": pd.concat(test_rows, ignore_index=True),
    }


# ── Distribution analysis ─────────────────────────────────────────────────────

def class_distribution(df: pd.DataFrame) -> pd.Series:
    """Return clip/frame counts per class, sorted descending.

    Args:
        df: DataFrame with a class_name column.

    Returns:
        pd.Series indexed by class_name, values are counts, sorted descending.
    """
    return df["class_name"].value_counts()


# ── Split persistence ─────────────────────────────────────────────────────────

def save_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Save train/val/test split DataFrames as CSV files.

    Creates output_dir (and parents) if it doesn't exist.

    Args:
        splits: Dict with "train", "val", "test" DataFrames.
        output_dir: Directory to write {train,val,test}.csv.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        split_df.to_csv(out / f"{name}.csv", index=False)


def load_splits(splits_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load train/val/test CSV files from a splits directory.

    Args:
        splits_dir: Directory containing train.csv, val.csv, test.csv.

    Returns:
        Dict with "train", "val", "test" DataFrames with original dtypes.

    Raises:
        FileNotFoundError: If any of the three CSV files is missing.
    """
    d = Path(splits_dir)
    result: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        p = d / f"{split}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Split file not found: {p}")
        result[split] = pd.read_csv(p)
    return result
