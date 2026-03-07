"""Tests for B2 Task 5: Dataset preparation (BoxingVI + Olympic Boxing)."""

from pathlib import Path

import pandas as pd
import pytest

from track_b.b2.dataset import (
    BOXINGVI_CLASSES,
    BOXINGVI_LABEL_MAP,
    MERGED_LABEL_MAP,
    OLYMPIC_CLASSES,
    OLYMPIC_TO_BOXINGVI,
    class_distribution,
    load_splits,
    make_splits,
    save_splits,
    scan_boxingvi,
    scan_olympic_boxing,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def boxingvi_dir(tmp_path: Path) -> Path:
    """Minimal BoxingVI-style directory with 5 fake clips per class."""
    root = tmp_path / "boxingvi"
    for cls in BOXINGVI_CLASSES:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(5):
            (cls_dir / f"clip_{i:03d}.mp4").touch()
    return root


@pytest.fixture
def olympic_dir(tmp_path: Path) -> Path:
    """Minimal Olympic Boxing-style directory with 10 fake files per class."""
    root = tmp_path / "olympic_boxing"
    for cls in OLYMPIC_CLASSES:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(10):
            (cls_dir / f"clip_{i:04d}.mp4").touch()
    return root


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Labeled DataFrame with 10 clips per BoxingVI class (60 rows total)."""
    rows = []
    for cls in BOXINGVI_CLASSES:
        for i in range(10):
            rows.append({
                "video_path": f"/fake/{cls}/clip_{i:03d}.mp4",
                "class_name": cls,
                "label": BOXINGVI_LABEL_MAP[cls],
                "dataset": "boxingvi",
            })
    return pd.DataFrame(rows)


# ── scan_boxingvi ─────────────────────────────────────────────────────────────

class TestScanBoxingVI:
    def test_returns_dataframe(self, boxingvi_dir):
        df = scan_boxingvi(boxingvi_dir)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, boxingvi_dir):
        df = scan_boxingvi(boxingvi_dir)
        assert {"video_path", "class_name", "label", "dataset"}.issubset(df.columns)

    def test_finds_all_classes(self, boxingvi_dir):
        df = scan_boxingvi(boxingvi_dir)
        assert set(df["class_name"].unique()) == set(BOXINGVI_CLASSES)

    def test_clip_count(self, boxingvi_dir):
        df = scan_boxingvi(boxingvi_dir)
        assert len(df) == len(BOXINGVI_CLASSES) * 5

    def test_labels_match_map(self, boxingvi_dir):
        df = scan_boxingvi(boxingvi_dir)
        for _, row in df.iterrows():
            assert row["label"] == BOXINGVI_LABEL_MAP[row["class_name"]]

    def test_dataset_column_is_boxingvi(self, boxingvi_dir):
        df = scan_boxingvi(boxingvi_dir)
        assert (df["dataset"] == "boxingvi").all()

    def test_ignores_non_video_files(self, boxingvi_dir):
        (boxingvi_dir / "jab" / "readme.txt").touch()
        (boxingvi_dir / "jab" / "labels.csv").touch()
        df = scan_boxingvi(boxingvi_dir)
        for path in df["video_path"]:
            assert Path(path).suffix in (".mp4", ".avi", ".mov", ".webm")

    def test_ignores_unknown_class_dirs(self, boxingvi_dir):
        unknown = boxingvi_dir / "unknown_punch"
        unknown.mkdir()
        (unknown / "clip.mp4").touch()
        df = scan_boxingvi(boxingvi_dir)
        assert "unknown_punch" not in df["class_name"].values

    def test_raises_if_dir_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scan_boxingvi(tmp_path / "does_not_exist")

    def test_raises_if_no_valid_class_dirs(self, tmp_path):
        root = tmp_path / "empty_boxingvi"
        root.mkdir()
        with pytest.raises(ValueError, match="No valid BoxingVI"):
            scan_boxingvi(root)

    def test_accepts_string_path(self, boxingvi_dir):
        df = scan_boxingvi(str(boxingvi_dir))
        assert len(df) > 0

    def test_video_paths_are_absolute(self, boxingvi_dir):
        df = scan_boxingvi(boxingvi_dir)
        for path in df["video_path"]:
            assert Path(path).is_absolute()


# ── scan_olympic_boxing ───────────────────────────────────────────────────────

class TestScanOlympicBoxing:
    def test_returns_dataframe(self, olympic_dir):
        df = scan_olympic_boxing(olympic_dir)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, olympic_dir):
        df = scan_olympic_boxing(olympic_dir)
        assert {"video_path", "class_name", "label", "dataset"}.issubset(df.columns)

    def test_finds_known_classes(self, olympic_dir):
        df = scan_olympic_boxing(olympic_dir)
        assert "jab" in df["class_name"].values
        assert "block" in df["class_name"].values
        assert "body_hook" in df["class_name"].values

    def test_clip_count(self, olympic_dir):
        df = scan_olympic_boxing(olympic_dir)
        assert len(df) == len(OLYMPIC_CLASSES) * 10

    def test_labels_in_merged_map(self, olympic_dir):
        df = scan_olympic_boxing(olympic_dir)
        for _, row in df.iterrows():
            assert row["label"] == MERGED_LABEL_MAP[row["class_name"]]

    def test_dataset_column_is_olympic(self, olympic_dir):
        df = scan_olympic_boxing(olympic_dir)
        assert (df["dataset"] == "olympic_boxing").all()

    def test_class_map_remapping(self, tmp_path):
        """class_map renames a non-canonical directory to a canonical class."""
        root = tmp_path / "olympic_custom"
        (root / "BoxingJab").mkdir(parents=True)
        (root / "BoxingJab" / "clip_001.mp4").touch()
        df = scan_olympic_boxing(root, class_map={"BoxingJab": "jab"})
        assert len(df) == 1
        assert df.iloc[0]["class_name"] == "jab"
        assert df.iloc[0]["label"] == MERGED_LABEL_MAP["jab"]

    def test_class_map_partial_remap(self, tmp_path):
        """class_map only needs to cover non-canonical names."""
        root = tmp_path / "olympic_partial"
        (root / "jab").mkdir(parents=True)
        (root / "jab" / "clip.mp4").touch()
        (root / "BoxingCross").mkdir(parents=True)
        (root / "BoxingCross" / "clip.mp4").touch()
        df = scan_olympic_boxing(root, class_map={"BoxingCross": "cross"})
        classes = set(df["class_name"].values)
        assert "jab" in classes
        assert "cross" in classes

    def test_ignores_unrecognized_dirs_without_map(self, tmp_path):
        """Directories not in MERGED_LABEL_MAP and not in class_map are silently skipped."""
        root = tmp_path / "olympic_extra"
        (root / "jab").mkdir(parents=True)
        (root / "jab" / "clip.mp4").touch()
        (root / "weird_action").mkdir(parents=True)
        (root / "weird_action" / "clip.mp4").touch()
        df = scan_olympic_boxing(root)
        assert "weird_action" not in df["class_name"].values
        assert len(df) == 1

    def test_accepts_image_files(self, tmp_path):
        """Olympic Boxing dataset may contain image frames."""
        root = tmp_path / "olympic_frames"
        (root / "jab").mkdir(parents=True)
        (root / "jab" / "frame_001.jpg").touch()
        (root / "jab" / "frame_002.png").touch()
        df = scan_olympic_boxing(root)
        assert len(df) == 2

    def test_raises_if_dir_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scan_olympic_boxing(tmp_path / "does_not_exist")

    def test_raises_if_no_valid_classes(self, tmp_path):
        root = tmp_path / "olympic_bad"
        (root / "unknown_class").mkdir(parents=True)
        (root / "unknown_class" / "clip.mp4").touch()
        with pytest.raises(ValueError, match="No valid Olympic Boxing"):
            scan_olympic_boxing(root)

    def test_accepts_string_path(self, olympic_dir):
        df = scan_olympic_boxing(str(olympic_dir))
        assert len(df) > 0


# ── make_splits ───────────────────────────────────────────────────────────────

class TestMakeSplits:
    def test_returns_three_keys(self, small_df):
        splits = make_splits(small_df)
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_no_overlap_between_splits(self, small_df):
        splits = make_splits(small_df)
        train_paths = set(splits["train"]["video_path"])
        val_paths = set(splits["val"]["video_path"])
        test_paths = set(splits["test"]["video_path"])
        assert train_paths.isdisjoint(val_paths)
        assert train_paths.isdisjoint(test_paths)
        assert val_paths.isdisjoint(test_paths)

    def test_all_items_assigned(self, small_df):
        splits = make_splits(small_df)
        total = sum(len(s) for s in splits.values())
        assert total == len(small_df)

    def test_train_is_largest(self, small_df):
        splits = make_splits(small_df)
        assert len(splits["train"]) > len(splits["val"])
        assert len(splits["train"]) > len(splits["test"])

    def test_all_classes_in_train(self, small_df):
        splits = make_splits(small_df)
        assert set(splits["train"]["class_name"]) == set(BOXINGVI_CLASSES)

    def test_all_classes_in_val(self, small_df):
        splits = make_splits(small_df)
        assert set(splits["val"]["class_name"]) == set(BOXINGVI_CLASSES)

    def test_reproducible_with_same_seed(self, small_df):
        s1 = make_splits(small_df, seed=42)
        s2 = make_splits(small_df, seed=42)
        pd.testing.assert_frame_equal(s1["train"], s2["train"])
        pd.testing.assert_frame_equal(s1["val"], s2["val"])
        pd.testing.assert_frame_equal(s1["test"], s2["test"])

    def test_different_seeds_give_different_splits(self, small_df):
        s1 = make_splits(small_df, seed=42)
        s2 = make_splits(small_df, seed=99)
        # With 60 rows and 6 classes, different seeds should produce different orderings
        assert not s1["train"]["video_path"].equals(s2["train"]["video_path"])

    def test_raises_on_bad_ratios(self, small_df):
        with pytest.raises(ValueError, match="sum to 1.0"):
            make_splits(small_df, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_raises_on_empty_df(self):
        with pytest.raises(ValueError, match="empty"):
            make_splits(pd.DataFrame())

    def test_custom_ratios(self, small_df):
        splits = make_splits(small_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        assert len(splits["train"]) > len(splits["val"])


# ── class_distribution ────────────────────────────────────────────────────────

class TestClassDistribution:
    def test_returns_series(self, small_df):
        dist = class_distribution(small_df)
        assert isinstance(dist, pd.Series)

    def test_all_classes_present(self, small_df):
        dist = class_distribution(small_df)
        for cls in BOXINGVI_CLASSES:
            assert cls in dist.index

    def test_counts_correct(self, small_df):
        dist = class_distribution(small_df)
        for cls in BOXINGVI_CLASSES:
            assert dist[cls] == 10  # 10 clips per class in small_df

    def test_sorted_descending(self, small_df):
        # Add extra jab clips to make it the most frequent
        extra = pd.DataFrame([{
            "video_path": f"/fake/jab/extra_{i}.mp4",
            "class_name": "jab",
            "label": 0,
            "dataset": "boxingvi",
        } for i in range(5)])
        df = pd.concat([small_df, extra], ignore_index=True)
        dist = class_distribution(df)
        assert dist.index[0] == "jab"  # highest count first
        assert dist.iloc[0] > dist.iloc[1]

    def test_works_on_split_subset(self, small_df):
        splits = make_splits(small_df)
        dist = class_distribution(splits["train"])
        assert isinstance(dist, pd.Series)
        assert len(dist) > 0


# ── save_splits / load_splits ─────────────────────────────────────────────────

class TestSaveLoadSplits:
    def test_roundtrip(self, small_df, tmp_path):
        splits = make_splits(small_df)
        save_splits(splits, tmp_path / "splits")
        loaded = load_splits(tmp_path / "splits")
        for name in ("train", "val", "test"):
            pd.testing.assert_frame_equal(
                splits[name].reset_index(drop=True),
                loaded[name].reset_index(drop=True),
            )

    def test_creates_three_csv_files(self, small_df, tmp_path):
        splits = make_splits(small_df)
        save_splits(splits, tmp_path / "splits")
        for name in ("train", "val", "test"):
            assert (tmp_path / "splits" / f"{name}.csv").exists()

    def test_creates_parent_dirs(self, small_df, tmp_path):
        splits = make_splits(small_df)
        nested = tmp_path / "deep" / "nested" / "splits"
        save_splits(splits, nested)
        assert nested.exists()

    def test_raises_if_split_missing(self, small_df, tmp_path):
        splits = make_splits(small_df)
        splits_dir = tmp_path / "partial_splits"
        splits_dir.mkdir()
        splits["train"].to_csv(splits_dir / "train.csv", index=False)
        splits["val"].to_csv(splits_dir / "val.csv", index=False)
        # No test.csv
        with pytest.raises(FileNotFoundError, match="test.csv"):
            load_splits(splits_dir)

    def test_column_names_preserved(self, small_df, tmp_path):
        splits = make_splits(small_df)
        save_splits(splits, tmp_path / "splits")
        loaded = load_splits(tmp_path / "splits")
        assert set(loaded["train"].columns) == {"video_path", "class_name", "label", "dataset"}

    def test_accepts_string_path(self, small_df, tmp_path):
        splits = make_splits(small_df)
        save_splits(splits, str(tmp_path / "splits"))
        loaded = load_splits(str(tmp_path / "splits"))
        assert "train" in loaded


# ── Label map consistency ─────────────────────────────────────────────────────

class TestLabelMapConsistency:
    def test_boxingvi_classes_in_merged_map(self):
        for cls in BOXINGVI_CLASSES:
            assert cls in MERGED_LABEL_MAP

    def test_olympic_extra_classes_in_merged_map(self):
        extra_classes = ["body_jab", "body_cross", "body_hook", "block"]
        for cls in extra_classes:
            assert cls in MERGED_LABEL_MAP

    def test_merged_map_labels_are_unique(self):
        labels = list(MERGED_LABEL_MAP.values())
        assert len(labels) == len(set(labels)), "Duplicate labels in MERGED_LABEL_MAP"

    def test_boxingvi_map_labels_are_unique(self):
        labels = list(BOXINGVI_LABEL_MAP.values())
        assert len(labels) == len(set(labels))

    def test_olympic_to_boxingvi_covers_all_olympic_classes(self):
        for cls in OLYMPIC_CLASSES:
            assert cls in OLYMPIC_TO_BOXINGVI, f"{cls} missing from OLYMPIC_TO_BOXINGVI"

    def test_olympic_to_boxingvi_targets_are_valid(self):
        for olympic_cls, boxingvi_cls in OLYMPIC_TO_BOXINGVI.items():
            if boxingvi_cls is not None:
                assert boxingvi_cls in BOXINGVI_LABEL_MAP, (
                    f"OLYMPIC_TO_BOXINGVI[{olympic_cls!r}] = {boxingvi_cls!r} "
                    "not in BOXINGVI_LABEL_MAP"
                )

    def test_jab_cross_map_directly(self):
        assert OLYMPIC_TO_BOXINGVI["jab"] == "jab"
        assert OLYMPIC_TO_BOXINGVI["cross"] == "cross"

    def test_body_classes_map_to_none(self):
        for cls in ["body_jab", "body_cross", "body_hook", "block"]:
            assert OLYMPIC_TO_BOXINGVI[cls] is None
