"""Download the Olympic Boxing Punch Classification dataset from Kaggle.

Stefanski et al., MDPI Entropy 2024. 312K frames, 8 classes.
Kaggle: https://www.kaggle.com/datasets/piotrstefaskiue/olympic-boxing-punch-classification-video-dataset

Usage:
    # Requires Kaggle API credentials (~/.kaggle/kaggle.json)
    python scripts/download_olympic_boxing.py --output data/olympic_boxing

Setup:
    1. Create a Kaggle account at https://kaggle.com
    2. Generate API token: Account → API → Create New API Token
    3. Save kaggle.json to ~/.kaggle/kaggle.json (chmod 600)
    4. Install: pip install kaggle
    5. Accept dataset license at the Kaggle dataset URL above

License:
    Non-commercial use only (verify at dataset page before large-scale use).
    See Open Question in specs/track-b-spec.md regarding license compatibility.

Notes:
    - The downloaded directory structure determines actual class names.
    - After download, run with --verify to check structure and get a class_map
      if directory names differ from expected.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from track_b.b2.dataset import MERGED_LABEL_MAP, OLYMPIC_CLASSES

KAGGLE_DATASET = "piotrstefaskiue/olympic-boxing-punch-classification-video-dataset"
KAGGLE_URL = (
    "https://www.kaggle.com/datasets/"
    "piotrstefaskiue/olympic-boxing-punch-classification-video-dataset"
)


def download_olympic_boxing(output_dir: Path) -> None:
    """Download Olympic Boxing dataset from Kaggle into output_dir.

    Skips download if output_dir already has contents (idempotent).
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Olympic Boxing dataset already present at {output_dir}. Skipping.")
        _verify_structure(output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from Kaggle: {KAGGLE_DATASET} ...")
    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "--dataset",
            KAGGLE_DATASET,
            "--path",
            str(output_dir),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: kaggle download failed:\n{result.stderr}", file=sys.stderr)
        print(
            "\nTroubleshooting:\n"
            "  1. Install kaggle CLI: pip install kaggle\n"
            "  2. Set up credentials: https://www.kaggle.com/docs/api\n"
            "     (~/.kaggle/kaggle.json with chmod 600)\n"
            f"  3. Accept license at: {KAGGLE_URL}\n"
            "  4. Or download manually and extract to the output directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("Olympic Boxing dataset downloaded successfully.")
    _verify_structure(output_dir)


def _verify_structure(root: Path) -> None:
    """Check directory structure and print class_map if names differ from expected."""
    found_dirs = sorted(d.name for d in root.iterdir() if d.is_dir())
    print(f"Found {len(found_dirs)} directories: {found_dirs}")

    expected = set(OLYMPIC_CLASSES)
    found_set = set(found_dirs)
    missing = expected - found_set
    unrecognized = found_set - set(MERGED_LABEL_MAP.keys())

    if missing:
        print(f"WARNING: Expected class directories not found: {sorted(missing)}")

    if unrecognized:
        print(
            f"\nUnrecognized directory names: {sorted(unrecognized)}\n"
            "These will be ignored by scan_olympic_boxing() unless you pass a class_map.\n"
            "Example class_map to use in dataset.py or your processing script:\n"
        )
        # Print a suggested class_map skeleton
        print("    class_map = {")
        for name in sorted(unrecognized):
            print(f'        "{name}": "???",  # TODO: map to canonical class name')
        print("    }")
        print(f"\nCanonical class names: {sorted(MERGED_LABEL_MAP.keys())}")
    else:
        # Count files per class
        for cls in sorted(found_set & set(MERGED_LABEL_MAP.keys())):
            cls_dir = root / cls
            n_files = sum(1 for f in cls_dir.iterdir() if f.is_file())
            print(f"  {cls:20s}: {n_files} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Olympic Boxing Kaggle dataset")
    parser.add_argument(
        "--output",
        default="data/olympic_boxing",
        help="Output directory (default: data/olympic_boxing)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify structure of existing download, skip re-download",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    if args.verify:
        if not output_dir.exists():
            print(f"ERROR: Directory not found: {output_dir}", file=sys.stderr)
            sys.exit(1)
        _verify_structure(output_dir)
    else:
        download_olympic_boxing(output_dir)


if __name__ == "__main__":
    main()
