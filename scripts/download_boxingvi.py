"""Download the BoxingVI boxing dataset.

BoxingVI (arXiv 2511.16524, Nov 2024): 6,915 video clips, 6 punch types, 18 athletes.
GitHub: https://github.com/Bikudebug/BoxingVI

Usage:
    python scripts/download_boxingvi.py --output data/boxingvi

Notes:
    - Requires git to be installed.
    - Dataset size: check GitHub for current storage size.
    - Subsequent calls are no-ops if output directory already exists and is non-empty.
    - After download, run verify mode to check directory structure:
        python scripts/download_boxingvi.py --verify --output data/boxingvi
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from track_b.b2.dataset import BOXINGVI_CLASSES

BOXINGVI_GITHUB = "https://github.com/Bikudebug/BoxingVI"


def download_boxingvi(output_dir: Path) -> None:
    """Clone BoxingVI from GitHub into output_dir.

    Skips download if output_dir already has contents (idempotent).
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"BoxingVI already present at {output_dir}. Skipping download.")
        _verify_structure(output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cloning BoxingVI from {BOXINGVI_GITHUB} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", BOXINGVI_GITHUB, str(output_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: git clone failed:\n{result.stderr}", file=sys.stderr)
        print(
            "Alternative: Download manually from https://github.com/Bikudebug/BoxingVI "
            "and extract to the output directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("BoxingVI downloaded successfully.")
    _verify_structure(output_dir)


def _verify_structure(root: Path) -> None:
    """Check that expected class directories exist and report any mismatches."""
    found = {d.name for d in root.iterdir() if d.is_dir()}
    expected = set(BOXINGVI_CLASSES)
    missing = expected - found
    extra = found - expected - {".git"}  # ignore git internals

    if missing:
        print(f"WARNING: Expected class directories not found: {sorted(missing)}")
        print(
            "  Review actual directory names and update BOXINGVI_LABEL_MAP if needed."
        )
    else:
        clip_counts = {}
        for cls in BOXINGVI_CLASSES:
            cls_dir = root / cls
            clips = list(cls_dir.glob("*.mp4")) + list(cls_dir.glob("*.avi"))
            clip_counts[cls] = len(clips)
        total = sum(clip_counts.values())
        print(f"Structure OK. {total} clips across {len(BOXINGVI_CLASSES)} classes:")
        for cls, count in clip_counts.items():
            print(f"  {cls:20s}: {count} clips")

    if extra:
        print(f"Note: Extra directories (not BoxingVI classes): {sorted(extra)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BoxingVI dataset")
    parser.add_argument(
        "--output", default="data/boxingvi", help="Output directory (default: data/boxingvi)"
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
        download_boxingvi(output_dir)


if __name__ == "__main__":
    main()
