#!/usr/bin/env python3
"""Adapt tracked video poses plus UVE red/blue/non-boxer scores for detection.

Input NPZ fields: ``track_ids[T,N]``, ``joints_2d[T,N,J,2]``,
``joints_3d[T,N,J,3]``, and ``identity_probabilities[T,N,3]`` ordered as
``red, blue, non_boxer``.  The result is a canonical pickle directly accepted
by ``detect_punch_events.py``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.uve_tracks import canonical_pose_payload, load_track_npz  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cadence", type=int, default=10)
    args = parser.parse_args()
    payload = canonical_pose_payload(load_track_npz(str(args.tracks)), args.cadence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(payload, file)
    print(f"Canonical red/blue pose tracks -> {args.output}")


if __name__ == "__main__":
    main()
