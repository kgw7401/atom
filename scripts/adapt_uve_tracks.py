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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.uve_tracks import canonical_pose_payload, load_track_npz  # noqa: E402
from atom.rtmw_pose import interpolate_all_gaps, interpolate_short_gaps, smooth_sequence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cadence", type=int, default=10)
    parser.add_argument("--identity-threshold", type=float, default=.35)
    parser.add_argument("--continuity-bonus", type=float, default=.25)
    parser.add_argument("--spatial-bonus", type=float, default=1.5)
    parser.add_argument("--require-two-boxers", action="store_true")
    parser.add_argument("--interpolate-all-missing", action="store_true")
    parser.add_argument("--max-gap", type=int, default=15)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--metadata-from", type=Path, default=None,
                        help="Optional existing pose payload supplying video dimensions and source metadata.")
    args = parser.parse_args()
    payload = canonical_pose_payload(
        load_track_npz(str(args.tracks)), args.cadence,
        identity_threshold=args.identity_threshold,
        continuity_bonus=args.continuity_bonus,
        require_two_boxers=args.require_two_boxers,
        spatial_bonus=args.spatial_bonus,
    )
    for side in ("red", "blue"):
        valid = np.asarray(payload[f"{side}_track_ids"]) >= 0
        interpolate = interpolate_all_gaps if args.interpolate_all_missing else (
            lambda values, mask: interpolate_short_gaps(values, mask, args.max_gap)
        )
        payload[f"pose_{side}_2d"] = smooth_sequence(interpolate(payload[f"pose_{side}_2d"], valid), args.smooth_window)
        payload[f"pose_{side}_3d"] = smooth_sequence(interpolate(payload[f"pose_{side}_3d"], valid), args.smooth_window)
        payload[f"valid_{side}"] = valid
    if args.metadata_from:
        with args.metadata_from.open("rb") as file:
            metadata = pickle.load(file)
        for key in (
            "source_video", "source_start_frame", "source_end_frame_exclusive",
            "fps", "width", "height", "tracker", "identity_checkpoint",
        ):
            if key in metadata:
                payload[key] = metadata[key]
    payload["source"] = "UVE-refined tracked RTMW pose; no inference-time GT"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(payload, file)
    print(f"Canonical red/blue pose tracks -> {args.output}")


if __name__ == "__main__":
    main()
