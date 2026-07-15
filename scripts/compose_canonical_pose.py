#!/usr/bin/env python3
"""Compose and optionally smooth canonical 2D/3D boxer pose sources."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-2d", type=Path, required=True)
    parser.add_argument("--pose-3d", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-3d", type=float, default=1.0)
    parser.add_argument("--smooth-window", type=int, default=1, help="Centered odd-width moving average.")
    parser.add_argument("--interpolate-all-missing", action="store_true",
                        help="Diagnostic oracle-association mode: interpolate every missing RTMW frame.")
    return parser.parse_args()


def interpolate_missing(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    known = np.flatnonzero(valid)
    if not len(known):
        return output
    frames = np.arange(len(output))
    flattened = output.reshape(len(output), -1)
    for column in range(flattened.shape[1]):
        flattened[:, column] = np.interp(frames, known, flattened[known, column])
    return flattened.reshape(output.shape)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return values
    padding = window // 2
    padded = np.pad(values, ((padding, padding),) + ((0, 0),) * (values.ndim - 1), mode="edge")
    cumulative = np.cumsum(padded, axis=0, dtype=np.float64)
    cumulative = np.concatenate((np.zeros_like(cumulative[:1]), cumulative), axis=0)
    return ((cumulative[window:] - cumulative[:-window]) / window).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.scale_3d <= 0:
        raise ValueError("scale-3d must be positive")
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise ValueError("smooth-window must be a positive odd integer")
    with args.pose_2d.open("rb") as file:
        source_2d = pickle.load(file)
    with args.pose_3d.open("rb") as file:
        source_3d = pickle.load(file)
    payload = {}
    for side in ("red", "blue"):
        pose_2d = np.asarray(source_2d[f"pose_{side}_2d"], dtype=np.float32)
        pose_3d = np.asarray(source_3d[f"pose_{side}_3d"], dtype=np.float32) * args.scale_3d
        if len(pose_2d) != len(pose_3d):
            raise ValueError(f"2D and 3D frame counts differ for {side}")
        if args.interpolate_all_missing:
            if f"valid_{side}" in source_2d:
                pose_2d = interpolate_missing(pose_2d, np.asarray(source_2d[f"valid_{side}"], dtype=bool))
            if f"valid_{side}" in source_3d:
                pose_3d = interpolate_missing(pose_3d, np.asarray(source_3d[f"valid_{side}"], dtype=bool))
        payload[f"pose_{side}_2d"] = smooth(pose_2d, args.smooth_window)
        payload[f"pose_{side}_3d"] = smooth(pose_3d, args.smooth_window)
    payload.update({
        "format": "atom-canonical-boxer-pose-v1",
        "source": "composed diagnostic pose",
        "pose_2d_source": str(args.pose_2d.resolve()),
        "pose_3d_source": str(args.pose_3d.resolve()),
        "scale_3d": args.scale_3d,
        "smooth_window": args.smooth_window,
        "interpolate_all_missing": args.interpolate_all_missing,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(payload, file)
    print(f"Composed canonical pose -> {args.output}")


if __name__ == "__main__":
    main()
