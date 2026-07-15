"""Compact, deterministic pose features for the first BoxingWeb baseline."""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from atom.boxingweb import OracleWindowSample


# These mappings match the ground-truth-pose path in the official BoxingWeb
# dataset loader.  They select 14 semantic joints per boxer, preserving its
# actor-first / opponent-second representation.
JOINT_INDEX = np.array([24, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7])
JOINT_INDEX_GT = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14])

TASK_LABELS = {
    "technique": {"straight": 0, "hook": 1, "uppercut": 2},
    "distance": {"long": 0, "medium": 1, "close": 2},
    "target": {"head": 0, "chest": 1, "abdomen": 2},
    "effect": {"ineffective": 0, "effective": 1},
}


@lru_cache(maxsize=2)
def _load_pose(pose_path: str) -> dict[str, Any]:
    with Path(pose_path).open("rb") as file:
        return pickle.load(file)


def task_targets(sample: OracleWindowSample) -> dict[str, int]:
    """Map raw BoxingWeb labels to the four official attribute tasks."""

    technique = sample.labels["technique"]
    return {
        "technique": TASK_LABELS["technique"][technique[1:]],
        "distance": TASK_LABELS["distance"][sample.labels["distance"]],
        "target": TASK_LABELS["target"][sample.labels["target"]],
        "effect": TASK_LABELS["effect"][sample.labels["effect"]],
    }


def hand_value(sample: OracleWindowSample) -> float:
    """Use the annotated striking hand as an input, as in the official model."""

    return float(sample.labels["technique"].startswith("r"))


def _reindex(pose: np.ndarray) -> np.ndarray:
    """Map raw BoxingWeb ground-truth joints to the official positions."""

    mapped = pose.copy()
    mapped[:, JOINT_INDEX] = mapped[:, JOINT_INDEX_GT]
    return mapped


def _safe_unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-8)


def _rolling_median(values: np.ndarray, window: int = 15) -> np.ndarray:
    """Return an edge-padded rolling median for one scalar time series."""

    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    radius = window // 2
    padded = np.pad(np.asarray(values), (radius, radius), mode="edge")
    return np.array([np.median(padded[index:index + window]) for index in range(len(values))])


def _local_3d(person: np.ndarray, normalize_scale: bool = False) -> np.ndarray:
    """Express selected joints in each boxer's torso-aligned local frame.

    Monocular 3D estimators do not preserve a stable metric scale from frame
    to frame.  ``normalize_scale`` divides by a rolling-median torso length,
    retaining articulation while suppressing that nuisance variation.
    """

    origin = 0.5 * (person[:, 2] + person[:, 1])
    x_axis = person[:, 2] - person[:, 1]
    z_axis = _safe_unit(0.5 * (person[:, 17] + person[:, 16]) - origin)
    y_axis = _safe_unit(np.cross(z_axis, x_axis))
    x_axis = _safe_unit(np.cross(y_axis, z_axis))
    rotation = np.stack((x_axis, y_axis, z_axis), axis=-1)
    local = np.einsum("tjc,tcd->tjd", person[:, JOINT_INDEX] - origin[:, None, :], rotation)
    if normalize_scale:
        shoulder_center = 0.5 * (person[:, 17] + person[:, 16])
        torso_length = np.linalg.norm(shoulder_center - origin, axis=-1)
        valid = np.isfinite(torso_length) & (torso_length > 1e-6)
        fallback = float(np.median(torso_length[valid])) if valid.any() else 1.0
        torso_length = np.where(valid, torso_length, fallback)
        scale = np.maximum(_rolling_median(torso_length), max(fallback * 0.25, 1e-6))
        local = local / scale[:, None, None]
    return local


def _local_2d(actor: np.ndarray, opponent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center both fighters on the actor and scale by its shoulder width."""

    origin = 0.5 * (actor[:, 2] + actor[:, 1])
    shoulder_width = np.linalg.norm(actor[:, 17] - actor[:, 16], axis=-1, keepdims=True)
    shoulder_width = np.maximum(shoulder_width, 1e-4)
    return (
        (actor[:, JOINT_INDEX] - origin[:, None, :]) / shoulder_width[:, None, :],
        (opponent[:, JOINT_INDEX] - origin[:, None, :]) / shoulder_width[:, None, :],
    )


def _resample(sequence: np.ndarray, frames: int) -> np.ndarray:
    if sequence.shape[0] == frames:
        return sequence
    source = np.linspace(0.0, 1.0, num=sequence.shape[0])
    target = np.linspace(0.0, 1.0, num=frames)
    flattened = sequence.reshape(sequence.shape[0], -1)
    result = np.empty((frames, flattened.shape[1]), dtype=np.float32)
    for column in range(flattened.shape[1]):
        result[:, column] = np.interp(target, source, flattened[:, column])
    return result.reshape((frames,) + sequence.shape[1:])


def extract_pose_feature(sample: OracleWindowSample, data_root: Path, frames: int = 16) -> np.ndarray:
    """Return a `[frames, 28, 5]` actor-relative feature sequence.

    The five channels are normalized 2D x/y plus torso-local 3D x/y/z. The
    first 14 joints belong to the attacker and the next 14 to the opponent.
    """

    return extract_pose_interval(
        data_root / sample.pose_path,
        sample.labels["side"],
        sample.event_start_frame,
        sample.event_end_frame,
        frames=frames,
    )


def extract_pose_interval(pose_path: Path, side: str, start_frame: int, end_frame: int, frames: int = 16) -> np.ndarray:
    """Return the actor-relative pose tensor for any proposed punch interval."""

    if frames < 2:
        raise ValueError("frames must be at least 2")
    if side not in {"red", "blue"}:
        raise ValueError(f"Expected red or blue side, got {side!r}")
    raw = _load_pose(str(pose_path))
    start, end = start_frame, end_frame + 1
    red_2d = _reindex(np.asarray(raw["pose_red_2d"])[start:end])
    blue_2d = _reindex(np.asarray(raw["pose_blue_2d"])[start:end])
    red_3d = _reindex(np.asarray(raw["pose_red_3d"])[start:end])
    blue_3d = _reindex(np.asarray(raw["pose_blue_3d"])[start:end])

    if side == "red":
        actor_2d, opponent_2d = red_2d, blue_2d
        actor_3d, opponent_3d = red_3d, blue_3d
    else:
        actor_2d, opponent_2d = blue_2d, red_2d
        actor_3d, opponent_3d = blue_3d, red_3d
    actor_2d, opponent_2d = _local_2d(actor_2d, opponent_2d)
    feature = np.concatenate(
        (
            np.concatenate((actor_2d, opponent_2d), axis=1),
            np.concatenate((_local_3d(actor_3d), _local_3d(opponent_3d)), axis=1),
        ),
        axis=2,
    )
    return np.nan_to_num(_resample(feature, frames), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_match_pose_features(pose_path: Path) -> np.ndarray:
    """Return `[time, 140]` canonical red-relative features for event detection.

    Unlike ``extract_pose_feature``, this function does not know who will
    attack.  It keeps red as the reference fighter and includes the blue
    fighter relative to that reference, so one temporal model can predict a
    punch-activity timeline for both red and blue.
    """

    with pose_path.open("rb") as file:
        raw: dict[str, Any] = pickle.load(file)
    red_2d = _reindex(np.asarray(raw["pose_red_2d"]))
    blue_2d = _reindex(np.asarray(raw["pose_blue_2d"]))
    red_3d = _reindex(np.asarray(raw["pose_red_3d"]))
    blue_3d = _reindex(np.asarray(raw["pose_blue_3d"]))
    frame_count = min(red_2d.shape[0], blue_2d.shape[0], red_3d.shape[0], blue_3d.shape[0])
    red_2d, blue_2d = _local_2d(red_2d[:frame_count], blue_2d[:frame_count])
    feature = np.concatenate(
        (
            np.concatenate((red_2d, blue_2d), axis=1),
            np.concatenate((_local_3d(red_3d[:frame_count]), _local_3d(blue_3d[:frame_count])), axis=1),
        ),
        axis=2,
    )
    return np.nan_to_num(feature.reshape(frame_count, -1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_boxer_pose_features(
    pose_path: Path,
    side: str,
    include_opponent: bool = False,
    feature_mode: str = "absolute",
) -> np.ndarray:
    """Return BoxMind-style ``[T, 28, 5]`` pose for one focal boxer.

    By default only the focal boxer's 14 joints are returned, following the
    paper's independent-per-boxer formulation. ``include_opponent`` retains
    the 28-joint actor/opponent representation for controlled comparison.
    """

    if side not in {"red", "blue"}:
        raise ValueError(f"Expected red or blue side, got {side!r}")
    modes = {
        "absolute", "local", "absolute-motion", "local-motion", "hybrid-motion", "hybrid-multiscale",
        "absolute-normalized-3d", "local-normalized-3d", "absolute-depth",
        "hybrid-depth-motion", "hybrid-motion-3d", "hybrid-multiscale-3d",
    }
    if feature_mode not in modes:
        raise ValueError(f"Unknown feature mode {feature_mode!r}; expected one of {sorted(modes)}")
    raw = _load_pose(str(pose_path))
    red_2d = _reindex(np.asarray(raw["pose_red_2d"]))
    blue_2d = _reindex(np.asarray(raw["pose_blue_2d"]))
    red_3d = _reindex(np.asarray(raw["pose_red_3d"]))
    blue_3d = _reindex(np.asarray(raw["pose_blue_3d"]))
    frame_count = min(red_2d.shape[0], blue_2d.shape[0], red_3d.shape[0], blue_3d.shape[0])
    if side == "red":
        actor_2d, opponent_2d = red_2d[:frame_count], blue_2d[:frame_count]
        actor_3d, opponent_3d = red_3d[:frame_count], blue_3d[:frame_count]
    else:
        actor_2d, opponent_2d = blue_2d[:frame_count], red_2d[:frame_count]
        actor_3d, opponent_3d = blue_3d[:frame_count], red_3d[:frame_count]
    absolute_2d = actor_2d[:, JOINT_INDEX]
    opponent_absolute_2d = opponent_2d[:, JOINT_INDEX]
    local_2d, opponent_local_2d = _local_2d(actor_2d, opponent_2d)
    base_mode = {
        "absolute-normalized-3d": "absolute",
        "local-normalized-3d": "local",
        "absolute-depth": "absolute",
        "hybrid-depth-motion": "hybrid-motion",
        "hybrid-motion-3d": "hybrid-motion",
        "hybrid-multiscale-3d": "hybrid-multiscale",
    }.get(feature_mode, feature_mode)
    if base_mode.startswith("local"):
        pose_2d, opponent_pose_2d = local_2d, opponent_local_2d
    else:
        pose_2d, opponent_pose_2d = absolute_2d, opponent_absolute_2d
    if base_mode in {"hybrid-motion", "hybrid-multiscale"}:
        pose_2d = np.concatenate((absolute_2d, local_2d), axis=2)
        opponent_pose_2d = np.concatenate((opponent_absolute_2d, opponent_local_2d), axis=2)
    if base_mode == "hybrid-multiscale":
        pose_parts = [pose_2d]
        opponent_parts = [opponent_pose_2d]
        for lag in (1, 3, 6):
            pose_parts.append(pose_2d - np.concatenate((np.repeat(pose_2d[:1], lag, axis=0), pose_2d[:-lag]), axis=0))
            opponent_parts.append(opponent_pose_2d - np.concatenate((np.repeat(opponent_pose_2d[:1], lag, axis=0), opponent_pose_2d[:-lag]), axis=0))
        pose_2d = np.concatenate(pose_parts, axis=2)
        opponent_pose_2d = np.concatenate(opponent_parts, axis=2)
    elif base_mode.endswith("motion"):
        velocity = np.diff(pose_2d, axis=0, prepend=pose_2d[:1])
        opponent_velocity = np.diff(opponent_pose_2d, axis=0, prepend=opponent_pose_2d[:1])
        pose_2d = np.concatenate((pose_2d, velocity), axis=2)
        opponent_pose_2d = np.concatenate((opponent_pose_2d, opponent_velocity), axis=2)
    depth_only = feature_mode in {"absolute-depth", "hybrid-depth-motion"}
    normalized_3d = feature_mode.endswith("-3d") or depth_only
    motion_3d = feature_mode in {"hybrid-depth-motion", "hybrid-motion-3d", "hybrid-multiscale-3d"}
    derived_2d_only = not normalized_3d and (base_mode.endswith("motion") or base_mode == "hybrid-multiscale")
    if include_opponent:
        pose_2d = np.concatenate((pose_2d, opponent_pose_2d), axis=1)
    if derived_2d_only:
        return np.nan_to_num(pose_2d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    pose_3d = _local_3d(actor_3d, normalize_scale=normalized_3d)
    if include_opponent:
        pose_3d = np.concatenate((pose_3d, _local_3d(opponent_3d, normalize_scale=normalized_3d)), axis=1)
    if depth_only:
        pose_3d = pose_3d[..., 2:3]
    if motion_3d:
        if base_mode == "hybrid-multiscale":
            parts = [pose_3d]
            for lag in (1, 3, 6):
                previous = np.concatenate((np.repeat(pose_3d[:1], lag, axis=0), pose_3d[:-lag]), axis=0)
                parts.append(pose_3d - previous)
            pose_3d = np.concatenate(parts, axis=2)
        else:
            pose_3d = np.concatenate((pose_3d, np.diff(pose_3d, axis=0, prepend=pose_3d[:1])), axis=2)
        features = np.concatenate((pose_2d, pose_3d), axis=2)
    else:
        features = np.concatenate((pose_2d, pose_3d), axis=2)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def select_pose_feature_channels(features: np.ndarray, pose_channels: int, feature_mode: str) -> np.ndarray:
    """Select 2D or 2D+3D channels while retaining derived motion layouts."""

    if feature_mode.endswith("-3d") or feature_mode in {"absolute-depth", "hybrid-depth-motion"}:
        if pose_channels != 5:
            raise ValueError("3D feature modes require --pose-channels 5.")
        return features
    if feature_mode.endswith("motion") or feature_mode == "hybrid-multiscale":
        if pose_channels != 2:
            raise ValueError("Motion feature modes currently support 2D source pose only.")
        return features
    return features[..., :pose_channels]
