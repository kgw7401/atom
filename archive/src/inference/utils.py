"""Shared inference utilities for LSTM model loading and window preprocessing.

Extracted from scripts/realtime_classify.py so both CLI scripts and the
server analysis pipeline can reuse the same logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.lstm import BoxingLSTM
from src.preprocessing.pipeline import PipelineConfig


def load_model(
    checkpoint_path: Path = Path("models/lstm_best.pt"),
) -> tuple[BoxingLSTM, list[str], np.ndarray, np.ndarray]:
    """Load pretrained LSTM. Returns (model, class_names, scaler_mean, scaler_scale)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = BoxingLSTM(
        input_size=ckpt["input_size"],
        num_classes=len(ckpt["class_names"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, ckpt["class_names"], ckpt["scaler_mean"], ckpt["scaler_scale"]


def preprocess_window(
    buffer: np.ndarray,
    cfg: PipelineConfig,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray | None:
    """Preprocess a single window (window_size, 33, 4) -> (window_size, K*C) scaled.

    Returns (window_size, features) float32 array, or None if invalid.
    """
    K = len(cfg.keypoint_indices)
    C = 3 if cfg.use_z else 2

    # Select keypoints
    selected = buffer[:, cfg.keypoint_indices, :]  # (W, K, 4)
    coords = selected[:, :, :C].copy()
    vis = selected[:, :, 3]

    # Mask low visibility
    mask = vis < cfg.visibility_threshold
    coords[mask] = np.nan

    # Interpolate NaN
    N = coords.shape[0]
    for k in range(K):
        for c in range(C):
            series = coords[:, k, c]
            nans = np.isnan(series)
            if nans.all() or not nans.any():
                continue
            valid = ~nans
            coords[:, k, c] = np.interp(np.arange(N), np.where(valid)[0], series[valid])

    # Check NaN ratio
    if np.isnan(coords).mean() > cfg.nan_ratio_threshold:
        return None

    # Normalize: hip center + shoulder scale
    lh, rh = cfg.left_hip_idx, cfg.right_hip_idx
    ls, rs = cfg.left_shoulder_idx, cfg.right_shoulder_idx

    hip_center = (coords[:, lh, :] + coords[:, rh, :]) / 2.0
    coords = coords - hip_center[:, np.newaxis, :]

    shoulder_dist = np.linalg.norm(coords[:, ls, :2] - coords[:, rs, :2], axis=1)
    shoulder_dist = np.maximum(shoulder_dist, 1e-6)
    coords = coords / shoulder_dist[:, np.newaxis, np.newaxis]

    if cfg.use_velocity:
        velocity = np.concatenate(
            [np.zeros((1, coords.shape[1], coords.shape[2]), dtype=coords.dtype),
             np.diff(coords, axis=0)],
            axis=0,
        )
        coords = np.concatenate([coords, velocity], axis=2)  # (W, K, 2*C)

    # (window_size, K, C) -> (window_size, K*C)
    T = coords.shape[0]
    features = coords.reshape(T, -1).astype(np.float32)

    # Fill NaN survivors with 0 (hip-centered, so 0 = body center)
    np.nan_to_num(features, nan=0.0, copy=False)

    # StandardScaler (same as training)
    features = ((features - scaler_mean) / scaler_scale).astype(np.float32)

    return features
