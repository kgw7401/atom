"""Keypoint preprocessing pipeline.

Transforms raw (N, 33, 4) keypoint sequences into normalized, windowed
samples ready for classification.

Steps:
    1. Select upper-body keypoints (33 → 9)
    2. Mask low-visibility points as NaN
    3. Interpolate NaN gaps
    4. Normalize: hip-center translation + shoulder-width scaling
    5. Savitzky-Golay smoothing
    6. Sliding window (30 frames, stride 5)
    7. Drop windows with excessive NaN
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import savgol_filter


@dataclass
class PipelineConfig:
    """Pipeline parameters loaded from config YAML."""

    keypoint_indices: list[int]
    window_size: int
    stride: int
    visibility_threshold: float
    nan_ratio_threshold: float
    smoothing_window: int
    smoothing_polyorder: int
    # Indices within the selected keypoints for normalization
    left_hip_idx: int   # index of left_hip in selected keypoints
    right_hip_idx: int  # index of right_hip in selected keypoints
    left_shoulder_idx: int
    right_shoulder_idx: int

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/boxing.yaml") -> PipelineConfig:
        with open(path) as f:
            cfg = yaml.safe_load(f)

        indices = cfg["keypoints"]["indices"]
        names = cfg["keypoints"]["names"]
        pipe = cfg["pipeline"]

        return cls(
            keypoint_indices=indices,
            window_size=pipe["window_size"],
            stride=pipe["stride"],
            visibility_threshold=pipe["visibility_threshold"],
            nan_ratio_threshold=pipe["nan_ratio_threshold"],
            smoothing_window=pipe["smoothing"]["window_length"],
            smoothing_polyorder=pipe["smoothing"]["polyorder"],
            left_hip_idx=names.index("left_hip"),
            right_hip_idx=names.index("right_hip"),
            left_shoulder_idx=names.index("left_shoulder"),
            right_shoulder_idx=names.index("right_shoulder"),
        )


class PreprocessingPipeline:
    """Transforms raw keypoints into classification-ready windows."""

    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig.from_yaml()

    def process(self, raw_keypoints: np.ndarray) -> np.ndarray:
        """Full pipeline: raw (N, 33, 4) → windows (W, window_size, K, 3).

        Args:
            raw_keypoints: (N, 33, 4) array [x, y, z, visibility]

        Returns:
            (W, window_size, num_keypoints, 3) float32 array of valid windows.
            Returns empty array with correct shape if no valid windows.
        """
        K = len(self.cfg.keypoint_indices)

        # 1. Select keypoints: (N, 33, 4) → (N, K, 4)
        selected = raw_keypoints[:, self.cfg.keypoint_indices, :]

        # 2. Split coords and visibility
        coords = selected[:, :, :3].copy()  # (N, K, 3) - x, y, z
        vis = selected[:, :, 3]              # (N, K)

        # 3. Mask low-visibility as NaN
        mask = vis < self.cfg.visibility_threshold
        coords[mask] = np.nan

        # 4. Interpolate NaN gaps (per keypoint, per coordinate)
        coords = self._interpolate_nans(coords)

        # 5. Normalize: hip-center + shoulder scale
        coords = self._normalize(coords)

        # 6. Savitzky-Golay smoothing
        coords = self._smooth(coords)

        # 7. Sliding window
        windows = self._sliding_window(coords)

        # 8. Drop windows with too many NaNs
        windows = self._drop_nan_windows(windows)

        return windows

    def _interpolate_nans(self, coords: np.ndarray) -> np.ndarray:
        """Linear interpolation of NaN values along time axis."""
        N, K, C = coords.shape
        for k in range(K):
            for c in range(C):
                series = coords[:, k, c]
                nans = np.isnan(series)
                if nans.all() or not nans.any():
                    continue
                valid = ~nans
                coords[:, k, c] = np.interp(
                    np.arange(N),
                    np.where(valid)[0],
                    series[valid],
                )
        return coords

    def _normalize(self, coords: np.ndarray) -> np.ndarray:
        """Translate to hip center, scale by shoulder width."""
        lh = self.cfg.left_hip_idx
        rh = self.cfg.right_hip_idx
        ls = self.cfg.left_shoulder_idx
        rs = self.cfg.right_shoulder_idx

        # Hip center: (N, 3)
        hip_center = (coords[:, lh, :] + coords[:, rh, :]) / 2.0
        coords = coords - hip_center[:, np.newaxis, :]

        # Shoulder width for scale: (N,)
        shoulder_dist = np.linalg.norm(
            coords[:, ls, :2] - coords[:, rs, :2], axis=1
        )
        # Avoid division by zero
        shoulder_dist = np.maximum(shoulder_dist, 1e-6)
        coords = coords / shoulder_dist[:, np.newaxis, np.newaxis]

        return coords

    def _smooth(self, coords: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter along time axis."""
        N = coords.shape[0]
        wl = self.cfg.smoothing_window
        if N < wl:
            return coords  # too short to smooth

        # Apply per keypoint per coordinate
        N, K, C = coords.shape
        for k in range(K):
            for c in range(C):
                series = coords[:, k, c]
                if np.isnan(series).any():
                    continue
                coords[:, k, c] = savgol_filter(
                    series, wl, self.cfg.smoothing_polyorder
                )
        return coords

    def _sliding_window(self, coords: np.ndarray) -> np.ndarray:
        """Create overlapping windows: (N, K, 3) → (W, window_size, K, 3)."""
        N = coords.shape[0]
        ws = self.cfg.window_size
        stride = self.cfg.stride

        if N < ws:
            return np.empty((0, ws, coords.shape[1], 3), dtype=np.float32)

        starts = range(0, N - ws + 1, stride)
        windows = np.stack([coords[i : i + ws] for i in starts])
        return windows.astype(np.float32)

    def _drop_nan_windows(self, windows: np.ndarray) -> np.ndarray:
        """Remove windows where NaN ratio exceeds threshold."""
        if windows.size == 0:
            return windows

        nan_ratio = np.isnan(windows).mean(axis=(1, 2, 3))
        valid = nan_ratio <= self.cfg.nan_ratio_threshold
        return windows[valid]
