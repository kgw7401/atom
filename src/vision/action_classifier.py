"""LSTM inference → action segments.

Loads the pretrained BoxingLSTM model and classifies raw keypoints from
MediaPipe into boxing action segments (guard, jab, cross, hooks, etc.).

Heavy dependency (torch) is imported lazily so the module can be
imported without it installed.

Reference: spec/roadmap.md Phase 2b
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.state.constants import PUNCH_NAMES
from src.state.types import ActionSegment


# ---------------------------------------------------------------------------
# Classifier config (loaded from boxing.yaml — no torch needed)
# ---------------------------------------------------------------------------

@dataclass
class _ClassifierConfig:
    """Pipeline parameters needed for LSTM preprocessing."""

    keypoint_indices: list[int]
    window_size: int
    stride: int
    visibility_threshold: float
    nan_ratio_threshold: float
    left_hip_idx: int
    right_hip_idx: int
    left_shoulder_idx: int
    right_shoulder_idx: int
    use_velocity: bool
    use_z: bool

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/boxing.yaml") -> _ClassifierConfig:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        names = cfg["keypoints"]["names"]
        pipe = cfg["pipeline"]
        features = pipe.get("features", {})
        return cls(
            keypoint_indices=cfg["keypoints"]["indices"],
            window_size=pipe["window_size"],
            stride=pipe["stride"],
            visibility_threshold=pipe["visibility_threshold"],
            nan_ratio_threshold=pipe["nan_ratio_threshold"],
            left_hip_idx=names.index("left_hip"),
            right_hip_idx=names.index("right_hip"),
            left_shoulder_idx=names.index("left_shoulder"),
            right_shoulder_idx=names.index("right_shoulder"),
            use_velocity=features.get("velocity", False),
            use_z=features.get("use_z", True),
        )


# ---------------------------------------------------------------------------
# LSTM model loader (torch imported lazily)
# ---------------------------------------------------------------------------

def _load_lstm_model(checkpoint_path: Path) -> tuple:
    """Load BoxingLSTM from checkpoint.

    Returns (model, class_names, scaler_mean, scaler_scale).
    """
    import torch
    import torch.nn as nn

    class BoxingLSTM(nn.Module):
        """LSTM for boxing action classification (9 classes)."""

        def __init__(
            self, input_size: int, num_classes: int,
            hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BoxingLSTM(
        input_size=ckpt["input_size"],
        num_classes=len(ckpt["class_names"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, ckpt["class_names"], ckpt["scaler_mean"], ckpt["scaler_scale"]


# ---------------------------------------------------------------------------
# ActionClassifier
# ---------------------------------------------------------------------------

class ActionClassifier:
    """Classify boxing actions from raw keypoints using LSTM.

    Usage::

        classifier = ActionClassifier.load()
        segments = classifier.classify(raw_keypoints, timestamps_s)
    """

    def __init__(
        self,
        model,
        class_names: list[str],
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        config: _ClassifierConfig,
        confidence_threshold: float = 0.7,
    ):
        self.model = model
        self.class_names = class_names
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.cfg = config
        self.confidence_threshold = confidence_threshold

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        confidence_threshold: float | None = None,
    ) -> ActionClassifier:
        """Load classifier from checkpoint and config files."""
        from server.config import settings

        if checkpoint_path is None:
            checkpoint_path = settings.lstm_checkpoint
        if config_path is None:
            config_path = settings.config_path
        if confidence_threshold is None:
            confidence_threshold = settings.confidence_threshold

        model, class_names, scaler_mean, scaler_scale = _load_lstm_model(checkpoint_path)
        config = _ClassifierConfig.from_yaml(config_path)

        return cls(
            model=model,
            class_names=class_names,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            config=config,
            confidence_threshold=confidence_threshold,
        )

    def classify(
        self,
        raw_keypoints: np.ndarray,
        timestamps_s: np.ndarray,
    ) -> list[ActionSegment]:
        """Classify raw keypoints into action segments.

        Args:
            raw_keypoints: (N, 33, 4) array from MediaPipe.
            timestamps_s: (N,) array of per-frame timestamps in seconds.

        Returns:
            List of ActionSegment with class_id, class_name, t_start, t_end.
        """
        detections = self._detect_per_window(raw_keypoints, timestamps_s)
        return self._merge_detections(detections)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_per_window(
        self, raw_keypoints: np.ndarray, timestamps_s: np.ndarray
    ) -> list[dict]:
        """Run LSTM on sliding windows to get per-frame detections."""
        import torch

        N = len(raw_keypoints)
        ws = self.cfg.window_size
        stride = self.cfg.stride
        detections = []

        for start in range(0, N - ws + 1, stride):
            window = raw_keypoints[start : start + ws]
            features = self._preprocess_window(window)
            if features is None:
                continue

            x = torch.from_numpy(features).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(x)
                proba = torch.softmax(logits, dim=1)[0].numpy()

            pred_idx = int(proba.argmax())
            action = self.class_names[pred_idx]
            conf = float(proba[pred_idx])

            if action != "guard" and conf < self.confidence_threshold:
                guard_idx = self.class_names.index("guard")
                action = "guard"
                conf = float(proba[guard_idx])

            detections.append({
                "timestamp_s": float(timestamps_s[start + ws - 1]),
                "action": action,
                "confidence": conf,
            })

        return detections

    def _preprocess_window(self, window: np.ndarray) -> np.ndarray | None:
        """Preprocess a single window (ws, 33, 4) → (ws, features) scaled.

        Replicates the Phase 1 preprocess_window logic:
        select → mask → interpolate → normalize → velocity → flatten → scale.
        """
        cfg = self.cfg
        K = len(cfg.keypoint_indices)
        C = 3 if cfg.use_z else 2

        selected = window[:, cfg.keypoint_indices, :]  # (ws, K, 4)
        coords = selected[:, :, :C].copy()
        vis = selected[:, :, 3]

        # Mask low visibility
        low_vis = vis < cfg.visibility_threshold
        coords[low_vis] = np.nan

        # Interpolate NaN per keypoint per coordinate
        T = coords.shape[0]
        for k in range(K):
            for c in range(C):
                series = coords[:, k, c]
                nans = np.isnan(series)
                if nans.all() or not nans.any():
                    continue
                valid = ~nans
                coords[:, k, c] = np.interp(
                    np.arange(T), np.where(valid)[0], series[valid]
                )

        if np.isnan(coords).mean() > cfg.nan_ratio_threshold:
            return None

        # Normalize: hip center + shoulder scale
        hip_center = (
            coords[:, cfg.left_hip_idx, :] + coords[:, cfg.right_hip_idx, :]
        ) / 2.0
        coords = coords - hip_center[:, np.newaxis, :]

        shoulder_dist = np.linalg.norm(
            coords[:, cfg.left_shoulder_idx, :2]
            - coords[:, cfg.right_shoulder_idx, :2],
            axis=1,
        )
        shoulder_dist = np.maximum(shoulder_dist, 1e-6)
        coords = coords / shoulder_dist[:, np.newaxis, np.newaxis]

        if cfg.use_velocity:
            velocity = np.concatenate(
                [
                    np.zeros((1, coords.shape[1], coords.shape[2]), dtype=coords.dtype),
                    np.diff(coords, axis=0),
                ],
                axis=0,
            )
            coords = np.concatenate([coords, velocity], axis=2)

        features = coords.reshape(T, -1).astype(np.float32)
        np.nan_to_num(features, nan=0.0, copy=False)

        # StandardScaler (same as training)
        features = ((features - self.scaler_mean) / self.scaler_scale).astype(np.float32)

        return features

    def _merge_detections(self, detections: list[dict]) -> list[ActionSegment]:
        """Merge consecutive same-action detections into ActionSegments."""
        if not detections:
            return []

        action_names = ["guard"] + list(PUNCH_NAMES)
        action_to_id = {name: i for i, name in enumerate(action_names)}

        segments: list[ActionSegment] = []
        current_action = detections[0]["action"]
        run_start = detections[0]["timestamp_s"]
        run_end = run_start

        for det in detections[1:]:
            if det["action"] == current_action:
                run_end = det["timestamp_s"]
            else:
                class_id = action_to_id.get(current_action, 0)
                segments.append(ActionSegment(
                    class_id=class_id,
                    class_name=current_action,
                    t_start=run_start,
                    t_end=run_end,
                ))
                current_action = det["action"]
                run_start = det["timestamp_s"]
                run_end = run_start

        # Last segment
        class_id = action_to_id.get(current_action, 0)
        segments.append(ActionSegment(
            class_id=class_id,
            class_name=current_action,
            t_start=run_start,
            t_end=run_end,
        ))

        return segments
