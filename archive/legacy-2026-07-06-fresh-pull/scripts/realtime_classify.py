#!/usr/bin/env python3
"""Real-time boxing action classification from webcam using LSTM.

Loads a pretrained LSTM model and classifies live webcam feed
using a sliding window of keypoints.

Usage:
    python scripts/realtime_classify.py
    python scripts/realtime_classify.py --camera 1
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
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


def main():
    parser = argparse.ArgumentParser(description="Real-time action classification")
    parser.add_argument("--camera", "-c", type=int, default=0)
    args = parser.parse_args()

    # Load model
    print("Loading LSTM model...")
    cfg = PipelineConfig.from_yaml()
    model, class_names, scaler_mean, scaler_scale = load_model()
    print(f"Model ready. Classes: {class_names}")

    # Colors per class for display
    colors = [
        (0, 255, 255),   # yellow
        (0, 255, 0),     # green
        (255, 100, 0),   # blue
        (0, 100, 255),   # orange
        (255, 0, 255),   # magenta
        (255, 255, 0),   # cyan
        (0, 0, 255),     # red
        (255, 0, 0),     # blue
        (128, 255, 128), # light green
    ]

    # Setup webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise ValueError(f"Cannot open camera {args.camera}")

    # Setup MediaPipe (IMAGE mode for live)
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(Path("models/pose_landmarker.task").resolve())
    )
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    # Rolling buffer
    buffer = deque(maxlen=cfg.window_size)
    prediction = ""
    confidence = 0.0
    pred_color = (255, 255, 255)
    stride_counter = 0

    # Transition filter: after an attack, must return to guard before next attack.
    # Recovery motion from a punch looks like a different punch to the model,
    # but in boxing you always return to guard between actions.
    HOLD_SECONDS = 0.3
    CONFIDENCE_THRESHOLD = 0.7  # below this → fall back to guard
    last_attack_time = 0.0
    guard_idx = class_names.index("guard")
    last_was_attack = False  # True after a non-guard prediction

    print(f"Window: {cfg.window_size} frames, stride: {cfg.stride}")
    print("Press 'q' to quit")

    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            h, w = frame.shape[:2]
            display = frame.copy()

            # Extract pose
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect(mp_image)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                keypoints = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
                    dtype=np.float32,
                )
                buffer.append(keypoints)

                # Draw skeleton
                for lm in lms:
                    if lm.visibility and lm.visibility > 0.5:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)

                # Classify when buffer is full, every stride frames
                stride_counter += 1
                if len(buffer) == cfg.window_size and stride_counter >= cfg.stride:
                    stride_counter = 0
                    buf_array = np.stack(list(buffer))  # (window_size, 33, 4)
                    features = preprocess_window(buf_array, cfg, scaler_mean, scaler_scale)

                    if features is not None:
                        x = torch.from_numpy(features).unsqueeze(0)  # (1, T, F)
                        with torch.no_grad():
                            logits = model(x)
                            proba = torch.softmax(logits, dim=1)[0].numpy()
                        pred_idx = proba.argmax()
                        new_pred = class_names[pred_idx]
                        new_conf = proba[pred_idx]
                        now = time.time()

                        # Low confidence on transition windows → fall back to guard
                        if new_pred != "guard" and new_conf < CONFIDENCE_THRESHOLD:
                            new_pred = "guard"
                            new_conf = proba[guard_idx]
                            pred_idx = guard_idx

                        # Transition filter: attack→attack is invalid in boxing.
                        # After an attack, you must return to guard first.
                        # This blocks the recovery motion being misread as
                        # a different punch (e.g. hook recovery → "cross").
                        if new_pred != "guard" and last_was_attack and now - last_attack_time < HOLD_SECONDS:
                            new_pred = prediction  # keep previous attack during hold
                            new_conf = confidence

                        # Update state
                        if new_pred != "guard":
                            prediction = new_pred
                            confidence = new_conf
                            pred_color = colors[pred_idx % len(colors)]
                            last_attack_time = now
                            last_was_attack = True
                        elif now - last_attack_time >= HOLD_SECONDS:
                            prediction = new_pred
                            confidence = new_conf
                            pred_color = colors[pred_idx % len(colors)]
                            last_was_attack = False

            # Display prediction
            if prediction:
                label = f"{prediction.upper()} ({confidence:.0%})"
                cv2.putText(display, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, pred_color, 3)

            # FPS
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                display,
                f"FPS: {fps:.0f}  |  buf: {len(buffer)}/{cfg.window_size}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Atom — Real-time Classifier", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
