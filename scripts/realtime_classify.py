#!/usr/bin/env python3
"""Real-time action classification from webcam.

Trains an RF model on existing data, then classifies live webcam feed
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
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing.pipeline import PreprocessingPipeline, PipelineConfig


def train_model(
    keypoints_dir: Path = Path("data/keypoints"),
) -> tuple[RandomForestClassifier, list[str], PreprocessingPipeline]:
    """Train RF on all available data. Returns (model, class_names, pipeline)."""
    import json

    pipeline = PreprocessingPipeline()
    all_windows = []
    all_labels = []
    class_set = set()

    for npy_path in sorted(keypoints_dir.glob("*.npy")):
        meta_path = npy_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        action = meta["action"]
        class_set.add(action)
        raw = np.load(npy_path)
        windows = pipeline.process(raw)
        if windows.size == 0:
            continue
        all_windows.append(windows)
        all_labels.extend([action] * len(windows))

    class_names = sorted(class_set)
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    X = np.concatenate(all_windows, axis=0)
    X = X.reshape(X.shape[0], -1)
    y = np.array([label_to_idx[label] for label in all_labels])

    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    return rf, class_names, pipeline


def preprocess_window(
    buffer: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray | None:
    """Preprocess a single window buffer (window_size, 33, 4) → flat feature vector.

    Returns flattened (window_size * K * 3,) array, or None if invalid.
    """
    K = len(cfg.keypoint_indices)

    # Select keypoints
    selected = buffer[:, cfg.keypoint_indices, :]  # (W, K, 4)
    coords = selected[:, :, :3].copy()
    vis = selected[:, :, 3]

    # Mask low visibility
    mask = vis < cfg.visibility_threshold
    coords[mask] = np.nan

    # Interpolate NaN
    N = coords.shape[0]
    for k in range(K):
        for c in range(3):
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
            [np.zeros((1, coords.shape[1], 3), dtype=coords.dtype),
             np.diff(coords, axis=0)],
            axis=0,
        )
        coords = np.concatenate([coords, velocity], axis=2)  # (W, K, 6)

    return coords.flatten().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Real-time action classification")
    parser.add_argument("--camera", "-c", type=int, default=0)
    args = parser.parse_args()

    # Train model
    print("Training model on existing data...")
    model, class_names, pipeline = train_model()
    cfg = pipeline.cfg
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
                    features = preprocess_window(buf_array, cfg)

                    if features is not None:
                        proba = model.predict_proba(features.reshape(1, -1))[0]
                        pred_idx = proba.argmax()
                        prediction = class_names[pred_idx]
                        confidence = proba[pred_idx]
                        pred_color = colors[pred_idx % len(colors)]

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
