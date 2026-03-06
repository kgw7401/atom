#!/usr/bin/env python3
"""Classify actions in a video file using trained LSTM model.

Usage:
    python scripts/classify_video.py path/to/video.mp4
    python scripts/classify_video.py path/to/video.mp4 --model models/lstm_v2.pt
    python scripts/classify_video.py path/to/video.mp4 --model models/lstm_v2.pt --save output.mp4
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.inference.utils import load_model, preprocess_window
from src.preprocessing.pipeline import PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="Classify actions in video")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--model", "-m", type=str, default="models/lstm_best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save output video (optional)")
    args = parser.parse_args()

    # Load model
    print(f"Loading LSTM model from {args.model}...")
    cfg = PipelineConfig.from_yaml()
    model, class_names, scaler_mean, scaler_scale = load_model(Path(args.model))
    print(f"Model ready. Classes: {class_names}\n")

    # Colors per class
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

    # Setup video capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

    # Setup video writer if saving
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"Saving output to: {args.save}")

    # Setup MediaPipe
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

    # Transition filter
    HOLD_SECONDS = 0.15  # Reduced from 0.3 for faster combo recognition
    CONFIDENCE_THRESHOLD = 0.6  # Reduced from 0.7 for more responsive detection
    last_attack_time = 0.0
    guard_idx = class_names.index("guard") if "guard" in class_names else -1
    last_was_attack = False

    frame_count = 0
    start_time = time.time()

    print("\nProcessing video...")
    print("Press 'q' to quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
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
                        cx, cy = int(lm.x * width), int(lm.y * height)
                        cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)

                # Classify when buffer is full
                stride_counter += 1
                if len(buffer) == cfg.window_size and stride_counter >= cfg.stride:
                    stride_counter = 0
                    buf_array = np.stack(list(buffer))
                    features = preprocess_window(buf_array, cfg, scaler_mean, scaler_scale)

                    if features is not None:
                        x = torch.from_numpy(features).unsqueeze(0)
                        with torch.no_grad():
                            logits = model(x)
                            proba = torch.softmax(logits, dim=1)[0].numpy()
                        pred_idx = proba.argmax()
                        new_pred = class_names[pred_idx]
                        new_conf = proba[pred_idx]
                        now = time.time()

                        # Low confidence → fall back to guard
                        if guard_idx >= 0 and new_pred != "guard" and new_conf < CONFIDENCE_THRESHOLD:
                            new_pred = "guard"
                            new_conf = proba[guard_idx]
                            pred_idx = guard_idx

                        # Transition filter
                        if guard_idx >= 0 and new_pred != "guard" and last_was_attack and now - last_attack_time < HOLD_SECONDS:
                            new_pred = prediction
                            new_conf = confidence

                        # Update state
                        if guard_idx < 0 or new_pred != "guard":
                            prediction = new_pred
                            confidence = new_conf
                            pred_color = colors[pred_idx % len(colors)]
                            last_attack_time = now
                            last_was_attack = True
                        elif guard_idx >= 0 and now - last_attack_time >= HOLD_SECONDS:
                            prediction = new_pred
                            confidence = new_conf
                            pred_color = colors[pred_idx % len(colors)]
                            last_was_attack = False

            # Display prediction
            if prediction:
                label = f"{prediction.upper()} ({confidence:.0%})"
                cv2.putText(display, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, pred_color, 3)

            # Progress
            progress = f"Frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)"
            cv2.putText(display, progress, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Show frame
            cv2.imshow("Video Classification", display)

            # Save frame
            if writer:
                writer.write(display)

            # Allow quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        landmarker.close()
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
    if args.save:
        print(f"Output saved to: {args.save}")


if __name__ == "__main__":
    main()
