#!/usr/bin/env python3
"""Real-time classification with debug info showing all class probabilities."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--model", "-m", type=str, default="models/lstm_best.pt",
                        help="Path to model checkpoint (default: models/lstm_best.pt)")
    args = parser.parse_args()

    # Load model
    print(f"Loading LSTM model from {args.model}...")
    cfg = PipelineConfig.from_yaml()
    model, class_names, scaler_mean, scaler_scale = load_model(Path(args.model))
    print(f"Model ready. Classes: {class_names}")

    # Setup webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise ValueError(f"Cannot open camera {args.camera}")

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

    print("\n=== Starting real-time classification ===")
    print("Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract keypoints
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            kp = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
            buffer.append(kp)

            # Predict when buffer is full
            if len(buffer) == cfg.window_size:
                window = np.array(buffer)
                X = preprocess_window(window, cfg, scaler_mean, scaler_scale)

                if X is None:
                    cv2.putText(frame, "Invalid keypoints", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imshow("Real-time Classification (Debug)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                with torch.no_grad():
                    logits = model(torch.from_numpy(X).unsqueeze(0))
                    probs = torch.softmax(logits, dim=1)[0]

                # Display all probabilities
                print("\n" + "="*60)
                sorted_indices = torch.argsort(probs, descending=True)
                for idx in sorted_indices:
                    prob = probs[idx].item()
                    class_name = class_names[idx]
                    bar = "█" * int(prob * 40)
                    print(f"{class_name:15s} {prob:6.2%} {bar}")

                # Draw top prediction on frame
                top_class = class_names[sorted_indices[0]]
                top_prob = probs[sorted_indices[0]].item()

                cv2.putText(frame, f"{top_class}: {top_prob:.1%}",
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                           1.5, (0, 255, 0), 3)

        cv2.imshow("Real-time Classification (Debug)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
