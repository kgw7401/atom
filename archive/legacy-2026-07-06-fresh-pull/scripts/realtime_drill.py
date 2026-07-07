#!/usr/bin/env python3
"""Real-time boxing drill trainer.

Instructs combos via TTS + visual overlay, detects execution with LSTM,
and scores accuracy, timing, and guard discipline.

Usage:
    python scripts/realtime_drill.py
    python scripts/realtime_drill.py --level 2 --count 15
    python scripts/realtime_drill.py --camera 1
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from scripts.realtime_classify import load_model, preprocess_window
from src.preprocessing.pipeline import PipelineConfig
from src.trainer.drill_engine import DrillEngine, DrillState


# --- Visual overlay ---

def render_drill_overlay(frame, overlay: dict, h: int, w: int) -> None:
    """Draw drill state on the frame."""
    state = overlay["state"]
    completed, total = overlay["session_progress"]

    # Session progress (top-right)
    if overlay["session_active"]:
        progress = f"Drill {completed + 1}/{total}"
        cv2.putText(frame, progress, (w - 220, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    if state == DrillState.IDLE:
        countdown = overlay.get("countdown")
        if countdown is not None and countdown > 0:
            text = f"Next in {countdown:.0f}..."
            sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            x = (w - sz[0]) // 2
            cv2.putText(frame, text, (x, h // 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)

    elif state in (DrillState.ANNOUNCING, DrillState.WAITING,
                   DrillState.EXECUTING, DrillState.GUARD_WATCH):
        instruction = overlay.get("instruction", "")
        if instruction:
            # Large centered instruction
            sz = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            x = (w - sz[0]) // 2
            color = (0, 255, 255)  # yellow
            cv2.putText(frame, instruction, (x, h // 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

            # Combo progress dots
            done, total_actions = overlay.get("combo_progress", (0, 0))
            if total_actions > 0:
                dot_y = h // 5 + 40
                dot_start = w // 2 - (total_actions * 30) // 2
                for i in range(total_actions):
                    cx = dot_start + i * 30
                    if i < done:
                        cv2.circle(frame, (cx, dot_y), 8, (0, 255, 0), -1)
                    else:
                        cv2.circle(frame, (cx, dot_y), 8, (100, 100, 100), 2)

        # State hint
        if state == DrillState.WAITING:
            cv2.putText(frame, "GO!", (w // 2 - 30, h // 5 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 200), 2)

    elif state == DrillState.EVALUATING:
        result = overlay.get("result")
        if result:
            # Success/fail
            if result["success"]:
                indicator, color = "OK", (0, 255, 0)
            else:
                indicator, color = "X", (0, 0, 255)
            sz = cv2.getTextSize(indicator, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)[0]
            x = (w - sz[0]) // 2
            cv2.putText(frame, indicator, (x, h // 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)

            # Feedback text
            feedback = result["feedback_text"]
            sz = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            x = (w - sz[0]) // 2
            cv2.putText(frame, feedback, (x, h // 5 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Score
            score_text = f"Score: {result['score']}"
            cv2.putText(frame, score_text, (w - 200, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# --- Main loop ---

def main():
    parser = argparse.ArgumentParser(description="Boxing drill trainer")
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--level", "-l", type=int, default=1,
                        help="Combo level (1=singles, 2=pairs, 3=advanced)")
    parser.add_argument("--count", "-n", type=int, default=10,
                        help="Number of drills per session")
    args = parser.parse_args()

    # Load model
    print("Loading LSTM model...")
    cfg = PipelineConfig.from_yaml()
    model, class_names, scaler_mean, scaler_scale = load_model()
    print(f"Model ready. Classes: {class_names}")

    # Init drill engine
    engine = DrillEngine()
    engine.start_session(level=args.level, count=args.count)

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
    prediction = "guard"
    confidence = 0.0
    stride_counter = 0

    # Confidence threshold for classification
    CONFIDENCE_THRESHOLD = 0.7
    guard_idx = class_names.index("guard")

    print(f"\nLevel {args.level} | {args.count} drills")
    print("Press 'q' to quit\n")

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

                # Classify
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

                        # Low confidence → guard fallback
                        if new_pred != "guard" and new_conf < CONFIDENCE_THRESHOLD:
                            new_pred = "guard"
                            new_conf = proba[guard_idx]

                        prediction = new_pred
                        confidence = new_conf

            # Feed prediction to drill engine
            now = time.time()
            engine.update(prediction, confidence, now)

            # Get overlay info and render
            overlay = engine.get_overlay_info()
            render_drill_overlay(display, overlay, h, w)

            # Show current prediction (smaller, bottom-left)
            label = f"{prediction.upper()} ({confidence:.0%})"
            cv2.putText(display, label, (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            # FPS
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                display,
                f"FPS: {fps:.0f}  |  buf: {len(buffer)}/{cfg.window_size}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (150, 150, 150),
                1,
            )

            cv2.imshow("Atom — Boxing Drill Trainer", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # End session check
            if not engine.session_active:
                # Show final frame for a moment
                cv2.putText(display, "SESSION COMPLETE", (w // 2 - 200, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow("Atom — Boxing Drill Trainer", display)
                cv2.waitKey(3000)
                break

    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
