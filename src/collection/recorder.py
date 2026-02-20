"""Video recorder for boxing pose data collection.

Records webcam video with live pose overlay and saves with the project
naming convention: {subject}_{action}_{angle}_{speed}.mp4

Usage::

    python -m src.collection.recorder -s charlie -a jab --angle front --speed normal
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import yaml

from src.extraction.pose_extractor import PoseExtractor


def load_config(config_path: str | Path = "configs/boxing.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_filename(subject: str, action: str, angle: str, speed: str) -> str:
    return f"{subject}_{action}_{angle}_{speed}.mp4"


def record(
    subject: str,
    action: str,
    angle: str,
    speed: str,
    output_dir: Path = Path("data/raw"),
    camera: int = 0,
    fps: float = 30.0,
    resolution: tuple[int, int] = (1280, 720),
):
    """Interactive recording session.

    Controls:
        r  — start / stop recording
        n  — next take (discard current & re-record same combo)
        q  — quit
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = make_filename(subject, action, angle, speed)
    filepath = output_dir / filename

    if filepath.exists():
        print(f"WARNING: {filename} already exists. Press 'r' to overwrite or 'q' to quit.")

    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise ValueError(f"Cannot open camera {camera}")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    # Use IMAGE mode extractor for live preview (no timestamp dependency)
    import mediapipe as mp

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(Path("models/pose_landmarker.task").resolve())
    )
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    recording = False
    writer = None
    frame_count = 0
    rec_start = 0.0

    label = f"{subject} | {action} | {angle} | {speed}"
    print(f"Session: {label}")
    print(f"Camera: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
    print(f"Output: {filename}")
    print("Controls: [r] record/stop  [q] quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()

            # Pose detection for preview
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect(mp_image)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                h, w = display.shape[:2]
                for lm in lms:
                    if lm.visibility and lm.visibility > 0.5:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)

            # Recording indicator
            if recording:
                elapsed = time.time() - rec_start
                cv2.circle(display, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    f"REC {elapsed:.1f}s  |  {frame_count} frames",
                    (50, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                writer.write(frame)
                frame_count += 1
            else:
                cv2.putText(
                    display,
                    f"READY  |  {filename}",
                    (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            # Labels at bottom
            cv2.putText(
                display,
                f"{action.upper()}  |  {angle}  |  {speed}",
                (10, actual_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Atom — Recorder", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                if not recording:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(filepath), fourcc, actual_fps, (actual_w, actual_h)
                    )
                    recording = True
                    frame_count = 0
                    rec_start = time.time()
                    print(f"  Recording: {filename}")
                else:
                    recording = False
                    writer.release()
                    writer = None
                    elapsed = time.time() - rec_start
                    print(f"  Saved: {filename} — {frame_count} frames ({elapsed:.1f}s)")

            elif key == ord("q"):
                break

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


def main():
    import argparse

    config = load_config()
    actions = config["actions"]
    angles = config["collection"]["angles"]
    speeds = config["collection"]["speeds"]

    parser = argparse.ArgumentParser(description="Record boxing pose videos")
    parser.add_argument("--subject", "-s", required=True, help="Subject name (e.g. charlie)")
    parser.add_argument("--action", "-a", required=True, choices=actions)
    parser.add_argument("--angle", required=True, choices=angles, help="Camera angle")
    parser.add_argument("--speed", required=True, choices=speeds, help="Movement speed")
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--output", "-o", type=str, default="data/raw")
    args = parser.parse_args()

    record(
        subject=args.subject,
        action=args.action,
        angle=args.angle,
        speed=args.speed,
        output_dir=Path(args.output),
        camera=args.camera,
    )


if __name__ == "__main__":
    main()
