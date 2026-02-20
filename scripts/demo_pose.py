#!/usr/bin/env python3
"""Demo: real-time pose estimation from webcam or video file.

Usage:
    python scripts/demo_pose.py              # webcam
    python scripts/demo_pose.py video.mp4    # video file
    python scripts/demo_pose.py 0 --save out.npy   # save keypoints
"""

import argparse
import time

import cv2
import numpy as np

from src.extraction.pose_extractor import PoseExtractor


def main():
    parser = argparse.ArgumentParser(description="Real-time pose estimation demo")
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="Video file path or camera index (default: 0 for webcam)",
    )
    parser.add_argument("--save", type=str, help="Save keypoints to .npy file")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    all_keypoints = []
    frame_count = 0
    fps_start = time.time()

    print(f"Opening source: {source}")
    print("Press 'q' to quit")

    with PoseExtractor() as extractor:
        for pose_frame in extractor.process_video(source, keep_images=True):
            frame_count += 1
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            annotated = extractor.draw_landmarks(pose_frame.image, pose_frame.keypoints)

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}  |  Frames: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Atom — Pose Estimation", annotated)

            all_keypoints.append(pose_frame.keypoints)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

    avg_fps = frame_count / (time.time() - fps_start) if frame_count else 0
    print(f"Processed {frame_count} frames | avg {avg_fps:.1f} FPS")

    if args.save and all_keypoints:
        kp_array = np.stack(all_keypoints)
        np.save(args.save, kp_array)
        print(f"Saved {kp_array.shape} to {args.save}")


if __name__ == "__main__":
    main()
