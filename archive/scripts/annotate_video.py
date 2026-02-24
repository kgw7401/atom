#!/usr/bin/env python3
"""Annotate action segments in existing videos.

Plays a video with pose overlay and timeline. Mark start/end of each
action segment, and save annotated keypoints ready for training.

Usage:
    python scripts/annotate_video.py path/to/video.mp4
    python scripts/annotate_video.py path/to/video.mp4 --subject youtube1

Controls:
    SPACE       pause / resume
    a / d       previous / next frame (when paused)
    A / D       jump -30 / +30 frames (when paused)
    1-7         select action class
    r           mark start of segment
    s           mark end + save segment
    u           undo last saved segment
    q           quit and show summary
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml


def load_config(config_path: str = "configs/boxing.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_all_keypoints(
    video_path: str,
    landmarker: mp.tasks.vision.PoseLandmarker,
) -> tuple[np.ndarray, int, float]:
    """First pass: extract keypoints from every frame.

    Returns:
        all_keypoints: (N, 33, 4) array
        total_frames: int
        fps: float
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_kps: list[np.ndarray] = []
    frame_idx = 0

    print(f"Extracting keypoints from {total} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            kps = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
                dtype=np.float32,
            )
        else:
            kps = np.full((33, 4), np.nan, dtype=np.float32)

        all_kps.append(kps)
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"  {frame_idx}/{total} ({pct:.0f}%)")

    cap.release()
    print(f"Done. {frame_idx} frames extracted.\n")

    return np.stack(all_kps), frame_idx, fps


def draw_timeline(
    display: np.ndarray,
    current_frame: int,
    total_frames: int,
    segment_start: int | None,
    saved_segments: list[tuple[int, int, str]],
    actions: list[str],
):
    """Draw a timeline bar with saved segments and current position."""
    h, w = display.shape[:2]
    y = h - 35
    x1, x2 = 10, w - 10
    tw = x2 - x1

    # Background bar
    cv2.rectangle(display, (x1, y - 4), (x2, y + 4), (60, 60, 60), -1)

    # Saved segment blocks
    colors = [
        (0, 255, 255), (0, 255, 0), (255, 100, 0), (0, 100, 255),
        (255, 0, 255), (255, 255, 0), (0, 0, 255),
    ]
    for start, end, action in saved_segments:
        sx = x1 + int(start / total_frames * tw)
        ex = x1 + int(end / total_frames * tw)
        aidx = actions.index(action) if action in actions else 0
        cv2.rectangle(display, (sx, y - 6), (max(ex, sx + 2), y + 6),
                       colors[aidx % len(colors)], -1)

    # Current marking range
    if segment_start is not None:
        sx = x1 + int(segment_start / total_frames * tw)
        pos_x = x1 + int(current_frame / total_frames * tw)
        cv2.rectangle(display, (sx, y - 8), (pos_x, y + 8), (0, 0, 255), 2)

    # Playhead
    pos_x = x1 + int(current_frame / total_frames * tw)
    cv2.line(display, (pos_x, y - 12), (pos_x, y + 12), (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description="Annotate action segments in video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--subject", "-s", default="video",
                        help="Subject name for saved files (default: video)")
    parser.add_argument("--config", default="configs/boxing.yaml")
    parser.add_argument("--output", "-o", default="data/keypoints",
                        help="Output directory for keypoints")
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"Error: {video_path} not found")
        return

    config = load_config(args.config)
    actions = config["actions"]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- First pass: extract keypoints ---
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(Path("models/pose_landmarker.task").resolve())
    )
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    all_keypoints, total_frames, fps = extract_all_keypoints(video_path, landmarker)
    landmarker.close()

    if total_frames == 0:
        print("Error: no frames extracted")
        return

    # --- Second pass: interactive annotation ---
    cap = cv2.VideoCapture(video_path)

    paused = True  # start paused so user can orient
    current_frame = 0
    selected_action = 0
    segment_start: int | None = None
    saved_segments: list[tuple[int, int, str]] = []

    # Scan existing files to avoid overwriting previous annotations
    segment_counter: dict[str, int] = {}
    for action in actions:
        existing = sorted(out_dir.glob(f"{args.subject}_{action}_seg*.npy"))
        if existing:
            # Parse highest seg number and start after it
            last = existing[-1].stem  # e.g. "youtube1_jab_seg004"
            num = int(last.rsplit("seg", 1)[1])
            segment_counter[action] = num + 1
        else:
            segment_counter[action] = 0

    existing_total = sum(segment_counter.values())
    if existing_total > 0:
        print(f"Found existing annotations — counters: {segment_counter}")

    frame_delay = max(1, int(1000 / fps))

    print(f"Video: {video_path}")
    print(f"Frames: {total_frames} | FPS: {fps:.1f} | Duration: {total_frames/fps:.1f}s")
    print(f"\nActions:")
    for i, action in enumerate(actions):
        print(f"  [{i + 1}] {action}")
    print(f"\nControls:")
    print(f"  SPACE = pause/resume")
    print(f"  a/d = step frame (A/D = jump 30 frames)")
    print(f"  1-{len(actions)} = select action")
    print(f"  r = mark start | s = save segment | u = undo")
    print(f"  q = quit\n")

    try:
        while True:
            # Read frame at current position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            h, w = display.shape[:2]

            # Draw pose skeleton
            kps = all_keypoints[current_frame]
            if not np.isnan(kps).all():
                for x, y_coord, _z, vis in kps:
                    if not np.isnan(vis) and vis > 0.5:
                        cx, cy = int(x * w), int(y_coord * h)
                        cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)

            # Timeline
            draw_timeline(display, current_frame, total_frames,
                          segment_start, saved_segments, actions)

            # Info text
            action_name = actions[selected_action]
            status = "PAUSED" if paused else "PLAYING"
            time_now = current_frame / fps
            time_total = total_frames / fps

            cv2.putText(display, f"[{selected_action + 1}] {action_name.upper()}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display,
                        f"{status}  |  {current_frame}/{total_frames}  |  "
                        f"{time_now:.1f}s / {time_total:.1f}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if segment_start is not None:
                seg_len = current_frame - segment_start
                seg_sec = seg_len / fps
                cv2.putText(display,
                            f"MARKING: {action_name} | start={segment_start} | "
                            f"{seg_len} frames ({seg_sec:.1f}s)",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            saved_text = f"Saved: {len(saved_segments)} segments"
            cv2.putText(display, saved_text,
                        (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Atom — Annotator", display)

            # Input handling
            wait_time = 0 if paused else frame_delay
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" "):
                paused = not paused

            elif key == ord("a"):
                # Step backward
                paused = True
                current_frame = max(0, current_frame - 1)

            elif key == ord("d"):
                # Step forward
                paused = True
                current_frame = min(total_frames - 1, current_frame + 1)

            elif key == ord("A"):
                # Jump backward 30 frames
                paused = True
                current_frame = max(0, current_frame - 30)

            elif key == ord("D"):
                # Jump forward 30 frames
                paused = True
                current_frame = min(total_frames - 1, current_frame + 30)

            elif key == ord("r"):
                segment_start = current_frame
                print(f"  START: frame {current_frame} [{action_name}]")

            elif key == ord("s"):
                if segment_start is not None and current_frame > segment_start:
                    segment_end = current_frame

                    # Slice keypoints for this segment
                    seg_kps = all_keypoints[segment_start:segment_end]

                    # Generate unique filename
                    count = segment_counter.get(action_name, 0)
                    segment_counter[action_name] = count + 1
                    stem = f"{args.subject}_{action_name}_seg{count:03d}"

                    npy_path = out_dir / f"{stem}.npy"
                    meta_path = out_dir / f"{stem}.json"

                    np.save(npy_path, seg_kps)

                    meta = {
                        "subject": args.subject,
                        "action": action_name,
                        "source_video": Path(video_path).name,
                        "start_frame": int(segment_start),
                        "end_frame": int(segment_end),
                        "num_frames": int(segment_end - segment_start),
                        "shape": list(seg_kps.shape),
                        "fps": fps,
                    }
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)

                    saved_segments.append((segment_start, segment_end, action_name))
                    dur = (segment_end - segment_start) / fps
                    print(f"  SAVED: {stem} — {seg_kps.shape[0]} frames ({dur:.1f}s)")

                    segment_start = None
                else:
                    print("  No valid segment. Press 'r' to mark start first.")

            elif key == ord("u"):
                if saved_segments:
                    start, end, act = saved_segments.pop()
                    segment_counter[act] -= 1
                    count = segment_counter[act]
                    stem = f"{args.subject}_{act}_seg{count:03d}"

                    npy_path = out_dir / f"{stem}.npy"
                    meta_path = out_dir / f"{stem}.json"
                    if npy_path.exists():
                        npy_path.unlink()
                    if meta_path.exists():
                        meta_path.unlink()
                    print(f"  UNDO: removed {stem}")
                else:
                    print("  Nothing to undo.")

            else:
                # Number keys 1-9 for action selection
                for i in range(min(len(actions), 9)):
                    if key == ord(str(i + 1)):
                        selected_action = i
                        action_name = actions[i]
                        print(f"  Action: [{i + 1}] {action_name}")
                        break

            # Advance frame when playing
            if not paused and key == 255:  # no key pressed
                current_frame += 1
                if current_frame >= total_frames:
                    current_frame = total_frames - 1
                    paused = True
                    print("  End of video. Paused.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Summary: {len(saved_segments)} segments saved to {out_dir}/")
    print(f"{'=' * 50}")
    for start, end, action in saved_segments:
        dur = (end - start) / fps
        print(f"  {action:15s}  frames {start:5d}-{end:5d}  ({dur:.1f}s)")

    if saved_segments:
        print(f"\nNext step: python scripts/train_baseline.py")


if __name__ == "__main__":
    main()
