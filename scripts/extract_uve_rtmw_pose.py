#!/usr/bin/env python3
"""Extract GT-free boxer poses with BoT-SORT, RTMW, and periodic UVE identity checks."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from atom.rtmw_pose import (  # noqa: E402
    interpolate_all_gaps,
    interpolate_short_gaps,
    rtmw_to_boxingweb,
    smooth_sequence,
)
from atom.uve_identity import appearance_descriptor, classify_appearances, load_uve_classifier  # noqa: E402
from atom.uve_tracks import canonical_pose_payload  # noqa: E402
from extract_rtmw3d_pose import batched_pose_inference, trusted_checkpoint_compatibility  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracks-output", type=Path, default=None)
    parser.add_argument("--identity-checkpoint", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--pose-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--confidence", type=float, default=.2)
    parser.add_argument("--cadence", type=int, default=10)
    parser.add_argument("--identity-threshold", type=float, default=.35)
    parser.add_argument("--continuity-bonus", type=float, default=.25)
    parser.add_argument("--spatial-bonus", type=float, default=1.5)
    parser.add_argument("--allow-missing-boxer", action="store_true",
                        help="Allow a red or blue assignment to be empty even when two candidates exist.")
    parser.add_argument("--max-detections", type=int, default=6)
    parser.add_argument("--max-pose-candidates", type=int, default=3)
    parser.add_argument("--batch-frames", type=int, default=32)
    parser.add_argument("--max-gap", type=int, default=15)
    parser.add_argument("--interpolate-all-missing", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive; defaults to the whole video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cadence < 1 or args.batch_frames < 1 or args.max_pose_candidates < 2:
        raise ValueError("cadence and batch sizes must be positive; max-pose-candidates must be at least 2")
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise ValueError("smooth-window must be a positive odd integer")
    trusted_checkpoint_compatibility()
    try:
        from ultralytics import YOLO
        sys.path.insert(0, str(args.pose_config.resolve().parents[1]))
        from mmengine.dataset import Compose
        from mmpose.apis import init_model
    except ModuleNotFoundError as error:
        raise RuntimeError("The UVE extractor requires Ultralytics and MMPose.") from error

    detector = YOLO(str(args.yolo))
    identity_model, identity_mean, identity_std = load_uve_classifier(args.identity_checkpoint, "cpu")
    pose_model = init_model(str(args.pose_config), str(args.pose_checkpoint), device=args.device)
    if hasattr(pose_model, "test_cfg"):
        pose_model.test_cfg["flip_test"] = False
    pose_model.cfg.model.test_cfg["flip_test"] = False
    pose_pipeline = Compose(pose_model.cfg.test_dataloader.dataset.pipeline)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open {args.video}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    end_frame = min(args.end_frame if args.end_frame is not None else total_frames, total_frames)
    if args.start_frame < 0 or end_frame <= args.start_frame:
        raise ValueError("Expected 0 <= start-frame < end-frame")
    frame_count = end_frame - args.start_frame
    slots = args.max_pose_candidates
    track_ids = np.full((frame_count, slots), -1, dtype=np.int64)
    joints_2d = np.zeros((frame_count, slots, 45, 2), dtype=np.float32)
    joints_3d = np.zeros((frame_count, slots, 45, 3), dtype=np.float32)
    probabilities = np.zeros((frame_count, slots, 3), dtype=np.float32)
    probabilities[..., 2] = 1.0
    boxes_output = np.zeros((frame_count, slots, 4), dtype=np.float32)
    last_probability: dict[int, np.ndarray] = {}
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    processed = 0
    started = time.time()
    try:
        for batch_start in range(args.start_frame, end_frame, args.batch_frames):
            frames = []
            for source_frame in range(batch_start, min(batch_start + args.batch_frames, end_frame)):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Unable to decode frame {source_frame}")
                frames.append(frame)
            pose_requests: list[tuple[np.ndarray, np.ndarray]] = []
            pose_metadata: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
            for offset, frame in enumerate(frames):
                source_frame = batch_start + offset
                local_frame = source_frame - args.start_frame
                result = detector.track(
                    frame, persist=True, tracker=args.tracker, classes=[0], conf=args.confidence,
                    device=args.device, verbose=False,
                )[0]
                boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                ids_tensor = result.boxes.id
                if not len(boxes) or ids_tensor is None:
                    continue
                ids = ids_tensor.int().cpu().numpy()
                areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
                keep = np.argsort(areas)[::-1][:args.max_detections]
                boxes, ids, areas = boxes[keep], ids[keep], areas[keep]
                refresh = local_frame % args.cadence == 0
                classify_indices = [index for index, track_id in enumerate(ids) if refresh or int(track_id) not in last_probability]
                if classify_indices:
                    descriptors = np.stack([appearance_descriptor(frame, boxes[index]) for index in classify_indices])
                    predicted = classify_appearances(identity_model, descriptors, identity_mean, identity_std)
                    for index, probability in zip(classify_indices, predicted):
                        last_probability[int(ids[index])] = probability
                candidate_probabilities = np.stack([
                    last_probability.get(int(track_id), np.array([.05, .05, .90], np.float32)) for track_id in ids
                ])
                area_fraction = areas / max(width * height, 1)
                boxer_score = 1.0 - candidate_probabilities[:, 2] + np.minimum(area_fraction * 4.0, .25)
                selected = np.argsort(boxer_score)[::-1][:slots]
                for slot, index in enumerate(selected):
                    pose_requests.append((frame, boxes[index]))
                    pose_metadata.append((local_frame, slot, int(ids[index]), boxes[index], candidate_probabilities[index]))
            if pose_requests:
                poses = batched_pose_inference(pose_model, pose_pipeline, pose_requests)
                for metadata, sample in zip(pose_metadata, poses):
                    local_frame, slot, track_id, box, probability = metadata
                    instances = sample.pred_instances
                    predicted = np.asarray(instances.keypoints)[0]
                    if "transformed_keypoints" in instances:
                        keypoints_2d = np.asarray(instances.transformed_keypoints)[0]
                        keypoints_3d = predicted
                    else:
                        keypoints_2d = predicted
                        keypoints_3d = np.zeros((133, 3), dtype=np.float32)
                    joints_2d[local_frame, slot], joints_3d[local_frame, slot] = rtmw_to_boxingweb(
                        keypoints_2d, keypoints_3d, width, height,
                    )
                    track_ids[local_frame, slot] = track_id
                    probabilities[local_frame, slot] = probability
                    boxes_output[local_frame, slot] = box
            processed += len(frames)
            if processed % 100 < len(frames) or processed == frame_count:
                rate = processed / max(time.time() - started, 1e-6)
                remaining = (frame_count - processed) / max(rate, 1e-6)
                print(f"frames={processed}/{frame_count} rate={rate:.1f}/s eta={remaining:.0f}s", flush=True)
    finally:
        capture.release()

    tracks = {
        "track_ids": track_ids,
        "joints_2d": joints_2d,
        "joints_3d": joints_3d,
        "identity_probabilities": probabilities,
        "boxes": boxes_output,
    }
    if args.tracks_output:
        args.tracks_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.tracks_output, **tracks)
    payload = canonical_pose_payload(
        tracks, cadence=args.cadence, identity_threshold=args.identity_threshold,
        continuity_bonus=args.continuity_bonus,
        require_two_boxers=not args.allow_missing_boxer,
        spatial_bonus=args.spatial_bonus,
    )
    for side in ("red", "blue"):
        valid = np.asarray(payload[f"{side}_track_ids"]) >= 0
        interpolate = interpolate_all_gaps if args.interpolate_all_missing else (
            lambda values, mask: interpolate_short_gaps(values, mask, args.max_gap)
        )
        payload[f"pose_{side}_2d"] = smooth_sequence(interpolate(payload[f"pose_{side}_2d"], valid), args.smooth_window)
        payload[f"pose_{side}_3d"] = smooth_sequence(interpolate(payload[f"pose_{side}_3d"], valid), args.smooth_window)
        payload[f"valid_{side}"] = valid
    payload.update({
        "source": "YOLO BoT-SORT + RTMW + periodic RGB UVE identity classifier; no inference-time GT",
        "source_video": str(args.video.resolve()),
        "source_start_frame": args.start_frame,
        "source_end_frame_exclusive": end_frame,
        "fps": fps,
        "width": width,
        "height": height,
        "tracker": args.tracker,
        "identity_checkpoint": str(args.identity_checkpoint.resolve()),
        "interpolate_all_missing": args.interpolate_all_missing,
        "smooth_window": args.smooth_window,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(payload, file)
    print(
        f"pose={args.output} red_coverage={payload['valid_red'].mean():.3f} "
        f"blue_coverage={payload['valid_blue'].mean():.3f}", flush=True,
    )


if __name__ == "__main__":
    main()
