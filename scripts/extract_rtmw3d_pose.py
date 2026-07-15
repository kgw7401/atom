#!/usr/bin/env python3
"""Extract video-estimated red/blue poses with YOLO + RTMW/RTMW3D.

This diagnostic front end uses a BoxingWeb GT-pose file only to associate
person detections with red and blue boxer identities. All output joint
coordinates are inferred from RGB video by RTMW3D.
"""

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

from atom.rtmw_pose import (  # noqa: E402
    bbox_iou,
    interpolate_all_gaps,
    interpolate_short_gaps,
    match_oracle_boxers,
    pose_bbox,
    rtmw_to_boxingweb,
    smooth_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--oracle-identity-pose", type=Path, required=True,
                        help="GT pose used only for red/blue detection association, never as output pose.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--pose-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--minimum-iou", type=float, default=0.05)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--interpolate-all-missing", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--inference-stride", type=int, default=1,
                        help="Run RGB models every N frames and interpolate skipped frames.")
    parser.add_argument("--flip-test", action="store_true",
                        help="Average original/flipped pose inference; slower and disabled by default for video.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive; defaults to the whole video.")
    return parser.parse_args()


def trusted_checkpoint_compatibility() -> None:
    """Restore pre-2.6 torch.load behavior for the official OpenMMLab file."""

    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_load


def batched_pose_inference(model, pipeline, requests: list[tuple[np.ndarray, np.ndarray]]):
    """Infer poses from multiple frames while casting generated labels for MPS."""

    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope

    scope = model.cfg.get("default_scope", "mmpose")
    if scope is not None:
        init_default_scope(scope)
    items = []
    for frame, bbox in requests:
        data = {"img": frame, "bbox": bbox[None], "bbox_score": np.ones(1, dtype=np.float32)}
        data.update(model.dataset_meta)
        item = pipeline(data)
        labels = item["data_samples"].gt_instance_labels
        for key, value in list(labels.items()):
            if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                labels.set_field(value.float(), key)
        items.append(item)
    with torch.no_grad():
        return model.test_step(pseudo_collate(items))


def main() -> None:
    args = parse_args()
    if args.start_frame < 0 or args.end_frame is not None and args.end_frame <= args.start_frame:
        raise ValueError("Expected 0 <= start-frame < end-frame")
    if args.batch_frames < 1:
        raise ValueError("batch-frames must be positive")
    if args.inference_stride < 1:
        raise ValueError("inference-stride must be positive")
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise ValueError("smooth-window must be a positive odd integer")
    trusted_checkpoint_compatibility()
    try:
        from ultralytics import YOLO
        # The official config imports the sibling ``rtmpose3d`` package.
        sys.path.insert(0, str(args.pose_config.resolve().parents[1]))
        from mmengine.dataset import Compose
        from mmpose.apis import init_model
    except ModuleNotFoundError as error:
        raise RuntimeError("Run this script in an environment containing ultralytics, MMPose, and RTMW3D.") from error

    with args.oracle_identity_pose.open("rb") as file:
        oracle = pickle.load(file)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {args.video}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    end_frame = min(args.end_frame if args.end_frame is not None else total_frames, total_frames)
    frame_count = end_frame - args.start_frame
    oracle_frames = min(len(np.asarray(oracle[key])) for key in ("pose_red_2d", "pose_blue_2d"))
    if oracle_frames == 0:
        raise ValueError("Oracle identity pose is empty.")

    detector = YOLO(str(args.yolo))
    pose_model = init_model(str(args.pose_config), str(args.pose_checkpoint), device=args.device)
    if hasattr(pose_model, "test_cfg"):
        pose_model.test_cfg["flip_test"] = args.flip_test
    pose_model.cfg.model.test_cfg["flip_test"] = args.flip_test
    pose_pipeline = Compose(pose_model.cfg.test_dataloader.dataset.pipeline)
    output_2d = {side: np.zeros((frame_count, 45, 2), dtype=np.float32) for side in ("red", "blue")}
    output_3d = {side: np.zeros((frame_count, 45, 3), dtype=np.float32) for side in ("red", "blue")}
    valid = {side: np.zeros(frame_count, dtype=bool) for side in ("red", "blue")}
    detection_iou = {side: np.zeros(frame_count, dtype=np.float32) for side in ("red", "blue")}
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    started = time.time()
    processed = 0
    try:
        for batch_start in range(args.start_frame, end_frame, args.batch_frames):
            batch_stop = min(batch_start + args.batch_frames, end_frame)
            frames = []
            for source_frame in range(batch_start, batch_stop):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Unable to decode frame {source_frame}")
                frames.append(frame)
            inference_items = [
                (offset, frame) for offset, frame in enumerate(frames)
                if (batch_start + offset - args.start_frame) % args.inference_stride == 0
                or batch_start + offset == end_frame - 1
            ]
            inference_frames = [frame for _, frame in inference_items]
            detection_results = detector.predict(
                inference_frames, classes=[0], conf=args.confidence, device=args.device,
                batch=args.batch_frames, verbose=False
            )
            pose_requests: list[tuple[np.ndarray, np.ndarray]] = []
            pose_metadata = []
            for (offset, frame), result in zip(inference_items, detection_results):
                source_frame = batch_start + offset
                local_frame = source_frame - args.start_frame
                oracle_frame = min(source_frame, oracle_frames - 1)
                boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                red_box = pose_bbox(oracle["pose_red_2d"][oracle_frame], width, height)
                blue_box = pose_bbox(oracle["pose_blue_2d"][oracle_frame], width, height)
                red_index, blue_index = match_oracle_boxers(boxes, red_box, blue_box, args.minimum_iou)
                for side, index, target in (("red", red_index, red_box), ("blue", blue_index, blue_box)):
                    if index is None:
                        continue
                    pose_requests.append((frame, boxes[index]))
                    pose_metadata.append((local_frame, side, boxes[index], target))
            if pose_requests:
                poses = batched_pose_inference(pose_model, pose_pipeline, pose_requests)
                for (local_frame, side, box, target), sample in zip(pose_metadata, poses):
                    instances = sample.pred_instances
                    predicted = np.asarray(instances.keypoints)[0]
                    if "transformed_keypoints" in instances:
                        keypoints_2d = np.asarray(instances.transformed_keypoints)[0]
                        keypoints_3d = predicted
                    else:
                        keypoints_2d = predicted
                        keypoints_3d = np.zeros((133, 3), dtype=np.float32)
                    output_2d[side][local_frame], output_3d[side][local_frame] = rtmw_to_boxingweb(
                        keypoints_2d, keypoints_3d, width, height
                    )
                    valid[side][local_frame] = True
                    if target is not None:
                        detection_iou[side][local_frame] = bbox_iou(box, target)
            processed += len(frames)
            if processed % 100 < len(frames) or processed == frame_count:
                elapsed = time.time() - started
                rate = processed / elapsed
                remaining = (frame_count - processed) / rate if rate else 0
                print(f"frames={processed}/{frame_count} rate={rate:.1f}/s eta={remaining:.0f}s", flush=True)
    finally:
        capture.release()

    interpolate = interpolate_all_gaps if args.interpolate_all_missing else (
        lambda values, mask: interpolate_short_gaps(values, mask, args.max_gap)
    )
    payload = {
        "pose_red_2d": smooth_sequence(interpolate(output_2d["red"], valid["red"]), args.smooth_window),
        "pose_blue_2d": smooth_sequence(interpolate(output_2d["blue"], valid["blue"]), args.smooth_window),
        "pose_red_3d": smooth_sequence(interpolate(output_3d["red"], valid["red"]), args.smooth_window),
        "pose_blue_3d": smooth_sequence(interpolate(output_3d["blue"], valid["blue"]), args.smooth_window),
        "valid_red": valid["red"],
        "valid_blue": valid["blue"],
        "detection_iou_red": detection_iou["red"],
        "detection_iou_blue": detection_iou["blue"],
        "format": "atom-canonical-boxer-pose-v1",
        "source": "YOLO person boxes + RTMW RGB pose + oracle red/blue association",
        "source_video": str(args.video.resolve()),
        "source_start_frame": args.start_frame,
        "source_end_frame_exclusive": end_frame,
        "fps": fps,
        "width": width,
        "height": height,
        "interpolate_all_missing": args.interpolate_all_missing,
        "smooth_window": args.smooth_window,
        "inference_stride": args.inference_stride,
        "flip_test": args.flip_test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(payload, file)
    print(
        f"pose={args.output} red_coverage={valid['red'].mean():.3f} "
        f"blue_coverage={valid['blue'].mean():.3f}", flush=True
    )


if __name__ == "__main__":
    main()
