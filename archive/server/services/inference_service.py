"""LSTM model loading and video classification service.

Loaded once at server startup via FastAPI lifespan. Reuses Phase 1
inference utilities from src/inference/utils.py.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from src.extraction.pose_extractor import PoseExtractor
from src.inference.utils import load_model, preprocess_window
from src.preprocessing.pipeline import PipelineConfig

from server.config import settings


class InferenceService:
    """Singleton service: loads model once, classifies uploaded videos."""

    def __init__(self):
        self.model, self.class_names, self.scaler_mean, self.scaler_scale = load_model(
            settings.lstm_checkpoint,
        )
        self.cfg = PipelineConfig.from_yaml(settings.config_path)
        self.confidence_threshold = settings.confidence_threshold

    def classify_video(self, video_path: str) -> list[dict]:
        """Full pipeline: video file -> list of {timestamp_s, action, confidence}.

        Runs MediaPipe pose extraction + LSTM classification on every
        sliding window of the video.
        """
        extractor = PoseExtractor(model_path=settings.pose_model)
        buffer: deque[np.ndarray] = deque(maxlen=self.cfg.window_size)
        results: list[dict] = []
        stride_counter = 0

        with extractor:
            for frame in extractor.process_video(video_path):
                buffer.append(frame.keypoints)
                stride_counter += 1

                if len(buffer) < self.cfg.window_size:
                    continue
                if stride_counter < self.cfg.stride:
                    continue

                stride_counter = 0
                buf_array = np.stack(list(buffer))
                features = preprocess_window(
                    buf_array, self.cfg, self.scaler_mean, self.scaler_scale,
                )
                if features is None:
                    continue

                x = torch.from_numpy(features).unsqueeze(0)
                with torch.no_grad():
                    logits = self.model(x)
                    proba = torch.softmax(logits, dim=1)[0].numpy()

                pred_idx = int(proba.argmax())
                action = self.class_names[pred_idx]
                conf = float(proba[pred_idx])

                # Low confidence fallback to guard
                if action != "guard" and conf < self.confidence_threshold:
                    guard_idx = self.class_names.index("guard")
                    action = "guard"
                    conf = float(proba[guard_idx])

                results.append({
                    "timestamp_s": frame.timestamp_ms / 1000.0,
                    "action": action,
                    "confidence": conf,
                })

        return results

    def segment_actions(self, detections: list[dict]) -> list[dict]:
        """Merge consecutive same-action detections into segments.

        Returns list of {action, start_s, end_s, avg_confidence}.
        """
        if not detections:
            return []

        segments: list[dict] = []
        current = detections[0].copy()
        run_confs = [current["confidence"]]

        for det in detections[1:]:
            if det["action"] == current["action"]:
                run_confs.append(det["confidence"])
                current["end_s"] = det["timestamp_s"]
            else:
                segments.append({
                    "action": current["action"],
                    "start_s": current["timestamp_s"],
                    "end_s": current.get("end_s", current["timestamp_s"]),
                    "avg_confidence": sum(run_confs) / len(run_confs),
                })
                current = det.copy()
                run_confs = [det["confidence"]]

        # Last segment
        segments.append({
            "action": current["action"],
            "start_s": current["timestamp_s"],
            "end_s": current.get("end_s", current["timestamp_s"]),
            "avg_confidence": sum(run_confs) / len(run_confs),
        })

        return segments
