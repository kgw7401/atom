"""Appearance features and classifier used by the UVE-style track refiner.

The BoxMind UVE classifier consumes 4D-Humans UV images. Those private
classifier weights are not published, so this module keeps the same
red/blue/non-boxer interface while using a compact RGB appearance descriptor.
It can later be replaced by a UV-image encoder without changing tracking code.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


IDENTITY_CLASSES = ("red", "blue", "non_boxer")
APPEARANCE_FEATURE_COUNT = 94


def _normalized_histogram(values: np.ndarray, bins: int, upper: float) -> np.ndarray:
    histogram, _ = np.histogram(values, bins=bins, range=(0.0, upper))
    total = histogram.sum()
    return (histogram / total if total else histogram).astype(np.float32)


def _region_features(region: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    colorful = (saturation > 60) & (value > 35)
    red = colorful & ((hue < 15) | (hue >= 165))
    blue = colorful & (hue >= 90) & (hue < 140)
    white = (saturation < 45) & (value > 150)
    dark = value < 55
    pixels = float(region.shape[0] * region.shape[1])
    mean = region.reshape(-1, 3).mean(axis=0) / 255.0
    std = region.reshape(-1, 3).std(axis=0) / 255.0
    return np.concatenate((
        _normalized_histogram(hue, 18, 180.0),
        _normalized_histogram(saturation, 8, 256.0),
        _normalized_histogram(value, 8, 256.0),
        np.array([red.sum() / pixels, blue.sum() / pixels, white.sum() / pixels, dark.sum() / pixels], np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    ))


def appearance_descriptor(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Return a fixed RGB appearance descriptor for one person detection."""

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("frame must be a BGR image with shape [H,W,3]")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    x1, x2 = np.clip((x1, x2), 0, width - 1)
    y1, y2 = np.clip((y1, y2), 0, height - 1)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.zeros(APPEARANCE_FEATURE_COUNT, dtype=np.float32)
    # The center strip suppresses ring ropes and spectators while retaining
    # the singlet and trunks that distinguish red- and blue-corner boxers.
    box_width, box_height = x2 - x1, y2 - y1
    cx1, cx2 = int(x1 + .15 * box_width), int(x2 - .15 * box_width)
    upper_y1, middle_y = int(y1 + .15 * box_height), int(y1 + .55 * box_height)
    lower_y2 = int(y1 + .90 * box_height)
    upper = image[max(0, upper_y1):max(upper_y1 + 1, middle_y), max(0, cx1):max(cx1 + 1, cx2)]
    lower = image[max(0, middle_y):max(middle_y + 1, lower_y2), max(0, cx1):max(cx1 + 1, cx2)]
    upper = cv2.resize(upper, (32, 32), interpolation=cv2.INTER_AREA)
    lower = cv2.resize(lower, (32, 32), interpolation=cv2.INTER_AREA)
    geometry = np.array((
        (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
        box_width / width, box_height / height,
        box_width * box_height / (width * height), box_width / max(box_height, 1.0),
    ), dtype=np.float32)
    descriptor = np.concatenate((_region_features(upper), _region_features(lower), geometry))
    if descriptor.shape != (APPEARANCE_FEATURE_COUNT,):
        raise RuntimeError(f"Unexpected descriptor shape: {descriptor.shape}")
    return np.nan_to_num(descriptor, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class UVEAppearanceClassifier(nn.Module):
    """Small replaceable red/blue/non-boxer appearance classifier."""

    def __init__(self, feature_count: int = APPEARANCE_FEATURE_COUNT, hidden: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_count, hidden), nn.GELU(), nn.Dropout(.1),
            nn.Linear(hidden, len(IDENTITY_CLASSES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def load_uve_classifier(path: Path, device: torch.device | str = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = UVEAppearanceClassifier(
        int(checkpoint.get("feature_count", APPEARANCE_FEATURE_COUNT)),
        int(checkpoint.get("hidden", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, np.asarray(checkpoint["feature_mean"], np.float32), np.asarray(checkpoint["feature_std"], np.float32)


def classify_appearances(
    model: nn.Module,
    descriptors: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    if len(descriptors) == 0:
        return np.empty((0, len(IDENTITY_CLASSES)), dtype=np.float32)
    normalized = (np.asarray(descriptors, np.float32) - mean) / std
    with torch.no_grad():
        logits = model(torch.from_numpy(normalized).to(device))
        return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
