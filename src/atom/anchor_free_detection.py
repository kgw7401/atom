"""BoxMind-style anchor-free temporal punch event detection primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


HANDS = ("left", "right")


@dataclass(frozen=True)
class AnchorFreeEvent:
    side: str
    hand: str
    start_frame: int
    end_frame: int
    score: float


class _Block(nn.Module):
    def __init__(self, channels: int, dilation: int, batch_norm: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        normalization = nn.BatchNorm1d if batch_norm else lambda _: nn.Identity()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            normalization(channels), nn.GELU(),
            nn.Conv1d(channels, channels, 1), normalization(channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(inputs + self.dropout(self.layers(inputs)))


class AnchorFreePunchDetector(nn.Module):
    """Shared TCN with independent left- and right-hand detection heads."""

    def __init__(self, feature_count: int, channels: int = 64, batch_norm: bool = False,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16), dropout: float = 0.0) -> None:
        super().__init__()
        self.input = nn.Conv1d(feature_count, channels, 1)
        self.tcn = nn.Sequential(*[_Block(channels, dilation, batch_norm, dropout) for dilation in dilations])
        self.head = nn.ModuleList([nn.Conv1d(channels, 3, 1) for _ in HANDS])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        temporal = self.tcn(self.input(features.reshape(features.shape[0], features.shape[1], -1).transpose(1, 2)))
        return torch.stack([head(temporal).transpose(1, 2) for head in self.head], dim=2)


class BidirectionalGRUPunchDetector(nn.Module):
    """Bidirectional recurrent alternative with the same anchor-free heads."""

    def __init__(self, feature_count: int, channels: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("GRU channels must be even")
        self.input = nn.Sequential(nn.Linear(feature_count, channels), nn.GELU())
        self.temporal = nn.GRU(
            channels, channels // 2, num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout,
        )
        self.head = nn.Linear(channels, len(HANDS) * 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flattened = features.reshape(features.shape[0], features.shape[1], -1)
        temporal, _ = self.temporal(self.input(flattened))
        return self.head(temporal).reshape(features.shape[0], features.shape[1], len(HANDS), 3)


class _MultiScaleBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.short = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.long = nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation)
        self.fuse = nn.Conv1d(2 * channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        branches = torch.cat((torch.nn.functional.gelu(self.short(inputs)), torch.nn.functional.gelu(self.long(inputs))), dim=1)
        return torch.nn.functional.gelu(inputs + self.dropout(self.fuse(branches)))


class MultiScalePunchDetector(nn.Module):
    """Parallel short/long temporal kernels for local motion and wider context."""

    def __init__(self, feature_count: int, channels: int = 64,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16), dropout: float = 0.0) -> None:
        super().__init__()
        self.input = nn.Conv1d(feature_count, channels, 1)
        self.temporal = nn.Sequential(*[_MultiScaleBlock(channels, dilation, dropout) for dilation in dilations])
        self.head = nn.ModuleList([nn.Conv1d(channels, 3, 1) for _ in HANDS])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flattened = features.reshape(features.shape[0], features.shape[1], -1).transpose(1, 2)
        temporal = self.temporal(self.input(flattened))
        return torch.stack([head(temporal).transpose(1, 2) for head in self.head], dim=2)


class TCNGRUPunchDetector(nn.Module):
    """TCN local-motion encoder followed by bidirectional recurrent context."""

    def __init__(self, feature_count: int, channels: int = 64,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16), dropout: float = 0.0) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("TCN-GRU channels must be even")
        self.input = nn.Conv1d(feature_count, channels, 1)
        self.local = nn.Sequential(*[_Block(channels, dilation, False, dropout) for dilation in dilations])
        self.context = nn.GRU(
            channels, channels // 2, num_layers=1, batch_first=True,
            bidirectional=True,
        )
        self.fuse = nn.Sequential(nn.Linear(2 * channels, channels), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(channels, len(HANDS) * 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flattened = features.reshape(features.shape[0], features.shape[1], -1)
        local = self.local(self.input(flattened.transpose(1, 2))).transpose(1, 2)
        context, _ = self.context(local)
        temporal = self.fuse(torch.cat((local, context), dim=-1)) + local
        return self.head(temporal).reshape(features.shape[0], features.shape[1], len(HANDS), 3)


def build_punch_detector(
    architecture: str,
    feature_count: int,
    channels: int = 64,
    batch_norm: bool = False,
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
    dropout: float = 0.0,
) -> nn.Module:
    if architecture == "tcn":
        return AnchorFreePunchDetector(feature_count, channels, batch_norm, dilations, dropout)
    if architecture == "bigru":
        return BidirectionalGRUPunchDetector(feature_count, channels, dropout)
    if architecture == "mstcn":
        return MultiScalePunchDetector(feature_count, channels, dilations, dropout)
    if architecture == "tcngru":
        return TCNGRUPunchDetector(feature_count, channels, dilations, dropout)
    raise ValueError(f"Unknown detector architecture: {architecture}")


def encode_targets(
    events: list[AnchorFreeEvent], frame_count: int, offset_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Encode center probability and positive-only boundary offsets for a round."""

    target = np.zeros((frame_count, len(HANDS), 3), dtype=np.float32)
    regression_mask = np.zeros((frame_count, len(HANDS)), dtype=bool)
    for event in events:
        index = HANDS.index(event.hand)
        start, end = max(0, event.start_frame), min(frame_count - 1, event.end_frame)
        for frame in range(start, end + 1):
            target[frame, index] = (
                1.0,
                (frame - start) / offset_scale,
                (end - frame) / offset_scale,
            )
            regression_mask[frame, index] = True
    return target, regression_mask


def temporal_iou(first: AnchorFreeEvent, second: AnchorFreeEvent) -> float:
    intersection = max(0, min(first.end_frame, second.end_frame) - max(first.start_frame, second.start_frame) + 1)
    union = max(first.end_frame, second.end_frame) - min(first.start_frame, second.start_frame) + 1
    return intersection / union if union else 0.0


def decode_events(
    logits: np.ndarray, side: str, threshold: float = 0.5, nms_iou: float = 0.5, offset_scale: float = 1.0
) -> list[AnchorFreeEvent]:
    """Decode offset predictions, then apply per-hand temporal NMS."""

    probability = 1.0 / (1.0 + np.exp(-logits[..., 0]))
    offsets = np.maximum(logits[..., 1:], 0.0)
    candidates: list[AnchorFreeEvent] = []
    for frame, values in enumerate(probability):
        for hand_index, score in enumerate(values):
            if not np.isfinite(score) or score < threshold or not np.isfinite(offsets[frame, hand_index]).all():
                continue
            left = probability[max(0, frame - 2):frame, hand_index]
            right = probability[frame + 1:frame + 3, hand_index]
            if (len(left) and score < left.max()) or (len(right) and score <= right.max()):
                continue
            hand = HANDS[hand_index]
            start = max(0, int(round(frame - offsets[frame, hand_index, 0] * offset_scale)))
            end = max(start, int(round(frame + offsets[frame, hand_index, 1] * offset_scale)))
            candidates.append(AnchorFreeEvent(side, hand, start, end, float(score)))
    kept: list[AnchorFreeEvent] = []
    for candidate in sorted(candidates, key=lambda event: event.score, reverse=True):
        if not any(candidate.hand == kept_event.hand and temporal_iou(candidate, kept_event) >= nms_iou for kept_event in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda event: (event.start_frame, event.side, event.hand))


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.99, gamma: float = 2.0) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    pt = torch.where(target > 0, probability, 1 - probability)
    weight = torch.where(target > 0, torch.full_like(target, alpha), torch.full_like(target, 1 - alpha))
    return (-weight * (1 - pt).pow(gamma) * torch.log(pt.clamp_min(1e-7))).mean()
