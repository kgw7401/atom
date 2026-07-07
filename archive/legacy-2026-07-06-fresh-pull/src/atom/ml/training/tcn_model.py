"""TCN (Temporal Convolutional Network) for per-frame punch classification.

Architecture:
    Input: (batch, 24, T)
    3x dilated Conv1D blocks (24→64, 64→64, 64→64) with BN + ReLU + Dropout
    Linear head: 64→4 classes (idle, jab, cross, hook)
    Output: (batch, 4, T) per-frame logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Class labels
CLASSES = ["idle", "jab", "cross", "hook"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}


class TCNBlock(nn.Module):
    """Single dilated causal convolution block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.trim = padding  # how many frames to trim from the right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.trim > 0:
            out = out[:, :, :-self.trim]  # causal: trim future frames
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class PunchTCN(nn.Module):
    """3-layer dilated TCN for punch classification."""

    def __init__(self, input_dim: int = 24, hidden_dim: int = 64, num_classes: int = NUM_CLASSES, dropout: float = 0.2):
        super().__init__()
        self.blocks = nn.Sequential(
            TCNBlock(input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
            TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
        )
        self.head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 24, T) input features.

        Returns:
            logits: (batch, num_classes, T) per-frame classification logits.
        """
        h = self.blocks(x)
        return self.head(h)
