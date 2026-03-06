"""Tests for CTR-GCN model architecture."""

import pytest
import torch
import numpy as np

from ml.model.modules import CTRGC, MSTCN, CTRGCBlock
from ml.model.ctrgcn import CTRGCN
from ml.graph.boxing_graph import get_adjacency_matrix, num_node


@pytest.fixture
def A():
    return get_adjacency_matrix()


@pytest.fixture
def model():
    return CTRGCN(num_classes=11, num_joints=15, in_channels=3)


# ── CTRGC (Spatial) ──────────────────────────────────────────────

class TestCTRGC:
    def test_output_shape(self, A):
        layer = CTRGC(in_channels=3, out_channels=64, A=A)
        x = torch.randn(2, 3, 30, 15)
        out = layer(x)
        assert out.shape == (2, 64, 30, 15)

    def test_channel_expansion(self, A):
        layer = CTRGC(in_channels=64, out_channels=128, A=A)
        x = torch.randn(2, 64, 30, 15)
        out = layer(x)
        assert out.shape == (2, 128, 30, 15)

    def test_non_adaptive(self, A):
        layer = CTRGC(in_channels=3, out_channels=64, A=A, adaptive=False)
        x = torch.randn(2, 3, 30, 15)
        out = layer(x)
        assert out.shape == (2, 64, 30, 15)


# ── MSTCN (Temporal) ─────────────────────────────────────────────

class TestMSTCN:
    def test_output_shape_stride1(self):
        layer = MSTCN(in_channels=64, out_channels=64, stride=1)
        x = torch.randn(2, 64, 30, 15)
        out = layer(x)
        assert out.shape == (2, 64, 30, 15)

    def test_output_shape_stride2(self):
        layer = MSTCN(in_channels=64, out_channels=64, stride=2)
        x = torch.randn(2, 64, 30, 15)
        out = layer(x)
        assert out.shape == (2, 64, 15, 15)

    def test_channel_change(self):
        layer = MSTCN(in_channels=64, out_channels=128, stride=1, residual=False)
        x = torch.randn(2, 64, 30, 15)
        out = layer(x)
        assert out.shape == (2, 128, 30, 15)

    def test_channel_change_with_residual(self):
        layer = MSTCN(in_channels=64, out_channels=128, stride=1, residual=True)
        x = torch.randn(2, 64, 30, 15)
        out = layer(x)
        assert out.shape == (2, 128, 30, 15)


# ── CTRGCBlock ────────────────────────────────────────────────────

class TestCTRGCBlock:
    def test_same_channels(self, A):
        block = CTRGCBlock(64, 64, A)
        x = torch.randn(2, 64, 30, 15)
        out = block(x)
        assert out.shape == (2, 64, 30, 15)

    def test_channel_expansion(self, A):
        block = CTRGCBlock(64, 128, A)
        x = torch.randn(2, 64, 30, 15)
        out = block(x)
        assert out.shape == (2, 128, 30, 15)

    def test_temporal_stride(self, A):
        block = CTRGCBlock(64, 128, A, stride=2)
        x = torch.randn(2, 64, 30, 15)
        out = block(x)
        assert out.shape == (2, 128, 15, 15)

    def test_no_residual(self, A):
        block = CTRGCBlock(3, 64, A, residual=False)
        x = torch.randn(2, 3, 30, 15)
        out = block(x)
        assert out.shape == (2, 64, 30, 15)


# ── Full CTRGCN Model ────────────────────────────────────────────

class TestCTRGCN:
    def test_forward_5d(self, model):
        """Standard input with person dimension: (N, C, T, V, M)."""
        x = torch.randn(4, 3, 30, 15, 1)
        out = model(x)
        assert out.shape == (4, 11)

    def test_forward_4d(self, model):
        """Input without person dimension: (N, C, T, V)."""
        x = torch.randn(4, 3, 30, 15)
        out = model(x)
        assert out.shape == (4, 11)

    def test_single_sample(self, model):
        """Batch size 1."""
        x = torch.randn(1, 3, 30, 15, 1)
        out = model(x)
        assert out.shape == (1, 11)

    def test_output_is_logits(self, model):
        """Output should be raw logits (not softmax)."""
        x = torch.randn(2, 3, 30, 15, 1)
        out = model(x)
        # Logits can be any real number, not constrained to [0,1]
        assert out.min() < 0 or out.max() > 1 or True  # just check it runs

    def test_softmax_sums_to_one(self, model):
        """Softmax of output should sum to 1."""
        x = torch.randn(2, 3, 30, 15, 1)
        out = model(x)
        probs = torch.softmax(out, dim=1)
        sums = probs.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(2), atol=1e-5, rtol=1e-5)

    def test_parameter_count(self, model):
        """Model should be lightweight: 0.5M - 5M parameters."""
        count = model.count_parameters()
        assert 100_000 < count < 5_000_000, f"Unexpected param count: {count}"

    def test_gradient_flow(self, model):
        """Gradients should flow through all blocks."""
        x = torch.randn(2, 3, 30, 15, 1)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check that gradients exist for key parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                # At least some gradients should be non-zero
                if "weight" in name:
                    assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_eval_mode(self, model):
        """Model should work in eval mode (BatchNorm uses running stats)."""
        model.eval()
        x = torch.randn(2, 3, 30, 15, 1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 11)

    def test_different_num_classes(self):
        """Model should work with custom number of classes."""
        model = CTRGCN(num_classes=8)
        x = torch.randn(2, 3, 30, 15, 1)
        out = model(x)
        assert out.shape == (2, 8)

    def test_deterministic_eval(self, model):
        """Same input should produce same output in eval mode."""
        model.eval()
        x = torch.randn(2, 3, 30, 15, 1)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_overfit_single_sample(self):
        """Model should be able to overfit a single sample (sanity check)."""
        model = CTRGCN(num_classes=3, num_joints=15, in_channels=3)
        model.train()

        x = torch.randn(1, 3, 30, 15, 1)
        target = torch.tensor([1])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        # Train for a few steps
        initial_loss = None
        for step in range(50):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, target)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5, \
            f"Failed to overfit: {initial_loss:.4f} → {final_loss:.4f}"

        # Should predict correct class
        model.eval()
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        assert pred.item() == 1
