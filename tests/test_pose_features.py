from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.pose_features import _local_2d, _local_3d, _resample, _rolling_median


class PoseFeatureTests(unittest.TestCase):
    def test_resample_preserves_first_and_last_frames(self) -> None:
        sequence = np.array([[[1.0]], [[3.0]], [[5.0]]], dtype=np.float32)
        result = _resample(sequence, 5)
        self.assertEqual(result.shape, (5, 1, 1))
        self.assertEqual(float(result[0, 0, 0]), 1.0)
        self.assertEqual(float(result[-1, 0, 0]), 5.0)
        self.assertEqual(float(result[2, 0, 0]), 3.0)

    def test_local_2d_is_translation_invariant(self) -> None:
        actor = np.zeros((2, 45, 2), dtype=np.float32)
        opponent = np.zeros_like(actor)
        actor[:, 1] = [0, 0]
        actor[:, 2] = [2, 0]
        actor[:, 16] = [0, 2]
        actor[:, 17] = [2, 2]
        opponent[:] = actor
        actor[1] += 10
        opponent[1] += 10
        local_actor, local_opponent = _local_2d(actor, opponent)
        np.testing.assert_allclose(local_actor[0], local_actor[1])
        np.testing.assert_allclose(local_opponent[0], local_opponent[1])

    def test_rolling_median_rejects_single_frame_scale_spike(self) -> None:
        result = _rolling_median(np.array([2.0, 2.0, 20.0, 2.0, 2.0]), window=3)
        np.testing.assert_allclose(result, 2.0)

    def test_scale_normalized_local_3d_ignores_uniform_body_scale(self) -> None:
        person = np.zeros((2, 45, 3), dtype=np.float32)
        person[:, 1] = [-1.0, 0.0, 0.0]
        person[:, 2] = [1.0, 0.0, 0.0]
        person[:, 16] = [-1.0, 0.0, 2.0]
        person[:, 17] = [1.0, 0.0, 2.0]
        person[:, 3] = [2.0, 0.0, 1.0]
        person[1] *= 3.0
        result = _local_3d(person, normalize_scale=True)
        np.testing.assert_allclose(result[0], result[1], atol=1e-6)
