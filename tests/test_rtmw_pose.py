from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.rtmw_pose import (  # noqa: E402
    interpolate_short_gaps,
    interpolate_all_gaps,
    match_oracle_boxers,
    rtmw_to_boxingweb,
    square_denormalize,
    square_normalize,
    smooth_sequence,
)


class RTMWPoseTests(unittest.TestCase):
    def test_square_coordinates_round_trip(self) -> None:
        points = np.array([[0, 0], [1273, 719], [500, 300]], dtype=np.float32)
        restored = square_denormalize(square_normalize(points, 1274, 720), 1274, 720)
        np.testing.assert_allclose(restored, points, atol=1e-4)

    def test_conversion_derives_neck_and_pelvis(self) -> None:
        pose_2d = np.zeros((133, 2), dtype=np.float32)
        pose_3d = np.zeros((133, 3), dtype=np.float32)
        pose_2d[5], pose_2d[6] = (100, 200), (300, 200)
        pose_3d[5], pose_3d[6] = (-1, 2, 3), (1, 2, 3)
        converted_2d, converted_3d = rtmw_to_boxingweb(pose_2d, pose_3d, 400, 300)
        np.testing.assert_allclose(converted_2d[1], square_normalize(np.array([[200, 200]]), 400, 300)[0])
        np.testing.assert_allclose(converted_3d[1], [0, 2, 3])

    def test_oracle_matching_requires_distinct_detections(self) -> None:
        detections = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=np.float32)
        self.assertEqual(match_oracle_boxers(detections, detections[1], detections[0]), (1, 0))

    def test_only_short_bounded_gaps_are_interpolated(self) -> None:
        values = np.array([[0.0], [0.0], [2.0], [0.0], [0.0], [5.0]], dtype=np.float32)
        valid = np.array([False, True, True, False, False, True])
        result = interpolate_short_gaps(values, valid, max_gap=2)
        self.assertEqual(float(result[0, 0]), 0.0)
        np.testing.assert_allclose(result[3:, 0], [3.0, 4.0, 5.0])

    def test_all_gaps_include_edges(self) -> None:
        values = np.array([[0.0], [1.0], [0.0], [3.0], [0.0]], dtype=np.float32)
        valid = np.array([False, True, False, True, False])
        np.testing.assert_allclose(interpolate_all_gaps(values, valid)[:, 0], [1, 1, 2, 3, 3])

    def test_centered_smoothing_preserves_length(self) -> None:
        values = np.array([[0.0], [3.0], [0.0]], dtype=np.float32)
        result = smooth_sequence(values, 3)
        self.assertEqual(result.shape, values.shape)
        np.testing.assert_allclose(result[:, 0], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
