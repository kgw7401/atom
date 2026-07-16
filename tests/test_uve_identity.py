from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.uve_identity import APPEARANCE_FEATURE_COUNT, appearance_descriptor


class UVEIdentityTests(unittest.TestCase):
    def test_descriptor_is_fixed_and_distinguishes_red_from_blue(self) -> None:
        red = np.zeros((100, 100, 3), dtype=np.uint8)
        red[:] = (0, 0, 255)
        blue = np.zeros_like(red)
        blue[:] = (255, 0, 0)
        box = np.array([10, 5, 90, 95], dtype=np.float32)
        red_descriptor = appearance_descriptor(red, box)
        blue_descriptor = appearance_descriptor(blue, box)
        self.assertEqual(red_descriptor.shape, (APPEARANCE_FEATURE_COUNT,))
        self.assertTrue(np.isfinite(red_descriptor).all())
        self.assertGreater(float(np.linalg.norm(red_descriptor - blue_descriptor)), 0.5)

    def test_invalid_box_returns_zero_descriptor(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        descriptor = appearance_descriptor(frame, np.array([5, 5, 5, 5]))
        np.testing.assert_array_equal(descriptor, 0.0)


if __name__ == "__main__":
    unittest.main()
