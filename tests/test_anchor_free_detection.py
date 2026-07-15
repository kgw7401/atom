import unittest

import numpy as np

from atom.anchor_free_detection import AnchorFreeEvent, build_punch_detector, decode_events, encode_targets
from atom.uve_tracks import refine_boxer_tracks


class AnchorFreeDetectionTests(unittest.TestCase):
    def test_bigru_matches_anchor_free_output_shape(self) -> None:
        import torch

        model = build_punch_detector("bigru", feature_count=10, channels=16)
        self.assertEqual(tuple(model(torch.zeros(2, 12, 5, 2)).shape), (2, 12, 2, 3))

    def test_mstcn_matches_anchor_free_output_shape(self) -> None:
        import torch

        model = build_punch_detector("mstcn", feature_count=10, channels=16, dilations=(1, 2))
        self.assertEqual(tuple(model(torch.zeros(2, 12, 5, 2)).shape), (2, 12, 2, 3))

    def test_scaled_offsets_round_trip_to_interval(self):
        target, _ = encode_targets([AnchorFreeEvent("red", "left", 10, 18, 1.0)], 32, offset_scale=32)
        logits = np.full_like(target, -20.0)
        logits[..., 1:] = target[..., 1:]
        logits[10:19, 0, 0] = 20.0
        events = decode_events(logits, "red", threshold=.5, offset_scale=32)
        self.assertEqual([(event.side, event.hand, event.start_frame, event.end_frame) for event in events], [("red", "left", 10, 18)])

    def test_uve_refiner_keeps_distinct_identities_in_a_block(self):
        tracks = {
            "track_ids": np.array([[7, 8], [7, 8]], dtype=np.int64),
            "joints_2d": np.array([[[[7., 0.]], [[8., 0.]]], [[[7., 1.]], [[8., 1.]]]]),
            "joints_3d": np.array([[[[7., 0., 0.]], [[8., 0., 0.]]], [[[7., 1., 0.]], [[8., 1., 0.]]]]),
            "identity_probabilities": np.array([[[.9, .05, .05], [.05, .9, .05]], [[.8, .1, .1], [.1, .8, .1]]]),
        }
        output = refine_boxer_tracks(tracks, cadence=10)
        np.testing.assert_array_equal(output["red_track_ids"], [7, 7])
        np.testing.assert_array_equal(output["blue_track_ids"], [8, 8])
        np.testing.assert_allclose(output["pose_red_2d"][:, 0, 0], [7, 7])


if __name__ == "__main__":
    unittest.main()
