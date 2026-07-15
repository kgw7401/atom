from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atom.boxingweb import VideoMetadata, build_oracle_index


class BuildOracleIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self._testMethodName).with_suffix("")

    def _write_match(self, events: list[dict]) -> None:
        match = self.root / "data_train" / "match_1"
        match.mkdir(parents=True)
        (match / "match_1.mp4").touch()
        (match / "match_1_pose_gt.pkl").touch()
        import json
        (match / "video_event.json").write_text(json.dumps(events))

    def test_clips_valid_interval_and_skips_known_invalid_intervals(self) -> None:
        self._write_match([
            {"name": "punching", "frame_begin": "10", "frame_end": "14", "side": "red", "technique": "lstraight", "distance": "long", "target": "head", "effect": "effective"},
            {"name": "punching", "frame_begin": "0", "frame_end": "0", "side": "blue", "technique": "rhook", "distance": "medium", "target": "head", "effect": "ineffective"},
            {"name": "punching", "frame_begin": "20", "frame_end": "15", "side": "blue", "technique": "rhook", "distance": "medium", "target": "head", "effect": "ineffective"},
            {"name": "punching", "frame_begin": "1", "frame_end": "3", "side": "blue", "technique": "rhook", "distance": "medium", "target": "head", "effect": "ineffective"},
            {"name": "set_side", "frame_begin": "0", "frame_end": "0"},
        ])
        metadata = VideoMetadata(fps=20, frame_count=30, width=100, height=100)
        with patch("atom.boxingweb.read_video_metadata", return_value=metadata):
            index = build_oracle_index(self.root, "train", context_seconds=0.25)

        self.assertEqual(len(index.samples), 1)
        sample = index.samples[0]
        self.assertEqual((sample.clip_start_frame, sample.clip_end_exclusive), (5, 20))
        self.assertEqual(index.skipped_events, {"interval_too_short": 1, "reversed_interval": 1, "zero_zero_placeholder": 1})

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
