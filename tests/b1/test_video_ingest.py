"""Tests for B1 Task 1: Video ingestion."""

from pathlib import Path

import pytest

from track_b.b1.video_ingest import (
    _is_youtube_url,
    extract_frames_analysis,
    extract_frames_training,
)


class TestIsYoutubeUrl:
    def test_youtube_com(self):
        assert _is_youtube_url("https://www.youtube.com/watch?v=abc123")

    def test_youtu_be(self):
        assert _is_youtube_url("https://youtu.be/abc123")

    def test_local_file(self):
        assert not _is_youtube_url("/path/to/video.mp4")

    def test_other_url(self):
        assert not _is_youtube_url("https://example.com/video.mp4")


class TestTrainingMode:
    def test_extracts_all_frames(self, fixture_video_2s: Path):
        metadata, frames = extract_frames_training(fixture_video_2s)
        assert len(frames) == 60  # 2s * 30fps

    def test_metadata_correct(self, fixture_video_2s: Path):
        metadata, _ = extract_frames_training(fixture_video_2s)
        assert metadata.width == 320
        assert metadata.height == 240
        assert metadata.fps == pytest.approx(30.0, abs=1.0)
        assert metadata.total_frames == 60
        assert metadata.duration == pytest.approx(2.0, abs=0.1)

    def test_frame_timestamps_sequential(self, fixture_video_2s: Path):
        _, frames = extract_frames_training(fixture_video_2s)
        for i, frame in enumerate(frames):
            assert frame.frame_number == i
            expected_ts = i / 30.0
            assert frame.timestamp == pytest.approx(expected_ts, abs=0.001)

    def test_frame_images_valid(self, fixture_video_2s: Path):
        _, frames = extract_frames_training(fixture_video_2s)
        for frame in frames:
            assert frame.image.shape == (240, 320, 3)
            assert frame.image.dtype.name == "uint8"

    def test_first_last_timestamp(self, fixture_video_2s: Path):
        _, frames = extract_frames_training(fixture_video_2s)
        assert frames[0].timestamp == pytest.approx(0.0)
        assert frames[-1].timestamp == pytest.approx(59 / 30.0, abs=0.001)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_frames_training("/nonexistent/video.mp4")


class TestAnalysisMode:
    def test_same_fps_extracts_all(self, fixture_video_2s: Path):
        """When target_fps matches source, all frames are extracted."""
        metadata, frames = extract_frames_analysis(fixture_video_2s, target_fps=30)
        assert len(frames) == 60

    def test_downsamples_fps(self, fixture_video_10s_60fps: Path):
        """60fps source sampled at 30fps → ~300 frames."""
        _, frames = extract_frames_analysis(fixture_video_10s_60fps, target_fps=30)
        assert len(frames) == pytest.approx(300, abs=5)

    def test_low_fps_sampling(self, fixture_video_10s_60fps: Path):
        """60fps source at 10fps → ~100 frames."""
        _, frames = extract_frames_analysis(fixture_video_10s_60fps, target_fps=10)
        assert len(frames) == pytest.approx(100, abs=5)

    def test_timestamps_reflect_source_time(self, fixture_video_10s_60fps: Path):
        """Timestamps should be in source video time, not output frame index."""
        _, frames = extract_frames_analysis(fixture_video_10s_60fps, target_fps=30)
        # First frame at ~0.0s, last frame near ~10.0s
        assert frames[0].timestamp == pytest.approx(0.0, abs=0.05)
        assert frames[-1].timestamp == pytest.approx(10.0, abs=0.2)

    def test_frame_numbers_sequential(self, fixture_video_10s_60fps: Path):
        """Output frame numbers should be sequential 0, 1, 2, ..."""
        _, frames = extract_frames_analysis(fixture_video_10s_60fps, target_fps=30)
        for i, frame in enumerate(frames):
            assert frame.frame_number == i

    def test_time_range_trimming(self, fixture_video_10s_60fps: Path):
        """Extract only 2s-5s range → ~90 frames at 30fps."""
        _, frames = extract_frames_analysis(
            fixture_video_10s_60fps, target_fps=30, start_time=2.0, end_time=5.0
        )
        assert len(frames) == pytest.approx(90, abs=5)
        # First frame timestamp should be ~2.0s
        assert frames[0].timestamp == pytest.approx(2.0, abs=0.1)
        # Last frame timestamp should be ~5.0s
        assert frames[-1].timestamp == pytest.approx(5.0, abs=0.2)

    def test_start_time_only(self, fixture_video_10s_60fps: Path):
        """Start at 5s, go to end → ~150 frames at 30fps."""
        _, frames = extract_frames_analysis(
            fixture_video_10s_60fps, target_fps=30, start_time=5.0
        )
        assert len(frames) == pytest.approx(150, abs=5)
        assert frames[0].timestamp == pytest.approx(5.0, abs=0.1)

    def test_metadata_reflects_full_video(self, fixture_video_10s_60fps: Path):
        """Metadata should reflect the full source video, not the trimmed range."""
        metadata, _ = extract_frames_analysis(
            fixture_video_10s_60fps, target_fps=30, start_time=2.0, end_time=5.0
        )
        assert metadata.total_frames == 600
        assert metadata.duration == pytest.approx(10.0, abs=0.1)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_frames_analysis("/nonexistent/video.mp4")

    def test_higher_target_fps_extracts_all(self, fixture_video_2s: Path):
        """If target_fps > source_fps, all frames are extracted."""
        _, frames = extract_frames_analysis(fixture_video_2s, target_fps=60)
        assert len(frames) == 60
