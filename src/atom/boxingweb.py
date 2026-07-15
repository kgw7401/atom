"""Read BoxingWeb oracle-window samples without modifying the source dataset.

An oracle-window sample uses the annotated punch interval to locate a clip.  It
is for attribute classification only: event detection is deliberately out of
scope here because it must not receive the ground-truth interval at inference.
"""

from __future__ import annotations

import json
import pickle
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal


Split = Literal["train", "test"]
ATTRIBUTE_FIELDS = ("side", "technique", "distance", "target", "effect")


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class OracleWindowSample:
    """One valid labeled punch and its surrounding temporal context.

    Event boundaries are inclusive because that is how BoxingWeb labels them.
    Clip boundaries are half-open (`clip_end_exclusive`) for unambiguous frame
    slicing in RGB and pose tensors.
    """

    split: Split
    match_id: str
    event_index: int
    video_path: str
    pose_path: str
    event_start_frame: int
    event_end_frame: int
    clip_start_frame: int
    clip_end_exclusive: int
    fps: float
    labels: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OracleIndex:
    """Serializable index plus the reason labels were excluded."""

    schema_version: int
    data_root: str
    context_seconds: float
    samples: tuple[OracleWindowSample, ...]
    skipped_events: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "data_root": self.data_root,
            "context_seconds": self.context_seconds,
            "sample_count": len(self.samples),
            "skipped_events": self.skipped_events,
            "samples": [sample.to_dict() for sample in self.samples],
        }


def read_video_metadata(video_path: Path) -> VideoMetadata:
    """Read exact stream metadata using ffprobe, without decoding frames."""

    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,width,height,nb_frames",
        "-of", "json", str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream: {video_path}")
    stream = streams[0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    if denominator == "0" or stream.get("nb_frames") in (None, "N/A"):
        raise ValueError(f"Incomplete video metadata: {video_path}")
    return VideoMetadata(
        fps=float(numerator) / float(denominator),
        frame_count=int(stream["nb_frames"]),
        width=int(stream["width"]),
        height=int(stream["height"]),
    )


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _paths(match_dir: Path) -> tuple[Path, Path]:
    videos = sorted(match_dir.glob("*.mp4"))
    poses = sorted(match_dir.glob("*_pose_gt.pkl"))
    if len(videos) != 1 or len(poses) != 1:
        raise ValueError(f"Expected one MP4 and one pose pickle in {match_dir}")
    return videos[0], poses[0]


def build_oracle_index(data_root: Path, split: Split, context_seconds: float = 0.0) -> OracleIndex:
    """Build a deterministic index of valid non-empty annotated punches.

    The data-audit rules are intentional: `0-0` is an unlabeled placeholder,
    reversed ranges are unusable, and all clip bounds are clamped to the
    video's frame range. To match the official BoxingWeb classifier, only
    intervals where `4 <= end - start <= 30` are included. No annotation file
    is modified.
    """

    if context_seconds < 0:
        raise ValueError("context_seconds must be non-negative")
    root = data_root.expanduser().resolve()
    split_dir = root / f"data_{split}"
    if not split_dir.is_dir():
        raise ValueError(f"Missing split directory: {split_dir}")

    samples: list[OracleWindowSample] = []
    skipped: Counter[str] = Counter()
    for match_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        video_path, pose_path = _paths(match_dir)
        metadata = read_video_metadata(video_path)
        events = json.loads((match_dir / "video_event.json").read_text())
        if not isinstance(events, list):
            raise ValueError(f"Expected an event list: {match_dir / 'video_event.json'}")

        context_frames = round(context_seconds * metadata.fps)
        for event_index, event in enumerate(events):
            if not isinstance(event, dict) or event.get("name") != "punching":
                continue
            labels = {field: str(event.get(field, "")) for field in ATTRIBUTE_FIELDS}
            if any(not value for value in labels.values()):
                skipped["missing_attribute"] += 1
                continue
            start = _integer(event.get("frame_begin"))
            end = _integer(event.get("frame_end"))
            if start is None or end is None:
                skipped["non_integer_interval"] += 1
            elif start == 0 and end == 0:
                skipped["zero_zero_placeholder"] += 1
            elif end < start:
                skipped["reversed_interval"] += 1
            elif start < 0 or end >= metadata.frame_count:
                skipped["interval_outside_video"] += 1
            elif end - start < 4:
                skipped["interval_too_short"] += 1
            elif end - start > 30:
                skipped["interval_too_long"] += 1
            else:
                clip_start = max(0, start - context_frames)
                clip_end_exclusive = min(metadata.frame_count, end + 1 + context_frames)
                samples.append(
                    OracleWindowSample(
                        split=split,
                        match_id=match_dir.name,
                        event_index=event_index,
                        video_path=str(video_path.relative_to(root)),
                        pose_path=str(pose_path.relative_to(root)),
                        event_start_frame=start,
                        event_end_frame=end,
                        clip_start_frame=clip_start,
                        clip_end_exclusive=clip_end_exclusive,
                        fps=metadata.fps,
                        labels=labels,
                    )
                )
    return OracleIndex(
        schema_version=1,
        data_root=str(root),
        context_seconds=context_seconds,
        samples=tuple(samples),
        skipped_events=dict(sorted(skipped.items())),
    )


def write_oracle_index(index: OracleIndex, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2) + "\n")


def iter_rgb_frames(sample: OracleWindowSample, data_root: Path) -> Iterator[Any]:
    """Yield BGR frames for the sample's clip, in source-frame order.

    OpenCV is imported only here so metadata indexing remains lightweight.  RGB
    conversion, resizing, and temporal subsampling are model-specific and must
    be applied by the caller.
    """

    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("RGB decoding needs opencv-python") from error
    capture = cv2.VideoCapture(str(data_root / sample.video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {sample.video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, sample.clip_start_frame)
        for _ in range(sample.clip_start_frame, sample.clip_end_exclusive):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Unable to decode frame in {sample.video_path}")
            yield frame
    finally:
        capture.release()


def load_pose_window(sample: OracleWindowSample, data_root: Path) -> dict[str, Any]:
    """Return pose arrays clipped to the same half-open frame range as RGB."""

    with (data_root / sample.pose_path).open("rb") as file:
        pose = pickle.load(file)
    if not isinstance(pose, dict):
        raise ValueError(f"Expected pose dictionary: {sample.pose_path}")
    return {
        key: value[sample.clip_start_frame:sample.clip_end_exclusive]
        for key, value in pose.items()
        if key.startswith("pose_")
    }
