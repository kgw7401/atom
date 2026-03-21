"""RecordingService — interactive CLI tool for recording audio chunks."""

from __future__ import annotations

import json
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from atom.services.audio_service import _load_dict, generate_checklist

# ── Constants ────────────────────────────────────────────────────────────

TARGET_DBFS = -16
MIN_DURATION_MS = 200

CATEGORIES = ("strike_atom", "strike_phrase", "defense", "cue")


# ── Recording plan ───────────────────────────────────────────────────────


def _categorize_text(text: str, data: dict) -> str:
    """Return the category for a chunk text."""
    if text in data["chunks"]["strike_atoms"]:
        return "strike_atom"
    if text in data["chunks"]["strike_phrases"]:
        return "strike_phrase"
    if text in data["chunks"]["defense"]:
        return "defense"
    cue_calls = {c["call"] for c in data.get("cues", [])}
    if text in cue_calls:
        return "cue"
    if text.endswith("라운드 시작합니다"):
        return "round_intro"
    return "unknown"


def _reuse_count_for(text: str, data: dict) -> int:
    """Count how many combos use this chunk."""
    assembly = {
        k: v for k, v in data.get("combo_assembly", {}).items()
        if not k.startswith("_")
    }
    count = 0
    for chunk_list in assembly.values():
        for chunk_text in chunk_list:
            if chunk_text == text:
                count += 1
    return count


@dataclass
class RecordingItem:
    """Single recording to make: a text + variant number."""

    text: str
    variant: int
    category: str
    reuse_count: int

    @property
    def filename(self) -> str:
        return f"{self.text}_{self.variant}.mp3"


def generate_recording_plan(category_filter: str | None = None) -> list[RecordingItem]:
    """Build ordered list of recordings needed, expanding suggested_takes into variants."""
    checklist = generate_checklist()
    data = _load_dict()

    items: list[RecordingItem] = []
    for entry in checklist:
        cat = _categorize_text(entry["text"], data)
        if category_filter and category_filter != "all" and cat != category_filter:
            continue
        for v in range(1, entry["suggested_takes"] + 1):
            items.append(RecordingItem(
                text=entry["text"],
                variant=v,
                category=cat,
                reuse_count=entry["reuse_count"],
            ))

    return items


# ── Progress persistence ─────────────────────────────────────────────────


class RecordingProgress:
    """JSON-based progress tracker for recording sessions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict = {"done": {}, "skipped": {}}
        self._load()

    def _load(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path) as f:
                self._data = json.load(f)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def _key(self, item: RecordingItem) -> str:
        return f"{item.text}_{item.variant}"

    def is_done(self, item: RecordingItem) -> bool:
        return self._key(item) in self._data["done"]

    def is_skipped(self, item: RecordingItem) -> bool:
        return self._key(item) in self._data["skipped"]

    def mark_done(self, item: RecordingItem, path: str) -> None:
        self._data["done"][self._key(item)] = {
            "path": path,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        # Remove from skipped if it was there
        self._data["skipped"].pop(self._key(item), None)
        self._save()

    def mark_skipped(self, item: RecordingItem) -> None:
        self._data["skipped"][self._key(item)] = {
            "skipped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save()

    def summary(self, total: int) -> dict:
        return {
            "total": total,
            "done": len(self._data["done"]),
            "skipped": len(self._data["skipped"]),
            "remaining": total - len(self._data["done"]) - len(self._data["skipped"]),
        }


# ── Audio device detection ───────────────────────────────────────────────


@dataclass
class AudioDevice:
    index: int
    name: str


def detect_audio_devices() -> list[AudioDevice]:
    """Parse ffmpeg avfoundation device list for audio input devices."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return []

    # ffmpeg prints device list to stderr
    output = result.stderr
    devices: list[AudioDevice] = []
    in_audio = False

    for line in output.splitlines():
        if "AVFoundation audio devices:" in line:
            in_audio = True
            continue
        if in_audio:
            # Lines look like: [AVFoundation ...] [0] MacBook Pro Microphone
            m = re.search(r"\[(\d+)\]\s+(.+)", line)
            if m:
                devices.append(AudioDevice(index=int(m.group(1)), name=m.group(2)))
            elif not line.strip():
                break

    return devices


# ── Recording / normalization / playback ─────────────────────────────────


def record_chunk(device_index: int, output_path: Path) -> subprocess.Popen:
    """Start ffmpeg recording from microphone. Returns Popen — caller stops it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "avfoundation",
            "-thread_queue_size", "1024",
            "-i", f":{device_index}",
            "-ac", "1",
            "-ar", "44100",
            "-codec:a", "libmp3lame",
            "-q:a", "2",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_recording(proc: subprocess.Popen) -> None:
    """Gracefully stop ffmpeg recording."""
    if proc.stdin:
        proc.stdin.write(b"q")
        proc.stdin.flush()
    proc.wait(timeout=5)


def normalize_audio(path: Path) -> float:
    """Normalize audio file to TARGET_DBFS, trim silence. Returns duration in seconds."""
    from pydub import AudioSegment
    from pydub.effects import normalize

    audio = AudioSegment.from_mp3(str(path))

    # Trim leading/trailing silence (threshold: -40 dBFS)
    def _detect_leading_silence(sound: AudioSegment, threshold: float = -40.0, chunk_size: int = 10) -> int:
        trim_ms = 0
        while trim_ms < len(sound) and sound[trim_ms:trim_ms + chunk_size].dBFS < threshold:
            trim_ms += chunk_size
        return trim_ms

    start_trim = _detect_leading_silence(audio)
    end_trim = _detect_leading_silence(audio.reverse())
    audio = audio[start_trim:len(audio) - end_trim]

    # Normalize to target dBFS
    change_in_dbfs = TARGET_DBFS - audio.dBFS
    audio = audio.apply_gain(change_in_dbfs)

    audio.export(str(path), format="mp3")
    return len(audio) / 1000.0


def denoise_audio(path: Path) -> None:
    """Apply noise reduction + highpass filter using ffmpeg only."""
    tmp = path.with_suffix(".tmp.mp3")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(path),
            "-af", "highpass=f=80,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
            "-codec:a", "libmp3lame", "-q:a", "2",
            str(tmp),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr else "ffmpeg failed")
    tmp.replace(path)


def denoise_directory(directory: Path) -> dict:
    """Apply denoise + normalize to all mp3 files in a directory.

    Returns {processed, errors}.
    """
    processed = 0
    errors: list[str] = []

    files = sorted(directory.glob("*.mp3"))
    total = len(files)

    for i, f in enumerate(files, 1):
        print(f"  [{i}/{total}] {f.name}...", end=" ", flush=True)
        try:
            denoise_audio(f)
            print("done")
            processed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errors.append(f"{f.name}: {e}")

    return {"processed": processed, "errors": errors}


def get_audio_duration(path: Path) -> float:
    """Get duration in seconds using pydub."""
    from pydub import AudioSegment

    audio = AudioSegment.from_mp3(str(path))
    return len(audio) / 1000.0


def play_audio(path: Path) -> None:
    """Play audio file using afplay (macOS)."""
    subprocess.run(["afplay", str(path)], check=True)


# ── Interactive recording session ────────────────────────────────────────


def run_recording_session(
    output_dir: Path,
    progress_path: Path,
    category_filter: str | None = None,
    device_index: int | None = None,
) -> None:
    """Main interactive recording loop."""
    print("=== Atom Audio Recording Studio ===\n")

    # 1. Detect audio devices
    if device_index is None:
        print("Detecting audio devices...")
        devices = detect_audio_devices()
        if not devices:
            print("No audio input devices found. Is ffmpeg installed?", file=sys.stderr)
            raise SystemExit(1)

        for dev in devices:
            print(f"  [{dev.index}] {dev.name}")

        if len(devices) == 1:
            device_index = devices[0].index
            print(f"Using: {devices[0].name}")
        else:
            try:
                choice = input(f"Select device [0]: ").strip()
                device_index = int(choice) if choice else 0
            except (ValueError, EOFError):
                device_index = 0
    print()

    # 2. Load plan + progress
    plan = generate_recording_plan(category_filter)
    progress = RecordingProgress(progress_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out already-done items
    remaining = [item for item in plan if not progress.is_done(item)]
    total = len(plan)
    s = progress.summary(total)
    print(f"Recording plan: {total} recordings, {s['done']} done, {s['skipped']} skipped")
    print()

    if not remaining:
        print("All recordings complete!")
        return

    print("Controls: ENTER=start/stop  p=play  a=accept  r=redo  s=skip  q=quit\n")

    # Handle Ctrl+C gracefully
    interrupted = False

    def _handle_sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\nInterrupted — saving progress...")

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        _recording_loop(remaining, plan, progress, output_dir, device_index)
    finally:
        signal.signal(signal.SIGINT, original_handler)

    s = progress.summary(total)
    print(f"\nSession summary: {s['done']}/{total} done, {s['skipped']} skipped, {s['remaining']} remaining")


def _recording_loop(
    remaining: list[RecordingItem],
    plan: list[RecordingItem],
    progress: RecordingProgress,
    output_dir: Path,
    device_index: int,
) -> None:
    """Inner loop for recording items one by one."""
    total = len(plan)
    done_offset = total - len(remaining)

    for i, item in enumerate(remaining):
        position = done_offset + i + 1

        # Count total variants for this text
        variants_for_text = [it for it in plan if it.text == item.text]
        variant_of = len(variants_for_text)

        print("─" * 40)
        print(f"[{position}/{total}] {item.category}")
        print(f"  >> {item.text} <<   (variant {item.variant}/{variant_of}, reuse: {item.reuse_count})")
        print()

        output_path = output_dir / item.filename

        while True:
            try:
                cmd = input("Press ENTER to start recording... ").strip().lower()
            except EOFError:
                return

            if cmd == "q":
                return
            if cmd == "s":
                progress.mark_skipped(item)
                print("  Skipped.\n")
                break
            if cmd != "":
                continue

            # Record
            print("\033[91m● RECORDING...\033[0m press ENTER to stop")
            start_time = time.time()
            proc = record_chunk(device_index, output_path)

            try:
                input()
            except EOFError:
                pass

            elapsed = time.time() - start_time
            stop_recording(proc)

            # Check minimum duration
            if elapsed < MIN_DURATION_MS / 1000:
                print(f"  Too short ({elapsed:.2f}s) — misfire, try again.\n")
                if output_path.exists():
                    output_path.unlink()
                continue

            # Normalize
            try:
                duration = normalize_audio(output_path)
                print(f"  Recorded: {duration:.2f}s, normalized to {TARGET_DBFS} dBFS\n")
            except Exception as e:
                print(f"  Normalization error: {e}. Raw file kept.\n")
                duration = elapsed

            # Review loop
            accepted = _review_loop(output_path, item, progress, output_dir)
            if accepted is None:
                # quit
                return
            if accepted:
                break
            # redo — loop back to record again


def _review_loop(
    output_path: Path,
    item: RecordingItem,
    progress: RecordingProgress,
    output_dir: Path,
) -> bool | None:
    """Post-recording review. Returns True=accepted, False=redo, None=quit."""
    while True:
        try:
            cmd = input("  [p]lay [a]ccept [r]edo [s]kip [q]uit > ").strip().lower()
        except EOFError:
            return None

        if cmd == "p":
            try:
                play_audio(output_path)
            except Exception as e:
                print(f"  Playback error: {e}")
        elif cmd == "a":
            progress.mark_done(item, str(output_path))
            print(f"  Saved: {output_path.relative_to(output_dir.parent) if output_dir.parent in output_path.parents else output_path}\n")
            return True
        elif cmd == "r":
            if output_path.exists():
                output_path.unlink()
            print("  Redoing...\n")
            return False
        elif cmd == "s":
            progress.mark_skipped(item)
            if output_path.exists():
                output_path.unlink()
            print("  Skipped.\n")
            return True  # move on
        elif cmd == "q":
            return None
