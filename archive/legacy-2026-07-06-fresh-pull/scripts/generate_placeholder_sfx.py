"""Generate placeholder impact SFX files for the Atom boxing app.

Creates short noise-burst MP3 files (~50-150ms) that serve as placeholders
until real impact sounds are recorded/sourced.

Requires: pydub, ffmpeg
Usage: python scripts/generate_placeholder_sfx.py
"""

import random
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import WhiteNoise

OUTPUT_DIR = Path(__file__).parent.parent / "mobile" / "assets" / "sfx" / "impact"

# Strike types: 3 variants each
STRIKES = {
    "jab": {"duration_ms": 60, "volume_db": -18},
    "cross": {"duration_ms": 80, "volume_db": -16},
    "hook": {"duration_ms": 100, "volume_db": -14},
    "uppercut": {"duration_ms": 120, "volume_db": -13},
    "body": {"duration_ms": 90, "volume_db": -15},
}

# Defense types: 1 variant each
DEFENSE = {
    "slip": {"duration_ms": 50, "volume_db": -22},
    "duck": {"duration_ms": 70, "volume_db": -22},
    "weave": {"duration_ms": 80, "volume_db": -21},
    "back": {"duration_ms": 60, "volume_db": -23},
}

# Chain rattle: 2 variants
CHAIN = {
    "chain": {"duration_ms": 150, "volume_db": -20},
}


def make_impact(duration_ms: int, volume_db: int, seed: int) -> AudioSegment:
    """Generate a short noise burst with fade-out to simulate impact."""
    random.seed(seed)
    noise = WhiteNoise().to_audio_segment(duration=duration_ms + 40)
    # Apply volume and quick fade-out for punch feel
    noise = noise + volume_db
    noise = noise.fade_in(5).fade_out(duration_ms // 2)
    # Trim to target duration
    return noise[:duration_ms]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    # Strikes: 3 variants each
    for name, cfg in STRIKES.items():
        for variant in range(1, 4):
            audio = make_impact(cfg["duration_ms"], cfg["volume_db"], seed=hash(f"{name}_{variant}"))
            path = OUTPUT_DIR / f"{name}_{variant}.mp3"
            audio.export(str(path), format="mp3", bitrate="64k")
            count += 1
            print(f"  {path.name} ({len(audio)}ms)")

    # Defense: 1 variant each
    for name, cfg in DEFENSE.items():
        audio = make_impact(cfg["duration_ms"], cfg["volume_db"], seed=hash(name))
        path = OUTPUT_DIR / f"{name}.mp3"
        audio.export(str(path), format="mp3", bitrate="64k")
        count += 1
        print(f"  {path.name} ({len(audio)}ms)")

    # Chain rattle: 2 variants
    for variant in range(1, 3):
        cfg = CHAIN["chain"]
        audio = make_impact(cfg["duration_ms"], cfg["volume_db"], seed=hash(f"chain_{variant}"))
        path = OUTPUT_DIR / f"chain_{variant}.mp3"
        audio.export(str(path), format="mp3", bitrate="64k")
        count += 1
        print(f"  {path.name} ({len(audio)}ms)")

    print(f"\nGenerated {count} placeholder SFX files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
