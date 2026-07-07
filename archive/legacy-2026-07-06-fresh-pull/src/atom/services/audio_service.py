"""AudioService — generate and manage combo/cue audio via edge-tts."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import AudioChunk

_DICT_PATH = Path(__file__).parent.parent / "data" / "combo_dictionary.json"
_AUDIO_DIR = Path(__file__).parent.parent.parent.parent / "data" / "audio"

# ── edge-tts voice settings ──────────────────────────────────────────────

EDGE_VOICE = "ko-KR-InJoonNeural"
EDGE_SETTINGS = {"rate": "+15%", "pitch": "+30Hz", "volume": "+50%"}


def _load_dict() -> dict:
    with open(_DICT_PATH) as f:
        return json.load(f)


def get_all_texts() -> dict:
    """Get all unique combo calls and cues from the dictionary.

    Returns {combos: list[str], cues: list[str], total: int}.
    """
    data = _load_dict()
    combos = []
    for level in ["basic", "intermediate", "advanced"]:
        for c in data["combos"][level]:
            combos.append(c["call"])
    cues = [c["call"] for c in data["cues"]]
    return {"combos": combos, "cues": cues, "total": len(combos) + len(cues)}


async def generate_tts(
    texts: list[str],
    output_dir: Path,
    voice: str | None = None,
    rate: str | None = None,
    pitch: str | None = None,
    volume: str | None = None,
) -> dict:
    """Generate TTS audio for texts using edge-tts.

    Args:
        texts: List of texts to generate.
        output_dir: Directory to save MP3 files.
        voice: edge-tts voice name (default: ko-KR-InJoonNeural).
        rate/pitch/volume: Override default settings.

    Returns:
        {generated: int, skipped: int, errors: list[str]}
    """
    import edge_tts

    v = voice or EDGE_VOICE
    r = rate or EDGE_SETTINGS["rate"]
    p = pitch or EDGE_SETTINGS["pitch"]
    vol = volume or EDGE_SETTINGS["volume"]

    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    errors: list[str] = []

    for text in texts:
        out_path = output_dir / f"{text}.mp3"
        if out_path.exists():
            skipped += 1
            continue

        try:
            comm = edge_tts.Communicate(text, v, rate=r, pitch=p, volume=vol)
            await comm.save(str(out_path))
            size_kb = out_path.stat().st_size / 1024
            generated += 1
            print(f"  {text} → {out_path.name} ({size_kb:.1f} KB)")
        except Exception as e:
            errors.append(f"{text}: {e}")
            print(f"  [ERROR] {text}: {e}")

    return {"generated": generated, "skipped": skipped, "errors": errors}


async def import_audio(directory: Path, session: AsyncSession) -> dict:
    """Import audio files from a directory into AudioChunk table.

    Expected filename: {text}.mp3

    Returns {imported, skipped, errors}.
    """
    imported = 0
    skipped = 0
    errors: list[str] = []

    for audio_file in sorted(directory.glob("*.mp3")):
        text = audio_file.stem
        variant = 1

        result = await session.execute(
            select(AudioChunk).where(
                AudioChunk.text == text,
                AudioChunk.variant == variant,
            )
        )
        if result.scalar_one_or_none():
            skipped += 1
            continue

        duration_ms = 0
        try:
            from mutagen.mp3 import MP3
            audio = MP3(str(audio_file))
            duration_ms = int(audio.info.length * 1000)
        except Exception:
            pass

        chunk = AudioChunk(
            text=text,
            variant=variant,
            audio_path=f"chunks/{audio_file.name}",
            duration_ms=duration_ms,
        )
        session.add(chunk)
        imported += 1

    if imported > 0:
        await session.commit()

    return {"imported": imported, "skipped": skipped, "errors": errors}


async def validate_audio(session: AsyncSession) -> dict:
    """Verify every combo + cue has an audio file.

    Returns {total, covered, missing: list[str]}.
    """
    texts_info = get_all_texts()
    all_texts = texts_info["combos"] + texts_info["cues"]

    result = await session.execute(select(AudioChunk.text).distinct())
    available = {r[0] for r in result.all()}

    missing = [t for t in all_texts if t not in available]

    return {
        "total": len(all_texts),
        "covered": len(all_texts) - len(missing),
        "missing": missing,
    }
