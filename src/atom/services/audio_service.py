"""AudioService — manage audio chunks for combo assembly."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import AudioChunk

_DICT_PATH = Path(__file__).parent.parent / "data" / "combo_dictionary.json"
_CHUNKS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "audio" / "chunks"


def _load_dict() -> dict:
    with open(_DICT_PATH) as f:
        return json.load(f)


def generate_checklist() -> list[dict]:
    """Analyze combo dictionary and return recording checklist.

    Returns list of {text, reuse_count, suggested_takes} sorted by reuse count.
    """
    data = _load_dict()
    assembly = {
        k: v for k, v in data.get("combo_assembly", {}).items()
        if not k.startswith("_")
    }

    # Count how many combos use each chunk
    chunk_usage: Counter[str] = Counter()
    for chunk_list in assembly.values():
        for chunk_text in chunk_list:
            chunk_usage[chunk_text] += 1

    # All unique chunks from definition
    all_chunks = (
        data["chunks"]["strike_atoms"]
        + data["chunks"]["strike_phrases"]
        + data["chunks"]["defense"]
    )

    checklist = []
    for text in all_chunks:
        count = chunk_usage.get(text, 0)
        takes = 3 if count >= 6 else 2 if count >= 4 else 1
        checklist.append({
            "text": text,
            "reuse_count": count,
            "suggested_takes": takes,
        })

    # Add cues
    for cue in data.get("cues", []):
        checklist.append({
            "text": cue["call"],
            "reuse_count": 0,
            "suggested_takes": 1,
        })

    # Add round intros
    for r in range(1, 13):
        checklist.append({
            "text": f"{r}라운드 시작합니다",
            "reuse_count": 0,
            "suggested_takes": 1,
        })

    # Sort: most-reused chunks first
    checklist.sort(key=lambda x: -x["reuse_count"])
    return checklist


async def import_chunks(directory: Path, session: AsyncSession) -> dict:
    """Import audio files from a directory into AudioChunk table.

    Expected filename format: {text}_{variant}.mp3  (e.g. 원투_1.mp3)
    Or: {text}.mp3 (variant defaults to 1)

    Returns {imported, skipped, errors}.
    """
    _CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    imported = 0
    skipped = 0
    errors: list[str] = []

    for audio_file in sorted(directory.glob("*.mp3")):
        stem = audio_file.stem
        # Parse filename: text_variant or just text
        if "_" in stem and stem.rsplit("_", 1)[1].isdigit():
            text, variant_str = stem.rsplit("_", 1)
            variant = int(variant_str)
        else:
            text = stem
            variant = 1

        # Check if already exists
        result = await session.execute(
            select(AudioChunk).where(
                AudioChunk.text == text,
                AudioChunk.variant == variant,
            )
        )
        if result.scalar_one_or_none():
            skipped += 1
            continue

        # Copy to chunks directory
        dest = _CHUNKS_DIR / audio_file.name
        try:
            shutil.copy2(audio_file, dest)
        except Exception as e:
            errors.append(f"{audio_file.name}: {e}")
            continue

        # Get duration (basic: use mutagen if available, else 0)
        duration_ms = 0
        try:
            from mutagen.mp3 import MP3
            audio = MP3(str(dest))
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


async def validate_chunks(session: AsyncSession) -> dict:
    """Verify every combo can be assembled from available AudioChunks.

    Returns {total_combos, covered, missing: [{combo, missing_chunks}]}.
    """
    data = _load_dict()
    assembly = {
        k: v for k, v in data.get("combo_assembly", {}).items()
        if not k.startswith("_")
    }

    # Get all available chunk texts
    result = await session.execute(select(AudioChunk.text).distinct())
    available = {r[0] for r in result.all()}

    missing_combos = []
    for combo_text, chunk_texts in assembly.items():
        missing = [ct for ct in chunk_texts if ct not in available]
        if missing:
            missing_combos.append({"combo": combo_text, "missing_chunks": missing})

    return {
        "total_combos": len(assembly),
        "covered": len(assembly) - len(missing_combos),
        "missing": missing_combos,
    }
