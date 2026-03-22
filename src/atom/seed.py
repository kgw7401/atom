"""Seed data for Atom — creates default user profile and templates on first run."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import AudioChunk, ProgramDayTemplate, ProgramProgress, SessionTemplate, UserProfile

CHUNKS_DIR = Path("data/audio/chunks")


async def seed_all(session: AsyncSession) -> dict[str, int]:
    """Seed reference data. Returns counts of inserted records."""
    counts = {
        "profile": await _seed_profile(session),
        "templates": await _seed_templates(session),
        "audio_chunks": await _seed_audio_chunks(session),
        "programs": await _seed_programs(session),
    }
    await session.commit()
    return counts


async def _seed_profile(session: AsyncSession) -> int:
    """Ensure a default user profile exists."""
    result = await session.execute(select(UserProfile))
    profile = result.scalar_one_or_none()
    if profile is None:
        session.add(UserProfile(experience_level="beginner", goal=""))
        return 1
    return 0


async def _seed_templates(session: AsyncSession) -> int:
    """Seed session templates if none exist."""
    result = await session.execute(select(func.count()).select_from(SessionTemplate))
    existing = result.scalar() or 0
    if existing > 0:
        return 0

    from atom.seed_templates import TEMPLATES

    count = 0
    for t in TEMPLATES:
        session.add(
            SessionTemplate(
                name=t["name"],
                level=t["level"],
                topic=t["topic"],
                segments_json=t["segments_json"],
            )
        )
        count += 1
    return count


async def _seed_programs(session: AsyncSession) -> int:
    """Seed program day templates if none exist."""
    result = await session.execute(select(func.count()).select_from(ProgramDayTemplate))
    existing = result.scalar() or 0
    if existing > 0:
        return 0

    from atom.seed_programs import get_program_templates

    count = 0
    for t in get_program_templates():
        session.add(ProgramDayTemplate(**t))
        count += 1

    # Ensure a ProgramProgress row exists
    result = await session.execute(select(func.count()).select_from(ProgramProgress))
    if (result.scalar() or 0) == 0:
        session.add(ProgramProgress(level="beginner", week=1, current_day=1))

    return count


async def _seed_audio_chunks(session: AsyncSession) -> int:
    """Scan data/audio/chunks/ and seed AudioChunk rows for any missing files."""
    if not CHUNKS_DIR.exists():
        return 0

    result = await session.execute(select(AudioChunk.text, AudioChunk.variant))
    existing = {(r[0], r[1]) for r in result.all()}

    count = 0
    for mp3 in sorted(CHUNKS_DIR.glob("*.mp3")):
        text = mp3.stem  # e.g. "원투" from "원투.mp3"
        variant = 1
        if (text, variant) in existing:
            continue

        duration_ms = 0
        try:
            from mutagen.mp3 import MP3
            audio = MP3(str(mp3))
            duration_ms = int(audio.info.length * 1000)
        except Exception:
            pass

        session.add(AudioChunk(
            text=text,
            variant=variant,
            audio_path=f"chunks/{mp3.name}",
            duration_ms=duration_ms,
        ))
        count += 1

    return count
