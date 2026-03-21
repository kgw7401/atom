"""Atom CLI — entry point for all commands."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import click

from atom.models.base import DEFAULT_DB_DIR, get_db_path, init_db, async_session
from atom.seed import seed_all
from atom.services.session_service import SessionService

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # atom-a/
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


def run(coro):
    """Run an async coroutine from sync Click command."""
    return asyncio.run(coro)


@click.group()
def cli() -> None:
    """Atom — AI Boxing Coach & Drill Platform."""
    pass


# ── init ─────────────────────────────────────────────────────────────────

@cli.command()
def init() -> None:
    """Initialize database, run migrations, and seed data."""
    run(_init())


async def _init() -> None:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Database: {db_path}")

    if ALEMBIC_INI.exists():
        click.echo("Running migrations...")
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), "upgrade", "head"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            click.echo(f"Migration failed: {result.stderr}", err=True)
            raise SystemExit(1)
        click.echo("Migrations complete.")
    else:
        click.echo("No alembic.ini found, creating tables directly...")
        await init_db()

    click.echo("Seeding data...")
    async with async_session() as session:
        counts = await seed_all(session)

    for entity, count in counts.items():
        if count > 0:
            click.echo(f"  Seeded {count} {entity}")
        else:
            click.echo(f"  {entity}: already seeded")

    click.echo("Atom initialized successfully.")


# ── serve ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default=8000, show_default=True, help="Bind port")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev mode)")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the API server (for mobile app and external access)."""
    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn not installed. Run: pip install uvicorn", err=True)
        raise SystemExit(1)
    click.echo(f"Starting Atom API server at http://{host}:{port}")
    click.echo("Mobile app: set server URL to your local IP (e.g. http://192.168.x.x:8000)")
    click.echo("Press Ctrl+C to stop.\n")
    uvicorn.run("atom.api.app:app", host=host, port=port, reload=reload)


# ── voices ───────────────────────────────────────────────────────────────

@cli.command("voices")
def voices_list() -> None:
    """List available TTS voices for use with --voice."""
    try:
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
        lines = result.stdout.strip().splitlines()
        click.echo(f"{'Voice':<24} {'Language':<12} Sample")
        click.echo("-" * 60)
        for line in sorted(lines):
            sample_start = line.find("# ")
            sample = line[sample_start + 2:] if sample_start != -1 else ""
            prefix = line[:sample_start].strip() if sample_start != -1 else line.strip()
            parts = prefix.split()
            # Language code is like "en_US" — has underscore at index 2
            lang = ""
            if parts and len(parts[-1]) >= 3 and parts[-1][2] == "_":
                lang = parts[-1]
                name = " ".join(parts[:-1])
            else:
                name = " ".join(parts)
            click.echo(f"{name:<24} {lang:<12} {sample[:30]}")
    except FileNotFoundError:
        click.echo("TTS not available on this platform (macOS only).")


# ── session ──────────────────────────────────────────────────────────────

@cli.group()
def session() -> None:
    """Manage drill sessions."""
    pass


@session.command("generate")
@click.option(
    "--level", "-l",
    type=click.Choice(["beginner", "intermediate", "advanced"]),
    default="beginner",
    show_default=True,
    help="Difficulty level",
)
@click.option("--rounds", "-r", default=3, show_default=True, help="Number of rounds")
@click.option("--duration", "-d", default=180, show_default=True, help="Round duration (seconds)")
@click.option("--rest", default=30, show_default=True, help="Rest between rounds (seconds)")
def session_generate(level: str, rounds: int, duration: int, rest: int) -> None:
    """Generate a session plan from templates."""
    run(_session_generate(level, rounds, duration, rest))


async def _session_generate(level: str, rounds: int, duration: int, rest: int) -> None:
    click.echo(f"Generating plan: level={level}, rounds={rounds}, duration={duration}s...")

    try:
        async with async_session() as s:
            svc = SessionService(s)
            result = await svc.generate_plan(
                level=level,
                rounds=rounds,
                round_duration_sec=duration,
                rest_sec=rest,
            )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    plan = result["plan"]
    plan_id = result["id"]
    click.echo(f"\nPlan generated")
    click.echo(f"  Template: {result['template_name']} — {result['template_topic']}")
    click.echo(f"  Audio: {'ready' if result['audio_ready'] else 'not recorded yet'}")
    click.echo(f"  Rounds: {len(plan['rounds'])}")

    for rnd in plan["rounds"]:
        n = rnd["round"]
        segs = rnd["segments"]
        click.echo(f"\n  Round {n} — {len(segs)} segments:")
        for seg in segs[:8]:
            chunks_info = ""
            if seg.get("chunks"):
                n_chunks = len(seg["chunks"])
                if n_chunks > 1:
                    chunks_info = f" ({n_chunks} chunks)"
            click.echo(f"    {seg['text']}{chunks_info}")
        if len(segs) > 8:
            click.echo(f"    ... and {len(segs) - 8} more")

    click.echo(f"\nPlan ID: {plan_id}")


@session.command("history")
@click.option("--limit", "-n", default=10, show_default=True, help="Number of sessions to show")
def session_history(limit: int) -> None:
    """List past drill sessions."""
    run(_session_history(limit))


async def _session_history(limit: int) -> None:
    from sqlalchemy import select, desc
    from atom.models.tables import SessionLog

    async with async_session() as s:
        result = await s.execute(
            select(SessionLog).order_by(desc(SessionLog.started_at)).limit(limit)
        )
        logs = list(result.scalars().all())

    if not logs:
        click.echo("No sessions found.")
        return

    click.echo(f"{'Date':<20} {'Duration':>8} {'Rounds':>8} {'Segments':>9} {'Status':<10}")
    click.echo("-" * 60)
    for log in logs:
        date_str = log.started_at.strftime("%Y-%m-%d %H:%M")
        dur = f"{log.total_duration_sec:.0f}s"
        rounds = f"{log.rounds_completed}/{log.rounds_total}"
        click.echo(
            f"{date_str:<20} {dur:>8} {rounds:>8} "
            f"{log.segments_delivered:>9} {log.status:<10}"
        )
    click.echo(f"\n{len(logs)} session(s)")


# ── profile ───────────────────────────────────────────────────────────

@cli.group()
def profile() -> None:
    """View and update user profile."""
    pass


@profile.command("show")
def profile_show() -> None:
    """Show current user profile and training stats."""
    run(_profile_show())


async def _profile_show() -> None:
    from atom.services.profile_service import ProfileService
    async with async_session() as s:
        svc = ProfileService(s)
        p = await svc.get_profile()

    if p is None:
        click.echo("No profile found. Run `atom init` first.")
        return

    click.echo(f"Experience:       {p.experience_level}")
    click.echo(f"Goal:             {p.goal or '(not set)'}")
    click.echo(f"Total sessions:   {p.total_sessions}")
    click.echo(f"Total training:   {p.total_training_minutes:.1f} min")
    if p.last_session_at:
        click.echo(f"Last session:     {p.last_session_at.strftime('%Y-%m-%d %H:%M')}")


@profile.command("set")
@click.option(
    "--experience",
    type=click.Choice(["beginner", "intermediate", "advanced"]),
    default=None,
)
@click.option("--goal", default=None, help="Training goal (free text)")
def profile_set(experience: str | None, goal: str | None) -> None:
    """Update profile settings."""
    if not experience and not goal:
        click.echo("Nothing to update. Use --experience or --goal.")
        return
    run(_profile_set(experience, goal))


async def _profile_set(experience: str | None, goal: str | None) -> None:
    from atom.services.profile_service import ProfileService
    kwargs = {}
    if experience:
        kwargs["experience_level"] = experience
    if goal:
        kwargs["goal"] = goal
    async with async_session() as s:
        svc = ProfileService(s)
        p = await svc.update_profile(**kwargs)
    click.echo(f"Profile updated.")
    click.echo(f"  Experience: {p.experience_level}")
    click.echo(f"  Goal:       {p.goal or '(not set)'}")


# ── audio ──────────────────────────────────────────────────────────

@cli.group()
def audio() -> None:
    """Generate and manage TTS audio for combos and cues."""
    pass


@audio.command("sample")
@click.option("--rate", default=None, help="Speech rate (e.g. +15%)")
@click.option("--pitch", default=None, help="Pitch (e.g. +30Hz)")
@click.option("--volume", default=None, help="Volume (e.g. +50%)")
@click.option("--voice", default=None, help="edge-tts voice name")
def audio_sample(rate: str | None, pitch: str | None, volume: str | None, voice: str | None) -> None:
    """Generate a few TTS samples to test voice settings."""
    run(_audio_sample(rate, pitch, volume, voice))


async def _audio_sample(
    rate: str | None, pitch: str | None, volume: str | None, voice: str | None,
) -> None:
    from atom.services.audio_service import generate_tts, get_all_texts, EDGE_SETTINGS, EDGE_VOICE

    texts_info = get_all_texts()
    combos = texts_info["combos"]
    cues = texts_info["cues"]

    # Pick samples: short/medium/long combo + 2 cues
    combos_sorted = sorted(combos, key=len)
    sample_texts = [
        combos_sorted[0],
        combos_sorted[len(combos_sorted) // 2],
        combos_sorted[-1],
        cues[0],
        cues[-1],
    ]

    settings = {
        "rate": rate or EDGE_SETTINGS["rate"],
        "pitch": pitch or EDGE_SETTINGS["pitch"],
        "volume": volume or EDGE_SETTINGS["volume"],
    }
    click.echo(f"Voice: {voice or EDGE_VOICE}")
    click.echo(f"Settings: {settings}")
    click.echo()

    output_dir = Path("data/audio/samples")
    result = await generate_tts(
        texts=sample_texts,
        output_dir=output_dir,
        voice=voice,
        rate=settings["rate"],
        pitch=settings["pitch"],
        volume=settings["volume"],
    )

    click.echo(f"\nGenerated: {result['generated']}, Skipped: {result['skipped']}")
    click.echo(f"Files in: {output_dir.resolve()}")


@audio.command("generate")
@click.option("--rate", default=None, help="Speech rate (e.g. +15%)")
@click.option("--pitch", default=None, help="Pitch (e.g. +30Hz)")
@click.option("--volume", default=None, help="Volume (e.g. +50%)")
@click.option("--voice", default=None, help="edge-tts voice name")
@click.option("--no-import", is_flag=True, help="Skip auto-import to DB")
def audio_generate(
    rate: str | None, pitch: str | None, volume: str | None,
    voice: str | None, no_import: bool,
) -> None:
    """Generate TTS audio for all combos and cues."""
    run(_audio_generate(rate, pitch, volume, voice, no_import))


async def _audio_generate(
    rate: str | None, pitch: str | None, volume: str | None,
    voice: str | None, no_import: bool,
) -> None:
    from atom.services.audio_service import generate_tts, get_all_texts, import_audio

    texts_info = get_all_texts()
    all_texts = texts_info["combos"] + texts_info["cues"]

    click.echo(f"Total: {texts_info['total']} ({len(texts_info['combos'])} combos, {len(texts_info['cues'])} cues)")

    output_dir = Path("data/audio/chunks")

    result = await generate_tts(
        texts=all_texts,
        output_dir=output_dir,
        voice=voice,
        rate=rate,
        pitch=pitch,
        volume=volume,
    )

    click.echo(f"\nGenerated: {result['generated']}")
    click.echo(f"Skipped (already exist): {result['skipped']}")
    if result["errors"]:
        click.echo(f"Errors: {len(result['errors'])}")
        for err in result["errors"]:
            click.echo(f"  {err}")

    # Auto-import to DB
    if not no_import and result["generated"] > 0:
        click.echo(f"\nImporting to database...")
        async with async_session() as session:
            imp = await import_audio(output_dir, session)
        click.echo(f"Imported: {imp['imported']}, DB skipped: {imp['skipped']}")


@audio.command("validate")
def audio_validate() -> None:
    """Check that all combos and cues have audio files."""
    run(_audio_validate())


async def _audio_validate() -> None:
    from atom.services.audio_service import validate_audio

    async with async_session() as session:
        result = await validate_audio(session)

    click.echo(f"Total: {result['total']}")
    click.echo(f"Covered: {result['covered']}")

    if result["missing"]:
        click.echo(f"\nMissing audio for {len(result['missing'])} texts:")
        for text in result["missing"][:20]:
            click.echo(f"  {text}")
        if len(result["missing"]) > 20:
            click.echo(f"  ... and {len(result['missing']) - 20} more")
    else:
        click.echo("\nAll combos and cues covered!")


if __name__ == "__main__":
    cli()
