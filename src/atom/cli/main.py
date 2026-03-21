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


# ── chunks ──────────────────────────────────────────────────────────

@cli.group()
def chunks() -> None:
    """Manage audio chunks for combo assembly."""
    pass


@chunks.command("checklist")
def chunks_checklist() -> None:
    """Generate recording checklist (what to record and how many takes)."""
    from atom.services.audio_service import generate_checklist

    items = generate_checklist()
    total_recordings = sum(i["suggested_takes"] for i in items)

    click.echo(f"{'Text':<24} {'Reuse':>5} {'Takes':>5}")
    click.echo("-" * 38)

    for item in items:
        click.echo(f"{item['text']:<24} {item['reuse_count']:>5} {item['suggested_takes']:>5}")

    click.echo(f"\nTotal items: {len(items)}")
    click.echo(f"Total recordings needed: {total_recordings}")


@chunks.command("import")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
def chunks_import(directory: Path) -> None:
    """Import audio files from a directory into the database.

    Expected filenames: {text}_{variant}.mp3 or {text}.mp3
    """
    run(_chunks_import(directory))


async def _chunks_import(directory: Path) -> None:
    from atom.services.audio_service import import_chunks

    async with async_session() as session:
        result = await import_chunks(directory, session)

    click.echo(f"Imported: {result['imported']}")
    click.echo(f"Skipped (already exists): {result['skipped']}")
    if result["errors"]:
        click.echo(f"Errors: {len(result['errors'])}")
        for err in result["errors"]:
            click.echo(f"  {err}")


@chunks.command("validate")
def chunks_validate() -> None:
    """Check that all combos can be assembled from available audio chunks."""
    run(_chunks_validate())


async def _chunks_validate() -> None:
    from atom.services.audio_service import validate_chunks

    async with async_session() as session:
        result = await validate_chunks(session)

    click.echo(f"Total combos: {result['total_combos']}")
    click.echo(f"Covered: {result['covered']}")

    if result["missing"]:
        click.echo(f"\nMissing chunks for {len(result['missing'])} combos:")
        for item in result["missing"][:20]:
            missing_str = ", ".join(item["missing_chunks"])
            click.echo(f"  {item['combo']}: needs [{missing_str}]")
        if len(result["missing"]) > 20:
            click.echo(f"  ... and {len(result['missing']) - 20} more")
    else:
        click.echo("\nAll combos fully covered!")


if __name__ == "__main__":
    cli()
