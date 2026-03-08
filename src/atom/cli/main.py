"""Atom CLI — entry point for all commands."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import click

from atom.models.base import DEFAULT_DB_DIR, get_db_path, init_db, async_session
from atom.seed import seed_all
from atom.services.combo_service import (
    ComboError,
    ComboService,
)
from atom.services.session_service import PlanValidationError, SessionService

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


# ── combo ────────────────────────────────────────────────────────────────

@cli.group()
def combo() -> None:
    """Manage combinations."""
    pass


@combo.command("list")
@click.option("--complexity", type=int, default=None, help="Filter by complexity (action count)")
def combo_list(complexity: int | None) -> None:
    """List all combinations."""
    run(_combo_list(complexity))


async def _combo_list(complexity: int | None) -> None:
    async with async_session() as session:
        svc = ComboService(session)
        combos = await svc.list(complexity=complexity)

    if not combos:
        click.echo("No combinations found.")
        return

    # Header
    click.echo(f"{'Name':<16} {'Actions':<40} {'Cx':>2}  {'Type':<6}")
    click.echo("-" * 68)
    for c in combos:
        actions_str = " → ".join(c.actions)
        kind = "system" if c.is_system else "user"
        click.echo(f"{c.display_name:<16} {actions_str:<40} {c.complexity:>2}  {kind:<6}")

    click.echo(f"\n{len(combos)} combo(s)")


@combo.command("add")
@click.argument("display_name")
@click.argument("actions", nargs=-1, required=True)
def combo_add(display_name: str, actions: tuple[str, ...]) -> None:
    """Create a new combo. Example: atom combo add "원투훅" jab cross lead_hook"""
    run(_combo_add(display_name, list(actions)))


async def _combo_add(display_name: str, actions: list[str]) -> None:
    try:
        async with async_session() as session:
            svc = ComboService(session)
            combo = await svc.create(display_name, actions)
        click.echo(f"Created: {combo.display_name} ({' → '.join(combo.actions)}) [id: {combo.id[:8]}]")
    except ComboError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@combo.command("show")
@click.argument("id_or_name")
def combo_show(id_or_name: str) -> None:
    """Show combo details."""
    run(_combo_show(id_or_name))


async def _combo_show(id_or_name: str) -> None:
    try:
        async with async_session() as session:
            svc = ComboService(session)
            c = await svc.get(id_or_name)
        click.echo(f"Name:       {c.display_name}")
        click.echo(f"Actions:    {' → '.join(c.actions)}")
        click.echo(f"Complexity: {c.complexity}")
        click.echo(f"Type:       {'system' if c.is_system else 'user'}")
        click.echo(f"ID:         {c.id}")
    except ComboError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@combo.command("edit")
@click.argument("id_or_name")
@click.option("--display-name", default=None, help="New display name")
@click.option("--actions", default=None, help="New actions (space-separated)")
def combo_edit(id_or_name: str, display_name: str | None, actions: str | None) -> None:
    """Edit a user combo."""
    kwargs = {}
    if display_name is not None:
        kwargs["display_name"] = display_name
    if actions is not None:
        kwargs["actions"] = actions.split()
    if not kwargs:
        click.echo("Nothing to update. Use --display-name or --actions.")
        return
    run(_combo_edit(id_or_name, kwargs))


async def _combo_edit(id_or_name: str, kwargs: dict) -> None:
    try:
        async with async_session() as session:
            svc = ComboService(session)
            combo = await svc.update(id_or_name, **kwargs)
        click.echo(f"Updated: {combo.display_name} ({' → '.join(combo.actions)})")
    except ComboError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@combo.command("delete")
@click.argument("id_or_name")
@click.confirmation_option(prompt="Are you sure you want to delete this combo?")
def combo_delete(id_or_name: str) -> None:
    """Delete a user combo."""
    run(_combo_delete(id_or_name))


async def _combo_delete(id_or_name: str) -> None:
    try:
        async with async_session() as session:
            svc = ComboService(session)
            combo = await svc.get(id_or_name)
            name = combo.display_name
            await svc.delete(id_or_name)
        click.echo(f"Deleted: {name}")
    except ComboError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


# ── session ──────────────────────────────────────────────────────────────

@cli.group()
def session() -> None:
    """Manage drill sessions."""
    pass


@session.command("start")
@click.option(
    "--template", "-t",
    type=click.Choice(["fundamentals", "combos", "mixed"]),
    default=None,
    help="Session template",
)
@click.option("--prompt", "-p", default=None, help="Natural language customization")
@click.option("--no-llm", is_flag=True, help="Use fallback planner (no LLM)")
@click.option("--no-tts", is_flag=True, help="Disable voice output")
@click.option("--voice", default=None, help="TTS voice name (see: atom voices)")
def session_start(
    template: str | None, prompt: str | None, no_llm: bool, no_tts: bool, voice: str | None
) -> None:
    """Generate a drill plan, preview it, and run the session."""
    run(_session_start(template, prompt, no_llm, no_tts, voice))


async def _session_start(
    template: str | None, prompt: str | None, no_llm: bool, no_tts: bool, voice: str | None = None
) -> None:
    from atom.services.session_engine import SessionEngine

    # Interactive template selection if not provided
    if template is None:
        click.echo("Select a template:")
        click.echo("  1) fundamentals  — 기본기 (singles & doubles, slow pace)")
        click.echo("  2) combos        — 콤비네이션 (3-4 action sequences)")
        click.echo("  3) mixed         — 종합 (offense + defense, varied)")
        choice = click.prompt("Template", type=click.IntRange(1, 3))
        template = ["fundamentals", "combos", "mixed"][choice - 1]

    # Optional prompt
    if prompt is None and not no_llm:
        prompt = click.prompt(
            "Any preferences? (enter to skip)",
            default="",
            show_default=False,
        )
        if not prompt.strip():
            prompt = None

    click.echo(f"\nGenerating plan: template={template}" +
               (f', prompt="{prompt}"' if prompt else "") + "...")

    # Get LLM client if available
    llm_client = None
    if not no_llm:
        try:
            from atom.services.llm_client import LLMClient
            llm_client = LLMClient()
        except RuntimeError:
            click.echo("(No API key — using fallback planner)")

    try:
        async with async_session() as s:
            svc = SessionService(s)
            result = await svc.generate_plan(template, user_prompt=prompt, llm_client=llm_client)
    except PlanValidationError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    plan = result["plan"]
    plan_id = result["id"]
    click.echo(f"\nPlan generated ({result['llm_model']})")
    click.echo(f"  Focus: {plan.get('focus', template)}")
    click.echo(f"  Duration: ~{plan.get('total_duration_minutes', '?')} min")
    click.echo(f"  Rounds: {len(plan['rounds'])}")

    for rnd in plan["rounds"]:
        n = rnd["round_number"]
        dur = rnd["duration_seconds"]
        rest = rnd["rest_after_seconds"]
        instrs = rnd["instructions"]
        click.echo(f"\n  Round {n} ({dur}s + {rest}s rest) — {len(instrs)} combos:")
        for instr in instrs[:5]:  # Show first 5
            t = instr["timestamp_offset"]
            name = instr["combo_display_name"]
            actions = " → ".join(instr["actions"])
            click.echo(f"    {t:5.1f}s  {name:<12} ({actions})")
        if len(instrs) > 5:
            click.echo(f"    ... and {len(instrs) - 5} more")

    click.echo(f"\nPlan ID: {plan_id}")

    # Ask to run
    if not click.confirm("\nStart session?", default=True):
        click.echo("Session cancelled. Plan saved — run later with: atom session run " + plan_id[:8])
        return

    # Run session
    engine = SessionEngine(
        plan=plan,
        plan_id=plan_id,
        tts_enabled=not no_tts,
        voice=voice or "Yuna",
        on_output=click.echo,
    )

    # Handle Ctrl+C gracefully
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(2, engine.abort)  # SIGINT

    try:
        await engine.run()
    finally:
        loop.remove_signal_handler(2)

    # Save session log
    async with async_session() as s:
        log = await engine.save_log(s)

    click.echo(f"\nSession log saved: {log.id[:8]} (status: {log.status})")


@session.command("history")
@click.option("--limit", "-n", default=10, show_default=True, help="Number of sessions to show")
def session_history(limit: int) -> None:
    """List past drill sessions."""
    run(_session_history(limit))


async def _session_history(limit: int) -> None:
    from atom.services.profile_service import ProfileService
    async with async_session() as s:
        svc = ProfileService(s)
        logs = await svc.list_sessions(limit=limit)

    if not logs:
        click.echo("No sessions found.")
        return

    click.echo(f"{'Date':<20} {'Template':<14} {'Duration':>8} {'Rounds':>8} {'Combos':>7} {'Status':<10}")
    click.echo("-" * 72)
    for log in logs:
        date_str = log.started_at.strftime("%Y-%m-%d %H:%M")
        dur = f"{log.total_duration_sec:.0f}s"
        rounds = f"{log.rounds_completed}/{log.rounds_total}"
        click.echo(
            f"{date_str:<20} {log.template_name:<14} {dur:>8} {rounds:>8} "
            f"{log.combos_delivered:>7} {log.status:<10}"
        )
    click.echo(f"\n{len(logs)} session(s)")


@session.command("show")
@click.argument("session_id")
def session_show(session_id: str) -> None:
    """Show session details and delivery log."""
    run(_session_show(session_id))


async def _session_show(session_id: str) -> None:
    from atom.services.profile_service import ProfileService
    async with async_session() as s:
        svc = ProfileService(s)
        # Support short ID prefix
        logs = await svc.list_sessions(limit=200)
        log = next((l for l in logs if l.id.startswith(session_id)), None)

    if log is None:
        click.echo(f"Session not found: {session_id}", err=True)
        raise SystemExit(1)

    click.echo(f"Session:    {log.id}")
    click.echo(f"Template:   {log.template_name}")
    click.echo(f"Status:     {log.status}")
    click.echo(f"Started:    {log.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if log.completed_at:
        click.echo(f"Completed:  {log.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"Duration:   {log.total_duration_sec:.0f}s")
    click.echo(f"Rounds:     {log.rounds_completed}/{log.rounds_total}")
    click.echo(f"Combos:     {log.combos_delivered}")

    events = (log.delivery_log_json or {}).get("events", [])
    combo_events = [e for e in events if e.get("type") == "combo_called"]
    if combo_events:
        click.echo(f"\nDelivery log ({len(combo_events)} combos called):")
        for e in combo_events[:10]:
            name = e.get("combo_display_name", "?")
            actions = " → ".join(e.get("actions", []))
            ts = e.get("ts", 0)
            rnd = e.get("round", "?")
            click.echo(f"  R{rnd} {ts:6.1f}s  {name:<14} ({actions})")
        if len(combo_events) > 10:
            click.echo(f"  ... and {len(combo_events) - 10} more")


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
    click.echo(f"Session freq:     {p.session_frequency:.1f}/week")
    if p.last_session_at:
        click.echo(f"Last session:     {p.last_session_at.strftime('%Y-%m-%d %H:%M')}")

    pref = p.template_preference_json or {}
    if pref:
        click.echo(f"\nTemplate preference:")
        for t, count in sorted(pref.items(), key=lambda x: -x[1]):
            click.echo(f"  {t:<16} {count}x")

    exposure = p.combo_exposure_json or {}
    if exposure:
        top = sorted(exposure.items(), key=lambda x: -x[1])[:5]
        click.echo(f"\nTop 5 combos drilled:")
        for name, count in top:
            click.echo(f"  {name:<16} {count}x")


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


if __name__ == "__main__":
    cli()
