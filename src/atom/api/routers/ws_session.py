"""WebSocket endpoint for real-time session execution."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from atom.models.base import async_session
from atom.services.session_engine import SessionEngine

router = APIRouter(tags=["websocket"])

# Registry of active sessions: plan_id → engine
# Allows reconnecting clients to replay missed events
_active_sessions: dict[str, SessionEngine] = {}


@router.websocket("/ws/sessions/{plan_id}")
async def session_websocket(websocket: WebSocket, plan_id: str):
    """Run a drill session and stream events to the client in real time.

    Protocol (server → client JSON):
      {"type": "session_start", "plan": {...}}
      {"type": "round_start", "round": N, "total": N, "duration_sec": N}
      {"type": "combo_call", "name": "...", "actions": [...], "ts": N}
      {"type": "round_end", "round": N}
      {"type": "rest_start", "round": N, "rest_sec": N}
      {"type": "rest_end", "round": N}
      {"type": "session_end", "status": "...", ...}

    Protocol (client → server JSON):
      {"type": "abort"}
      {"type": "reconnect", "last_ts": N}   # replay events after ts N
    """
    await websocket.accept()

    # Fetch drill plan from DB
    from atom.models.tables import DrillPlan

    async with async_session() as db:
        result = await db.execute(select(DrillPlan).where(DrillPlan.id == plan_id))
        plan_row = result.scalar_one_or_none()

    if plan_row is None:
        await websocket.send_json({"type": "error", "message": f"Plan '{plan_id}' not found"})
        await websocket.close()
        return

    plan = plan_row.plan_json

    # Check if this plan already has an active session (reconnect case)
    existing_engine = _active_sessions.get(plan_id)
    if existing_engine is not None:
        await _handle_reconnect(websocket, existing_engine)
        return

    # New session
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    engine = SessionEngine(
        plan=plan,
        plan_id=plan_id,
        tts_enabled=False,  # client handles TTS via expo-speech
        on_event=queue.put_nowait,
    )
    _active_sessions[plan_id] = engine

    try:
        await websocket.send_json({"type": "session_start", "plan": plan})
        await asyncio.gather(
            _run_engine(engine, queue),
            _stream_events(websocket, engine, queue),
            _listen_client(websocket, engine),
        )
    except WebSocketDisconnect:
        pass  # client disconnected — engine keeps running, stays in registry
    finally:
        if not engine._aborted and engine.state.value == "session_end":
            # Session fully done — remove from registry and save log
            _active_sessions.pop(plan_id, None)
            async with async_session() as db:
                await engine.save_log(db)
        # If still running (client disconnected), keep in registry for reconnect


async def _run_engine(engine: SessionEngine, queue: asyncio.Queue) -> None:
    try:
        await engine.run()
    finally:
        queue.put_nowait(None)  # sentinel — signals stream_events to stop


async def _stream_events(
    websocket: WebSocket,
    engine: SessionEngine,
    queue: asyncio.Queue,
) -> None:
    while True:
        event = await queue.get()
        if event is None:
            break
        try:
            await websocket.send_json(event)
        except Exception:
            engine.abort()
            break


async def _listen_client(websocket: WebSocket, engine: SessionEngine) -> None:
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "abort":
                engine.abort()
                break
    except (WebSocketDisconnect, Exception):
        pass  # client disconnected; engine keeps running


async def _handle_reconnect(websocket: WebSocket, engine: SessionEngine) -> None:
    """Replay missed events for a reconnecting client."""
    try:
        # Wait for client's reconnect message with last known timestamp
        data = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
        last_ts = float(data.get("last_ts", -1)) if data.get("type") == "reconnect" else -1
    except Exception:
        last_ts = -1

    # Replay all events after last_ts
    missed = [e for e in engine.events if e.ts > last_ts]
    for event in missed:
        try:
            await websocket.send_json(event.to_dict())
        except Exception:
            return

    # If session ended already, close
    if engine.state.value == "session_end":
        return

    # Otherwise hook into the running engine's event stream
    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    original_on_event = engine.on_event
    current_ts = engine.events[-1].ts if engine.events else 0

    def combined_on_event(event: dict) -> None:
        if original_on_event:
            original_on_event(event)
        if event.get("ts", 0) > current_ts:
            queue.put_nowait(event)

    engine.on_event = combined_on_event

    try:
        await asyncio.gather(
            _stream_events(websocket, engine, queue),
            _listen_client(websocket, engine),
        )
    except WebSocketDisconnect:
        pass
    finally:
        engine.on_event = original_on_event
