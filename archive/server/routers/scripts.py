"""Script generation endpoint."""

from __future__ import annotations

import json

from fastapi import APIRouter

from server.models.db import Script, async_session
from server.models.schemas import ScriptGenerateRequest, ScriptResponse
from server.services.script_generator import generate_script

router = APIRouter(tags=["scripts"])


@router.post("/scripts/generate", response_model=ScriptResponse)
async def create_script(req: ScriptGenerateRequest) -> ScriptResponse:
    result = generate_script(level=req.level, duration_seconds=req.duration_seconds)

    # Persist to DB so analysis pipeline can load it later
    async with async_session() as db:
        script = Script(
            id=str(result.script_id),
            level=result.level,
            duration_seconds=req.duration_seconds,
            instructions_json=json.dumps(
                [inst.model_dump() for inst in result.instructions]
            ),
        )
        db.add(script)
        await db.commit()

    return result
