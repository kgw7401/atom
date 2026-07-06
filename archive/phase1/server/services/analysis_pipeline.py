"""Analysis pipeline orchestrator — ties together inference, verification, and reporting.

Called as a background task after video upload completes.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from server.models.db import ActionStat, Session, SessionReport, Script, async_session
from server.services.inference_service import InferenceService
from server.services.report_generator import generate_report
from server.services.verification_engine import verify_session


async def run_analysis(session_id: str, inference: InferenceService) -> None:
    """Run full analysis pipeline for an uploaded session video.

    Steps:
    1. Load session + script from DB
    2. Classify video with InferenceService
    3. Segment actions
    4. Verify against script instructions
    5. Generate report
    6. Save report + action stats to DB
    7. Update session status
    """
    async with async_session() as db:
        # Load session
        session = await db.get(Session, session_id)
        if not session or not session.video_path:
            return

        # Update status
        session.status = "analyzing"
        session.analysis_progress = 0.1
        await db.commit()

        # Load script
        script = await db.get(Script, session.script_id)
        if not script:
            session.status = "failed"
            await db.commit()
            return

        instructions = json.loads(script.instructions_json)

        try:
            # Classify video
            session.analysis_progress = 0.2
            await db.commit()

            detections = inference.classify_video(session.video_path)

            session.analysis_progress = 0.6
            await db.commit()

            # Segment actions
            segments = inference.segment_actions(detections)

            # Verify against script
            session.analysis_progress = 0.8
            await db.commit()

            verifications = verify_session(instructions, segments, level=script.level)

            # Generate report
            report = generate_report(uuid.UUID(session_id), verifications)

            # Save report
            db_report = SessionReport(
                session_id=session_id,
                overall_score=report.overall_score,
                summary_json=report.summary.model_dump_json(),
                instruction_results_json=json.dumps(
                    [r.model_dump() for r in report.instructions]
                ),
                coaching_json=report.coaching.model_dump_json(),
            )
            db.add(db_report)

            # Save per-action stats for digital twin
            await _save_action_stats(db, session.user_id, session_id, verifications)

            # Update session
            session.status = "completed"
            session.analysis_progress = 1.0
            session.completed_at = datetime.now(timezone.utc)
            await db.commit()

        except Exception as e:
            session.status = "failed"
            await db.commit()
            raise


async def _save_action_stats(db, user_id: str, session_id: str, verifications) -> None:
    """Aggregate per-action stats from verification results."""
    action_data: dict[str, dict] = {}

    for v in verifications:
        if v.type != "attack":
            continue
        for action in v.expected_actions:
            if action not in action_data:
                action_data[action] = {
                    "attempts": 0, "successes": 0,
                    "reactions": [], "scores": [],
                }
            action_data[action]["attempts"] += 1
            if action in v.detected_actions:
                action_data[action]["successes"] += 1
            if v.reaction_time is not None:
                action_data[action]["reactions"].append(v.reaction_time)
            action_data[action]["scores"].append(v.score)

    for action, data in action_data.items():
        reactions = data["reactions"]
        scores = data["scores"]
        stat = ActionStat(
            user_id=user_id,
            session_id=session_id,
            action=action,
            attempts=data["attempts"],
            successes=data["successes"],
            avg_reaction_time=(sum(reactions) / len(reactions)) if reactions else None,
            avg_score=(sum(scores) / len(scores)) if scores else None,
        )
        db.add(stat)
