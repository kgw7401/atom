"""TemplateService — pick a template, build randomized round plans."""

from __future__ import annotations

import asyncio
import io
import json
import random
from pathlib import Path

from pydub import AudioSegment
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import AudioChunk, DrillPlan, SessionTemplate

AUDIO_DIR = Path("data/audio")

_DICT_PATH = Path(__file__).parent.parent / "data" / "combo_dictionary.json"
_assembly_cache: dict | None = None


def _load_assembly() -> dict:
    """Load combo assembly mapping (cached)."""
    global _assembly_cache
    if _assembly_cache is None:
        with open(_DICT_PATH) as f:
            data = json.load(f)
        _assembly_cache = {
            k: v for k, v in data.get("combo_assembly", {}).items()
            if not k.startswith("_")
        }
    return _assembly_cache


class TemplateService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def pick_template(self, level: str) -> SessionTemplate:
        """Pick a random template for the level, avoiding the last 3 used."""
        recent_result = await self.session.execute(
            select(DrillPlan.template_id)
            .where(DrillPlan.template_id.isnot(None))
            .order_by(desc(DrillPlan.created_at))
            .limit(3)
        )
        recent_ids = [r[0] for r in recent_result.all()]

        query = select(SessionTemplate).where(SessionTemplate.level == level)
        if recent_ids:
            query = query.where(SessionTemplate.id.notin_(recent_ids))

        result = await self.session.execute(query)
        candidates = list(result.scalars().all())

        if not candidates:
            result = await self.session.execute(
                select(SessionTemplate).where(SessionTemplate.level == level)
            )
            candidates = list(result.scalars().all())

        if not candidates:
            raise ValueError(f"No templates found for level '{level}'")

        return random.choice(candidates)

    async def build_round_plan(
        self,
        template: SessionTemplate,
        rounds: int,
        round_duration_sec: int,
    ) -> dict:
        """Build round plan with weighted sampling.

        - Combos sampled from combo_pool using weights
        - Audio chunks resolved per segment
        """
        assembly = _load_assembly()

        pool = template.segments_json["combo_pool"]

        combo_calls = [item["call"] for item in pool]
        combo_weights = [item["weight"] for item in pool]

        total_target = max(5, round(round_duration_sec / 3.5))

        rounds_list = []
        for r in range(1, rounds + 1):
            sequence = random.choices(combo_calls, weights=combo_weights, k=total_target)

            round_segments = []
            for text in sequence:
                chunks = await self._resolve_chunks(text, assembly)
                round_segments.append({"text": text, "chunks": chunks})

            rounds_list.append({"round": r, "segments": round_segments})

        return {"rounds": rounds_list}

    async def _resolve_chunks(self, text: str, assembly: dict) -> list[dict]:
        """Resolve a segment text into a single audio clip.

        Full combo audio is generated as one file per combo/cue,
        so we look up the segment text directly in AudioChunk.
        """
        result = await self.session.execute(
            select(AudioChunk).where(AudioChunk.text == text)
        )
        available = list(result.scalars().all())
        if available:
            chosen = random.choice(available)
            return [{
                "text": text,
                "clip_url": f"/audio/{chosen.audio_path}",
                "duration_ms": chosen.duration_ms,
            }]
        return [{
            "text": text,
            "clip_url": "",
            "duration_ms": 0,
        }]

    async def assemble_round_audio(
        self, plan: dict, plan_id: str, pause_ms: int = 1500,
    ) -> dict:
        """Concatenate segment MP3s into one continuous MP3 per round.

        Returns enriched plan with `audio_url` and `timestamps` per round.
        """
        plan_dir = AUDIO_DIR / plan_id
        plan_dir.mkdir(parents=True, exist_ok=True)

        enriched_rounds = []
        for rnd in plan["rounds"]:
            round_num = rnd["round"]
            segments = rnd["segments"]

            final_audio = AudioSegment.empty()
            timestamps: list[dict] = []
            cursor_ms = 0

            for i, seg in enumerate(segments):
                # Find the first chunk with a clip_url
                clip_url = ""
                for chunk in seg.get("chunks", []):
                    if chunk.get("clip_url"):
                        clip_url = chunk["clip_url"]
                        break

                if not clip_url:
                    # No audio file — skip this segment in the assembled audio
                    continue

                # clip_url is e.g. "/audio/chunks/원투_1.mp3" → "data/audio/chunks/원투_1.mp3"
                file_path = AUDIO_DIR / clip_url.removeprefix("/audio/")
                if not file_path.exists():
                    continue

                seg_audio = await asyncio.to_thread(
                    AudioSegment.from_mp3, str(file_path),
                )

                timestamps.append({
                    "start_ms": cursor_ms,
                    "end_ms": cursor_ms + len(seg_audio),
                    "text": seg["text"],
                })

                final_audio += seg_audio
                cursor_ms += len(seg_audio)

                # Add silence between segments (not after last)
                if i < len(segments) - 1:
                    final_audio += AudioSegment.silent(duration=pause_ms)
                    cursor_ms += pause_ms

            if len(final_audio) == 0:
                enriched_rounds.append(rnd)
                continue

            # Export per-round MP3
            round_path = plan_dir / f"round_{round_num}.mp3"
            buf = io.BytesIO()
            await asyncio.to_thread(
                final_audio.export, buf, format="mp3", bitrate="128k",
            )
            round_path.write_bytes(buf.getvalue())

            audio_url = f"/audio/{plan_id}/round_{round_num}.mp3"
            enriched_rounds.append({
                **rnd,
                "audio_url": audio_url,
                "timestamps": timestamps,
            })

        return {**plan, "rounds": enriched_rounds}
