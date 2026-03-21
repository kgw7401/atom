"""TemplateService — pick a template, shuffle segments, resolve audio chunks."""

from __future__ import annotations

import json
import random
from pathlib import Path

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from atom.models.tables import AudioChunk, DrillPlan, SessionTemplate

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
        """Shuffle segments per round, resolve audio chunks.

        - intro/outro are pinned (not shuffled)
        - segments are shuffled and scaled to round duration
        - Round number injected into intro text
        """
        assembly = _load_assembly()

        intro_texts = template.segments_json.get("intro", [])
        segment_texts = template.segments_json.get("segments", [])
        outro_texts = template.segments_json.get("outro", [])

        # Scale segment count to round duration (~3.5s per segment avg)
        est_sec_per_seg = 3.5
        target_count = max(5, round(round_duration_sec / est_sec_per_seg))
        target_count = min(target_count, len(segment_texts))

        rounds_list = []
        for r in range(1, rounds + 1):
            shuffled = list(segment_texts)
            random.shuffle(shuffled)
            selected = shuffled[:target_count]

            # Inject round number into intro
            round_intro = [
                t.replace("시작합니다", f"{r}라운드 시작합니다")
                for t in intro_texts
            ]
            all_texts = round_intro + selected + outro_texts

            # Resolve audio chunks per segment
            round_segments = []
            for text in all_texts:
                chunks = await self._resolve_chunks(text, assembly)
                round_segments.append({"text": text, "chunks": chunks})

            rounds_list.append({"round": r, "segments": round_segments})

        return {"rounds": rounds_list}

    async def _resolve_chunks(self, text: str, assembly: dict) -> list[dict]:
        """Resolve a segment text into audio chunks."""
        chunk_texts = assembly.get(text)
        if not chunk_texts:
            # Cues, intro, outro — the text itself is one chunk
            chunk_texts = [text]

        chunks = []
        for ct in chunk_texts:
            result = await self.session.execute(
                select(AudioChunk).where(AudioChunk.text == ct)
            )
            available = list(result.scalars().all())
            if available:
                chosen = random.choice(available)
                chunks.append({
                    "text": ct,
                    "clip_url": chosen.audio_path,
                    "duration_ms": chosen.duration_ms,
                })
            else:
                chunks.append({
                    "text": ct,
                    "clip_url": "",
                    "duration_ms": 0,
                })
        return chunks
