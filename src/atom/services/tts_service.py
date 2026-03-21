"""TTS Service — ElevenLabs per-round audio with natural coaching flow.

Generates one continuous audio per round using convert_with_timestamps,
then inserts controlled pauses between segments for user execution time.
Returns exact timestamps for mobile UI synchronization.

Flow:
  1. Join segment texts into one script (space-separated)
  2. Single convert_with_timestamps call → audio + character-level alignment
  3. Extract each segment's audio slice using alignment positions
  4. Insert controlled silence (pause_sec) between segments
  5. Export final MP3 with precise timestamps

Environment variables:
  ELEVENLABS_API_KEY   — required
  ELEVENLABS_VOICE_ID  — optional (default: Adam multilingual)
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
from pathlib import Path

from pydub import AudioSegment

AUDIO_DIR = Path("data/audio")

# Default voice (Korean-capable, energetic male)
DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # "Adam" — multilingual

# Round-level voice settings (balanced baseline — text content drives emotion)
DEFAULT_STABILITY = 0.45
DEFAULT_SIMILARITY = 0.75
DEFAULT_SPEED = 1.0


class TTSService:
    def __init__(self, api_key: str, voice_id: str | None = None):
        self.api_key = api_key
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)

    async def generate_session_audio(self, plan: dict, plan_id: str) -> dict:
        """Generate per-round concatenated audio with timestamps."""
        plan_dir = AUDIO_DIR / plan_id
        plan_dir.mkdir(parents=True, exist_ok=True)

        enriched_rounds = []

        for rnd in plan["rounds"]:
            round_num = rnd["round"]
            segments = rnd["segments"]

            audio_bytes, timestamps, script = await self._generate_round_audio(
                segments,
            )

            round_path = plan_dir / f"round_{round_num}.mp3"
            round_path.write_bytes(audio_bytes)

            size_kb = len(audio_bytes) / 1024
            duration_sec = sum(
                t["end_ms"] for t in timestamps[-1:]
            ) / 1000 if timestamps else 0
            print(
                f"[TTS] Round {round_num}: {size_kb:.1f} KB, "
                f"~{duration_sec:.0f}s, {len(segments)} segments"
            )

            audio_url = f"/audio/{plan_id}/round_{round_num}.mp3"
            enriched_rounds.append({
                **rnd,
                "script": script,
                "audio_url": audio_url,
                "timestamps": timestamps,
            })

        return {**plan, "rounds": enriched_rounds}

    async def _generate_round_audio(
        self, segments: list[dict],
    ) -> tuple[bytes, list[dict], str]:
        """Generate one continuous audio for a round with controlled pauses.

        Returns (mp3_bytes, timestamps, full_script).
        """
        from elevenlabs import AsyncElevenLabs
        from elevenlabs.types import VoiceSettings

        client = AsyncElevenLabs(api_key=self.api_key)

        # ── 1. Build full script and track segment character positions ──
        texts = [seg["text"] for seg in segments]
        seg_char_ranges: list[tuple[int, int]] = []
        cursor = 0
        for i, text in enumerate(texts):
            seg_char_ranges.append((cursor, cursor + len(text)))
            cursor += len(text)
            if i < len(texts) - 1:
                cursor += 1  # space separator

        full_script = " ".join(texts)

        # ── 2. Single convert_with_timestamps call ─────────────────────
        response = await client.text_to_speech.convert_with_timestamps(
            voice_id=self.voice_id,
            text=full_script,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=DEFAULT_STABILITY,
                similarity_boost=DEFAULT_SIMILARITY,
                style=0.0,
                use_speaker_boost=True,
                speed=DEFAULT_SPEED,
            ),
        )

        # ── 3. Decode audio ────────────────────────────────────────────
        raw_audio = await asyncio.to_thread(
            AudioSegment.from_mp3,
            io.BytesIO(base64.b64decode(response.audio_base_64)),
        )
        total_audio_ms = len(raw_audio)

        # ── 4. Resolve segment audio boundaries from alignment ─────────
        alignment = response.alignment
        char_starts = alignment.character_start_times_seconds
        char_ends = alignment.character_end_times_seconds

        # Verify alignment matches our script length
        use_estimation = len(alignment.characters) != len(full_script)
        if use_estimation:
            print(
                f"[TTS] Alignment mismatch: {len(alignment.characters)} chars "
                f"vs {len(full_script)} script chars — using estimation"
            )

        # ── 5. Build final audio with controlled pauses ────────────────
        final_audio = AudioSegment.empty()
        timestamps: list[dict] = []
        out_cursor_ms = 0

        for i, seg in enumerate(segments):
            char_start, char_end = seg_char_ranges[i]

            if use_estimation:
                # Fallback: linear estimation from character positions
                total_chars = max(len(full_script), 1)
                seg_start_ms = int(char_start / total_chars * total_audio_ms)
                seg_end_ms = int(char_end / total_chars * total_audio_ms)
            else:
                seg_start_ms = int(char_starts[char_start] * 1000)
                seg_end_ms = int(char_ends[char_end - 1] * 1000)

            # Clamp to audio bounds
            seg_start_ms = max(0, min(seg_start_ms, total_audio_ms))
            seg_end_ms = max(seg_start_ms, min(seg_end_ms, total_audio_ms))

            seg_audio = raw_audio[seg_start_ms:seg_end_ms]

            timestamps.append({
                "start_ms": out_cursor_ms,
                "end_ms": out_cursor_ms + len(seg_audio),
                "text": seg["text"],
                "tempo": seg.get("tempo", "medium"),
                "intensity": seg.get("intensity", "medium"),
            })

            final_audio += seg_audio
            out_cursor_ms += len(seg_audio)

            # Add controlled pause between segments (not after last)
            if i < len(segments) - 1:
                pause_ms = int(seg.get("pause_sec", 1.0) * 1000)
                pause_ms = max(300, min(pause_ms, 3000))  # clamp 0.3-3.0s
                final_audio += AudioSegment.silent(duration=pause_ms)
                out_cursor_ms += pause_ms

        # ── 6. Export MP3 ──────────────────────────────────────────────
        buf = io.BytesIO()
        await asyncio.to_thread(
            final_audio.export, buf, format="mp3", bitrate="128k",
        )

        return buf.getvalue(), timestamps, full_script
