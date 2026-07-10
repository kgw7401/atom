#!/usr/bin/env python3
"""Analyze a boxing sparring video with the Gemini Files API.

Usage:
    GEMINI_API_KEY=... python analyze_sparring.py round.mp4 \
      --me "blue headgear and black T-shirt"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google import genai


DEFAULT_MODEL = "gemini-3.5-flash"

ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_summary": {
            "type": "string",
            "description": "A concise Korean summary of the visible sparring context.",
        },
        "meaningful_exchanges": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "description": "Timestamp in MM:SS format.",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "Timestamp in MM:SS format.",
                    },
                    "opponent_trigger": {
                        "type": "string",
                        "description": "What the opponent visibly did to create the situation.",
                    },
                    "my_response": {
                        "type": "string",
                        "description": "What the identified boxer visibly did in response.",
                    },
                    "problem_category": {
                        "type": "string",
                        "enum": [
                            "reaction_delay",
                            "distance_or_entry",
                            "post_defense_inaction",
                            "retreat_pattern",
                            "no_issue",
                            "unclear",
                        ],
                    },
                    "visual_evidence": {
                        "type": "string",
                        "description": "Only directly visible evidence; no inferred punch impact.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "start_time",
                    "end_time",
                    "opponent_trigger",
                    "my_response",
                    "problem_category",
                    "visual_evidence",
                    "confidence",
                ],
            },
        },
        "repeated_pattern": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [
                        "reaction_delay",
                        "distance_or_entry",
                        "post_defense_inaction",
                        "retreat_pattern",
                        "no_clear_pattern",
                    ],
                },
                "description": {"type": "string"},
                "supporting_timestamps": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["category", "description", "supporting_timestamps"],
        },
        "next_round_rule": {
            "type": "string",
            "description": "One short, executable rule for the next sparring round.",
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Occlusion, camera angle, timing, or other reasons the analysis is uncertain.",
        },
    },
    "required": [
        "video_summary",
        "meaningful_exchanges",
        "repeated_pattern",
        "next_round_rule",
        "limitations",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a sparring video to Gemini and save structured feedback as JSON."
    )
    parser.add_argument("video", type=Path, help="Path to a local sparring video.")
    parser.add_argument(
        "--me",
        required=True,
        help='How to identify you in the video, e.g. "blue headgear and black T-shirt".',
    )
    parser.add_argument(
        "--context",
        default="",
        help="Optional context for this round, such as a drill or a specific concern.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Defaults to <video-name>.analysis.json.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Maximum seconds to wait for Gemini to process the uploaded video (default: 300).",
    )
    parser.add_argument(
        "--keep-remote-file",
        action="store_true",
        help="Do not delete the Gemini Files API upload after analysis.",
    )
    return parser.parse_args()


def get_file_state_name(file: Any) -> str:
    state = getattr(file, "state", None)
    if state is None:
        return "UNKNOWN"
    return getattr(state, "name", str(state).rsplit(".", 1)[-1]).upper()


def wait_until_active(client: genai.Client, file: Any, timeout_seconds: int) -> Any:
    deadline = time.monotonic() + timeout_seconds
    current_file = file

    while True:
        state = get_file_state_name(current_file)
        if state == "ACTIVE":
            return current_file
        if state == "FAILED":
            raise RuntimeError("Gemini failed to process this video file.")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Gemini did not finish processing the video within {timeout_seconds} seconds."
            )

        print(f"Video processing state: {state}. Retrying in 5 seconds...", file=sys.stderr)
        time.sleep(5)
        current_file = client.files.get(name=current_file.name)


def build_prompt(me: str, context: str) -> str:
    extra_context = f"\nAdditional round context: {context}" if context else ""
    return f"""
Analyze this boxing sparring video in Korean.

The boxer to analyze is: {me}.{extra_context}

The goal is to identify decision-making failures under pressure, not to give generic form advice.
Use only visible evidence. Do not state that a punch landed, that someone was hurt, or that intent was known unless it is visually unambiguous. When the camera angle, occlusion, or video sampling prevents a reliable conclusion, choose `unclear`, lower confidence, and record the limitation.

Find at most eight meaningful exchanges. Focus on these categories:
- reaction_delay: the boxer visibly reacts late or freezes after an opponent action.
- distance_or_entry: the boxer enters or remains at an unhelpful visible distance.
- post_defense_inaction: after visibly defending, the boxer has no clear follow-up action such as moving, returning, or clinching.
- retreat_pattern: under pressure, the boxer repeatedly retreats in a straight line without a visible exit decision.
- no_issue: a meaningful exchange without one of the above problems.
- unclear: the moment cannot be judged from the video.

Return one repeated pattern only when at least two exchanges support it. End with one short, specific, executable rule for the next sparring round. The rule must describe an observable action, not a mindset.
""".strip()


def default_output_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}.analysis.json")


def main() -> int:
    args = parse_args()

    if not args.video.is_file():
        print(f"Video file not found: {args.video}", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("--timeout must be greater than zero.", file=sys.stderr)
        return 2

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set.", file=sys.stderr)
        return 2

    output_path = args.output or default_output_path(args.video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)
    uploaded_file = None

    try:
        print(f"Uploading {args.video.name}...", file=sys.stderr)
        uploaded_file = client.files.upload(file=str(args.video))
        active_file = wait_until_active(client, uploaded_file, args.timeout)

        print(f"Analyzing with {args.model}...", file=sys.stderr)
        response = client.interactions.create(
            model=args.model,
            input=[
                {
                    "type": "video",
                    "uri": active_file.uri,
                    "mime_type": active_file.mime_type,
                },
                {"type": "text", "text": build_prompt(args.me, args.context)},
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": ANALYSIS_SCHEMA,
            },
        )

        analysis = json.loads(response.output_text)
        result = {
            "metadata": {
                "created_at": datetime.now(UTC).isoformat(),
                "model": args.model,
                "source_video": str(args.video),
                "source_video_bytes": args.video.stat().st_size,
                "fighter_description": args.me,
                "context": args.context or None,
            },
            "analysis": analysis,
        }
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved analysis to {output_path}", file=sys.stderr)
        return 0
    except json.JSONDecodeError as error:
        print(f"Gemini returned invalid JSON: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1
    finally:
        if uploaded_file is not None and not args.keep_remote_file:
            try:
                client.files.delete(name=uploaded_file.name)
                print("Deleted the uploaded Gemini file.", file=sys.stderr)
            except Exception as error:
                print(f"Could not delete the uploaded Gemini file: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
