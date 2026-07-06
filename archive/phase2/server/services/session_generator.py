"""
Session Generator Service.

Generates drill session plans using LLM with 3-Layer Context:
  Layer 1: User profile (experience, goals, preferences)
  Layer 2: Recent session history and mastery tracking
  Layer 3: Boxing domain knowledge (combos, drills, progressions)

Output: Structured JSON drill plan with rounds, combos, and TTS timings.
"""

import json
from typing import Any, Dict, List, Optional

from server.services.llm_service import LLMClient


# System prompt template
SYSTEM_PROMPT = """You are an expert boxing coach generating personalized drill sessions.

Your task is to create a drill session plan based on the user's profile, training history, and current skill level.

Output must be valid JSON with this structure:
{
  "session_type": "drill",
  "focus": "<combo or skill focus>",
  "total_duration_minutes": <int>,
  "rounds": [
    {
      "round_number": <int>,
      "duration_seconds": <int>,
      "rest_after_seconds": <int>,
      "instructions": [
        {
          "timestamp_offset": <float in seconds from round start>,
          "combo_name": "<Korean name>",
          "expected_actions": ["action1", "action2", ...]
        }
      ]
    }
  ]
}

Available actions: jab, cross, lead_hook, rear_hook, lead_uppercut, rear_uppercut, lead_bodyshot, rear_bodyshot, slip, duck, backstep

Guidelines:
- Keep sessions 6-12 minutes total (3-5 rounds)
- Rounds should be 2-3 minutes each
- Rest 30-60 seconds between rounds
- Call combos every 5-10 seconds during rounds
- Progress from simple to complex within session
- Focus on 2-3 combos per session for better learning
"""


class SessionGenerator:
    """Generates personalized drill sessions using LLM.

    Args:
        llm_client: LLM client instance
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        if llm_client is None:
            llm_client = LLMClient(provider='anthropic')
        self.llm_client = llm_client

    async def generate(
        self,
        user_profile: Dict[str, Any],
        recent_sessions: Optional[List[Dict]] = None,
        combo_mastery: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a drill session plan.

        Args:
            user_profile: User info (experience_level, goals, preferences)
            recent_sessions: Optional list of recent session summaries
            combo_mastery: Optional dict of combo mastery states

        Returns:
            Session plan dict (validated JSON structure)
        """
        # Build 3-Layer Context
        context = self._build_context(user_profile, recent_sessions, combo_mastery)

        # Generate with LLM
        try:
            plan = await self.llm_client.complete_json(
                prompt=context,
                system=SYSTEM_PROMPT,
                max_tokens=1500,
                temperature=0.8,
            )

            # Validate structure
            self._validate_plan(plan)

            return plan

        except Exception as e:
            # Fallback to hardcoded plan on LLM failure
            print(f"LLM generation failed: {e}, using fallback plan")
            return self._fallback_plan(user_profile)

    def _build_context(
        self,
        user_profile: Dict[str, Any],
        recent_sessions: Optional[List[Dict]],
        combo_mastery: Optional[Dict[str, Any]],
    ) -> str:
        """Build 3-Layer Context prompt."""
        # Layer 1: User Profile
        context = f"# User Profile\n"
        context += f"Experience: {user_profile.get('experience_level', 'beginner')}\n"
        context += f"Goal: {user_profile.get('goal', 'general fitness')}\n\n"

        # Layer 2: Recent History
        if recent_sessions:
            context += f"# Recent Sessions ({len(recent_sessions)})\n"
            for i, session in enumerate(recent_sessions[:3], 1):
                context += f"{i}. {session.get('focus', 'drill')}: "
                context += f"{session.get('success_rate', 0):.0%} success rate\n"
            context += "\n"

        # Layer 3: Mastery State
        if combo_mastery:
            context += "# Combo Mastery\n"
            for combo, data in list(combo_mastery.items())[:5]:
                status = data.get('status', 'new')
                rate = data.get('drill_success_rate', 0.0)
                context += f"- {combo}: {status} ({rate:.0%})\n"
            context += "\n"

        context += "Generate a personalized drill session for this user."
        return context

    def _validate_plan(self, plan: Dict) -> None:
        """Validate session plan structure."""
        required_keys = ['session_type', 'focus', 'total_duration_minutes', 'rounds']
        for key in required_keys:
            if key not in plan:
                raise ValueError(f"Missing required key: {key}")

        if not isinstance(plan['rounds'], list) or len(plan['rounds']) == 0:
            raise ValueError("Rounds must be non-empty list")

        for round_data in plan['rounds']:
            if 'instructions' not in round_data:
                raise ValueError("Round missing instructions")

    def _fallback_plan(self, user_profile: Dict) -> Dict:
        """Hardcoded fallback plan for beginners."""
        experience = user_profile.get('experience_level', 'beginner')

        if experience == 'beginner':
            return {
                'session_type': 'drill',
                'focus': '기본 원투 콤보',
                'total_duration_minutes': 6,
                'rounds': [
                    {
                        'round_number': 1,
                        'duration_seconds': 120,
                        'rest_after_seconds': 45,
                        'instructions': [
                            {
                                'timestamp_offset': 5.0,
                                'combo_name': '원투',
                                'expected_actions': ['jab', 'cross'],
                            },
                            {
                                'timestamp_offset': 15.0,
                                'combo_name': '원투',
                                'expected_actions': ['jab', 'cross'],
                            },
                            {
                                'timestamp_offset': 25.0,
                                'combo_name': '원투',
                                'expected_actions': ['jab', 'cross'],
                            },
                        ],
                    },
                ],
            }
        else:
            return {
                'session_type': 'drill',
                'focus': '콤비네이션 드릴',
                'total_duration_minutes': 9,
                'rounds': [
                    {
                        'round_number': 1,
                        'duration_seconds': 150,
                        'rest_after_seconds': 60,
                        'instructions': [
                            {
                                'timestamp_offset': 5.0,
                                'combo_name': '원투쓰리',
                                'expected_actions': ['jab', 'cross', 'lead_hook'],
                            },
                            {
                                'timestamp_offset': 15.0,
                                'combo_name': '원투쓰리투',
                                'expected_actions': ['jab', 'cross', 'lead_hook', 'cross'],
                            },
                        ],
                    },
                ],
            }
