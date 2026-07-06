"""
Feedback Generator Service.

Generates personalized coach feedback from drill analysis results using LLM.

Input: DrillResult with combo stats and match details
Output: Natural Korean coaching feedback (3-5 sentences)
"""

from typing import Any, Dict, Optional

from ml.pipeline.types import DrillResult
from server.services.llm_service import LLMClient


SYSTEM_PROMPT = """You are a supportive boxing coach providing feedback in Korean.

Your feedback should:
- Be encouraging and specific
- Highlight what went well
- Suggest ONE concrete improvement area
- Use natural, conversational Korean
- Be 3-5 sentences total
- End with motivation for next session

Tone: Friendly coach, not formal teacher.
"""


class FeedbackGenerator:
    """Generates coach feedback from drill analysis.

    Args:
        llm_client: LLM client instance
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        if llm_client is None:
            llm_client = LLMClient(provider='anthropic')
        self.llm_client = llm_client

    async def generate(
        self,
        drill_result: DrillResult,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate coach feedback from analysis.

        Args:
            drill_result: Analysis results from session matcher
            user_profile: Optional user info for personalization

        Returns:
            Korean feedback text (3-5 sentences)
        """
        # Build analysis summary
        summary = self._build_summary(drill_result, user_profile)

        try:
            feedback = await self.llm_client.complete(
                prompt=summary,
                system=SYSTEM_PROMPT,
                max_tokens=300,
                temperature=0.8,
            )

            return feedback.strip()

        except Exception as e:
            print(f"LLM feedback generation failed: {e}, using fallback")
            return self._fallback_feedback(drill_result)

    def _build_summary(
        self, drill_result: DrillResult, user_profile: Optional[Dict]
    ) -> str:
        """Build analysis summary for LLM."""
        summary = "# Drill Session Analysis\n\n"

        # Overall performance
        success_rate = drill_result.overall_success_rate
        summary += f"Overall Success Rate: {success_rate:.0f}%\n"
        summary += f"Total Instructions: {drill_result.total_instructions}\n"
        summary += f"  Success: {drill_result.total_success}\n"
        summary += f"  Partial: {drill_result.total_partial}\n"
        summary += f"  Miss: {drill_result.total_miss}\n\n"

        # Per-combo breakdown
        if drill_result.combo_stats:
            summary += "# Combo Performance\n"
            for combo_key, stat in drill_result.combo_stats.items():
                summary += f"- {stat.combo_name}: "
                summary += f"{stat.successes}/{stat.attempts} "
                summary += f"({stat.success_rate:.0%})\n"
            summary += "\n"

        # User context
        if user_profile:
            experience = user_profile.get('experience_level', 'beginner')
            summary += f"User Experience: {experience}\n\n"

        summary += "Generate encouraging Korean feedback for this session."
        return summary

    def _fallback_feedback(self, drill_result: DrillResult) -> str:
        """Hardcoded fallback feedback."""
        success_rate = drill_result.overall_success_rate

        if success_rate >= 80:
            return (
                "정말 잘하셨어요! 오늘 세션에서 콤보 정확도가 아주 높았습니다. "
                "이 실력을 유지하면서 다음엔 조금 더 빠른 템포로 도전해보세요. "
                "계속 이렇게만 하시면 금방 마스터하실 거예요!"
            )
        elif success_rate >= 50:
            return (
                "좋아요, 잘하고 계십니다! 콤보의 흐름을 이해하고 계시네요. "
                "다음 세션에선 동작 사이의 연결을 좀 더 부드럽게 만드는 데 집중해보세요. "
                "꾸준히 하시면 더 좋아질 거예요. 파이팅!"
            )
        else:
            return (
                "괜찮습니다, 처음이니까요! 새로운 콤보를 배우는 건 시간이 필요해요. "
                "각 동작을 천천히 연습하면서 폼을 확실히 익히는 데 집중하세요. "
                "급하게 하지 마시고, 정확하게 하는 것부터 시작해봐요. 다음엔 더 잘하실 거예요!"
            )
