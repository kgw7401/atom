"""Seed data for training programs — Beginner/Intermediate/Advanced Week 1."""

from __future__ import annotations


def _parse_segments(raw: str) -> list[dict]:
    """Parse a segment string like 'combo, [cue!], combo' into structured list."""
    items = [s.strip() for s in raw.split(",") if s.strip()]
    result = []
    for item in items:
        if item.startswith("[") and item.endswith("]"):
            result.append({"text": item[1:-1], "is_cue": True})
        else:
            result.append({"text": item, "is_cue": False})
    return result


# ──────────────────────────────────────────────────────────────
# Beginner Week 1
# ──────────────────────────────────────────────────────────────

BEGINNER_WEEK1 = [
    # Day 1: 잽 & 원투
    {
        "level": "beginner",
        "week": 1,
        "day_number": 1,
        "theme": "잽 & 원투",
        "theme_description": "모든 것의 시작 — 잽과 크로스의 기본기를 다진다",
        "coach_comment": "첫 날이야! 잽은 복싱의 가장 중요한 펀치다. 팔을 쭉 뻗고, 빠르게 당겨. 오늘은 폼에 집중하자.",
        "r1_segments": "잽, 잽, 잽잽, [가드!], 잽, 잽, [턱 당겨!], 잽잽, 잽, 잽, [눈!], 잽잽, 잽, 잽",
        "r2_segments": "원투, 잽, 원투, 잽잽, [스텝!], 원투, 잽, 원투, [호흡!], 잽잽, 원투, 원투, [가드!], 원투",
        "r3_segments": "원투, 원투, 잽잽, 원투, [더 빠르게!], 원투, 잽잽, 원투, 원투, [강하게!], 잽잽, 원투, 원투, [좋아!]",
        "finisher_type": "nonstop",
        "finisher_segments": "원투, 원투, 원투, 원투, 원투, [더!], 원투, 원투, 원투, 원투, [좋아!], 원투, 원투, 원투, 원투, [더 빠르게!], 원투, 원투, 원투, [멈추지 마!], 원투, 원투, 원투, 원투, [거의 다 왔어!], 원투, 원투, [마지막!], 원투",
    },
    # Day 2: 리듬
    {
        "level": "beginner",
        "week": 1,
        "day_number": 2,
        "theme": "리듬",
        "theme_description": "복싱은 춤이다 — 잽과 더블잽을 번갈아 치며 리듬감을 깨운다",
        "coach_comment": "오늘은 리듬이야. 같은 콤보도 타이밍이 다르면 완전히 다른 펀치가 된다. 힘 빼고, 리듬에 집중해.",
        "r1_segments": "잽, 잽잽, 잽, 잽잽, [호흡!], 잽, 잽잽, 잽, 잽잽, [힘 빼!], 잽, 잽잽, 잽, [스텝!]",
        "r2_segments": "원투, 잽잽, 원투, 잽잽, [스텝!], 원투, 잽잽, 원투, [힘 빼!], 잽잽, 원투, 잽잽, 원투, [가드!]",
        "r3_segments": "원투, 잽잽, 원투, 원투, [더 빠르게!], 잽잽, 원투, 원투, 잽잽, [강하게!], 원투, 원투, 원투, [좋아!]",
        "finisher_type": "countdown",
        "finisher_segments": "원투, 원투, 원투, 원투, 원투, [가드!], 원투, 원투, 원투, 원투, [더!], 원투, 원투, 원투, [강하게!], 원투, 원투, [더 빠르게!], 원투, [좋아!], 원투, 원투, 원투, 원투, 원투, [멈추지 마!], 원투, 원투, 원투, 원투, [거의 다 왔어!], 원투, 원투, [마지막!]",
    },
    # Day 3: 훅
    {
        "level": "beginner",
        "week": 1,
        "day_number": 3,
        "theme": "훅",
        "theme_description": "파괴력의 시작 — 팔꿈치 90도, 몸통 회전이 핵심",
        "coach_comment": "훅은 KO 펀치야. 팔로 치는 게 아니라 허리로 친다. 팔꿈치 각도 유지하고, 몸통을 돌려.",
        "r1_segments": "잽 훅, 잽 훅, 양훅, [가드!], 잽 훅, 양훅, [턱 당겨!], 잽 훅, 잽 훅, 양훅, [눈!], 잽 훅",
        "r2_segments": "투 훅, 잽 훅, 양훅, 훅 투, [스텝!], 투 훅, 잽 훅, [가드!], 양훅, 훅 투, 투 훅, [호흡!], 잽 훅",
        "r3_segments": "투 훅, 양훅, 훅 투, [강하게!], 잽 훅, 양훅, 투 훅, [더 빠르게!], 훅 투, 양훅, 잽 훅, [좋아!], 투 훅",
        "finisher_type": "body_blitz",
        "finisher_segments": "바디 훅, 바디 훅, 바디 훅, 바디 훅, [더!], 바디 훅, 바디 훅, 바디 훅, 바디 훅, [강하게!], 바디 훅, 바디 훅, 바디 훅, 바디 훅, [멈추지 마!], 바디 훅, 바디 훅, 바디 훅, [거의 다 왔어!], 바디 훅, 바디 훅, [마지막!], 바디 훅",
    },
    # Day 4: 바디워크
    {
        "level": "beginner",
        "week": 1,
        "day_number": 4,
        "theme": "바디워크",
        "theme_description": "레벨 체인지의 기술 — 무릎을 굽혀서 내려간다",
        "coach_comment": "바디샷은 가장 과소평가된 무기야. 레벨 체인지가 칼로리 소모를 극대화한다. 허리로 숙이지 말고, 무릎을 굽혀.",
        "r1_segments": "투 바디, 바디 투, 투 바디, [가드!], 바디 훅, 투 바디, [눈!], 바디 투, 바디 훅, 투 바디, [턱 당겨!], 바디 투",
        "r2_segments": "바디 훅, 원투, 투 바디, 잽 훅, [스텝!], 바디 투, 원투, 바디 훅, [호흡!], 잽 훅, 투 바디, 원투, [가드!]",
        "r3_segments": "바디 훅, 원투, 바디 투, [강하게!], 양훅, 투 바디, 원투, [더 빠르게!], 바디 훅, 잽 훅, 바디 투, [좋아!], 원투",
        "finisher_type": "nonstop",
        "finisher_segments": "바디 훅, 바디 훅, 바디 훅, 바디 훅, 바디 훅, [더!], 바디 훅, 바디 훅, 바디 훅, 바디 훅, [강하게!], 바디 훅, 바디 훅, 바디 훅, 바디 훅, [멈추지 마!], 바디 훅, 바디 훅, 바디 훅, [거의 다 왔어!], 바디 훅, 바디 훅, [마지막!]",
    },
    # Day 5: 어퍼컷
    {
        "level": "beginner",
        "week": 1,
        "day_number": 5,
        "theme": "어퍼컷",
        "theme_description": "인파이팅의 무기 — 다리에서 시작해서 주먹으로 끝난다",
        "coach_comment": "어퍼컷은 근거리 펀치야. 와인드업 없이 컴팩트하게. 다리에서 파워가 나온다.",
        "r1_segments": "양어퍼, 어퍼 투, 양어퍼, [가드!], 어퍼 투, 양어퍼, [턱 당겨!], 어퍼 투, 양어퍼, [호흡!], 어퍼 투, 양어퍼",
        "r2_segments": "원투, 양어퍼, 잽 훅, 어퍼 투, [스텝!], 양어퍼, 원투, [힘 빼!], 잽 훅, 양어퍼, 어퍼 투, [가드!], 원투",
        "r3_segments": "양어퍼, 양훅, 원투, [강하게!], 어퍼 투, 양훅, 양어퍼, [더 빠르게!], 원투, 훅 투, 양어퍼, [좋아!], 양훅",
        "finisher_type": "chain",
        "finisher_segments": "양어퍼, 양훅, 원투, 양어퍼, 양훅, [더!], 원투, 양어퍼, 양훅, 원투, [강하게!], 양어퍼, 양훅, 원투, 양어퍼, [멈추지 마!], 양훅, 원투, 양어퍼, [거의 다 왔어!], 양훅, 원투, [마지막!]",
    },
    # Day 6: 체력
    {
        "level": "beginner",
        "week": 1,
        "day_number": 6,
        "theme": "체력",
        "theme_description": "멈추지 않는 연습 — 단순한 콤보를 쉬지 않고 쏟아낸다",
        "coach_comment": "오늘은 기술이 아니라 볼륨이야. 복잡한 생각 없이, 몸이 자동으로 움직이게 해. 쉬지 마.",
        "r1_segments": "원투, 잽잽, 원투, 잽잽, [호흡!], 원투, 잽잽, 원투, [힘 빼!], 잽잽, 원투, 잽잽, [스텝!], 원투",
        "r2_segments": "원투, 잽잽, 잽 훅, 원투, [스텝!], 양훅, 잽잽, 원투, [호흡!], 잽 훅, 양훅, 원투, [가드!], 잽잽",
        "r3_segments": "원투, 양훅, 잽잽, 원투, [더 빠르게!], 잽 훅, 원투, 양훅, [강하게!], 원투, 잽잽, 원투, [좋아!], 양훅",
        "finisher_type": "nonstop",
        "finisher_segments": "잽, 잽, 잽, 잽, 잽, 잽, [더!], 잽, 잽, 잽, 잽, 잽, [강하게!], 잽, 잽, 잽, 잽, 잽, [멈추지 마!], 잽, 잽, 잽, 잽, [거의 다 왔어!], 잽, 잽, 잽, [마지막!], 잽",
    },
    # Day 7: 통합
    {
        "level": "beginner",
        "week": 1,
        "day_number": 7,
        "theme": "통합",
        "theme_description": "이번 주의 모든 것 — 다양한 콤보에 반응하는 스파링 시뮬레이션",
        "coach_comment": "마지막 날이야! 이번 주에 배운 모든 것을 합친다. 다양한 콤보가 나올 때 바로 반응해. 잘 해왔어!",
        "r1_segments": "잽, 원투, 잽잽, 잽 훅, [가드!], 양훅, 원투, 잽잽, [스텝!], 양어퍼, 원투, 바디 훅, [눈!]",
        "r2_segments": "원투, 잽 훅, 양훅, 바디 훅, [호흡!], 어퍼 투, 투 훅, 바디 투, [스텝!], 양어퍼, 훅 투, 원투, [가드!]",
        "r3_segments": "원투, 양훅, 잽 훅, 바디 훅, [강하게!], 양어퍼, 투 훅, 원투, [더 빠르게!], 양훅, 바디 투, 원투, [좋아!]",
        "finisher_type": "countdown",
        "finisher_segments": "원투, 양훅, 잽 훅, 바디 훅, 원투, [가드!], 양어퍼, 원투, 양훅, 잽 훅, [더!], 원투, 바디 훅, 양훅, [강하게!], 원투, 양어퍼, [멈추지 마!], 원투, [마지막!], 양훅, 잽 훅, 바디 훅, 원투, 양어퍼, [거의 다 왔어!], 원투, 양훅, 바디 훅, [마지막!]",
    },
]


def get_program_templates() -> list[dict]:
    """Return all program day templates in DB-ready format."""
    result = []
    for day in BEGINNER_WEEK1:
        result.append({
            "level": day["level"],
            "week": day["week"],
            "day_number": day["day_number"],
            "theme": day["theme"],
            "theme_description": day["theme_description"],
            "coach_comment": day["coach_comment"],
            "r1_segments_json": _parse_segments(day["r1_segments"]),
            "r2_segments_json": _parse_segments(day["r2_segments"]),
            "r3_segments_json": _parse_segments(day["r3_segments"]),
            "finisher_json": {
                "type": day["finisher_type"],
                "segments": _parse_segments(day["finisher_segments"]),
            },
        })
    return result
