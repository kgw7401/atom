핵심은 **템플릿은 '코치의 두뇌', TTS는 '코치의 입', 앱은 '리듬 엔진'**으로 나누는 거야.

⸻

1. 전체 구조 (MVP 아키텍처)

[Client]
↓
[Backend API]
↓
[Template Picker + Segment Shuffler]
↓
[Structured Session JSON]
↓
[TTS (Pre-generation)]
↓
[Audio Playback]

⸻

2. 핵심 설계 포인트

❌ 이전 구조
• 매번 LLM으로 세션 생성 → 비용 + 지연

⸻

✅ 현재 구조
• 난이도별 사전 제작 템플릿 (beginner/intermediate/advanced)
• 템플릿 = 4개 블록 (warmup → main → pressure → cooldown)
• 블록 내 세그먼트를 매번 셔플 → 체감 다양성 확보
• LLM 비용 0, 지연 최소화

👉 이게 "코치 느낌"의 핵심

⸻

3. 템플릿 설계

(1) 블록 구조

각 템플릿은 4개 블록으로 구성:

| 블록      | 역할                    | 템포   | 강도   |
|-----------|------------------------|--------|--------|
| warmup    | 몸풀기, 기본 콤비       | slow   | low    |
| main      | 핵심 드릴, 콤비네이션   | medium | medium |
| pressure  | 몰아치기, 연속 콤비     | fast   | high   |
| cooldown  | 정리, 호흡              | slow   | low    |

⸻

(2) 세그먼트 구조 (표준)

{
  "text": "원투훅!",
  "pause_sec": 2.0,
  "tempo": "medium",
  "intensity": "medium"
}

• text: TTS로 읽을 코칭 멘트
• pause_sec: 음성 후 정적 시간 (0.5~3.0초)
• tempo: slow | medium | fast
• intensity: low | medium | high

⸻

(3) 다양성 확보 방법

① 블록 순서는 고정 (warmup → main → pressure → cooldown)
② 블록 내 세그먼트 순서를 매번 셔플
③ warmup 첫 세그먼트, cooldown 마지막 세그먼트는 고정 (인트로/아웃트로)
④ "최근 3개" 템플릿 제외 후 랜덤 선택
⑤ 라운드 시간에 따라 세그먼트 수 스케일링

👉 같은 템플릿이어도 매번 다른 느낌

⸻

4. API 설계

(1) 입력

POST /api/sessions/plan

{
  "level": "beginner",
  "rounds": 3,
  "round_duration_sec": 180,
  "rest_sec": 30
}

⸻

(2) 출력

{
  "id": "...",
  "llm_model": "template",
  "rounds": 3,
  "round_duration_sec": 180,
  "rest_sec": 30,
  "plan": {
    "rounds": [
      {
        "round": 1,
        "segments": [
          {"text": "자, 1라운드 시작합니다.", "pause_sec": 1.5, "tempo": "slow", "intensity": "low"},
          {"text": "잽!", "pause_sec": 2.5, "tempo": "slow", "intensity": "low"},
          ...
        ]
      }
    ]
  },
  "audio_ready": true
}

⸻

5. TTS 설계 (퀄리티 핵심)

(1) 생성 방식
• 템플릿 선택 → 셔플 → TTS 생성 (세션당 1회)
• 라운드 단위로 MP3 생성 + 타임스탬프

⸻

(2) tempo 매핑

| tempo  | rate |
|--------|------|
| slow   | 0.85 |
| medium | 1.0  |
| fast   | 1.2  |

⸻

(3) intensity 매핑

| intensity | pitch | volume |
|-----------|-------|--------|
| low       | -2    | soft   |
| medium    | 0     | normal |
| high      | +2    | loud   |

⸻

(4) TTS 엔진
• MVP: ElevenLabs (감정 표현 우수)
• 대안: Google Cloud Text-to-Speech

⸻

6. 클라이언트 구조 (React Native)

핵심 컴포넌트

1. Session Player
   • 셔플된 세션 JSON + 오디오 재생

2. Audio Playback
   • 라운드 단위 MP3 + 타임스탬프 동기화
   • 50ms 폴링으로 현재 위치 추적

3. Timer Engine
   • 라운드 시간 관리
   • 휴식 시간 카운트다운

⸻

7. UX 설계

세션 시작 전
• 난이도 선택 (초급 / 중급 / 고급)
• 라운드 수 / 시간 / 휴식 설정
• "세션 생성" → 3초 내 시작 목표

⸻

진행 중
• 음성 코칭 (TTS)
• 남은 시간 표시
• intensity 시각화
• 세그먼트 텍스트 동기화

⸻

라운드 끝
• "수고했어요. 30초 휴식."
• 이것도 TTS로 → 몰입감 유지

⸻

8. MVP에서 반드시 넣어야 하는 것

✅ 1. 라운드별 intensity 상승
• 워밍업 → 메인 → 압박 → 쿨다운

⸻

✅ 2. 디펜스 강제 포함
• 슬립, 덕킹, 가드

⸻

✅ 3. 압박 멘트
• "멈추지 마!"
• "계속!"
• "좋아!"

👉 이게 "코치 느낌"의 핵심

⸻

9. 비용 전략

| 항목        | 이전        | 현재         |
|-------------|-------------|--------------|
| LLM 호출    | 세션마다 1회 | 0회 (오프라인 제작) |
| TTS 호출    | 세션마다     | 세션마다 (동일)     |
| 콘텐츠 제작 | 실시간 생성  | 사전 제작 + 셔플    |

👉 LLM 비용 완전 제거. TTS만 런타임 비용.

⸻

10. 확장 (다음 단계)
• 템플릿 20개+로 확장 (콘텐츠 제작 별도 작업)
• AirPods 센서 → 반응형 코칭
• 동작 인식 → 자세 피드백
• 반복 학습 → 개인화

⸻

11. 핵심 한 줄

👉 "LLM은 콘텐츠 '사전 제작 도구'로, 앱은 '리듬 셔플 엔진'으로"
