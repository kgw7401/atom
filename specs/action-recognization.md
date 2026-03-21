🥊 0. 전체 방향 (핵심 한 줄)

MVP는 “동작 따라했는지 검증” → 이후 “왜 틀렸는지 분석”으로 확장

⸻

🧠 1. 전체 시스템 구조

[TTS Session (정답)]
↓
[User 수행 + 영상 업로드]
↓
[Pose Extraction (MediaPipe)]
↓
[Action Recognition (TCN)]
↓
[Event Sequence 생성]
↓
[정답 시퀀스와 비교]
↓
[피드백 생성 (LLM)]

⸻

🔑 2. MVP 목표 (명확하게)

👉 지금 단계에서 할 것:
• jab / cross / hook 구분
• 시퀀스 비교
• “맞았는지 / 틀렸는지” 피드백

⸻

👉 하지 않을 것:
• 고급 전략 분석 ❌
• BoxMind full 구조 ❌
• 실시간 처리 ❌

⸻

🧩 3. 핵심 기술 스택

1️⃣ Pose Estimation
• MediaPipe (추론용)

⸻

2️⃣ Action Recognition

👉 모델:
• Temporal Convolutional Network

⸻

👉 이유:
• 짧은 시간 패턴 (펀치) 인식에 최적
• 빠르고 구현 쉬움
• 데이터 적어도 됨

⸻

3️⃣ 데이터
• BoxingVI (AlphaPose 기반 keypoint)

⸻

⚠️ 4. 가장 중요한 문제 (데이터 mismatch)

문제
• 학습: AlphaPose
• 추론: MediaPipe

👉 그대로 쓰면 성능 망함

⸻

해결 (핵심🔥)

👉 “좌표가 아니라 feature로 통일”

⸻

반드시 해야 할 것

1. 정규화

x = x - hip_center
y = y - hip_center
x /= shoulder_width
y /= shoulder_width

⸻

2. velocity 추가

dx = x*t - x*{t-1}
dy = y*t - y*{t-1}

⸻

3. joint subset 사용
   • shoulder / elbow / wrist 중심

⸻

👉 핵심:

❗ “pose estimator가 아니라 representation을 맞춘다”

⸻

🧠 5. 모델 구조 (간단 버전)

Input: (time, features)

→ Conv1D (dilation=1)
→ Conv1D (dilation=2)
→ Conv1D (dilation=4)

→ Linear
→ Softmax (jab / cross / hook)

⸻

👉 출력:

frame-level label

⸻

🧩 6. Event 추출

문제

jab jab jab jab

👉 몇 번 친 건지 모름

⸻

해결

if label changes:
new event

⸻

👉 결과

[
{ "type": "jab" },
{ "type": "cross" }
]

⸻

🧠 7. 시퀀스 비교 (핵심 로직🔥)

정답

[jab, jab, cross]

유저

[jab, cross]

⸻

해결 방법

👉 DTW or edit distance

⸻

결과

{
"score": 0.8,
"missed": ["jab"]
}

⸻

🤖 8. 피드백 생성

LLM 입력:

{
"expected": ["jab", "jab", "cross"],
"actual": ["jab", "cross"]
}

⸻

출력:

"두 번째 잽이 빠졌어요. 콤비네이션을 끝까지 이어가세요."

⸻

🧠 9. BoxMind 접근 (확장 전략)

핵심 아이디어

👉 “동작 = 하나의 라벨” ❌
👉 “동작 = 여러 속성 조합” ⭕

⸻

예

Right hand + Straight + Head

⸻

장점
• 일반화 가능
• 더 정밀한 피드백

⸻

단점
• 라벨링 어려움
• 구현 복잡

⸻

🔥 10. 최적 전략 (중요)

❌ 바로 BoxMind 하지 마라

⸻

✅ 이렇게 가라

Step 1 (지금)

jab / cross / hook

⸻

Step 2

feature 추가:

trajectory (직선/곡선)
velocity (속도)

⸻

Step 3

{
"type": "hook",
"trajectory": "outer"
}

⸻

👉 “BoxMind-lite”

⸻

🧠 11. 핵심 인사이트 5개

1️⃣

❗ 모델보다 데이터 표현이 더 중요하다

⸻

2️⃣

❗ TCN은 “짧은 시간 패턴” 문제에 최적이다

⸻

3️⃣

❗ segmentation보다 classification이 먼저다 (MVP 기준)

⸻

4️⃣

❗ 완벽한 정확도보다 “일관된 피드백”이 중요하다

⸻

5️⃣

❗ 처음부터 복잡하게 만들면 실패한다

⸻

🚀 12. 다음 단계 (실행 플랜)

1️⃣ 바로 해야 할 것
• MediaPipe → keypoint 추출
• feature engineering 구현
• TCN 학습

⸻

2️⃣ 그 다음
• event extraction
• sequence matching

⸻

3️⃣ 마지막
• LLM 피드백 연결

⸻

🔥 최종 한 줄 정리

👉 “MVP는 단순하게 시작하고, 구조는 확장 가능하게 설계하라”
