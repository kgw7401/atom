# AI 스파링 구간 코치 스펙

## 1. 제품 정의

사용자가 스파링 또는 복싱 훈련 영상을 업로드하고 특정 구간을 선택하면, AI가 그 구간을 분석해 문제점, 더 나은 선택지, 후속 질문 답변, 훈련 드릴을 제공하는 시스템이다.

현재 목표는 BoxMind처럼 경기 전략을 최적화하거나 승률을 예측하는 것이 아니다. BoxMind의 **atomic event -> indicator -> reasoning** 구조를 참고해, 사용자가 선택한 구간에 대해 실용적인 코칭 피드백을 만드는 것이 목표다.

## 2. 사용자 흐름

```text
1. 사용자가 스파링 영상을 업로드한다.
2. 사용자가 구간을 선택한다. 예: 01:12-01:18
3. 시스템이 해당 구간의 이벤트와 움직임 feature를 추출한다.
4. AI가 상황, 문제점, 더 나은 선택지를 설명한다.
5. 사용자가 채팅으로 후속 질문을 한다.
6. AI가 같은 구간 맥락을 유지하며 답한다.
7. 마지막에 관련 훈련 드릴을 추천한다.
```

예시 피드백:

```text
상대가 전진하며 거리를 좁혔고, 당신은 대부분 직선으로 후퇴했습니다.
진입을 끊는 잽, 각도 전환, 프레임, 클린치가 없었습니다.

문제는 상대의 라인을 바꾸지 못한 채 공간만 내준 것입니다.
더 나은 선택지는 잽 후 사이드스텝, 체크훅 후 피벗, 가까워졌을 때 프레임/클린치입니다.

추천 드릴:
- 잽 + 사이드스텝
- 체크훅 + 피벗
- 로프 탈출
```

## 3. 핵심 접근

상황 자체를 라벨링하지 않는다. 복싱 상황은 경우의 수가 너무 많기 때문이다.

대신 다음 구조를 사용한다.

```text
영상 구간
  -> atomic event 추출
  -> 움직임 / 공간 feature 추출
  -> indicator 구성
  -> reasoning
  -> 피드백 + 드릴
```

개념 구분:

```text
Atomic Event:
  관찰 가능한 최소 행동 단위.

Indicator:
  이벤트와 trace를 조합해 만든 파생 지표.

Situation:
  이벤트와 indicator를 바탕으로 해석한 전술적 맥락.

Feedback:
  문제점, 선택지, 훈련 제안.
```

## 4. BoxMind 참고점

BoxMind는 펀치 이벤트를 다음처럼 정의한다.

```text
e = (
  t_start,
  t_end,
  a_hand,
  a_dist,
  a_tech,
  a_target,
  a_eff
)
```

즉 하나의 펀치를 시간, 손, 거리, 기술, 타깃, 효과의 조합으로 표현한다.  
그다음 여러 이벤트를 모아 indicator를 만든다.

우리도 이 철학을 따른다. 다만 목표는 경기 전략 최적화가 아니라, 선택 구간에 대한 코칭 피드백이다.

## 5. V1 Atomic Event

V1에서는 BoxingWeb 데이터와 호환되는 펀치 중심 atomic event부터 사용한다.

```text
PunchAtomicEvent = (
  t_start,
  t_end,
  actor,
  hand,
  distance,
  technique,
  target,
  effect
)
```

필드:

```text
actor:
  user | opponent | red | blue | unknown

hand:
  left | right | lead | rear | unknown

distance:
  long | mid | close | unknown

technique:
  straight | hook | uppercut | unknown

target:
  head | body | unknown

effect:
  effective | ineffective | blocked | missed | unknown
```

예시:

```json
{
  "type": "punch",
  "t_start": 3.2,
  "t_end": 3.5,
  "actor": "opponent",
  "hand": "right",
  "distance": "mid",
  "technique": "straight",
  "target": "head",
  "effect": "ineffective"
}
```

이 이벤트가 모든 복싱 움직임을 표현하지는 않는다. 하지만 BoxingWeb이 이미 이 라벨을 제공하고, BoxMind 구조와도 맞기 때문에 첫 출발점으로 가장 현실적이다.

## 6. BoxingWeb으로 가능한 것과 부족한 것

가능한 것:

```text
펀치 timeline
red / blue actor
펀치 시작/종료
손, 기술, 거리, 타깃, 효과
red / blue 두 선수의 2D/3D pose
```

부족한 것:

```text
가드 상태
풋워크 종류
피벗
슬립 / 롤 / 블록
직선 후퇴
각도 만들기
상대 압박
코너 / 로프 압박
놓친 전술적 선택지
```

따라서 V1에서는 고수준 상황을 supervised label로 학습하지 않는다.  
대신 pose와 tracking에서 계산 가능한 feature를 indicator로 사용한다.

## 7. 움직임 Indicator

BoxingWeb의 `pose_gt.pkl`에서 다음 feature를 계산한다.

```text
두 선수 사이 거리 trace
거리 변화 속도
사용자 / 상대 중심점 trajectory
전진 / 후퇴 displacement
좌우 displacement
각도 변화
상대 펀치 이후 반응 타이밍
```

예시 indicator:

```text
1.2초 동안 거리가 감소함
사용자가 대부분 후퇴함
좌우 이동량이 낮음
상대 진입 후 0.5초 안에 반격이 없음
```

reasoning layer는 이를 보고 다음처럼 해석한다.

```text
상대가 전진 압박했다.
사용자는 직선 후퇴에 가까웠다.
상대의 진입을 끊지 못했다.
```

이 문장들은 atomic label이 아니라 reasoning 결과다.

## 8. 시스템 아키텍처

V1:

```text
BoxingWeb segment
  -> video_event.json에서 PunchAtomicEvent 생성
  -> pose_gt.pkl에서 movement indicator 계산
  -> event + indicator context 구성
  -> 코칭 피드백 생성
  -> 채팅 follow-up 지원
```

나중에 사용자 raw video로 확장:

```text
raw video
  -> pose estimation
  -> punch detection
  -> punch attribute classification
  -> movement indicator
  -> coaching reasoning
```

## 9. Reasoning 입력 형식

reasoning model에는 구조화된 context를 넣는다.

```json
{
  "segment": {
    "start": 12.0,
    "end": 18.0
  },
  "events": [
    {
      "type": "punch",
      "t_start": 12.4,
      "t_end": 12.7,
      "actor": "opponent",
      "hand": "left",
      "distance": "mid",
      "technique": "straight",
      "target": "head",
      "effect": "ineffective"
    }
  ],
  "indicators": {
    "distance_trend": "closing",
    "user_movement": "mostly_backward",
    "lateral_exit": false,
    "user_counter_within_0_5s": false
  }
}
```

이 context를 바탕으로 AI는 상황 요약, 문제점, 선택지, 드릴을 생성한다.

## 10. 채팅 코칭

사용자는 분석 후 계속 질문할 수 있다.

예:

```text
여기서 왜 직선 후퇴가 안 좋아?
체크훅이 가능했을까?
잽이 나아, 클린치가 나아?
이걸 고치려면 무슨 훈련을 해야 해?
```

AI는 선택 구간의 event와 indicator를 근거로 답해야 한다.  
영상에서 보이지 않거나 추출하지 못한 정보는 아는 척하지 않는다.

## 11. 데이터 전략

### Phase 1: BoxingWeb

가장 먼저 사용한다.

```text
/Users/kgw7401/Downloads/boxingweb
```

구현할 것:

```text
BoxingWeb adapter
PunchAtomicEvent schema
event timeline export
pose 기반 movement indicator
segment feedback prototype
```

### Phase 2: BoxingVI

필요하면 나중에 추가한다.

주요 용도:

```text
punch detector
jab / cross / hook / uppercut classifier
raw video event extraction
```

BoxingVI는 distance, target, effect, 두 선수 간 전술 맥락에는 상대적으로 덜 유용하다.

### Phase 3: 사용자 영상

나중에 필요한 것:

```text
pose extraction
punch detection
punch attribute classification
user / opponent identity tracking
segment-level feedback
```

## 12. MVP 범위

첫 MVP:

```text
BoxingWeb 라운드와 선택 구간이 주어졌을 때,
event timeline과 tactical feedback을 생성한다.
```

MVP 입력:

```text
BoxingWeb round folder
segment start/end
사용자 side: red 또는 blue
선택적 사용자 질문
```

MVP 출력:

```text
event timeline
movement indicator
구간 진단
문제점
더 나은 선택지
훈련 드릴
채팅용 context
```

검증 질문:

```text
구조화된 펀치 이벤트와 움직임 indicator만으로도
사용자에게 의미 있는 코칭 피드백을 만들 수 있는가?
```

## 13. 마일스톤

### Milestone 1: 스키마와 데이터 로더

```text
PunchAtomicEvent 정의
BoxingWeb adapter 구현
video_event.json 정규화
segment filtering 지원
```

### Milestone 2: Motion Indicator

```text
pose_gt.pkl 로드
두 선수 중심점 trace 계산
거리 변화 계산
전진 / 후퇴 / 좌우 이동 계산
펀치 주변 반응 타이밍 계산
```

### Milestone 3: Feedback Prototype

```text
event + indicator context 구성
구간 요약 생성
문제점 / 선택지 / 드릴 생성
```

### Milestone 4: Chat Context

```text
선택 구간 context 저장
후속 질문 지원
같은 event와 indicator를 근거로 답변
```

### Milestone 5: Raw Video 확장

```text
pose extraction
punch detection
punch attribute classification
사용자 업로드 영상 적용
```

## 14. 즉시 다음 작업

초기 구조:

```text
src/
  sparring_coach/
    __init__.py
    events.py
    datasets/
      __init__.py
      boxingweb.py
    indicators/
      __init__.py
      motion.py
scripts/
  inspect_boxingweb_events.py
```

첫 명령어 목표:

```text
python scripts/inspect_boxingweb_events.py \
  --root /Users/kgw7401/Downloads/boxingweb
```

기대 출력:

```text
round 수
punch event 수
라벨 분포
정규화된 PunchAtomicEvent 예시
```

이것이 가장 낮은 리스크의 첫 구현이다. 모델 학습이나 raw video 분석 전에 이벤트 기반을 먼저 만든다.
