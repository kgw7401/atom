# AI 스파링 구간 코치 스펙

## 1. 목표

사용자가 스파링 또는 복싱 훈련 영상에서 특정 구간을 선택하면, AI가 그 구간을 보고 다음을 제공하는 시스템을 만든다.

- 현재 어떤 상황이었는지
- 무엇이 문제였는지
- 그 순간 가능한 대응 옵션은 무엇이었는지
- 가장 추천되는 선택은 무엇인지
- 이를 개선하기 위한 훈련 드릴은 무엇인지
- 사용자의 후속 질문에 코치처럼 답변

현재 목표는 BoxMind처럼 경기 전체 전략을 최적화하는 것이 아니다. 지금 가장 먼저 검증할 것은 **AI가 짧은 복싱 구간을 보고 실제로 유용한 코칭을 할 수 있는가**이다.

## 2. 핵심 판단

초기 접근은 **VLM-first**로 간다.

장기적으로는 atomic event model, pose/motion indicator, VLM, coaching reasoning을 결합한 hybrid 구조가 더 안정적이다. 하지만 개인 프로젝트의 시간과 리소스를 고려하면, 처음부터 event 추출 모델을 학습하는 것은 너무 무겁다.

따라서 MVP에서는 다음 방식을 우선한다.

```text
사용자가 선택한 5-15초 영상 구간
  -> VLM이 시간순 관찰 JSON 생성
  -> 복싱 코칭 지식 / 원칙 검색
  -> LLM이 상황 해석과 코칭 생성
  -> 사용자가 채팅으로 후속 질문
```

이 방식은 모델 학습 없이 바로 사용 경험을 검증할 수 있고, 나중에 BoxMind/BoxingWeb 기반 event model을 자연스럽게 추가할 수 있다.

## 3. VLM-first의 정확한 의미

여기서 말하는 VLM-first는 **Temporal-grounded VLM 모델을 새로 학습하거나 논문 구현으로 바로 점프한다는 뜻이 아니다.**

초기 의미는 더 단순하다.

```text
일반 VLM에 짧은 복싱 영상 구간을 먼저 넣어본다.
그다음 프롬프트와 입력 구조를 조정해 시간순 관찰을 만들게 한다.
```

즉 첫 단계는 다음과 같다.

```text
5-15초 복싱 영상 구간
  -> 일반 VLM에 입력
  -> 상황 설명 / 코칭을 바로 시켜본다
  -> 답변이 얼마나 쓸모 있는지 확인한다
```

그다음 단계에서 temporal-grounded 방식으로 다듬는다.

```text
영상 구간
  -> 1초 단위 또는 주요 프레임 단위로 샘플링
  -> 각 프레임/clip에 시간 정보를 붙인다
  -> VLM에게 시간순 observation JSON을 만들게 한다
  -> 별도 coaching layer가 피드백을 생성한다
```

정리하면 다음과 같다.

```text
하지 않는 것:
  Temporal-grounded VLM 연구 모델을 처음부터 구현하거나 학습

하는 것:
  일반 VLM을 사용하되, 입력/출력을 시간순으로 구조화해서 실험
```

따라서 V1의 실제 출발점은 **VLM smoke test**다. 먼저 영상을 태워보고, VLM이 어떤 부분을 잘 보고 어떤 부분을 놓치는지 확인한다. 그 결과를 바탕으로 observation JSON, coaching prompt, RAG 지식을 설계한다.

## 4. 접근법 비교

| 접근 | 장점 | 단점 | 현재 판단 |
|---|---|---|---|
| Atomic event model first | 구조적이고 장기적으로 안정적 | 데이터/학습/검출 파이프라인이 무거움 | 지금은 후순위 |
| Pose/action recognition first | 움직임 분석 근거가 좋음 | 코칭 MVP까지 거리가 멂 | 보조 레이어 |
| VLM-only | 가장 빠르게 테스트 가능 | 시간순 근거와 일관성이 약함 | 단독 사용은 위험 |
| Temporal-grounded VLM + Coaching RAG | 빠르고, 코칭 목적에 가깝고, 확장 가능 | 프롬프트/지식 구조 설계 필요 | V1 최선 |
| Hybrid architecture | 가장 안정적이고 제품화에 유리 | 구현 범위가 큼 | 장기 목표 |

여기서 "최선"은 학술적으로 가장 완벽한 모델이 아니라, **현재 목표에 가장 가깝고 개인이 실행 가능한 선택**이라는 뜻이다.

## 5. V1 아키텍처

```text
1. Segment Input Layer
   사용자가 영상에서 분석할 구간을 선택한다.

2. Temporal Sampling Layer
   선택 구간을 시간순 프레임 또는 짧은 clip 단위로 나눈다.

3. VLM Observation Layer
   VLM이 영상 구간을 보고 보이는 사실만 시간순으로 정리한다.

4. Coaching Knowledge Layer
   복싱 원칙, 상황별 대응, 드릴 지식을 검색하거나 주입한다.

5. Coaching Reasoning Layer
   관찰 결과와 코칭 지식을 바탕으로 피드백을 생성한다.

6. Chat Coach Layer
   같은 구간 맥락을 유지하며 사용자의 후속 질문에 답한다.
```

핵심은 VLM에게 바로 "코칭해줘"라고 시키지 않는 것이다. 먼저 VLM은 **관찰자** 역할을 하고, 코칭 레이어가 **해석과 조언**을 담당한다.

## 6. VLM Observation

VLM의 첫 번째 역할은 코칭이 아니라 관찰이다.

예시 입력:

```text
이 복싱 영상 구간을 시간순으로 관찰해줘.
확실히 보이는 사실과 불확실한 추정을 구분해줘.
코칭 조언은 아직 하지 말고, 장면에서 관찰 가능한 내용만 구조화해줘.
```

예시 출력:

```json
{
  "segment": {
    "start_sec": 72.0,
    "end_sec": 78.0
  },
  "timeline": [
    {
      "time": "72.0-73.2",
      "observation": "상대가 전진하며 거리를 좁힌다.",
      "confidence": "medium"
    },
    {
      "time": "73.2-74.1",
      "observation": "사용자는 주로 직선으로 뒤로 이동한다.",
      "confidence": "medium"
    },
    {
      "time": "74.1-75.0",
      "observation": "상대가 앞손 또는 직선성 펀치로 진입을 시도한다.",
      "confidence": "low"
    }
  ],
  "spatial_context": {
    "distance_trend": "closing",
    "user_movement": "retreating",
    "opponent_movement": "advancing",
    "lateral_movement": "low"
  },
  "uncertainties": [
    "정확한 펀치 종류는 프레임만으로 확신하기 어렵다.",
    "피격 여부는 영상 화질에 따라 불확실하다."
  ]
}
```

이 구조를 쓰면 VLM의 판단 근거가 남고, 나중에 atomic event model이나 pose feature를 같은 JSON에 추가하기 쉽다.

## 7. Coaching Reasoning

코칭 레이어는 VLM observation과 복싱 지식을 바탕으로 답변한다.

출력 형식:

```text
1. 상황 요약
2. 핵심 문제
3. 가능했던 선택지
4. 가장 추천하는 대응
5. 다음 훈련 드릴
6. 사용자가 물어볼 만한 후속 질문 제안
```

예시:

```text
상대가 전진 압박을 걸었고, 당신은 직선으로 뒤로 빠지면서 거리를 다시 만들지 못했습니다.
이 상황의 핵심 문제는 상대의 진입 라인을 바꾸지 못한 것입니다.

가능한 선택지는 세 가지입니다.
1. 앞손 잽으로 진입을 끊고 사이드로 빠지기
2. 상대가 들어오는 타이밍에 체크훅 후 피벗
3. 거리가 이미 가까워졌다면 프레임 또는 클린치로 흐름 끊기

가장 먼저 연습할 것은 잽 + 사이드스텝입니다.
```

## 8. Coaching RAG

VLM이 장면을 볼 수 있어도, 좋은 코칭을 항상 안정적으로 생성한다고 보장할 수는 없다. 따라서 코칭 원칙과 드릴을 별도 지식으로 관리한다.

초기에는 복잡한 벡터 DB 없이 Markdown 문서 또는 JSON 파일로 시작한다.

예시 지식 단위:

```json
{
  "situation": "opponent_pressure_straight_retreat",
  "symptoms": [
    "상대가 전진한다",
    "사용자가 직선으로 후퇴한다",
    "좌우 이동이 적다"
  ],
  "principles": [
    "압박을 받을 때는 상대의 진입 라인을 바꿔야 한다.",
    "직선 후퇴만 반복하면 상대의 후속타 사정거리 안에 남기 쉽다."
  ],
  "options": [
    "잽으로 진입 끊기",
    "사이드스텝",
    "체크훅 후 피벗",
    "프레임 또는 클린치"
  ],
  "drills": [
    "jab + side step",
    "check hook + pivot",
    "rope escape drill"
  ]
}
```

## 9. BoxMind / BoxingWeb의 위치

BoxMind/BoxingWeb은 V1의 필수 시작점이 아니라, 장기적으로 정확도와 재현성을 높이는 보강 자산이다.

BoxMind 참고 구조:

```text
atomic event -> indicator -> reasoning
```

BoxMind의 punch event:

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

우리 프로젝트에서의 장기 활용:

```text
VLM observation이 불확실한 부분
  -> BoxingWeb 기반 punch event model로 보강

공간/움직임에 대한 추정
  -> pose 기반 motion indicator로 보강

코칭 판단의 근거
  -> event + motion + VLM observation을 함께 사용
```

즉 BoxMind/BoxingWeb은 버리는 것이 아니라, **VLM-first MVP 이후 hybrid architecture로 가기 위한 정확도 보강 레이어**로 둔다.

## 10. 장기 Hybrid 아키텍처

```text
Video Segment
  -> VLM Observation
  -> Atomic Event Model
  -> Pose / Motion Evidence
  -> Segment Evidence JSON
  -> Coaching RAG
  -> Coach Reasoning
  -> Chat Coach
```

장기적으로 Segment Evidence JSON은 다음 정보를 포함한다.

```json
{
  "vlm_observations": [],
  "atomic_events": [],
  "motion_indicators": {},
  "uncertainties": [],
  "coaching_context": {}
}
```

이 구조의 장점:

- VLM-only보다 판단 근거가 명확하다.
- event model만 쓰는 것보다 상황 이해가 풍부하다.
- 나중에 사용자 영상이 늘어날수록 데이터 축적이 가능하다.
- 실패 케이스를 event, motion, prompt, coaching knowledge 중 어디서 생겼는지 분리해서 개선할 수 있다.

## 11. MVP 범위

V1에서 할 것:

- 사용자가 짧은 영상 구간을 넣을 수 있게 한다.
- VLM이 시간순 observation JSON을 생성한다.
- 간단한 coaching knowledge 문서를 만든다.
- observation + knowledge 기반 코칭 피드백을 생성한다.
- 같은 구간에 대해 채팅으로 후속 질문을 할 수 있게 한다.

V1에서 하지 않을 것:

- 직접 atomic event model 학습
- 전체 영상 자동 이벤트 검출
- 복잡한 pose pipeline 구축
- 상황 label dataset 제작
- BoxMind식 18개 indicator 재현
- 경기 승률 또는 전략 최적화

## 12. 검증 기준

MVP의 성공 기준은 모델 정확도 숫자가 아니다. 다음 질문에 답하는 것이다.

- 사용자가 받은 피드백이 실제로 쓸모 있는가?
- AI가 상황을 시간순으로 이해하는가?
- 조언이 너무 일반적이지 않고, 해당 구간에 맞게 구체적인가?
- 후속 질문에 코치처럼 이어서 답할 수 있는가?
- VLM만으로 부족한 정보가 무엇인지 드러나는가?
- 나중에 event model이나 motion indicator가 필요한 이유가 명확해지는가?

## 13. 구현 순서

### Phase 1: VLM-first prototype

1. 영상 구간 입력 방식 정리
2. 샘플 영상 3-5개의 5-15초 구간 준비
3. 일반 VLM에 영상을 바로 넣고 코칭 smoke test
4. VLM이 잘 보는 것과 놓치는 것 기록
5. 프레임/clip sampling 구현
6. VLM observation prompt 작성
7. observation JSON schema 정의
8. coaching prompt 작성
9. 간단한 coaching knowledge 문서 작성
10. structured observation 기반 코칭 품질 확인

### Phase 2: Coaching RAG 정리

1. 자주 나오는 상황별 코칭 원칙 작성
2. 대응 옵션과 드릴 library 작성
3. observation과 지식 매칭 방식 구현
4. 답변 형식 안정화

### Phase 3: BoxingWeb 활용

1. BoxingWeb 데이터 구조 재확인
2. `video_event.json`을 `atomic_events` 포맷으로 변환
3. `pose_gt.pkl`에서 간단한 motion indicator 계산
4. VLM observation과 event/motion evidence를 비교
5. VLM이 놓치는 부분을 event model로 보강할 수 있는지 평가

### Phase 4: Hybrid 확장

1. punch atomic event model 학습 또는 재사용
2. 사용자 영상에서 event 후보 추출
3. pose 기반 distance / pressure / retreat feature 추가
4. Segment Evidence JSON 통합
5. 실제 사용자 영상 실패 케이스 수집

## 14. 현재 결론

지금 당장 가장 좋은 선택은 **Temporal-grounded VLM + 가벼운 Coaching RAG**이다.

이유:

- 사용자가 원하는 경험에 가장 가깝다.
- 모델 학습 없이 빠르게 검증할 수 있다.
- 개인 프로젝트의 시간과 리소스에 맞다.
- VLM-only보다 근거와 일관성을 확보할 수 있다.
- 나중에 BoxMind/BoxingWeb 기반 event model을 붙이기 쉽다.

따라서 현재 프로젝트의 첫 목표는 "좋은 event model 만들기"가 아니라, **선택된 복싱 구간을 보고 AI가 코치처럼 유용한 피드백을 줄 수 있는지 검증하는 것**이다.
