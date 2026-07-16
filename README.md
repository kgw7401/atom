# Atom

개인 스파링 영상을 분석해, 경기를 어떻게 풀어나가면 되는지와 다음에 무엇을 훈련하면 되는지를 검증 가능한 근거와 함께 조언하는 개인 AI 복싱 코치를 만드는 프로젝트다.

코치의 방법론은 "전체를 보되, 한 번에 하나를 처방한다"이다. 경기 전반을 관찰하되, 라운드당 최우선 문제 1개, 행동 규칙 1개, 훈련 메뉴 1~2개만 낸다. 관찰 클래스는 코치가 말하고 싶은데 근거가 없는 것이 생길 때만 수요 주도로 추가한다.

현재는 제품이나 자동 채점 시스템을 만드는 단계가 아니다. 먼저 다음 질문을 검증한다.

> 영상에서 실제로 판단을 놓친 장면을 시간 근거와 함께 찾고, 다음 라운드에서 시험할 수 있는 피드백으로 바꿀 수 있는가?

## 현재 기술 방향

관찰의 중심은 범용 VLM이 아니라 모션과 복싱 이벤트를 추출하는 전용 CV 파이프라인이다.

```text
스파링 영상
  -> 선수 추적과 포즈/모션
  -> 펀치와 방어, 풋워크 이벤트
  -> 거리와 압박 상태
  -> 상대 행동 -> 나의 선택 -> 결과
  -> 최우선 문제 하나
  -> 다음 라운드 규칙과 훈련 처방
```

BoxMind의 `atomic event -> tactical indicator -> strategy` 분해를 출발점으로 삼는다. 로컬에 확보한 공식 BoxingWeb 데이터셋으로 펀치 이벤트 계층을 먼저 재현하고, 개인 스파링에 필요한 방어·풋워크·의사결정 계층을 추가한다.

VLM/LLM은 영상 속 사실을 단독 판정하지 않는다. 구조화된 이벤트를 코치 언어로 설명하고, 낮은 확신도의 장면을 보조 검토하고, 관찰된 문제에 연결된 훈련 메뉴를 표현하는 역할로 제한한다.

## 문서

- [프로젝트 방향](docs/project-direction.md): 코치 지향점, 범위 경계, 검증 루프, 성공 기준
- [실험 기록](docs/experiment-history.md): Gemini/VLM 실험 결과와 기술 방향을 바꾼 근거
- [BoxMind 재현 계획](docs/boxmind-baseline-plan.md): BoxingWeb 자산, 구현 단계, 단계별 검증 기준
- [첫 영상 피드백](results/IMG_0574.feedback.md): 초기 Gemini 분석과 사용자 검토 기록
- [첫 속성 분류 기준선 결과](results/pose-baseline-report.json): pose-only 모델의 test 성능
- [GT pose 이벤트 검출 결과](results/pose-event-detector-pose-report.json): pose-only 시간적 펀치 검출 성능
- [RGB 융합 실험 결과](results/pose-event-detector-rgb-report.json): pose-guided RGB 움직임 특징 결합 성능
- [영상 포즈 이벤트 검출 결과](results/video-pose-event-detection-report.json): YOLO+RTMW3D 실제 MP4 전처리와 GT/영상 포즈 비교
- [RTMW 재학습 결과](results/rtmw-adapted-event-detection-report.json): 전체 50경기 영상 포즈 추출, RTMW 분포 재학습 및 앙상블 평가
- [UVE 검증 결과](results/uve-validation-report.json): GT 없는 red/blue/non-boxer 재식별, IDF1 및 이벤트 성능 비교

## 현재 상태

BoxingWeb의 구조 감사와 GT pose 검증을 완료하고, YOLO+RTMW 영상 포즈에서 anchor-free 펀치 검출기를 재학습했다. 상대 선수까지의 거리·접근 방향을 명시적으로 추가한 교차 시간축 앙상블은 검증 F1 0.725를 기록했다. 이후 진단용 GT red/blue association을 제거하기 위해 BoT-SORT 위치 트랙과 매 10프레임 RGB 외형 3분류를 결합한 UVE 호환 모듈을 구현했다. 검증 4경기에서 identity IDF1은 0.898, UVE 포즈 펀치 검출 F1은 0.650이며, 동일 모델의 oracle association F1 0.687 대비 0.036 하락했다. BoxMind의 비공개 UV-map 분류기 대신 RGB descriptor를 사용한 결과이므로, 다음 단계는 UVE 분포로 검출기를 재학습하고 더 강한 appearance embedding으로 신원 정밀도를 높이는 것이다.
