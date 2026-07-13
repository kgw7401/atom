# BoxMind 재현 기반 계획

## 1. 목적

BoxMind의 전체 전략 최적화 시스템을 복제하는 것이 아니라, 개인 스파링 의사결정 분석에 필요한 하위 관찰 계층을 재현한다.

첫 기술 목표는 다음과 같다.

> 전체 스파링 영상을 입력하면 누가 언제 어떤 펀치를 했고, 당시 거리와 결과 후보가 무엇인지 근거와 함께 타임라인으로 출력한다.

## 2. 보유 자산

로컬 데이터 경로는 `~/boxingweb`이다. 공식 BoxingWeb 공개 저장소와 같은 train/test 경기 및 파일 구성을 가진다.

```text
boxingweb/
  data_train/  # 40 videos, 5,537 valid punch events
  data_test/   # 10 videos, 1,335 valid punch events
```

각 경기에는 다음이 있다.

- 전체 라운드 MP4
- `video_event.json`
- `*_pose_gt.pkl`

펀치 이벤트 라벨:

```text
frame_begin, frame_end
side
technique: l/r straight, hook, uppercut
distance: long, medium, close
target: head, chest, abdomen
effect: effective, ineffective
```

공식 저장소에는 RGB I3D, pose encoder, TCN/MoE fusion을 이용한 속성 분류 학습 코드가 공개되어 있다. 데이터와 외부 코드는 이 저장소에 복사하지 않고 경로와 버전을 명시해 참조한다.

## 3. 그대로 가져올 부분

- 정확한 시간 경계를 가진 atomic punch event 표현
- RGB와 포즈의 시간 정보 결합
- 기술, 거리, 표적, 효과의 다중 속성 분류
- 경기 단위 train/test 분리
- 하위 사건에서 상위 지표로 올라가는 계층 구조

## 4. 다르게 만들 부분

BoxMind는 경기 결과와 상대 전략 최적화가 중심이다. 이 프로젝트는 개인의 선택 개선이 중심이므로 다음을 우선한다.

- 내가 누구인지 지속적으로 추적
- 상대 전진과 거리 감소
- 직선 후퇴와 측면 탈출
- 방어 후 멈춤 또는 다음 행동
- 로프·코너 근접
- 같은 실패의 반복과 다음 라운드 실행 여부

승률 예측, 자동 채점과 18개 지표 전체는 초기 범위에서 제외한다.

## 5. 구현 단계

### A. 데이터 감사

- JSON 스키마와 pose pickle 구조 문서화
- 영상 FPS, 해상도, 이벤트 길이와 클래스 분포 확인
- 누락 값, 겹치는 펀치, 0프레임 이벤트 처리 규칙 결정
- 데이터 라이선스와 원본 출처 확인

완료 조건: 같은 입력에서 항상 같은 데이터 통계와 split을 만드는 검사 스크립트가 있다.

### B. Oracle-window 속성 분류

정답 `frame_begin/frame_end`를 사용해 펀치 클립을 자른다. 공식 코드와 의존성을 고정한 뒤 최소 한 번 train/test 평가를 재현한다.

측정 항목:

- 손을 제외한 trajectory: straight/hook/uppercut
- 손: left/right
- 거리, 표적, 효과
- 클래스별 precision, recall, F1과 confusion matrix

완료 조건: 공식 split에서 재현 가능한 기준선과 오류 샘플을 남긴다.

### C. 전체 영상 이벤트 검출

정답 구간을 입력하지 않고 프레임별 또는 구간별 punch/background와 시작·종료를 예측한다. 속성 분류와 이벤트 검출의 오류를 분리한다.

측정 항목:

- temporal IoU별 mAP
- event precision/recall/F1
- 경기당 거짓 양성 수
- 공격 선수 ID 정확도

완료 조건: test 경기 하나의 전체 타임라인을 자동 생성하고 정답과 겹쳐 볼 수 있다.

### D. 개인 영상 전이 검사

모델을 개인 스파링 영상에 그대로 적용한다. 방송 영상과 체육관 고정 카메라의 차이로 생기는 오류를 기록한다.

확인 항목:

- 선수 ID 유지
- 가림과 클린치에서의 추적 실패
- 작은 선수 크기와 카메라 각도
- 장갑·복장·조명의 도메인 차이

완료 조건: 추가 학습 전에 실패 유형과 필요한 개인 라벨의 최소 범위를 정한다.

### E. 의사결정에 필요한 이벤트 확장

개인 영상에서 다음 최소 클래스를 순차적으로 추가한다.

```text
defense: block, parry, slip, duck, lean_back
footwork: step_back, side_step, pivot, clinch
state: advancing, retreating, distance, rope_or_corner
```

완료 조건: 적어도 하나의 압박 장면을 `상대 행동 -> 내 반응 -> 결과`로 재구성할 수 있다.

## 6. 출력 계약 초안

```json
{
  "start_frame": 420,
  "end_frame": 438,
  "actor": "opponent",
  "event": "punch",
  "attributes": {
    "hand": "lead",
    "trajectory": "straight",
    "distance": "medium",
    "target": "head",
    "effect": "unclear"
  },
  "confidence": 0.81,
  "evidence": {
    "track_id": 2,
    "source": ["rgb", "pose"]
  }
}
```

상위 레이어는 이 이벤트를 수정하지 않고 참조한다. 모든 전략 문장은 해당 사건과 원본 프레임으로 역추적할 수 있어야 한다.

## 7. 중단 기준

다음 조건이면 모델 규모를 키우기 전에 문제를 다시 정의한다.

- 공식 test에서도 oracle-window 속성 분류가 재현되지 않는다.
- 전체 영상 이벤트 검출의 거짓 양성이 검토 시간을 줄이지 못할 정도로 많다.
- 개인 영상 도메인 차이가 적은 라벨 추가로 해결되지 않는다.
- 하위 사건 정확도가 높아져도 의사결정 장면과 행동 규칙으로 연결되지 않는다.

반대로 단계 C와 D에서 신뢰할 수 있는 타임라인이 나오면 방어·풋워크 데이터 수집으로 진행한다.
