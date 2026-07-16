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

상태: 완료. `scripts/audit_boxingweb.py`가 영상·이벤트·포즈 구조를 검사한다. 50경기·6,872개 펀치 이벤트가 문서의 split과 일치했다. `0-0` placeholder 311개와 역전 구간 5개는 제외하며, 포즈·영상 길이가 1~2프레임 다른 4경기는 공통 범위만 사용한다.

### B. Oracle-window 속성 분류

정답 `frame_begin/frame_end`를 사용해 펀치 클립을 자른다. 공식 코드와 의존성을 고정한 뒤 최소 한 번 train/test 평가를 재현한다.

첫 구현은 학습 스크립트가 유효한 정답 구간을 자동으로 인덱싱한다. 공식 BoxingWeb 구현과 비교할 수 있도록 `frame_end - frame_begin`이 4~30인 구간만 포함하며, 필요한 경우에만 앞뒤 문맥을 옵션으로 추가한다. 이벤트 구간은 양 끝을 포함하고, RGB·포즈 클립 경계는 끝이 제외되는 범위로 통일한다. 이 인덱스는 검출 모델에 사용하지 않는다.

상태: 2026-07-14에 pose-only 다중 과제 MLP를 최초 기준선으로 실행했다. 공식 코드와 동일한 데이터 split·유효 구간·속성 정의, 그리고 공격 손 입력을 쓰지만, 공식 RGB+TCN/MoE 모델의 재현은 아니다. test macro-F1은 기술 0.655, 거리 0.594, 표적 0.505, 효과 0.583(평균 0.584)이었다. 상세 혼동 행렬은 `results/pose-baseline-report.json`에 남긴다.

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

첫 실험은 GT pose를 입력으로 하는 시간적 검출기다. 각 프레임에서 red/blue의 펀치 활성 상태와 이벤트 시작·끝을 동시에 예측하고, 시작·끝 후보를 4~30프레임 길이의 이벤트로 짝지어 IoU별 event F1을 측정한다. 이 실험은 영상 포즈 추정·선수 추적의 오류를 분리해, 포즈 움직임 자체가 펀치 검출에 충분한지부터 확인한다.

상태: 2026-07-15에 36 train 경기·4 validation 경기·10 test 경기로 GT pose 시간적 검출기를 실행했다. validation으로 이벤트·활성 임계값을 고른 뒤, 더 넓은 시간 문맥을 사용하는 pose-only 모델의 test event F1은 IoU 0.1/0.3/0.5 기준 각각 0.655/0.621/0.540이었다. IoU 0.5에서 728개 거짓 양성과 496개 누락이 남는다. 특히 4~7프레임 펀치의 재현율은 0.250으로, 포즈만으로 빠른 동작의 경계를 정확히 잡는 데 한계가 보인다.

같은 split에서 GT pose가 안내하는 선수 crop의 저해상도 RGB 프레임 차분을 결합한 첫 융합 실험은 IoU 0.1/0.3/0.5 event F1이 0.586/0.545/0.460으로 하락했다. 따라서 이 단순 RGB 특징은 채택하지 않는다. 결과는 RGB 자체의 가치 판단이 아니라, 외형·움직임을 충분히 표현하지 못하는 이 특징 설계의 실패로 해석한다.

이후 논문의 독립 선수별 anchor-free 형식에 맞춘 TCN으로 교체했다. GT pose 전체 test에서 2D+3D F1은 0.750, 2D-only F1은 0.719였다. YOLO 사람 검출과 RTMW3D를 이용해 실제 MP4 한 경기 5,493프레임을 처리했으며, red/blue 대응 성공률은 각각 96.6%, 94.2%였다. 영상 RTMW 2D와 GT 3D를 결합하면 해당 경기의 GT 기준선과 같은 F1 0.748이 나왔지만, RTMW 단안 3D를 넣으면 0.152로 하락했다. 따라서 첫 실제 영상 경로는 단안 3D를 제외한 2D-only로 고정한다. 짧은 누락 보간과 3프레임 평활화 후 실제 영상 F1은 0.603이다. 현재 평가는 GT pose를 선수 red/blue 대응에만 사용한 진단 실험이며, UVE 외형 식별기로 이 oracle 의존성을 제거해야 한다.

2026-07-16에는 이 oracle association을 실제 UVE 호환 경로로 교체했다. 논문의 공개 설명대로 사람을 red boxer, blue boxer, non-boxer로 분류하고 10프레임마다 트랙 신원을 재검증한다. 4D-Humans UV-map 분류기 코드와 가중치는 공개되지 않아, 현재 구현은 BoT-SORT 위치 연속성과 훈련 경기에서 학습한 RGB 의상 descriptor를 사용한다. 검증 4경기 합산 ID precision/recall/IDF1은 0.899/0.897/0.898이다. 같은 상대 관계 펀치 검출기는 oracle association에서 F1 0.687, UVE 포즈에서 F1 0.650을 기록했다. 추론 시 GT 의존성은 제거됐지만, 논문의 UVE IDF1 0.985와는 차이가 있으므로 더 강한 appearance embedding과 UVE 분포 재학습이 필요하다.

후속 실험에서는 UVE+RTMW-m 2D 포즈를 train 40경기와 test 10경기에서 추출했다. UVE 분포 재학습 모델과 안정 RTMW 분포 모델, TCN/MSTCN/TCN-GRU를 train validation에서 조합한 결과 F1 0.704를 기록했다. 설정을 고정하고 전체 40경기로 재학습한 뒤 한 번만 수행한 held-out test 결과는 precision 0.732, recall 0.600, F1 0.659다. 추론 시 GT pose나 GT red/blue 연결을 사용하지 않는다. 최종 구성은 `results/rtmw-punch-detector-ensemble.json`, 상세 결과는 `results/event-detection-final-report.json`에 고정했다.

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
