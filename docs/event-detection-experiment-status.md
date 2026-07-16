# Event detection experiment status

Last saved: 2026-07-17 (Asia/Seoul)

## 구현 상태

영상에서 정답 포즈나 정답 선수 ID 없이 red/blue 선수의 left/right 펀치 시간 구간을 검출하는 경로가 완성됐다.

```text
MP4
  -> YOLO 사람 검출 + BoT-SORT
  -> 매 10프레임 RGB red/blue/non-boxer UVE 신원 보정
  -> RTMW-m 2D 포즈 + 누락 보간/평활화
  -> 상대 관계 운동 특징
  -> TCN/MSTCN/TCN-GRU 앙상블
  -> 손별 temporal NMS
  -> boxer, hand, start_frame, end_frame, score
```

최종 모델 명세는 `results/rtmw-punch-detector-ensemble.json`, 모델 파일은 `models/event-detector/`, 상세 결과는 `results/event-detection-final-report.json`에 있다.

## 고정된 성능

모델과 operating point는 마지막 4개 train 경기에서만 선택했다.

- UVE 포즈 validation: precision 0.710, recall 0.697, F1 0.704
- 전체 40개 train 경기 재학습 후 10개 held-out test 최초 평가:
  precision 0.732, recall 0.600, F1 0.659
- 기준: temporal IoU 0.5, 선수와 물리적 left/right hand 일치
- threshold 0.75, NMS IoU 0.2
- test 결과: TP 729, FP 267, FN 486

테스트 라벨은 모델·가중치·threshold·NMS를 모두 고정한 뒤 한 번만 평가했으며 튜닝에 사용하지 않았다.

## 해석

이전 최고 validation F1 0.725는 GT 포즈로 red/blue를 연결한 진단용 결과라 배포 성능이 아니다. 현재 F1 0.659는 실제 영상에서 UVE와 RTMW로 입력을 만드는 완전한 GT-free 경로의 held-out test 성능이다. 경기별 F1 범위는 0.553~0.766이고, precision보다 recall 손실이 크다.

BoxMind 논문의 0.783과는 아직 차이가 있다. 논문은 BoxingWeb 40경기 외에 공개되지 않은 BoxingStudio 30경기를 함께 학습하고, 코드와 가중치가 공개되지 않은 UV-map UVE 분류기를 사용한다. 현재 RGB UVE의 validation IDF1은 0.898로 논문의 0.985보다 낮다. 따라서 추가 데이터나 공식 UVE 없이 같은 수치를 주장할 수 없다.

## 재현 경로

대형 실험 자산은 다음 위치에 보존돼 있다.

```text
/Users/kgw7401/.local/share/atom-experiments/2026-07-16
```

- `rtmw2d-uve/train`: UVE+RTMW 포즈 40경기
- `rtmw2d-uve/test`: UVE+RTMW 포즈 10경기
- `rtmw2d-uve/tracks`: 프레임별 추적 및 신원 확률
- `rtmw2d-fullrate`: 진단용 안정 RTMW 포즈
- `artifacts`: 선택/재학습 checkpoint와 실험 report
- `models`, `mmpose-source`, `venv`: 외부 pose 모델과 실행 환경

raw MP4 전체 경로는 `scripts/detect_video_punch_events.py`, 이미 생성된 포즈 파일에서 이벤트만 출력하는 명령은 다음과 같다.

```bash
EXP=/Users/kgw7401/.local/share/atom-experiments/2026-07-16
$EXP/venv/bin/python scripts/detect_video_punch_events.py \
  --video match.mp4 --pose-output match.pose.pkl --output punch_events.json \
  --identity-checkpoint $EXP/artifacts/uve-rgb-appearance-classifier.pt \
  --pose-config $EXP/mmpose-source/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py \
  --pose-checkpoint $EXP/models/rtmw2d-m.pth \
  --yolo $EXP/models/yolo11n.pt
```

```bash
python scripts/detect_punch_events.py \
  --pose path/to/uve_pose.pkl \
  --ensemble results/rtmw-punch-detector-ensemble.json \
  --output punch_events.json
```

held-out test를 다시 튜닝 용도로 평가하지 않는다. 다음 성능 개선은 train 내부 교차검증으로만 수행하고, 새 독립 test 세트를 확보한 뒤 검증해야 한다.

## 다음 병목

이벤트 검출 레이어의 구현은 완료됐지만 논문 수준 성능은 달성하지 못했다. 우선순위는 다음과 같다.

1. 더 많은 실제 스파링/BoxingStudio 유사 라운드와 누락 후보를 추가 학습한다.
2. RGB 의상 descriptor를 UV-map 또는 강한 re-identification embedding으로 교체한다.
3. train-only 교차검증에서 빠른 펀치의 recall과 선수 가림 상황을 집중 개선한다.
4. 새 독립 평가 영상을 확보해 개선 모델을 비교한다.
