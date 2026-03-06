# Atom - Boxing Motion Digital Twin

권투 동작 인식 및 분석 플랫폼

## 📁 프로젝트 구조

```
atom/
├── archive/              # 메인 코드베이스
│   ├── src/
│   │   ├── extraction/   # MediaPipe 키포인트 추출
│   │   ├── preprocessing/# 전처리 파이프라인
│   │   ├── models/       # LSTM 모델
│   │   └── inference/    # 추론 유틸리티
│   ├── scripts/
│   │   ├── train_lstm.py         # 모델 학습
│   │   ├── realtime_classify.py  # 실시간 분류
│   │   └── extract_keypoints.py  # 키포인트 추출
│   ├── configs/
│   │   └── boxing.yaml   # 설정
│   └── models/
│       └── lstm_best.pt  # 학습된 모델 (99.7% 정확도)
├── data/                 # 데이터
├── server/               # FastAPI 서버
├── spec/                 # 스펙 문서
└── tests/                # 테스트
```

## 🥊 지원 동작 (9 classes)

1. **jab** - 잽
2. **cross** - 크로스
3. **lead_hook** - 리드 훅
4. **rear_hook** - 리어 훅
5. **lead_uppercut** - 리드 어퍼컷
6. **rear_uppercut** - 리어 어퍼컷
7. **lead_bodyshot** - 리드 바디샷
8. **rear_bodyshot** - 리어 바디샷
9. **guard** - 가드 자세

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# Rye 환경 사용
rye sync
```

### 2. 실시간 동작 인식
```bash
cd archive
PYTHONPATH=. python scripts/realtime_classify.py
```

### 3. 비디오에서 키포인트 추출
```bash
cd archive
PYTHONPATH=. python scripts/extract_keypoints.py \
    --raw-dir ../data/videos \
    --out-dir ../data/keypoints
```

### 4. 모델 학습
```bash
cd archive
PYTHONPATH=. python scripts/train_lstm.py \
    --epochs 100 \
    --batch-size 64
```

## 🎯 모델 성능

- **아키텍처**: LSTM (2-layer, 128 hidden)
- **입력**: 6-frame window, MediaPipe 33 keypoints
- **정확도**: 99.7% (test set)
- **추론 속도**: 실시간 (30 FPS)
- **파라미터**: 222K

## 📊 기술 스택

- **포즈 추정**: MediaPipe Pose Landmarker
- **모델**: PyTorch LSTM
- **전처리**: Hip-center normalization, Shoulder-width scaling
- **데이터 증강**: Scale jitter (0.85-1.15x)
- **서버**: FastAPI (Phase 2)

## 🔧 주요 기능

### 실시간 분류
- 웹캠에서 실시간 동작 인식
- Transition filter (guard → attack → guard)
- 신뢰도 기반 필터링

### 키포인트 추출
- MediaPipe 33 keypoints
- 가시성 기반 필터링
- NaN 보간 처리

### 전처리 파이프라인
- 11개 상체 키포인트 선택
- Hip-center 정규화
- Shoulder-width 스케일링
- Savitzky-Golay 평활화

## 📝 개발 히스토리

- **Phase 1**: LSTM 9-class 분류기 (99.7% 정확도)
- **Phase 2a**: State Engine 구현 (18-dim state vector)
- **Phase 2b-f**: REST API, Session runtime, Coaching reports

## 🎓 참고

자세한 내용은 `spec/` 디렉토리의 문서를 참조하세요:
- `spec/overview.md` - 시스템 철학
- `spec/state-vector.md` - 18차원 상태 벡터 수학 스펙
- `spec/runtime.md` - 세션 라이프사이클
- `spec/roadmap.md` - Phase 2 로드맵

## 📦 백업

이전 코드 (XGBoost/CTR-GCN)는 `atom_backup_ml_20260306.tar.gz`에 백업되어 있습니다.
