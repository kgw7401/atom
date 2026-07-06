# Archive로 전환 계획

## 현재 상태
- ✅ **archive/**: 작동하는 LSTM (OLD 모델, 성능 좋음)
- ❌ **ml/**: XGBoost/CTR-GCN 관련 코드 (사용 안함)
- ✅ **data/**: 유지
- ✅ **spec/**: 유지
- ✅ **server/**: 유지

## 변경 사항

### 삭제할 디렉토리/파일
```
ml/                          # 전체 삭제
checkpoints/                 # OLD 모델 백업 후 삭제
*.py (루트의 테스트 스크립트들)
```

### 유지할 것
```
archive/                     # 메인 코드베이스
data/                        # 데이터
spec/                        # 스펙 문서
server/                      # 서버 코드
configs/                     # 설정
```

### Archive 구조
```
archive/
├── src/
│   ├── extraction/          # MediaPipe 추출
│   ├── preprocessing/       # 전처리 파이프라인
│   ├── models/              # LSTM 모델
│   └── inference/           # 추론 유틸
├── scripts/
│   ├── train_lstm.py        # 학습
│   ├── realtime_classify.py # 실시간 분류
│   └── extract_keypoints.py # 키포인트 추출
├── configs/
│   └── boxing.yaml          # 설정
└── models/
    └── lstm_best.pt         # 학습된 모델 (OLD)
```

## 다음 단계

1. ml/ 디렉토리 삭제
2. 루트의 불필요한 스크립트 정리
3. README 업데이트
4. archive를 메인으로 사용
