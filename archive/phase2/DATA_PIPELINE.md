# 🔄 Atom 데이터 파이프라인

## 📊 데이터 플로우

```
1. RAW (원본 배치)
   └─> 10초 영상, 여러 동작 섞임
       └─> data/raw/v1_webcam/batches/jab_front_batch_*.mp4

2. SPLIT (개별 클립)
   └─> 배치를 1-2초 클립으로 분리
       └─> data/processed/splits/v1/jab/jab_front_001.mp4

3. KEYPOINTS (키포인트)
   └─> 개별 클립에서 추출
       └─> data/processed/keypoints/v1/jab_front_001.npy

4. TRAINING (학습)
   └─> 키포인트 .npy 파일 사용
       └─> train.py --data_dir data/processed/keypoints/merged
```

## 📁 정확한 폴더 구조

```
data/
├── raw/                                  # 원본 배치 영상
│   └── v1_webcam/
│       ├── metadata.json
│       └── batches/                      # 10초 배치 영상들
│           ├── jab_front_batch_*.mp4
│           ├── cross_left_batch_*.mp4
│           └── ...
│
├── processed/
│   ├── splits/                           # 쪼개진 개별 클립
│   │   └── v1/
│   │       ├── jab/
│   │       │   ├── jab_front_001.mp4
│   │       │   ├── jab_front_002.mp4
│   │       │   └── ... (30개)
│   │       ├── cross/
│   │       │   └── ... (30개)
│   │       └── ... (6 actions × 30 = 180개)
│   │
│   └── keypoints/                        # 키포인트 (학습 직접 사용!)
│       ├── v1/
│       │   ├── jab_front_001.npy
│       │   ├── cross_front_001.npy
│       │   └── ... (180개)
│       │
│       └── merged/                       # v1+v2+... 병합
│           └── ... (360개)
│
├── splits_txt/                           # Train/Val/Test 리스트
│   ├── train.txt                         # 키포인트 파일명 리스트
│   ├── val.txt
│   └── test.txt
│
└── labels.txt                            # 라벨 (키포인트 파일명 → action)
```

## 🔄 워크플로우

### 현재 (v1 웹캠 데이터)

```bash
1. 배치 녹화 (이미 완료)
   data/batches/*.mp4 (10초 배치 영상)

2. 개별 클립 분리 (이미 완료)
   ml/scripts/split_video.py 사용
   → data/splits/{action}/*.mp4

3. 키포인트 추출 (이미 완료)
   ml/scripts/extract_keypoints.py
   → data/keypoints/*.npy

4. 라벨 생성 (이미 완료)
   → data/labels.txt

5. Train/Val/Test split (이미 완료)
   → data/splits_txt/*.txt

6. 학습 (이미 완료)
   → checkpoints/boxing_6class_v1/
```

### 새로운 v2 데이터 수집 시

```bash
1. 배치 녹화
   iPhone으로 각 각도별 10초 배치 녹화
   → data/raw/v2_iphone_angles/batches/

2. 개별 클립 분리
   rye run python ml/scripts/split_video.py \
       --input data/raw/v2_iphone_angles/batches \
       --output data/processed/splits/v2

3. 키포인트 추출
   rye run python ml/scripts/extract_keypoints.py \
       --input data/processed/splits/v2 \
       --output data/processed/keypoints/v2

4. 라벨 생성
   rye run python ml/scripts/create_labels.py \
       --keypoints_dir data/processed/keypoints/v2 \
       --output data/processed/keypoints/v2_labels.txt

5. 데이터 병합
   rye run python ml/scripts/merge_datasets.py \
       --sources data/processed/keypoints/v1 \
                 data/processed/keypoints/v2 \
       --output data/processed/keypoints/merged \
       --labels_output data/labels_merged.txt

6. Split 재생성
   rye run python ml/scripts/create_splits.py \
       --keypoints_dir data/processed/keypoints/merged \
       --labels_file data/labels_merged.txt \
       --output_dir data/splits_txt_v2

7. 재학습
   rye run python ml/scripts/train.py \
       --data_dir data/processed/keypoints/merged \
       --labels_file data/labels_merged.txt \
       --train_split data/splits_txt_v2/train.txt \
       --val_split data/splits_txt_v2/val.txt \
       --output_dir checkpoints/boxing_v2_diverse
```

## 🎯 핵심 포인트

### ✅ RAW (보존용)
- **배치 영상**: 10초, 여러 동작 섞임
- **절대 삭제 금지**: 언제든 다시 처리 가능
- **백업 필수**: 외부 저장소에도 보관

### 🔧 PROCESSED (작업용)
- **개별 클립**: 1-2초, 동작 하나씩
- **키포인트**: 학습에 직접 사용
- **버전 관리**: v1, v2, v3... + merged

### 📊 TRAINING (최종)
- **키포인트 .npy**: 학습 입력
- **라벨 .txt**: 파일명 → 동작 매핑
- **Split .txt**: Train/Val/Test 리스트

## 💡 왜 이렇게?

### 배치 → 개별 클립?
- **편리한 녹화**: 한 번에 여러 동작
- **정확한 라벨링**: 나중에 천천히 쪼개기
- **유연한 처리**: 필요시 재분리 가능

### 원본 보존?
- **재처리 가능**: 알고리즘 개선 시
- **실험 반복**: 다른 전처리 시도
- **품질 보장**: 압축/변환 없이 원본 유지

## 📝 데이터 수집 팁

### iPhone으로 배치 녹화
1. 삼각대 고정
2. 각도별 폴더 준비
3. 각 각도에서:
   - 10초 녹화
   - jab 2개, cross 2개, hook 2개... 섞어서
   - 파일명: `mixed_batch_{angle}_{번호}.MOV`

### 나중에 쪼개기
```bash
# 자동 감지 or 수동 타임스탬프
rye run python ml/scripts/split_video.py \
    --input mixed_batch_front_001.MOV \
    --output splits/ \
    --timestamps 0.5,2.1,3.8,5.5...
```

## 🔍 현재 데이터 확인

```bash
# 배치 영상
ls -lh data/batches/

# 개별 클립
ls -lh data/splits/jab/
ls -lh data/splits/cross/

# 키포인트
ls -lh data/keypoints/ | wc -l

# 라벨
head -20 data/labels.txt

# Split 리스트
wc -l data/splits_txt/*.txt
```
