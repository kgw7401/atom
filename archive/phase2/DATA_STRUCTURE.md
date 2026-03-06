# Atom 데이터 구조 및 관리

## 📁 폴더 구조

```
data/
├── raw/                          # 원본 배치 영상 (절대 삭제하지 말 것!)
│   ├── v1_webcam/               # 버전 1: 웹캠 데이터
│   │   ├── collection_date.txt  # 2026-03-01
│   │   ├── metadata.json        # 촬영 환경 정보
│   │   ├── batches/             # 10초 배치 영상 (여러 동작 섞임)
│   │   │   ├── jab_front_batch_20260301_233833.mp4
│   │   │   ├── cross_left_batch_20260301_234152.mp4
│   │   │   └── ...
│   │
│   ├── v2_iphone_angles/        # 버전 2: iPhone 다양한 각도
│   │   ├── collection_date.txt  # 2026-03-03
│   │   ├── metadata.json
│   │   ├── angle_front/
│   │   ├── angle_left30/
│   │   └── ...
│   │
│   └── v3_distances/            # 버전 3: 거리 변화 (미래)
│       └── ...
│
├── processed/                    # 처리된 데이터
│   ├── splits/                  # 배치에서 쪼개진 개별 클립 (학습 직접 사용!)
│   │   ├── v1/                  # v1 배치에서 추출
│   │   │   ├── jab/
│   │   │   │   ├── jab_front_001.mp4
│   │   │   │   ├── jab_front_002.mp4
│   │   │   │   └── ...
│   │   │   ├── cross/
│   │   │   ├── lead_hook/
│   │   │   └── ...
│   │   │
│   │   └── v2/                  # v2 배치에서 추출
│   │       └── ...
│   │
│   ├── keypoints/               # 쪼개진 클립에서 추출한 키포인트
│   │   ├── v1/                  # v1 splits에서 추출
│   │   │   ├── jab_front_001.npy
│   │   │   ├── cross_front_001.npy
│   │   │   └── ...
│   │   │
│   │   ├── v2/                  # v2 원본에서 추출
│   │   │   └── ...
│   │   │
│   │   └── merged/              # 병합된 최신 버전 (학습용)
│   │       ├── jab_front_001.npy      # v1
│   │       ├── jab_angle30_001.npy    # v2
│   │       └── ...
│   │
│   └── metadata/                # 메타데이터
│       ├── labels_v1.txt
│       ├── labels_v2.txt
│       ├── labels_merged.txt    # 최신 병합 라벨
│       └── stats.json           # 데이터셋 통계
│
├── splits/                      # Train/Val/Test 분할
│   ├── v1/                      # v1 데이터 split
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── test.txt
│   │
│   ├── v2_merged/               # v1+v2 병합 split
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── test.txt
│   │
│   └── current -> v2_merged/    # 심볼릭 링크 (현재 사용 중)
│
└── archive/                     # 더 이상 안 쓰는 데이터
    └── old_experiments/

checkpoints/
├── boxing_v1_webcam/            # v1 데이터로 학습한 모델
│   ├── checkpoint_best.pth
│   ├── training_history.json
│   └── config.yaml
│
├── boxing_v2_diverse/           # v2 병합 데이터로 학습
│   └── ...
│
└── current -> boxing_v2_diverse/  # 현재 사용 중 모델
```

## 📝 metadata.json 예시

```json
{
  "version": "v2_iphone_angles",
  "collection_date": "2026-03-03",
  "camera": "iPhone 15 Pro",
  "resolution": "1920x1080",
  "fps": 30,
  "angles": ["front", "left30", "right30", "left60", "right60", "top"],
  "distance": "2.5m",
  "lighting": "indoor_bright",
  "background": "gym",
  "samples_per_action": 30,
  "total_samples": 180,
  "collector": "user",
  "notes": "Diverse angle collection for domain generalization"
}
```

## 🔄 워크플로우

### 1. 새로운 데이터 수집 시

```bash
# 1. 버전 디렉토리 생성
mkdir -p data/raw/v2_iphone_angles
cd data/raw/v2_iphone_angles

# 2. 메타데이터 작성
cat > metadata.json << 'JSON'
{
  "version": "v2_iphone_angles",
  "collection_date": "2026-03-03",
  ...
}
JSON

# 3. 영상 촬영 및 저장
# 각 각도별 폴더에 저장

# 4. 백업 (중요!)
cp -r data/raw/v2_iphone_angles /backup/location/
```

### 2. 키포인트 추출

```bash
# v2 원본 → v2 키포인트
python ml/scripts/extract_keypoints.py \
    --input data/raw/v2_iphone_angles \
    --output data/processed/keypoints/v2
```

### 3. 데이터 병합

```bash
# v1 + v2 → merged
python ml/scripts/merge_datasets.py \
    --sources data/processed/keypoints/v1 data/processed/keypoints/v2 \
    --output data/processed/keypoints/merged \
    --labels_output data/processed/metadata/labels_merged.txt
```

### 4. Split 생성

```bash
# merged 데이터로 split
python ml/scripts/create_splits.py \
    --keypoints_dir data/processed/keypoints/merged \
    --labels_file data/processed/metadata/labels_merged.txt \
    --output_dir data/splits/v2_merged
```

### 5. 학습

```bash
# merged 데이터로 학습
python ml/scripts/train.py \
    --data_dir data/processed/keypoints/merged \
    --labels_file data/processed/metadata/labels_merged.txt \
    --train_split data/splits/v2_merged/train.txt \
    --val_split data/splits/v2_merged/val.txt \
    --output_dir checkpoints/boxing_v2_diverse
```

## 💡 베스트 프랙티스

### ✅ DO:
1. **원본 항상 보존**: `data/raw/` 절대 삭제 금지
2. **버전 관리**: 각 수집 시점마다 v1, v2, v3...
3. **메타데이터 기록**: 촬영 환경, 날짜 등 상세히
4. **정기 백업**: 원본 데이터는 외부 저장소에도
5. **심볼릭 링크 사용**: `current` → 최신 버전
6. **Git에는 코드만**: 데이터는 .gitignore

### ❌ DON'T:
1. 원본 직접 수정
2. 버전 정보 없이 저장
3. 메타데이터 없이 수집
4. 백업 없이 삭제
5. 처리 과정 중간 파일을 원본과 섞기

## 🗂️ .gitignore 설정

```gitignore
# 데이터 파일들 (용량 크므로 git에서 제외)
data/raw/
data/processed/
data/splits/
checkpoints/*.pth
checkpoints/*.pt

# 메타데이터는 포함 (작은 파일)
!data/processed/metadata/*.json
!data/processed/metadata/*.txt

# 설정 파일은 포함
!checkpoints/*/config.yaml
!checkpoints/*/training_history.json
```

## 📊 데이터셋 통계 추적

`data/processed/metadata/stats.json`:

```json
{
  "last_updated": "2026-03-03T14:30:00",
  "total_samples": 360,
  "versions": {
    "v1_webcam": {
      "samples": 180,
      "date": "2026-03-01",
      "per_class": 30
    },
    "v2_iphone_angles": {
      "samples": 180,
      "date": "2026-03-03",
      "per_class": 30
    }
  },
  "class_distribution": {
    "jab": 60,
    "cross": 60,
    "lead_hook": 60,
    "rear_hook": 60,
    "lead_uppercut": 60,
    "rear_uppercut": 60
  },
  "environment_diversity": {
    "cameras": ["webcam", "iphone"],
    "angles": ["front", "left", "right", "left30", "right30", "left60", "right60", "top"],
    "distances": ["2.5m"],
    "lighting": ["indoor_bright"]
  }
}
```
