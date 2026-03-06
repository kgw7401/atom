# 🚀 Atom 데이터 관리 Quick Start

## Step 1: 기존 데이터 재구성 (지금 바로!)

```bash
# 현재 데이터를 프로페셔널 구조로 재구성
rye run python ml/scripts/reorganize_data.py
```

**실행 후:**
- `data/raw/v1_webcam/` - 원본 영상
- `data/processed/keypoints/v1/` - 키포인트
- `data/splits/v1/` - Train/Val/Test split
- `checkpoints/boxing_v1_webcam/` - 모델

---

## Step 2: 새로운 데이터 수집 (iPhone)

### 2.1 폴더 준비
```bash
mkdir -p data/raw/v2_iphone_angles/{angle_front,angle_left30,angle_right30,angle_left60,angle_right60,angle_top}
```

### 2.2 메타데이터 작성
```bash
cat > data/raw/v2_iphone_angles/metadata.json << 'JSON'
{
  "version": "v2_iphone_angles",
  "collection_date": "2026-03-03",
  "camera": "iPhone 15 Pro",
  "resolution": "1920x1080",
  "fps": 30,
  "angles": ["front", "left30", "right30", "left60", "right60", "top"],
  "distance": "2.5m",
  "lighting": "indoor_bright",
  "samples_per_action": 30,
  "total_samples": 180,
  "notes": "Diverse angle data for domain generalization"
}
JSON
```

### 2.3 영상 촬영
각 각도 폴더에서:
- jab 5개
- cross 5개
- lead_hook 5개
- rear_hook 5개
- lead_uppercut 5개
- rear_uppercut 5개

파일명: `{action}_{번호}.MOV` (예: `jab_001.MOV`)

---

## Step 3: 새 데이터 처리

### 3.1 키포인트 추출
```bash
rye run python ml/scripts/extract_keypoints.py \
    --input data/raw/v2_iphone_angles \
    --output data/processed/keypoints/v2
```

### 3.2 라벨 생성
```bash
rye run python ml/scripts/create_labels.py \
    --keypoints_dir data/processed/keypoints/v2 \
    --output data/processed/metadata/labels_v2.txt
```

### 3.3 데이터 병합
```bash
rye run python ml/scripts/merge_datasets.py \
    --sources data/processed/keypoints/v1 data/processed/keypoints/v2 \
    --output data/processed/keypoints/merged \
    --labels_output data/processed/metadata/labels_merged.txt
```

### 3.4 Split 생성
```bash
rye run python ml/scripts/create_splits.py \
    --keypoints_dir data/processed/keypoints/merged \
    --labels_file data/processed/metadata/labels_merged.txt \
    --output_dir data/splits/v2_merged

# current 심볼릭 링크 업데이트
cd data/splits
rm -f current
ln -s v2_merged current
```

---

## Step 4: 재학습

```bash
rye run python ml/scripts/train.py \
    --data_dir data/processed/keypoints/merged \
    --labels_file data/processed/metadata/labels_merged.txt \
    --train_split data/splits/v2_merged/train.txt \
    --val_split data/splits/v2_merged/val.txt \
    --output_dir checkpoints/boxing_v2_diverse \
    --epochs 80 \
    --batch_size 16 \
    --lr 0.01
```

**학습 완료 후:**
```bash
# current 체크포인트 링크 업데이트
cd checkpoints
rm -f current
ln -s boxing_v2_diverse current
```

---

## Step 5: 테스트

### 실시간 테스트
```bash
rye run python ml/scripts/realtime_demo.py \
    --checkpoint checkpoints/current/checkpoint_best.pth \
    --confidence 0.5
```

### 영상 파일 테스트
```bash
rye run python ml/scripts/test_live.py \
    --video data/IMG_9101.MOV \
    --checkpoint checkpoints/current/checkpoint_best.pth \
    --confidence 0.5
```

### 평가
```bash
rye run python ml/scripts/evaluate.py \
    --checkpoint checkpoints/current/checkpoint_best.pth \
    --data_dir data/processed/keypoints/merged \
    --labels_file data/processed/metadata/labels_merged.txt \
    --test_split data/splits/current/test.txt \
    --benchmark
```

---

## 📊 데이터 현황 확인

```bash
# 전체 통계
cat data/processed/metadata/stats.json | python -m json.tool

# 각 버전 메타데이터
cat data/raw/v1_webcam/metadata.json | python -m json.tool
cat data/raw/v2_iphone_angles/metadata.json | python -m json.tool

# 라벨 확인
head -20 data/processed/metadata/labels_merged.txt
wc -l data/processed/metadata/labels_*.txt
```

---

## 🔧 문제 해결

### 심볼릭 링크가 안 되는 경우
```bash
# 절대 경로로 생성
cd data/splits
ln -sf "$(pwd)/v2_merged" current
```

### 중복 파일 확인
```bash
# 병합 시 중복 체크
find data/processed/keypoints/merged -name "*.npy" | \
    awk -F/ '{print $NF}' | sort | uniq -d
```

### 데이터셋 크기 확인
```bash
du -sh data/raw/*/
du -sh data/processed/keypoints/*/
```

---

## 📝 체크리스트

데이터 수집 전:
- [ ] 메타데이터 작성
- [ ] 폴더 구조 준비
- [ ] 백업 계획

데이터 수집 후:
- [ ] 원본 백업
- [ ] 키포인트 추출
- [ ] 라벨 생성
- [ ] 데이터 병합

학습 전:
- [ ] Split 생성 확인
- [ ] 데이터 분포 확인
- [ ] 이전 모델 백업

학습 후:
- [ ] Test 정확도 확인
- [ ] 실시간 테스트
- [ ] 체크포인트 보관
