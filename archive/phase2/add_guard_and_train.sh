#!/bin/bash
set -e  # Exit on error

echo "=================================================="
echo "Guard 클래스 추가 및 7-class LSTM 학습"
echo "=================================================="

# Step 1: Guard 키포인트 추출
echo ""
echo "Step 1/5: Guard 영상에서 키포인트 추출..."
# rye run python ml/scripts/extract_keypoints.py \
#     --input_dir data/raw/guard \
#     --output_dir data/processed/keypoints/guard \
#     --model_path pose_landmarker_heavy.task

# Step 2: Guard 라벨 생성
echo ""
echo "Step 2/5: Guard 라벨 생성..."
python3 << 'EOF'
from pathlib import Path

keypoints_dir = Path('data/processed/keypoints/guard')
labels_file = Path('data/processed/metadata/labels_guard.txt')
labels_file.parent.mkdir(parents=True, exist_ok=True)

with open(labels_file, 'w') as f:
    for npy_file in sorted(keypoints_dir.glob('*.npy')):
        filename = npy_file.stem
        f.write(f'{filename} guard\n')

guard_count = len(list(keypoints_dir.glob('*.npy')))
print(f'✅ Created {labels_file}')
print(f'✅ Total guard samples: {guard_count}')
EOF

# Step 3: 7-class 데이터셋 병합
echo ""
echo "Step 3/5: 7-class 데이터셋 병합..."
python3 << 'EOF'
import shutil
from pathlib import Path

# Copy 6-class to new directory
src = Path('data/processed/keypoints/merged')
dst = Path('data/processed/keypoints/merged_7class')
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
merged_count = len(list(src.glob('*.npy')))
print(f'✅ Copied {merged_count} files from merged')

# Add guard keypoints
guard_src = Path('data/processed/keypoints/guard')
guard_count = 0
for npy_file in guard_src.glob('*.npy'):
    shutil.copy(npy_file, dst / npy_file.name)
    guard_count += 1
print(f'✅ Added {guard_count} guard samples')

# Merge labels
labels_6 = Path('data/processed/metadata/labels_6class.txt')
labels_guard = Path('data/processed/metadata/labels_guard.txt')
labels_7 = Path('data/processed/metadata/labels_7class.txt')

with open(labels_7, 'w') as out:
    with open(labels_6) as f:
        out.write(f.read())
    with open(labels_guard) as f:
        out.write(f.read())

total_count = len(list(dst.glob('*.npy')))
print(f'✅ Created labels_7class.txt')
print(f'✅ Total 7-class samples: {total_count}')
EOF

# Step 4: 7-class splits 생성
echo ""
echo "Step 4/5: Train/Val/Test splits 생성..."
rye run python ml/scripts/create_splits.py \
    --labels data/processed/metadata/labels_7class.txt \
    --output_dir data/splits_txt/final_7class \
    --train_ratio 0.75 \
    --val_ratio 0.10

# Step 5: 7-class LSTM 학습
echo ""
echo "Step 5/5: 7-class LSTM 학습 시작..."
echo "이 작업은 시간이 걸립니다 (약 30-60분)..."
rye run python ml/scripts/train_lstm.py \
    --config ml/configs/boxing.yaml \
    --data_dir data/processed/keypoints/merged_7class \
    --label_path data/processed/metadata/labels_7class.txt \
    --train_list data/splits_txt/final_7class/train.txt \
    --val_list data/splits_txt/final_7class/val.txt \
    --checkpoint_dir checkpoints/boxing_lstm_7class_v1 \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --num_workers 8

echo ""
echo "=================================================="
echo "✅ 완료!"
echo "=================================================="
echo "체크포인트: checkpoints/boxing_lstm_7class_v1/"
echo ""
