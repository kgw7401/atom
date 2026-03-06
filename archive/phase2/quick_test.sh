#!/bin/bash
# 가장 간단한 테스트 - 기존 영상 중 하나 선택

echo "🥊 사용 가능한 영상들:"
echo ""
find data/batches -name "*.mp4" -type f | nl
echo ""
read -p "테스트할 영상 번호 입력: " num

video=$(find data/batches -name "*.mp4" -type f | sed -n "${num}p")

if [ -z "$video" ]; then
    echo "❌ 잘못된 번호입니다"
    exit 1
fi

echo ""
echo "🎯 테스트 중: $video"
echo ""

rye run python ml/scripts/test_live.py \
    --video "$video" \
    --checkpoint checkpoints/boxing_6class_v1/checkpoint_best.pth \
    --confidence 0.6

echo ""
echo "✅ 완료!"
