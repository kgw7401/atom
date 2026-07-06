#!/bin/bash
# 디버깅용 상세 테스트

echo "🔍 디버깅 테스트"
echo "================================"
echo ""

# Use a known good video from training
video="data/batches/cross_front_batch_20260301_235439.mp4"

echo "📹 테스트 영상: $video"
echo ""

# Test with VERY low confidence threshold
echo "🎯 신뢰도 임계값 0.3으로 테스트 (매우 낮음)..."
echo ""

rye run python ml/scripts/test_live.py \
    --video "$video" \
    --checkpoint checkpoints/boxing_6class_v1/checkpoint_best.pth \
    --confidence 0.3 \
    --stride 3

echo ""
echo "================================"
