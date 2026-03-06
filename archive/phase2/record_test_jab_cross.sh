#!/bin/bash
# 학습 데이터와 동일한 환경에서 Jab/Cross 테스트 영상 녹화

echo "🥊 Jab/Cross 테스트 영상 녹화"
echo "=============================="
echo ""
echo "학습 데이터와 동일한 환경:"
echo "- 웹캠 사용"
echo "- 카메라와 2-3m 거리"
echo "- 전신이 보이게"
echo "- 정면 각도"
echo ""
read -p "준비되셨으면 Enter..."

# 10초 동안 자유롭게 Jab/Cross
rye run python ml/scripts/record_with_webcam.py \
    --output data/test_videos \
    --action jab_cross_test \
    --interval 15 \
    --reps 1 \
    --countdown 3

video=$(ls -t data/test_videos/jab_cross_test*.mp4 | head -1)

echo ""
echo "📹 녹화 완료: $video"
echo ""
echo "🎯 테스트 중..."
echo ""

# 테스트
rye run python ml/scripts/test_live.py \
    --video "$video" \
    --checkpoint checkpoints/boxing_6class_v1/checkpoint_best.pth \
    --confidence 0.5

echo ""
echo "✅ 완료!"
echo ""
echo "이제 결과를 보고 Jab/Cross가 제대로 인식되는지 확인하세요!"
