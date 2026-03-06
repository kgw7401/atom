#!/bin/bash
# 각 동작별 테스트 예시

echo "🥊 CTR-GCN 모델 테스트 예시"
echo "======================================"
echo ""
echo "사용법: 아래 명령어 중 하나를 복사해서 실행하세요"
echo ""

# Find one example of each action
for action in jab cross lead_hook rear_hook lead_uppercut rear_uppercut; do
    video=$(find data/batches -name "${action}_*.mp4" -type f | head -1)
    if [ -n "$video" ]; then
        echo "# $action 테스트:"
        echo "rye run python ml/scripts/test_live.py --video \"$video\" --checkpoint checkpoints/boxing_6class_v1/checkpoint_best.pth --confidence 0.6"
        echo ""
    fi
done

echo "# 시각화 포함 (영상 재생):"
video=$(find data/batches -name "*.mp4" -type f | head -1)
echo "rye run python ml/scripts/test_live.py --video \"$video\" --checkpoint checkpoints/boxing_6class_v1/checkpoint_best.pth --visualize"
echo ""
echo "======================================"
