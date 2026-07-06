#!/bin/bash
# Quick test script for trained CTR-GCN model

echo "🥊 CTR-GCN Model Test"
echo "===================="
echo ""
echo "Options:"
echo "1. Record new test video"
echo "2. Test on existing video"
echo ""
read -p "Select (1 or 2): " choice

if [ "$choice" == "1" ]; then
    echo ""
    echo "📹 Recording test video..."
    echo "Perform 3-5 punches in front of camera"
    echo ""
    
    # Record 10 second test clip
    rye run python ml/scripts/record_with_webcam.py \
        --output data/test_videos \
        --action test_combo \
        --interval 10 \
        --reps 1 \
        --countdown 3
    
    # Find the recorded video
    video=$(ls -t data/test_videos/*.mp4 | head -1)
    
elif [ "$choice" == "2" ]; then
    read -p "Enter video path: " video
else
    echo "Invalid choice"
    exit 1
fi

echo ""
echo "🎯 Running inference on: $video"
echo ""

# Run inference
rye run python ml/scripts/test_live.py \
    --video "$video" \
    --checkpoint checkpoints/boxing_6class_v1/checkpoint_best.pth \
    --confidence 0.6 \
    --visualize

echo ""
echo "✅ Test complete!"
