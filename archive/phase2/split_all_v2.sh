#!/bin/bash
# v2 데이터 일괄 분리

echo "📹 v2 영상 자르기 시작"
echo "=============================="
echo ""

# 입력 디렉토리 (업로드한 영상들이 있는 곳)
INPUT_DIR="data/raw/v2_iphone_angles/batches"

# 출력 디렉토리
OUTPUT_DIR="data/processed/splits/v2"

# 각 동작별로 분리
ACTIONS=("jab" "cross" "lead_hook" "rear_hook" "lead_uppercut" "rear_uppercut")

for action in "${ACTIONS[@]}"; do
    echo ""
    echo "처리 중: $action"
    echo "─────────────────────────────"

    # 해당 동작의 모든 영상 찾기
    videos=$(find $INPUT_DIR -name "${action}_*.mp4" -o -name "${action}_*.MOV" 2>/dev/null)

    if [ -z "$videos" ]; then
        echo "⚠️  ${action} 영상을 찾을 수 없습니다"
        continue
    fi

    # 각 영상 처리
    for video in $videos; do
        basename=$(basename "$video" | sed 's/\.[^.]*$//')
        
        rye run python ml/scripts/split_video.py \
            --input "$video" \
            --duration 1.5 \
            --output_dir "$OUTPUT_DIR/$action" \
            --prefix "${basename}"
    done
done

echo ""
echo "=============================="
echo "✅ 완료!"
echo "=============================="
echo ""
echo "결과 확인:"
echo "  ls -lh $OUTPUT_DIR/*/"
echo ""
echo "다음 단계: 키포인트 추출"
echo "  rye run python ml/scripts/extract_keypoints.py \\"
echo "      --input $OUTPUT_DIR \\"
echo "      --output data/processed/keypoints/v2"
