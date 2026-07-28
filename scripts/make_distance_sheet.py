"""거리별 실제 장면을 늘어놓은 기준표를 만든다.

거리대 경계(`ranges.py` 의 RANGE_BANDS)를 사람이 직접 보고 정하기 위한 도구다.
"0.75m 가 클린치인가 근접 타격인가" 는 계산으로 답이 안 나온다. 체격과 리치에
따라 다르므로 본인이 장면을 보고 정해야 한다.

신뢰도가 높은 프레임만 골라서, 목표 거리에 가장 가까운 장면을 뽑아 격자로 붙인다.
같은 순간이 여러 번 나오지 않도록 서로 떨어진 프레임을 고른다.

사용법:
  python scripts/make_distance_sheet.py --positions out/IMG_0711.positions.csv \\
      --config configs/IMG_0711.json --out out/distance_scale.jpg

  # 특정 구간을 촘촘히 보고 싶을 때
  python scripts/make_distance_sheet.py ... --targets 0.4 0.5 0.6 0.7 0.8 0.9
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np

DEFAULT_TARGETS = [0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.40, 1.60, 1.85, 2.15]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS,
                    help="보고 싶은 거리들 (m)")
    ap.add_argument("--min-conf", type=float, default=0.8,
                    help="이 신뢰도 이상인 프레임만 후보로 삼는다")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--tile-width", type=int, default=640)
    ap.add_argument("--apart", type=int, default=90,
                    help="고른 장면끼리 최소 몇 프레임 떨어뜨릴지")
    args = ap.parse_args()

    with open(args.config) as f:
        video = os.path.expanduser(json.load(f)["video"])

    rows = [r for r in csv.DictReader(open(args.positions))
            if r["distance_m"] and float(r["confidence"]) >= args.min_conf]
    if not rows:
        print(f"신뢰도 {args.min_conf} 이상인 프레임이 없다. --min-conf 를 낮춰라.")
        return
    print(f"후보 프레임: {len(rows)} (신뢰도 >= {args.min_conf})")

    cap = cv2.VideoCapture(video)
    tw = args.tile_width
    th = int(tw * 9 / 16)

    tiles, used = [], set()
    for t in args.targets:
        pool = [r for r in rows if int(r["frame"]) not in used]
        if not pool:
            break
        best = min(pool, key=lambda r: abs(float(r["distance_m"]) - t))
        got = float(best["distance_m"])
        if abs(got - t) > 0.15:
            print(f"  {t:.2f}m: 가까운 장면이 없다 (가장 가까운 값 {got:.2f}m) — 건너뜀")
            continue
        fr_i = int(best["frame"])
        used.update(range(fr_i - args.apart, fr_i + args.apart))

        cap.set(cv2.CAP_PROP_POS_FRAMES, fr_i)
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(fr, (tw, th))
        cv2.rectangle(fr, (0, 0), (tw, 42), (0, 0, 0), -1)
        cv2.putText(fr, f"{got:.2f}m   t={float(best['time_sec']):.1f}s",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(fr, (0, 0), (tw - 1, th - 1), (90, 90, 90), 2)
        tiles.append(fr)
        print(f"  {t:.2f}m 요청 -> {got:.2f}m  (t={float(best['time_sec']):.1f}s)")

    cap.release()
    if not tiles:
        print("뽑은 장면이 없다.")
        return

    # 마지막 줄이 모자라면 검은 칸으로 채운다
    while len(tiles) % args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + args.cols])
                      for i in range(0, len(tiles), args.cols)])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\n저장: {args.out}  ({grid.shape[1]}x{grid.shape[0]})")
    print("이 표를 보고 `scripts/ranges.py` 의 RANGE_BANDS 경계를 정하면 된다.")


if __name__ == "__main__":
    main()
