"""1단계 최종 산출물을 만든다 — 프레임마다 두 사람의 바닥 위치와 거리.

입력: 추적 결과 CSV + 바닥 보정값 JSON
출력: 프레임당 한 줄짜리 CSV (docs/scope.md 에서 정한 형태)

    frame        프레임 번호
    time_sec     영상 시작부터 몇 초
    me_x, me_z   내 바닥 위치 (m). x=좌우, z=카메라에서 멀어지는 방향
    op_x, op_z   상대 바닥 위치 (m)
    distance_m   둘 사이 거리 (m)
    confidence   이 줄을 얼마나 믿을 수 있나 (0~1)

confidence 는 아래를 곱해서 만든다. 값을 지어내지 않고, 못 믿는 줄은 못 믿는다고 적는다.
    - 두 사람이 다 잡혔나
    - 정체를 색으로 판단했나, 직전 위치로 때웠나
    - 발이 화면 밖으로 잘렸나
    - 두 사람 박스가 얼마나 겹쳤나

사용법:
  python scripts/make_positions.py --tracks out/IMG_0711.tracks.csv \\
      --ground configs/IMG_0711.ground.json --out out/IMG_0711.positions.csv
"""

import argparse
import csv
import json
import math
from collections import defaultdict

import numpy as np

from ranges import MIN_PLAUSIBLE_M, RANGE_BANDS

# 신뢰도 감점표
W_METHOD = {"color": 1.0, "color_abs": 0.85, "prev": 0.4}
W_CLIPPED = 0.5
W_FOOT_SRC = {"ankle2": 1.0, "ankle1": 0.85, "bbox": 0.6}


def ground_xz(u, v, horizon_y, cam_h, focal, cx):
    """화면 좌표 (u,v) 를 바닥 좌표 (x,z) 미터로 바꾼다."""
    dv = v - horizon_y
    if dv <= 1:
        return None
    z = focal * cam_h / dv
    x = (u - cx) * cam_h / dv
    return x, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--ground", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth", type=int, default=5,
                    help="위치를 몇 프레임으로 평활화할지 (홀수, 1이면 안 함)")
    args = ap.parse_args()

    g = json.load(open(args.ground))
    horizon = g["fit"]["horizon_y"]
    cam_h = g["camera_height_m"]
    focal = g["focal_px"]
    cx = g["principal_x"]
    print(f"보정값: 지평선 y={horizon:.1f}px  카메라높이={cam_h:.2f}m  f={focal:.0f}px")

    by_frame = defaultdict(dict)
    fps = None
    for r in csv.DictReader(open(args.tracks)):
        role = r.get("role") or r["fighter"]
        by_frame[int(r["frame"])][role] = r
        if fps is None and float(r["time_sec"]) > 0:
            fps = int(r["frame"]) / float(r["time_sec"])

    frames = sorted(by_frame)
    n_total = frames[-1] + 1 if frames else 0

    rows = []
    n_impossible = 0
    for fi in range(n_total):
        rec = by_frame.get(fi, {})
        out = {"frame": fi, "time_sec": round(fi / fps, 3) if fps else ""}
        conf = 1.0
        pos = {}
        for role in ("me", "opponent"):
            r = rec.get(role)
            if r is None:
                conf = 0.0
                continue
            p = ground_xz(float(r["foot_x"]), float(r["foot_y"]), horizon, cam_h, focal, cx)
            if p is None:
                conf = 0.0
                continue
            pos[role] = p
            conf *= W_METHOD.get(r["id_method"], 0.4)
            conf *= W_FOOT_SRC.get(r["foot_src"], 0.6)
            if r["foot_clipped"] == "1":
                conf *= W_CLIPPED

        if len(pos) == 2:
            iou = float(next(iter(rec.values()))["box_iou"])
            # 많이 겹칠수록 검출 자체를 못 믿는다. 한 사람에 박스가 두 개 생기거나
            # 두 사람이 박스 하나로 합쳐지는데, 둘 다 거리를 줄이는 쪽으로 틀린다.
            if iou >= 0.35:
                conf *= 0.25
            elif iou >= 0.15:
                conf *= 0.7
            d = math.dist(pos["me"], pos["opponent"])
        else:
            d = None

        out.update({
            "me_x": round(pos["me"][0], 3) if "me" in pos else "",
            "me_z": round(pos["me"][1], 3) if "me" in pos else "",
            "op_x": round(pos["opponent"][0], 3) if "opponent" in pos else "",
            "op_z": round(pos["opponent"][1], 3) if "opponent" in pos else "",
            "distance_m": round(d, 3) if d is not None else "",
            "confidence": round(conf, 3),
        })
        rows.append(out)

    # 평활화: 60fps 라 프레임 단위 흔들림이 크다. 이동 중앙값으로 튀는 값을 눌러준다.
    if args.smooth > 1:
        k = args.smooth | 1
        half = k // 2
        for col in ("me_x", "me_z", "op_x", "op_z"):
            vals = [r[col] if r[col] != "" else np.nan for r in rows]
            arr = np.array(vals, float)
            sm = arr.copy()
            for i in range(len(arr)):
                w = arr[max(0, i - half): i + half + 1]
                w = w[~np.isnan(w)]
                if w.size:
                    sm[i] = float(np.median(w))
            for i, r in enumerate(rows):
                if r[col] != "":
                    r[col] = round(float(sm[i]), 3)
        # 평활화한 좌표로 거리를 다시 계산한다
        for r in rows:
            if r["me_x"] != "" and r["op_x"] != "":
                r["distance_m"] = round(
                    math.dist((r["me_x"], r["me_z"]), (r["op_x"], r["op_z"])), 3)

    # 물리적으로 불가능한 값 버리기. 반드시 평활화 뒤에 해야 한다.
    # 앞에서 버리면 위 재계산이 좌표로부터 값을 되살려낸다.
    for r in rows:
        if r["distance_m"] != "" and r["distance_m"] < MIN_PLAUSIBLE_M:
            n_impossible += 1
            r["distance_m"] = ""
            r["confidence"] = 0.0

    fields = ["frame", "time_sec", "me_x", "me_z", "op_x", "op_z", "distance_m", "confidence"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["distance_m"] != ""]
    good = [r for r in ok if r["confidence"] >= 0.6]
    ds = np.array([r["distance_m"] for r in good], float)
    print(f"\n저장: {args.out}  ({len(rows)} 프레임)")
    print(f"  거리가 나온 프레임      : {len(ok)} ({100*len(ok)/max(len(rows),1):.1f}%)")
    print(f"  그중 믿을 만한 것(>=0.6): {len(good)} ({100*len(good)/max(len(rows),1):.1f}%)")
    print(f"  물리적으로 불가능해 버림 : {n_impossible} (<{MIN_PLAUSIBLE_M}m, 검출이 겹쳐 무너진 경우)")
    if ds.size:
        print(f"\n  두 사람 거리 (믿을 만한 프레임만)")
        print(f"    중앙 {np.median(ds):.2f}m   25% {np.percentile(ds,25):.2f}m   "
              f"75% {np.percentile(ds,75):.2f}m")
        print(f"    최소 {ds.min():.2f}m   최대 {ds.max():.2f}m")
        for lo, hi, name, desc, _c in RANGE_BANDS:
            n = int(((ds >= lo) & (ds < hi)).sum())
            label = f"{name} ({desc})"
            print(f"    {label:<24} {n:>5} 프레임 ({100*n/ds.size:4.1f}%)  {n/fps:5.1f}초")


if __name__ == "__main__":
    main()
