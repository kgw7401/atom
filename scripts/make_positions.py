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

KP_MIN = 0.5          # 어깨·손목 키포인트를 믿을 최소 점수
REACH_PCTL = 98       # 사거리 반경으로 쓸 분위수 (최대값은 튀는 값에 약하다)


def ground_xz(u, v, horizon_y, cam_h, focal, cx):
    """화면 좌표 (u,v) 를 바닥 좌표 (x,z) 미터로 바꾼다."""
    dv = v - horizon_y
    if dv <= 1:
        return None
    z = focal * cam_h / dv
    x = (u - cx) * cam_h / dv
    return x, z


def px_to_m(px, foot_y, horizon_y, cam_h):
    """그 사람이 있는 깊이에서 화면 픽셀 길이를 미터로 바꾼다.

    깊이 Z = f·cam_h/(foot_y-v0) 이고 그 깊이의 배율은 Z/f 이므로,
    두 식에서 초점거리가 약분된다. f 를 몰라도 되는 값이다.
    """
    dv = foot_y - horizon_y
    return None if dv <= 1 else px * cam_h / dv


def reach_px(r):
    """발 중심에서 손목까지의 좌우 거리 중 긴 쪽 (픽셀).

    어깨~손목을 재면 팔 길이만 나오는데, 우리가 원하는 것은 "내 위치에서
    손이 얼마나 멀리 나가나" 다. 스탠스로 몸이 앞으로 나가는 것까지 포함해야
    하므로 발 중심을 기준으로 잰다.

    좌우 성분만 쓰는 이유: 손은 가슴 높이에 있어서 화면상 세로 거리에는
    높이가 섞인다. 좌우 성분은 그 깊이에서의 배율만 곱하면 바로 미터가 된다.
    카메라 쪽으로 뻗은 펀치는 짧게 나오지만, 영상 전체의 분위수를 쓰므로
    좌우로 뻗은 순간이 값을 결정한다.
    """
    try:
        if float(r["wr_conf"]) < KP_MIN:
            return None
    except (KeyError, ValueError):
        return None
    fx = float(r["foot_x"])
    best = max(abs(float(r["lwr_x"]) - fx), abs(float(r["rwr_x"]) - fx))
    return best or None


def facing_deg(r, horizon_y, focal, cx):
    """바닥 평면에서 본 정면 방향 (도).

    어깨선은 3차원에서 수평이다. 수평선은 연장하면 지평선 위의 한 점에서
    만나고, 그 점이 곧 그 방향의 소실점이다. 소실점이 화면 x=vx 에 있으면
    바닥 좌표에서의 방향은 (vx-cx, f) 다. 키를 가정할 필요가 없다.

    정면은 어깨 방향의 수직이다. 앞뒤 두 갈래 중 어느 쪽인지는 자세 모델이
    좌우 어깨를 구분해 준다는 점으로 정한다 — 카메라를 향해 서면 본인의
    왼쪽 어깨가 화면 오른쪽에 온다.
    """
    try:
        if float(r["sh_conf"]) < KP_MIN:
            return None
    except (KeyError, ValueError):
        return None
    lx, ly = float(r["lsh_x"]), float(r["lsh_y"])
    rx, ry = float(r["rsh_x"]), float(r["rsh_y"])
    dx, dy = lx - rx, ly - ry
    if math.hypot(dx, dy) < 8:          # 어깨가 겹쳐 보이면 방향을 못 정한다
        return None

    if abs(dy) < 1e-6:
        # 어깨선이 화면에서 완전히 수평 = 카메라 축과 나란한 방향
        sx, sz = 1.0, 0.0
    else:
        t = (horizon_y - ly) / dy       # 어깨선을 지평선까지 연장
        vx = lx + t * dx
        # 소실점 방향이 곧 3차원 방향 (vx-cx, f) 다.
        # t>0 이면 소실점이 왼쪽 어깨 너머에 있으므로 그 방향이 '오른어깨->왼어깨' 다.
        sx, sz = vx - cx, focal
        if t < 0:
            sx, sz = -sx, -sz
    n = math.hypot(sx, sz)
    sx, sz = sx / n, sz / n
    # 앞뒤 구분은 위에서 이미 끝났다. 자세 모델이 좌우 어깨를 구분해 주므로
    # '오른어깨->왼어깨' 벡터 자체가 몸이 어느 쪽을 향하는지를 담고 있다.
    # 그 벡터를 시계방향 90도 돌리면 정면이다.
    #   카메라를 향해 섬: 왼손이 +x 쪽 -> s=(1,0) -> 정면 (0,-1) = 카메라 쪽
    #   등을 돌림      : 왼손이 -x 쪽 -> s=(-1,0) -> 정면 (0,+1) = 카메라 반대
    return math.degrees(math.atan2(-sx, sz))


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

    # 사거리 반경은 선수마다 하나의 상수다. 영상 전체에서 팔을 가장 길게 뻗은
    # 값을 쓴다. 매 프레임 재면 팔을 접을 때마다 값이 흔들린다.
    reach_m = {}
    for role in ("me", "opponent"):
        vals = []
        for fi in frames:
            r = by_frame[fi].get(role)
            if r is None or r["foot_clipped"] == "1":
                continue
            px = reach_px(r)
            if px is None:
                continue
            m = px_to_m(px, float(r["foot_y"]), horizon, cam_h)
            if m:
                vals.append(m)
        if len(vals) >= 100:
            reach_m[role] = float(np.percentile(vals, REACH_PCTL))
            print(f"  {role:>8} 사거리 반경 {reach_m[role]:.2f} m  (표본 {len(vals)})")
        else:
            print(f"  {role:>8} 사거리 표본 부족 ({len(vals)}) — 비워 둔다")

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
        for role, tag in (("me", "me"), ("opponent", "op")):
            r = rec.get(role)
            fa = facing_deg(r, horizon, focal, cx) if r is not None else None
            out[f"{tag}_facing"] = round(fa, 1) if fa is not None else ""
            out[f"{tag}_reach"] = round(reach_m[role], 3) if role in reach_m else ""
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

    fields = ["frame", "time_sec", "me_x", "me_z", "op_x", "op_z",
              "me_facing", "op_facing", "me_reach", "op_reach",
              "distance_m", "confidence"]
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
