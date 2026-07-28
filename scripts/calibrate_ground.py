"""화면 좌표를 실제 바닥 좌표로 바꾸기 위한 보정값을 구한다.

원리:
  카메라가 고정돼 있고 사람이 바닥 위를 걸어다닌다면, 화면에서 사람의 키(픽셀)는
  발이 화면에서 얼마나 아래에 있는지에 정비례한다.

      키(px) = (실제키 / 카메라높이) x (발의 화면 y - 지평선 y)

  그래서 추적 데이터 수천 개로 이 직선을 맞추면 지평선 위치와 카메라 높이가 나온다.
  링 모서리를 찍을 필요가 없다.

  직선이 잘 맞는지(R^2)가 곧 "이 영상에서 바닥 평면 가정이 성립하는가"의 답이다.

구한 값으로 바닥 좌표를 만든다:
      깊이 Z = f x 카메라높이 / (발y - 지평선y)
      좌우 X = (발x - 화면중심x) x 카메라높이 / (발y - 지평선y)

  X 는 초점거리 f 가 없어도 되지만 Z 는 필요하다. f 는 두 가지로 얻는다.
    1) 카메라 기종의 화각으로 계산 (기본값: iPhone 14 Pro 1배율)
    2) --ring-side 로 링 실제 한 변 길이를 주면 그 값에 맞게 역산

사용법:
  python scripts/calibrate_ground.py --csv out/IMG_0711.tracks.csv
  python scripts/calibrate_ground.py --csv out/IMG_0711.tracks.csv \\
      --person-height 1.75 --out configs/IMG_0711.ground.json
"""

import argparse
import csv
import json
import math

import numpy as np

# iPhone 14 Pro 메인 카메라는 35mm 환산 24mm.
# 가로화각 = 2*atan(36/(2*24)) = 73.7도  ->  f(px) = (가로/2) / tan(화각/2)
IPHONE14PRO_EQ_MM = 24.0


def focal_px_from_equiv(width_px, equiv_mm=IPHONE14PRO_EQ_MM):
    hfov = 2 * math.atan(36.0 / (2 * equiv_mm))
    return (width_px / 2.0) / math.tan(hfov / 2.0)


def load_samples(path):
    """직선 맞추기에 쓸 만한 행만 고른다."""
    rows = list(csv.DictReader(open(path)))
    good = []
    for r in rows:
        if r["foot_clipped"] == "1":
            continue
        if not r["foot_src"].startswith("ankle"):
            continue
        if r["id_method"] == "prev":          # 정체가 덜 확실한 프레임은 제외
            continue
        if float(r["box_iou"]) >= 0.15:       # 겹친 순간은 박스 크기가 부정확
            continue
        fy = float(r["foot_y"])
        y1 = float(r["y1"])
        h = fy - y1                            # 머리끝 ~ 발목 픽셀 높이
        if h <= 50:
            continue
        good.append((r["fighter"], float(r["foot_x"]), fy, h))
    return rows, good


def fit_line(ys, hs):
    """h = a*y + b 최소자승. (a, b, R^2) 반환."""
    a, b = np.polyfit(ys, hs, 1)
    pred = a * ys + b
    ss_res = float(np.sum((hs - pred) ** 2))
    ss_tot = float(np.sum((hs - np.mean(hs)) ** 2))
    return float(a), float(b), (1 - ss_res / ss_tot if ss_tot > 0 else 0.0)


def fit_upper_envelope(ys, hs, n_bins=24, q=90, min_per_bin=25):
    """자세를 낮춘 프레임을 빼고 '똑바로 섰을 때'의 직선을 맞춘다.

    복서는 계속 무릎을 굽히고 상체를 숙이기 때문에 화면상 키가 크게 흔들린다.
    그냥 최소자승으로 맞추면 이 흔들림이 전부 오차로 들어간다.
    발 위치를 구간으로 나눠 각 구간의 상위 q% 키만 쓰면 서 있는 순간만 남는다.

    반환: (a, b, R^2, 쓴 점 개수)
    """
    edges = np.linspace(ys.min(), ys.max(), n_bins + 1)
    px, py = [], []
    for i in range(n_bins):
        m = (ys >= edges[i]) & (ys < edges[i + 1])
        if m.sum() < min_per_bin:
            continue
        px.append(float(np.mean(ys[m])))
        py.append(float(np.percentile(hs[m], q)))
    if len(px) < 4:
        return None
    px, py = np.array(px), np.array(py)
    a, b, r2 = fit_line(px, py)
    return a, b, r2, len(px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--person-height", type=float, default=None,
                    help="선수 실제 키(m). 주면 카메라 높이와 미터 좌표가 나온다")
    ap.add_argument("--ring-side", type=float, default=None,
                    help="링 한 변 실제 길이(m). 주면 초점거리를 이 값에 맞게 역산")
    ap.add_argument("--ring-far-left", type=float, nargs=2, default=[620, 690],
                    help="링 뒤쪽 왼쪽 모서리 화면 좌표 (눈대중 추정값)")
    ap.add_argument("--ring-far-right", type=float, nargs=2, default=[1750, 700])
    ap.add_argument("--quantile", type=float, default=90,
                    help="각 구간에서 상위 몇 %% 키를 '서 있는 자세'로 볼지")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, good = load_samples(args.csv)
    print(f"전체 {len(rows)}행 중 보정에 쓸 만한 행: {len(good)}")
    if len(good) < 200:
        print("표본이 너무 적다. 보정을 신뢰할 수 없다.")
        return

    ys = np.array([g[2] for g in good])
    hs = np.array([g[3] for g in good])

    a_ls, b_ls, r2_ls = fit_line(ys, hs)
    env = fit_upper_envelope(ys, hs, q=args.quantile)
    if env is None:
        print("구간별 표본이 부족해 상위 envelope 맞추기를 못 했다. 최소자승 결과를 쓴다.")
        a, b, r2 = a_ls, b_ls, r2_ls
        fit_kind = "least_squares"
    else:
        a, b, r2, n_bins_used = env
        fit_kind = f"upper_envelope_q{args.quantile}"

    v0 = -b / a

    print("\n=== 바닥 평면 직선 맞추기 ===")
    print(f"  전체 최소자승     : 기울기={a_ls:.4f}  R^2={r2_ls:.4f} (원본 {len(ys)}점)  "
          f"지평선={-b_ls/a_ls:.0f}px")
    if env is not None:
        print(f"  상위{args.quantile:.0f}% (선 자세): 기울기={a:.4f}  R^2={r2:.4f} (구간평균 {n_bins_used}점)  "
              f"지평선={v0:.0f}px")
        print("  주의: 위 두 R^2 는 서로 다른 데이터에 대한 값이라 직접 비교하면 안 된다.")
        print("        구간평균은 점 개수가 적고 이미 평균이 돼 있어서 R^2 가 원래 높게 나온다.")
    print(f"  발y 범위          : {ys.min():.0f} ~ {ys.max():.0f}  "
          f"(지평선은 이 범위 밖으로 한참 외삽해야 나오는 값이라 불안정하다)")
    print(f"  채택한 방식       : {fit_kind}")

    # 두 선수를 따로 맞춰본다. 서로 독립인데 지평선이 비슷하게 나오면 모델이 맞는 것이다.
    per = {}
    for name in ("light", "dark"):
        sel = [g for g in good if g[0] == name]
        if len(sel) < 300:
            continue
        ay = np.array([s[2] for s in sel])
        ah = np.array([s[3] for s in sel])
        pe = fit_upper_envelope(ay, ah, q=args.quantile)
        if pe is None:
            continue
        pa, pb, pr2, _ = pe
        per[name] = {"slope": pa, "intercept": pb, "r2": pr2, "n": len(sel), "horizon": -pb / pa}
        print(f"  [{name:>5}] n={len(sel):>5}  기울기={pa:.4f}  R^2={pr2:.4f}  지평선={-pb/pa:.0f}px")
    if len(per) == 2:
        ratio = per["light"]["slope"] / per["dark"]["slope"]
        gap = abs(per["light"]["horizon"] - per["dark"]["horizon"])
        print(f"  두 선수 키 비 (light/dark) : {ratio:.3f}")
        print(f"  두 선수가 각자 계산한 지평선 차이: {gap:.0f}px  "
              f"(작을수록 좋다. 서로 독립적인 계산이 일치한다는 뜻)")

    out = {
        "source_csv": args.csv,
        "frame_size": [args.width, args.height],
        "fit": {"kind": fit_kind, "slope": a, "intercept": b, "r2": r2,
                "horizon_y": v0, "n": len(good),
                "least_squares": {"slope": a_ls, "intercept": b_ls, "r2": r2_ls}},
        "per_fighter": per,
    }

    # --- 미터 단위로 바꾸기 ---
    if args.person_height:
        cam_h = args.person_height / a
        out["person_height_m"] = args.person_height
        out["camera_height_m"] = cam_h
        print("\n=== 미터 변환 ===")
        print(f"  기준 선수 키   : {args.person_height:.2f} m")
        print(f"  카메라 높이    : {cam_h:.2f} m  (바닥 기준)")

        cx = args.width / 2.0
        f_nominal = focal_px_from_equiv(args.width)
        f_used, f_src = f_nominal, "iPhone 14 Pro 24mm 환산값"

        if args.ring_side:
            # 링 뒤쪽 두 모서리 사이 실제 거리가 ring_side 가 되도록 f 를 역산
            (u1, v1), (u2, v2) = args.ring_far_left, args.ring_far_right
            d1, d2 = v1 - v0, v2 - v0
            if d1 > 0 and d2 > 0:
                X1, X2 = (u1 - cx) * cam_h / d1, (u2 - cx) * cam_h / d2
                k1, k2 = cam_h / d1, cam_h / d2          # Z = f * k
                dX, dk = X1 - X2, k1 - k2
                # (dX)^2 + (f*dk)^2 = ring_side^2
                rhs = args.ring_side ** 2 - dX ** 2
                if rhs > 0 and abs(dk) > 1e-9:
                    f_used = math.sqrt(rhs) / abs(dk)
                    f_src = f"링 한 변 {args.ring_side}m 기준 역산"
                else:
                    print("  ! 링 크기로 초점거리를 못 구했다. 모서리 좌표를 다시 확인해야 한다.")

        out["focal_px"] = f_used
        out["focal_source"] = f_src
        out["principal_x"] = cx
        print(f"  초점거리 f     : {f_used:.0f} px  ({f_src})")
        if not args.ring_side:
            print(f"  (참고) 링 실측값을 주면 f 를 검증할 수 있다: --ring-side 5.0")

        # 실제 데이터로 감을 잡을 수 있게, 관측된 거리 범위를 미터로 보여준다
        by_frame = {}
        for r in rows:
            if r["id_method"] == "prev" or r["foot_clipped"] == "1":
                continue
            by_frame.setdefault(int(r["frame"]), {})[r["fighter"]] = r
        ds = []
        for fr, d in by_frame.items():
            if len(d) != 2:
                continue
            pts = []
            for name in ("light", "dark"):
                u, v = float(d[name]["foot_x"]), float(d[name]["foot_y"])
                dv = v - v0
                if dv <= 1:
                    break
                pts.append(((u - cx) * cam_h / dv, f_used * cam_h / dv))
            if len(pts) == 2:
                ds.append(math.dist(pts[0], pts[1]))
        if ds:
            ds = np.array(ds)
            print(f"\n  두 선수 사이 거리 ({len(ds)} 프레임)")
            print(f"    최소 {ds.min():.2f}m / 25% {np.percentile(ds,25):.2f}m / "
                  f"중앙 {np.median(ds):.2f}m / 75% {np.percentile(ds,75):.2f}m / 최대 {ds.max():.2f}m")
            out["distance_m_summary"] = {
                "n": int(len(ds)), "min": float(ds.min()), "p25": float(np.percentile(ds, 25)),
                "median": float(np.median(ds)), "p75": float(np.percentile(ds, 75)),
                "max": float(ds.max()),
            }
            print("    * 복싱 스파링이면 중앙값이 대략 1~2m 나와야 말이 된다.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
