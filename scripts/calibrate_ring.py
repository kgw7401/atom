"""링 자체를 기준으로 카메라를 보정한다.

`calibrate_ground.py` 는 사람 키만으로 바닥 평면을 추정했다. 그 방식은 초점거리를
카메라 기종의 명목값으로 가정해야 하는데, 실제 영상은 줌으로 찍혀 있어서 그 가정이
틀렸다. 결과적으로 깊이가 절반 이하로 눌렸다 (깊이/폭 비율 0.43, 정사각형 링이면
1이어야 함).

이 스크립트는 링의 로프를 쓴다. 로프는 실제로 평행하므로 화면에서 만나는 점이
소실점이고, 링은 정사각형이므로 두 방향이 직각이다. 그 두 사실만으로 초점거리와
지평선이 나온다. 가정할 게 없다.

    좌우 방향 로프들  -> 소실점 V2
    깊이 방향 로프들  -> 소실점 V1
    두 방향이 직각    -> f^2 = -(V1-c)·(V2-c)
    지평선            = V1 과 V2 를 잇는 선

카메라 높이는 사람 키로 정한다 (그 부분은 기존과 같다).

사용법:
  python scripts/calibrate_ring.py --config configs/IMG_0711.json \\
      --tracks out/IMG_0711.tracks.csv --person-height 1.73 \\
      --out configs/IMG_0711.ground.json
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np


# 캔버스로 인정할 색 차이 한계 (CIELAB 거리). 발 위치에서 뽑은 기준색과 비교한다.
CANVAS_LAB_TOL = 26.0

# 캔버스를 찾을 때 선수 이동 범위에서 이만큼까지만 본다 (m).
# 링 밖 체육관 바닥이 캔버스와 색이 비슷하면 마스크가 새어나가는데,
# 선수는 링 안에만 있으므로 이 창 밖은 링일 수 없다.
CANVAS_SEARCH_MARGIN = 1.6


def background_frame(video, n_samples=60):
    """움직이는 사람을 지운 배경. 고르게 뽑은 프레임들의 중앙값."""
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    buf = []
    for i in np.linspace(0, total - 1, n_samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            buf.append(fr)
    cap.release()
    if not buf:
        raise RuntimeError("프레임을 읽지 못했다")
    return np.median(np.stack(buf), axis=0).astype(np.uint8), len(buf)


def detect_segments(bg, min_len=200):
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 40, 120)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=100,
                           minLineLength=min_len, maxLineGap=20)
    return np.empty((0, 4)) if segs is None else segs.reshape(-1, 4)


def seg_angle(s):
    return np.degrees(np.arctan2(float(s[3] - s[1]), float(s[2] - s[0])))


def ransac_vanishing_point(group, iters=4000, tol_deg=3.0, seed=0):
    """선분들이 한 점에서 만나도록 하는 소실점. 지지 선분 수가 최대인 것을 고른다."""
    if len(group) < 3:
        return None, []

    def line_of(s):
        a = np.array([s[0], s[1], 1.0])
        b = np.array([s[2], s[3], 1.0])
        L = np.cross(a, b)
        return L / max(np.linalg.norm(L[:2]), 1e-12)

    lines = [line_of(s) for s in group]
    mids = [np.array([(s[0] + s[2]) / 2, (s[1] + s[3]) / 2]) for s in group]
    dirs = []
    for s in group:
        d = np.array([s[2] - s[0], s[3] - s[1]], float)
        dirs.append(d / max(np.linalg.norm(d), 1e-12))

    rng = np.random.default_rng(seed)
    best_V, best_inliers = None, []
    for _ in range(iters):
        i, j = rng.choice(len(group), 2, replace=False)
        V = np.cross(lines[i], lines[j])
        if abs(V[2]) < 1e-12:
            continue
        V = V / V[2]
        inliers = []
        for k in range(len(group)):
            d2 = V[:2] - mids[k]
            n = np.linalg.norm(d2)
            if n < 1e-6:
                continue
            cos = abs(dirs[k] @ (d2 / n))
            if np.degrees(np.arccos(np.clip(cos, 0, 1))) < tol_deg:
                inliers.append(k)
        if len(inliers) > len(best_inliers):
            best_V, best_inliers = V, inliers
    return best_V, best_inliers


def find_ring_square(bg, V1, V2, horizon_y, focal, cam_h, cx, foot_pts=None,
                     ppm=110, pad=2.0, debug_dir=None):
    """링 캔버스의 네 변을 찾는다.

    먼저 바닥 평면을 링 축에 맞춰 펼친다. 그러면 링 변이 그림의 가로세로와
    나란해져서 경계를 찾기가 훨씬 쉽다 (원본에서는 비스듬한 사다리꼴이다).

    화면 아래가 잘려 링 앞쪽 변은 보이지 않는다. 하지만 링은 정사각형이므로,
    좌우 두 변 사이 거리가 곧 한 변의 길이이고 앞쪽 변의 위치가 정해진다.
    모서리를 짚을 필요가 없다.

    반환: dict (링 좌표계 정의와 네 변의 위치) 또는 None
    """
    H, W = bg.shape[:2]
    e1 = np.array([V2[0] - cx, focal]); e1 /= np.linalg.norm(e1)
    e2 = np.array([V1[0] - cx, focal]); e2 /= np.linalg.norm(e2)
    R = np.stack([e1, e2])                       # 카메라 바닥좌표 -> 링 좌표

    # 렌더 범위는 화면 아래쪽 절반의 바닥만 기준으로 잡는다.
    # 지평선 근처는 몇 픽셀이 수십 미터에 해당해서, 거기까지 포함하면 범위가
    # 폭발하고 정작 링이 있는 구간이 그림 밖으로 밀려난다.
    v_near = horizon_y + 0.45 * (H - horizon_y)
    us, vs = np.meshgrid(np.linspace(0, W - 1, 60), np.linspace(v_near, H - 1, 60))
    dv = vs - horizon_y
    Xc, Zc = (us - cx) * cam_h / dv, focal * cam_h / dv
    ab = np.einsum("ij,jkl->ikl", R, np.stack([Xc, Zc]))
    a0, a1 = ab[0].min() - pad, ab[0].max() + pad
    b0, b1 = ab[1].min() - pad, ab[1].max() + pad
    Wo, Ho = int((a1 - a0) * ppm), int((b1 - b0) * ppm)
    if Wo < 50 or Ho < 50:
        return None

    aa, bb = np.meshgrid(np.linspace(a0, a1, Wo), np.linspace(b1, b0, Ho))
    XZ = np.einsum("ij,jkl->ikl", R.T, np.stack([aa, bb]))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cx + focal * XZ[0] / XZ[1]
        v = horizon_y + focal * cam_h / XZ[1]
    bad = (XZ[1] <= 0.2) | ~np.isfinite(u) | ~np.isfinite(v)
    u[bad], v[bad] = -1, -1
    top = cv2.remap(bg, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))

    # 캔버스 색은 영상마다 다르다 (조명, 노출, 화이트밸런스). 고정 임계값을 쓰면
    # 다른 영상에서 캔버스의 일부만 잡힌다 — 실제로 두 번째 영상에서 그렇게 깨졌다.
    # 확실한 단서가 있다: 선수는 반드시 캔버스 위에 서 있다. 발이 있던 자리의
    # 색을 기준으로 삼으면 영상마다 알아서 맞춰진다.
    lab = cv2.cvtColor(top, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = None
    if foot_pts is not None and len(foot_pts):
        samples = []
        for X, Z in foot_pts:
            a, b = R @ np.array([X, Z])
            px, py = int((a - a0) * ppm), int((b1 - b) * ppm)
            if 3 <= px < Wo - 3 and 3 <= py < Ho - 3:
                samples.append(lab[py - 3:py + 4, px - 3:px + 4].reshape(-1, 3))
        if len(samples) >= 20:
            ref = np.median(np.concatenate(samples), axis=0)

    if ref is not None:
        d = np.linalg.norm(lab - ref, axis=2)
        m = (d < CANVAS_LAB_TOL).astype(np.uint8)
    else:
        # 발 표본이 없으면 예전 방식(푸른 계열 고정 범위)으로 물러난다
        hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        h, s, val = (hsv[:, :, i].astype(int) for i in range(3))
        m = ((h > 100) & (h < 122) & (s > 35) & (val > 60)).astype(np.uint8)
    # 선수가 다닌 범위 밖은 링일 수 없다. 색만으로는 링 밖 바닥과 구분이 안 되는
    # 영상이 있어서 (두 번째 영상이 그랬다) 물리적 제약으로 잘라낸다.
    if foot_pts is not None and len(foot_pts) >= 20:
        ab_f = np.array([R @ np.array([X, Z]) for X, Z in foot_pts])
        fa0, fa1 = np.percentile(ab_f[:, 0], [1, 99])
        fb0, fb1 = np.percentile(ab_f[:, 1], [1, 99])
        gate = np.zeros_like(m)
        x_lo = max(0, int((fa0 - CANVAS_SEARCH_MARGIN - a0) * ppm))
        x_hi = min(Wo, int((fa1 + CANVAS_SEARCH_MARGIN - a0) * ppm))
        y_lo = max(0, int((b1 - (fb1 + CANVAS_SEARCH_MARGIN)) * ppm))
        y_hi = min(Ho, int((b1 - (fb0 - CANVAS_SEARCH_MARGIN)) * ppm))
        gate[y_lo:y_hi, x_lo:x_hi] = 1
        m = m * gate

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None

    # 로프가 캔버스를 가로로 갈라 덩어리가 여러 개가 된다. 세로로 이어붙이는
    # 연산으로 메우려 했더니 배경 쪽으로도 다리를 놔서 먼쪽 경계가 부풀었다.
    # 대신 발이 실제로 닿은 덩어리만 고른다 — 배경에는 발이 닿지 않는다.
    canvas = None
    if foot_pts is not None and len(foot_pts) >= 20:
        hits = {}
        for X, Z in foot_pts:
            a, b = R @ np.array([X, Z])
            px, py = int((a - a0) * ppm), int((b1 - b) * ppm)
            if 0 <= px < Wo and 0 <= py < Ho and cc[py, px] > 0:
                hits[cc[py, px]] = hits.get(cc[py, px], 0) + 1
        keep = [k for k, v in hits.items() if v >= max(5, 0.01 * len(foot_pts))]
        if keep:
            canvas = np.isin(cc, keep).astype(np.uint8)
    if canvas is None:
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        canvas = (cc == big).astype(np.uint8)

    cols, rows = canvas.sum(0), canvas.sum(1)
    xs = np.nonzero(cols > 0.4 * cols.max())[0]
    ys = np.nonzero(rows > 0.4 * rows.max())[0]
    if len(xs) < 10 or len(ys) < 10:
        return None
    left, right = a0 + xs.min() / ppm, a0 + xs.max() / ppm
    far, near_seen = b1 - ys.min() / ppm, b1 - ys.max() / ppm
    side = right - left
    near = far - side                     # 정사각형이므로 앞쪽 변이 정해진다

    if debug_dir:
        vis = top.copy()
        vis[canvas > 0] = (0.6 * vis[canvas > 0] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.rectangle(vis, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 0, 255), 2)
        y_near = int((b1 - near) * ppm)
        if 0 <= y_near < Ho:
            cv2.line(vis, (xs.min(), y_near), (xs.max(), y_near), (255, 0, 255), 2)
        cv2.imwrite(f"{debug_dir}/ring_rectified.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])

    return {
        "axes": R.tolist(),
        "left_a": float(left), "right_a": float(right),
        "far_b": float(far), "near_b": float(near),
        "near_seen_b": float(near_seen),
        "side_m": float(side),
        "near_edge_inferred": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--person-height", type=float, required=True,
                    help="기준 선수(나)의 실제 키 (m)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug-dir", default=None, help="중간 이미지 저장 경로")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    video = os.path.expanduser(cfg["video"])
    me_shirt = cfg.get("me_shirt", "dark")

    bg, n_used = background_frame(video)
    H, W = bg.shape[:2]
    print(f"배경 합성: {n_used} 프레임 -> {W}x{H}")

    segs = detect_segments(bg)
    horiz = [s for s in segs if abs(seg_angle(s)) < 12]
    steep = [s for s in segs if -45 < seg_angle(s) < -14]
    print(f"선분 {len(segs)}개  ->  좌우계열 {len(horiz)}, 깊이계열 {len(steep)}")

    V2, in2 = ransac_vanishing_point(horiz)
    V1, in1 = ransac_vanishing_point(steep)
    if V1 is None or V2 is None:
        print("소실점을 못 찾았다. 로프가 충분히 안 보이는 영상일 수 있다.")
        return
    print(f"  V2 (좌우) = ({V2[0]:.0f}, {V2[1]:.0f})   지지 {len(in2)}/{len(horiz)}")
    print(f"  V1 (깊이) = ({V1[0]:.0f}, {V1[1]:.0f})   지지 {len(in1)}/{len(steep)}")

    # 링은 정사각형이므로 두 방향은 직각이다. 여기서 초점거리가 나온다.
    c = np.array([W / 2, H / 2])
    f2 = -((V1[:2] - c) @ (V2[:2] - c))
    if f2 <= 0:
        print(f"f^2 이 음수({f2:.0f}). 소실점 추정이 틀렸을 가능성이 높다.")
        return
    focal = float(np.sqrt(f2))
    equiv = 36 / (2 * np.tan(np.arctan(W / 2 / focal)))
    print(f"  초점거리 f = {focal:.0f} px  (35mm 환산 {equiv:.0f}mm)")

    # 지평선 = 두 소실점을 잇는 선. 화면 중앙에서의 높이를 대표값으로 쓴다.
    hz = np.cross(np.append(V1[:2], 1.0), np.append(V2[:2], 1.0))
    horizon_y = float(-(hz[0] * (W / 2) + hz[2]) / hz[1])
    print(f"  지평선 y = {horizon_y:.0f} (화면 중앙 기준)")

    # 카메라 높이는 사람 키로 정한다.
    rows = [r for r in csv.DictReader(open(args.tracks))]
    good = [r for r in rows
            if r["foot_clipped"] == "0" and r["foot_src"].startswith("ankle")
            and r["id_method"] != "prev" and float(r["box_iou"]) < 0.15
            and (r.get("role") or r["fighter"]) in ("me", me_shirt)]
    if len(good) < 200:
        print(f"카메라 높이 추정에 쓸 표본이 적다 ({len(good)}). 결과를 믿기 어렵다.")
    fy = np.array([float(r["foot_y"]) for r in good])
    hp = np.array([float(r["foot_y"]) - float(r["y1"]) for r in good])
    slope = float(np.median(hp / (fy - horizon_y)))
    cam_h = args.person_height / slope
    print(f"  카메라 높이 = {cam_h:.2f} m  (표본 {len(good)})")

    # 검증: 링은 정사각형이므로 선수 이동 범위의 깊이/폭 비가 1에 가까워야 한다.
    allrows = [r for r in rows if r["foot_clipped"] == "0" and r["foot_src"].startswith("ankle")]
    u = np.array([float(r["foot_x"]) for r in allrows])
    v = np.array([float(r["foot_y"]) for r in allrows])
    dv = v - horizon_y
    X = (u - W / 2) * cam_h / dv
    Z = focal * cam_h / dv
    wx = np.percentile(X, 99) - np.percentile(X, 1)
    wz = np.percentile(Z, 99) - np.percentile(Z, 1)
    print(f"\n검증 — 선수 이동 범위 (1~99%)")
    print(f"  좌우 {wx:.2f} m,  깊이 {wz:.2f} m,  비율 {wz/wx:.2f}")
    print("  링은 정사각형이므로 1에 가까울수록 좋다.")
    print("  화면 아래가 잘려 링 앞부분이 안 보이므로 1보다 작게 나오는 것이 정상이다.")

    # 폴더를 먼저 만든다. cv2.imwrite 는 경로가 없어도 예외 없이 조용히 실패한다.
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
    # 캔버스 색 기준으로 쓸 발 위치 (바닥 좌표). 선수는 반드시 캔버스 위에 있다.
    foot_pts = []
    for r in allrows:
        d = float(r["foot_y"]) - horizon_y
        if d > 1:
            foot_pts.append(((float(r["foot_x"]) - W / 2.0) * cam_h / d,
                             focal * cam_h / d))
    ring = find_ring_square(bg, V1, V2, horizon_y, focal, cam_h, W / 2.0,
                            foot_pts=foot_pts, debug_dir=args.debug_dir)
    if ring:
        print(f"\n링 캔버스 (링 좌표계, m)")
        print(f"  좌 {ring['left_a']:.2f}  우 {ring['right_a']:.2f}  -> 한 변 {ring['side_m']:.2f} m")
        print(f"  먼쪽 {ring['far_b']:.2f}   앞쪽 {ring['near_b']:.2f} (정사각형 조건으로 추정)")
        print(f"  실제로 보이는 앞쪽 끝: {ring['near_seen_b']:.2f}"
              f"  -> {ring['near_seen_b']-ring['near_b']:.2f} m 는 화면 밖")
    else:
        print("\n링 캔버스를 못 찾았다.")

    out = {
        "source": "ring_vanishing_points",
        "ring": ring,
        # 설정 파일에 적힌 그대로 둔다. 펼친 절대경로를 저장하면 남의 기계에서 못 쓴다.
        "source_video": cfg["video"],
        "frame_size": [W, H],
        "vanishing_points": {"depth": V1[:2].tolist(), "lateral": V2[:2].tolist(),
                             "support": [len(in1), len(in2)]},
        "focal_px": focal,
        "focal_35mm_equiv": float(equiv),
        "principal_x": W / 2.0,
        "person_height_m": args.person_height,
        "camera_height_m": float(cam_h),
        "fit": {"kind": "ring_horizon", "horizon_y": horizon_y, "slope": slope,
                "n": len(good)},
        "check": {"span_x_m": float(wx), "span_z_m": float(wz),
                  "depth_over_width": float(wz / wx)},
    }
    with open(args.out, "w") as fo:
        json.dump(out, fo, indent=2, ensure_ascii=False)
    print(f"\n저장: {args.out}")

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        cv2.imwrite(f"{args.debug_dir}/ring_bg.png", bg)
        vis = bg.copy()
        for k in in2:
            s = horiz[k]
            cv2.line(vis, (s[0], s[1]), (s[2], s[3]), (0, 255, 255), 2)
        for k in in1:
            s = steep[k]
            cv2.line(vis, (s[0], s[1]), (s[2], s[3]), (255, 120, 0), 2)
        cv2.imwrite(f"{args.debug_dir}/ring_lines.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"디버그 이미지: {args.debug_dir}/ring_bg.png, ring_lines.jpg")


if __name__ == "__main__":
    main()
