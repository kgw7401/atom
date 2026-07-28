"""위에서 본 링 그림을 만든다 — 원본 영상과 나란히 놓아 눈으로 검증한다.

`docs/spatial-model.md` 에서 정한 표현을 실제로 그린다.
선수 한 명은 점이 아니라 원 두 개(몸 / 손 도달 범위)와 정면 화살표다.

왼쪽에 원본, 오른쪽에 위에서 본 그림을 붙인다. 두 화면을 같이 보면서
점이 실제 움직임과 맞는지, 원이 겹치는 순간이 실제로 주고받는 순간인지 확인한다.

사용법:
  python scripts/make_topdown.py --config configs/IMG_0711.json \\
      --positions out/IMG_0711.positions.csv --ground configs/IMG_0711.ground.json \\
      --out out/topdown.mp4 --start 40 --duration 30
"""

import argparse
import csv
import json
import math
import os

import cv2
import numpy as np

from ranges import TextLayer

COL = {"me": (255, 140, 60), "opponent": (80, 220, 255)}   # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Board:
    """링 좌표(m) 를 그림 좌표(px) 로 바꿔주는 캔버스."""

    def __init__(self, w, h, x_range, z_range, margin=54):
        self.w, self.h, self.m = w, h, margin
        self.x0, self.x1 = x_range
        self.z0, self.z1 = z_range
        sx = (w - 2 * margin) / (self.x1 - self.x0)
        sz = (h - 2 * margin) / (self.z1 - self.z0)
        self.s = min(sx, sz)                       # 가로세로 비율을 유지한다
        self.cx = w / 2 - self.s * (self.x0 + self.x1) / 2
        self.cz = h / 2 + self.s * (self.z0 + self.z1) / 2

    def px(self, x, z):
        # z 가 클수록(카메라에서 멀수록) 위쪽에 그린다
        return int(self.cx + self.s * x), int(self.cz - self.s * z)

    def r(self, meters):
        return max(1, int(self.s * meters))


def draw_board(bg_h, bg_w, board, rec, text, ring=None, tx=0):
    """tx 는 글자를 찍을 때 더할 가로 오프셋.

    글자는 합쳐진 최종 화면에 한 번에 그리기 때문에, 오른쪽 판에 찍을 글자는
    왼쪽 원본의 너비만큼 밀어줘야 한다.
    """
    img = np.full((bg_h, bg_w, 3), 22, np.uint8)

    # 1m 격자
    for m in np.arange(math.ceil(board.x0), board.x1, 1.0):
        u, _ = board.px(m, board.z0)
        cv2.line(img, (u, 0), (u, bg_h), (46, 46, 46), 1)
    for m in np.arange(math.ceil(board.z0), board.z1, 1.0):
        _, v = board.px(board.x0, m)
        cv2.line(img, (0, v), (bg_w, v), (46, 46, 46), 1)

    if ring is not None:
        pts = np.array([board.px(x, z) for x, z in ring], np.int32)
        cv2.polylines(img, [pts], True, (70, 130, 70), 2)

    # 카메라 방향 표시
    text.add("카메라 쪽", (tx + bg_w // 2, bg_h - 26), size=20, color=(150, 150, 150), anchor="ms")

    pos = {}
    for role in ("me", "opponent"):
        r = rec.get(role)
        if r is None:
            continue
        x, z = r["x"], r["z"]
        pos[role] = (x, z)
        u, v = board.px(x, z)
        c = COL[role]

        # 손 도달 범위
        if r["reach"]:
            cv2.circle(img, (u, v), board.r(r["reach"]), tuple(int(q * 0.55) for q in c), 2)
        # 몸
        cv2.circle(img, (u, v), board.r(0.22), c, -1)
        cv2.circle(img, (u, v), board.r(0.22), (15, 15, 15), 2)

        # 정면 화살표. 바닥 좌표의 +z 는 화면에서 위쪽이므로 세로 부호를 뒤집는다.
        if r["facing"] is not None:
            a = math.radians(r["facing"])
            L = board.r(r["reach"] or 0.6) + 16
            cv2.arrowedLine(img, (u, v),
                            (int(u + L * math.cos(a)), int(v - L * math.sin(a))),
                            c, 3, tipLength=0.3)
        text.add("나" if role == "me" else "상대", (tx + u, v - board.r(0.22) - 10),
                 size=21, color=c, anchor="ms")

    # 두 사람을 잇는 선과 거리
    if len(pos) == 2:
        pa, pb = board.px(*pos["me"]), board.px(*pos["opponent"])
        cv2.line(img, pa, pb, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--ground", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--min-conf", type=float, default=0.5)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    video = os.path.expanduser(cfg["video"])

    rows = {}
    for r in csv.DictReader(open(args.positions)):
        rows[int(r["frame"])] = r

    # 링 경계. calibrate_ring.py 가 캔버스에서 검출한 것 (링 좌표계).
    # 있으면 그림 범위도 링에 맞추고, 좌표도 링 축으로 돌려서 그린다 —
    # 링이 화면에 반듯한 정사각형으로 나오게 하기 위해서다.
    g = json.load(open(args.ground))
    ring_info = g.get("ring")
    Rax = None
    ring_poly = None
    if ring_info:
        Rax = np.array(ring_info["axes"])          # 카메라 바닥좌표 -> 링 좌표
        L, Rt = ring_info["left_a"], ring_info["right_a"]
        Nb, Fb = ring_info["near_b"], ring_info["far_b"]
        ring_poly = [(L, Nb), (Rt, Nb), (Rt, Fb), (L, Fb)]
        pad = 0.5
        x_range = (L - pad, Rt + pad)
        z_range = (Nb - pad, Fb + pad)
        print(f"링: 한 변 {ring_info['side_m']:.2f} m  (앞쪽 변은 정사각형 조건으로 추정)")
    else:
        xs, zs = [], []
        for r in rows.values():
            for a, b in (("me_x", "me_z"), ("op_x", "op_z")):
                if r[a] and float(r["confidence"]) >= 0.6:
                    xs.append(float(r[a])); zs.append(float(r[b]))
        pad = 0.6
        x_range = (min(xs) - pad, max(xs) + pad)
        z_range = (min(zs) - pad, max(zs) + pad)
    print(f"그림 범위: x {x_range[0]:.1f}~{x_range[1]:.1f}  z {z_range[0]:.1f}~{z_range[1]:.1f} m")

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f0 = int(args.start * fps)
    f1 = int((args.start + args.duration) * fps) if args.duration else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    # 왼쪽 원본(축소) + 오른쪽 위에서 본 그림
    vh = 720
    vw = int(W * vh / H)
    bw = 720
    board = Board(bw, vh, x_range, z_range)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vw_out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw + bw, vh))

    text = TextLayer()
    fi = f0
    while fi < f1:
        ok, fr = cap.read()
        if not ok:
            break
        left = cv2.resize(fr, (vw, vh))

        r = rows.get(fi)
        rec = {}
        conf = float(r["confidence"]) if r else 0.0
        if r:
            for role, tag in (("me", "me"), ("opponent", "op")):
                if not r[f"{tag}_x"]:
                    continue
                x, z = float(r[f"{tag}_x"]), float(r[f"{tag}_z"])
                fa = float(r[f"{tag}_facing"]) if r[f"{tag}_facing"] else None
                if Rax is not None:
                    # 카메라 바닥좌표 -> 링 좌표로 회전
                    x, z = Rax @ np.array([x, z])
                    if fa is not None:
                        v = Rax @ np.array([math.cos(math.radians(fa)),
                                            math.sin(math.radians(fa))])
                        fa = math.degrees(math.atan2(v[1], v[0]))
                rec[role] = {
                    "x": x, "z": z, "facing": fa,
                    "reach": float(r[f"{tag}_reach"]) if r[f"{tag}_reach"] else None,
                }
        right = draw_board(vh, bw, board, rec if conf >= args.min_conf else {}, text,
                           ring=ring_poly, tx=vw)

        head = f"t={fi/fps:6.2f}s"
        if r and r["distance_m"] and conf >= args.min_conf:
            head += f"   거리 {float(r['distance_m']):.2f} m"
        elif conf < args.min_conf:
            head += "   측정 불가"
        text.add(head, (16, 14), size=26, color=(235, 235, 235), anchor="la")

        canvas = np.hstack([left, right])
        canvas = text.render(canvas)
        vw_out.write(canvas)
        fi += 1

    cap.release()
    vw_out.release()
    print(f"저장: {args.out}  ({fi - f0} frames, {vw + bw}x{vh})")


if __name__ == "__main__":
    main()
