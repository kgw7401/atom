"""추적 결과를 원본 영상 위에 그려서 눈으로 검증할 영상을 만든다.

정답 라벨을 만들지 않는다. 이 영상을 직접 보면서 점이 엉뚱한 사람에게 붙거나
튀는 구간을 찾는 것이 1단계의 검증 방법이다.

화면에 표시하는 것:
  - 화면 위쪽 거리대 막대. 지금 거리가 어느 구간인지 바늘로 표시한다
  - 선수별 색 점 (발 위치)와 박스
  - 두 사람을 잇는 선과 거리
  - 프레임 번호, 시간, 구분 방식(color/color_abs/prev)
  - 신뢰가 낮은 순간은 빨간 테두리와 빨간 글씨로 표시

거리대 구간은 `ranges.py` 한 곳에 있다. 리포트와 같은 기준을 쓴다.

사용법:
  python scripts/make_overlay.py --config configs/IMG_0711.json \
      --csv out/IMG_0711.tracks.csv --positions out/IMG_0711.positions.csv \
      --out out/IMG_0711.overlay.mp4
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from ranges import BAR_MAX_M, CONF_SHOW_MIN, RANGE_BANDS, TextLayer, band_of

COLORS = {          # BGR
    "light": (80, 220, 255),   # 노랑 - 흰 상의
    "dark": (255, 140, 60),    # 파랑 - 검정 상의
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_range_bar(fr, text, dist_m, conf, n_found=2):
    """화면 위쪽에 거리대 막대를 그린다.

    지금 거리가 어느 구간인지 눈으로 바로 보이게 하는 것이 목적이다.
    숫자만 보고 "이게 근거리인가 중거리인가" 를 매번 계산하지 않아도 되게 한다.
    """
    w = fr.shape[1]
    x0, x1 = 40, w - 40
    y0, y1 = 74, 116                       # 막대 위/아래
    span = x1 - x0

    def to_px(m):
        return int(x0 + span * min(m, BAR_MAX_M) / BAR_MAX_M)

    # 배경 (반투명 검정)
    pad = fr[0:170, :].copy()
    cv2.rectangle(pad, (0, 0), (w, 170), (0, 0, 0), -1)
    fr[0:170, :] = cv2.addWeighted(fr[0:170, :], 0.35, pad, 0.65, 0)

    # 신뢰도가 낮으면 구간을 아예 켜지 않는다.
    # "밀착 (신뢰도 낮음)" 이라고 쓰면 사람 눈에는 '밀착' 만 남는다.
    # 두 사람이 겹치면 검출이 무너져 거리가 실제보다 짧게 나오므로,
    # 그 순간을 밀착으로 보여주면 없는 클린치를 만들어내는 셈이다.
    trusted = dist_m is not None and conf >= CONF_SHOW_MIN
    cur = band_of(dist_m) if trusted else None

    # 구간별 색 칠하기. 지금 구간만 진하게, 나머지는 흐리게.
    for lo, hi, name, _desc, color in RANGE_BANDS:
        bx0, bx1 = to_px(lo), to_px(hi)
        if bx1 <= bx0:
            continue
        active = cur is not None and cur[0] == name
        c = color if active else tuple(int(v * 0.35) for v in color)
        cv2.rectangle(fr, (bx0, y0), (bx1, y1), c, -1)
        cv2.rectangle(fr, (bx0, y0), (bx1, y1), (30, 30, 30), 1)
        text.add(name, ((bx0 + bx1) // 2, (y0 + y1) // 2), size=22,
                 color=(255, 255, 255) if active else (170, 170, 170),
                 anchor="mm", outline=3)

    # 눈금
    for m in (0.8, 1.3, 1.9):
        mx = to_px(m)
        cv2.line(fr, (mx, y1), (mx, y1 + 7), (200, 200, 200), 2)
        text.add(f"{m:.1f}m", (mx, y1 + 11), size=17, color=(200, 200, 200),
                 anchor="ma", outline=2)

    # 현재 위치 바늘. 믿을 때만 채워서 그리고, 아니면 속 빈 회색 표시만 남긴다.
    if dist_m is not None:
        px = to_px(dist_m)
        if trusted:
            cv2.fillPoly(fr, [np.array([[px, y0 - 4], [px - 11, y0 - 20],
                                        [px + 11, y0 - 20]])], (255, 255, 255))
            cv2.line(fr, (px, y0), (px, y1), (255, 255, 255), 3)
        else:
            cv2.polylines(fr, [np.array([[px, y0 - 4], [px - 11, y0 - 20],
                                         [px + 11, y0 - 20]])], True, (140, 140, 140), 2)

    # 큰 글씨 요약
    if trusted:
        name, desc, color = cur
        text.add(f"{dist_m:.2f}m   {name} ({desc})", (x0, 34), size=34,
                 color=color, anchor="la", outline=4)
    else:
        if dist_m is None and n_found < 2:
            why = "한 명을 못 찾음"
        elif dist_m is None:
            # 두 명 다 찾았는데 거리가 버려졌다 = 물리적으로 불가능한 값이 나왔다.
            why = "두 사람이 겹쳐 검출이 한쪽으로 무너짐 (거리가 사람 몸보다 가까움)"
        else:
            why = f"두 사람이 겹쳐서 검출이 흔들림 (참고값 {dist_m:.2f}m, 신뢰도 {conf:.2f})"
        text.add(f"측정 불가 — {why}", (x0, 34), size=32,
                 color=(150, 150, 150), anchor="la", outline=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--positions", default=None,
                    help="make_positions.py 결과. 주면 거리를 px 대신 m 로 표시한다")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="시작 초")
    ap.add_argument("--duration", type=float, default=0.0, help="길이 초 (0이면 끝까지)")
    ap.add_argument("--trail", type=int, default=45, help="발자취를 몇 프레임 남길지")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    video = os.path.expanduser(cfg["video"])
    ring = np.array(cfg["ring_polygon"], np.int32)

    by_frame = defaultdict(dict)
    for r in csv.DictReader(open(args.csv)):
        by_frame[int(r["frame"])][r["fighter"]] = r

    # 거리를 버린 프레임도 담는다 (값은 None). 담지 않으면 아래에서 "거리 정보 없음"
    # 으로 착각해 화면상 픽셀 거리를 대신 띄우는데, 그러면 버린 값을 다른 옷을 입혀
    # 다시 보여주는 셈이 된다.
    metric = {}
    if args.positions:
        for r in csv.DictReader(open(args.positions)):
            d = float(r["distance_m"]) if r["distance_m"] else None
            metric[int(r["frame"])] = (d, float(r["confidence"]))

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    f0 = int(args.start * fps)
    f1 = int((args.start + args.duration) * fps) if args.duration else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    trail = defaultdict(list)
    text = TextLayer()
    fi = f0
    while fi < f1:
        ok, fr = cap.read()
        if not ok:
            break
        rec = by_frame.get(fi, {})

        # 링 영역 (걸러내기 기준선)
        cv2.polylines(fr, [ring], True, (0, 200, 0), 2)

        pts = {}
        for name, row in rec.items():
            col = COLORS[name]
            x1, y1, x2, y2 = (int(float(row[k])) for k in ("x1", "y1", "x2", "y2"))
            fx, fy = int(float(row["foot_x"])), int(float(row["foot_y"]))
            pts[name] = (fx, fy)

            cv2.rectangle(fr, (x1, y1), (x2, y2), col, 2)
            cv2.circle(fr, (fx, fy), 9, col, -1)
            cv2.circle(fr, (fx, fy), 9, (0, 0, 0), 2)

            tag = f"{row.get('role') or name}  {row['id_method']}"
            if row["foot_clipped"] == "1":
                tag += " CLIPPED"
            cv2.putText(fr, tag, (x1, max(20, y1 - 8)), FONT, 0.6, col, 2, cv2.LINE_AA)

            trail[name].append((fx, fy))
            trail[name] = trail[name][-args.trail:]

        # 발자취
        for name, tr in trail.items():
            for i in range(1, len(tr)):
                a = int(255 * i / len(tr))
                cv2.line(fr, tr[i - 1], tr[i], tuple(int(c * a / 255) for c in COLORS[name]), 2)

        # 두 사람 사이 거리
        if len(pts) == 2:
            (ax, ay), (bx, by) = pts["light"], pts["dark"]
            cv2.line(fr, (ax, ay), (bx, by), (255, 255, 255), 2)
            label, col = None, None
            if fi in metric:
                d_m, c = metric[fi]
                if d_m is not None:
                    # 못 믿는 값은 크게 띄우지 않는다. 괄호로 참고값임을 밝힌다.
                    trusted = c >= CONF_SHOW_MIN
                    label = f"{d_m:.2f}m" if trusted else f"({d_m:.2f}m?)"
                    col = (255, 255, 255) if trusted else (150, 150, 150)
                # d_m 이 None 이면 버린 값이다. 아무것도 쓰지 않는다.
            else:
                d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                label, col = f"{d:.0f}px", (255, 255, 255)
            if label:
                mx, my = (ax + bx) // 2, (ay + by) // 2
                cv2.putText(fr, label, (mx - 55, my - 12), FONT, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(fr, label, (mx - 55, my - 12), FONT, 1.0, col, 2, cv2.LINE_AA)

        # 화면 위쪽 거리대 막대
        d_m, conf = metric.get(fi, (None, 0.0))
        draw_range_bar(fr, text, d_m, conf, n_found=len(rec))

        # 막대 아래 상태 줄
        any_row = next(iter(rec.values()), None)
        iou = any_row["box_iou"] if any_row else "-"
        methods = "  ".join(f"{r.get('role') or n}={r['id_method']}" for n, r in sorted(rec.items()))
        status = (f"frame {fi}   t={fi/fps:.2f}s   found {len(rec)}/2   "
                  f"overlap={iou}   {methods or '-'}")
        cv2.putText(fr, status, (40, 158), FONT, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(fr, status, (40, 158), FONT, 0.62, (230, 230, 230), 1, cv2.LINE_AA)

        if len(rec) < 2 or any(r["id_method"] == "prev" for r in rec.values()):
            cv2.rectangle(fr, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

        fr = text.render(fr)

        vw.write(fr)
        fi += 1

    cap.release()
    vw.release()
    print(f"저장: {args.out}  ({fi - f0} frames)")


if __name__ == "__main__":
    main()
