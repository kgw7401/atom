"""추적 결과가 얼마나 쓸만한지 재고, 문제 구간의 시간을 뽑는다.

정답 라벨이 없으므로 "맞았나"는 못 잰다. 대신 물리적으로 말이 안 되는 것들을
찾는다. 사람은 1/60초에 순간이동할 수 없다.

재는 것:
  - 두 명이 다 잡힌 프레임 비율
  - 정체를 색으로 구분한 비율 (색을 못 믿고 위치로 때운 비율)
  - 발 위치가 화면 밖으로 잘린 비율
  - 순간이동 (한 프레임에 너무 많이 움직임) = 사람이 바뀌었을 가능성
  - 두 사람 박스가 겹친 구간 = 클린치/근접

사용법:
  python scripts/report_quality.py --csv out/IMG_0711.tracks.csv --out out/IMG_0711.quality.md
"""

import argparse
import csv
from collections import defaultdict

# 1프레임(1/60초)에 발이 이보다 많이 움직이면 사람이 바뀐 것으로 의심한다.
# 사람이 1/60초에 움직일 수 있는 거리는 화면상 수십 px 수준이다.
JUMP_PX = 120.0
CLINCH_IOU = 0.15


def segments(frames, fps, gap=15, min_len=6):
    """흩어진 프레임 번호를 연속 구간으로 묶는다. 짧은 것은 버린다."""
    if not frames:
        return []
    frames = sorted(frames)
    out, s, p = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - p > gap:
            if p - s >= min_len:
                out.append((s, p))
            s = f
        p = f
    if p - s >= min_len:
        out.append((s, p))
    return [(a, b, a / fps, b / fps) for a, b in out]


def fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=59.96)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    by_frame = defaultdict(dict)
    for r in rows:
        by_frame[int(r["frame"])][r["fighter"]] = r

    frames = sorted(by_frame)
    n_frames = frames[-1] + 1 if frames else 0
    fps = args.fps

    both = [f for f in frames if len(by_frame[f]) == 2]
    one = [f for f in frames if len(by_frame[f]) == 1]
    missing = [f for f in range(n_frames) if f not in by_frame]

    # 나 / 상대 를 따로 센다. 한쪽만 계속 놓치고 있는지 보려면 합계로는 알 수 없다.
    roles = {}
    for name in ("light", "dark"):
        got = [f for f in frames if name in by_frame[f]]
        if got:
            label = by_frame[got[0]][name].get("role", name)
            roles[label] = got

    by_prev = [f for f in frames if any(r["id_method"] == "prev" for r in by_frame[f].values())]
    clipped = [f for f in frames if any(r["foot_clipped"] == "1" for r in by_frame[f].values())]
    clinch = [f for f in both if float(next(iter(by_frame[f].values()))["box_iou"]) >= CLINCH_IOU]

    # 순간이동 탐지
    jumps = []
    last = {}
    for f in frames:
        for name, r in by_frame[f].items():
            x, y = float(r["foot_x"]), float(r["foot_y"])
            if name in last:
                pf, px, py = last[name]
                if f - pf <= 2:
                    d = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                    if d > JUMP_PX:
                        jumps.append((f, name, d))
            last[name] = (f, x, y)

    n = max(n_frames, 1)
    pct = lambda k: f"{100 * k / n:5.1f}%"

    L = []
    L.append("# 추적 품질 리포트\n")
    L.append(f"- 입력: `{args.csv}`")
    L.append(f"- 전체 프레임: {n_frames} ({n_frames/fps:.1f}초)\n")

    L.append("## 커버리지\n")
    L.append("| 항목 | 프레임 | 비율 |")
    L.append("|---|---:|---:|")
    L.append(f"| 두 명 다 잡힘 | {len(both)} | {pct(len(both))} |")
    L.append(f"| 한 명만 잡힘 | {len(one)} | {pct(len(one))} |")
    L.append(f"| 아무도 못 잡음 | {len(missing)} | {pct(len(missing))} |")
    for label, got in sorted(roles.items()):
        L.append(f"| 그중 `{label}` 잡힌 프레임 | {len(got)} | {pct(len(got))} |")
    L.append("")

    L.append("## 믿을 수 있는 정도\n")
    L.append("| 항목 | 프레임 | 비율 | 의미 |")
    L.append("|---|---:|---:|---|")
    L.append(f"| 색으로 구분 실패 | {len(by_prev)} | {pct(len(by_prev))} | 겹쳐서 옷 색을 못 봄. 직전 위치로 때움 |")
    L.append(f"| 발 위치 잘림 | {len(clipped)} | {pct(len(clipped))} | 발이 화면 아래로 나감. 위치 부정확 |")
    L.append(f"| 근접/클린치 | {len(clinch)} | {pct(len(clinch))} | 두 박스가 {int(CLINCH_IOU*100)}% 이상 겹침 |")
    L.append(f"| 순간이동 의심 | {len(jumps)} | {pct(len(jumps))} | 1프레임에 {JUMP_PX:.0f}px 초과 이동 = 사람 바뀌었을 수 있음 |")
    L.append("")

    for title, segs, why in [
        ("두 명을 다 못 잡은 구간", segments(one + missing, fps), "한 명이 가려졌거나 검출 실패"),
        ("옷 색으로 구분 못 한 구간", segments(by_prev, fps), "겹침. 이 구간의 정체는 덜 믿을 만함"),
        ("근접/클린치 구간", segments(clinch, fps), "붙어서 싸운 구간"),
    ]:
        L.append(f"## {title}\n")
        L.append(f"{why}\n")
        if not segs:
            L.append("없음\n")
            continue
        L.append("| 시작 | 끝 | 길이 |")
        L.append("|---|---|---:|")
        for a, b, ta, tb in segs[:25]:
            L.append(f"| {fmt(ta)} | {fmt(tb)} | {tb-ta:.2f}s |")
        if len(segs) > 25:
            L.append(f"\n(총 {len(segs)}구간 중 25개만 표시)")
        L.append("")

    if jumps:
        L.append("## 순간이동 의심 지점 (상위 20)\n")
        L.append("사람이 1/60초에 이만큼 움직일 수 없다. 정체가 뒤바뀐 지점일 가능성이 높다.\n")
        L.append("| 시간 | 대상 | 이동 거리 |")
        L.append("|---|---|---:|")
        for f, name, d in sorted(jumps, key=lambda z: -z[2])[:20]:
            L.append(f"| {fmt(f/fps)} | {name} | {d:.0f}px |")
        L.append("")

    with open(args.out, "w") as fo:
        fo.write("\n".join(L))
    print("\n".join(L[:32]))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
