"""위치 데이터에서 공간 스탯을 뽑는다.

`docs/spatial-model.md` 의 "스탯이 먼저, 분류는 나중" 원칙에 따라, 스타일 분류가
아니라 **해석 가능한 수치**를 만든다. 하나하나 눈으로 검증할 수 있어야 한다.

지금 계산하는 것:
  거리    — 분포, 사거리 관계 (일방 우위 / 동시 위험 / 안전)
  움직임  — 접근·후퇴 주도권, 전진/후퇴 시간 비, 이동량
  각도    — 각도 우위 시간, 피벗 빈도

아직 못 하는 것:
  거리 5단계 체류 시간 — 경계값이 무효가 됐고 '한 발 사거리' 가 없다
  링 관련 (중앙 점유, 코너 압박) — 링 경계가 영상마다 다르게 나온다

사용법:
  python scripts/make_stats.py --positions out/IMG_0711.positions.csv \\
      --out out/IMG_0711.stats.md
"""

import argparse
import csv
import json
import math

import numpy as np

BODY_R = 0.22          # 몸 반경 (m). 사거리 원이 상대 '몸' 에 닿는지 볼 때 쓴다
MIN_CONF = 0.6         # 이 신뢰도 미만은 계산에서 뺀다
VEL_HALF = 3           # 속도를 잴 때 앞뒤로 몇 프레임을 볼지 (±3 = 0.1초)
MAX_SPEED = 6.0        # 사람이 낼 수 없는 속도 (m/s). 이보다 크면 추적이 튄 것
# 피벗: 순간 각속도로 세면 잡음을 센다 (정면이 프레임당 1.3도 떨리면 초당 78도다).
# 짧은 창 동안 '한 방향으로 꾸준히' 돈 양을 보고, 한 번 센 뒤에는 잠시 쉰다.
# 정면 방향 신뢰도 하한.
# 이 값은 어깨 단서와 발선 단서가 얼마나 일치하는지다 (둘이 어긋나면 낮아진다).
# 어깨선만 쓰던 시절에는 기하 특이 구성 때문에 20% 프레임밖에 못 썼는데,
# 발선을 합친 뒤로는 0.5 하한에서 98~99% 가 남는다.
# 홀드아웃 검증: 오차 중앙 23도, 90도 이내 95~98%.
FACING_REL_MIN = 0.5

PIVOT_WIN = 0.30       # 회전량을 누적할 창 (초)
PIVOT_TURN = 45.0      # 그 창에서 이만큼 넘게 돌면 피벗 (도)
PIVOT_REFRACT = 0.40   # 한 번 세면 이 시간 동안은 다시 안 센다 (초)

# 사거리 원에 더할 '한 발' 여유 (m). 실측이 아니라 감도를 보여주기 위한 값들이다.
STEP_ALLOWANCES = [0.0, 0.2, 0.4, 0.6]


def load(path, fps_hint=59.96):
    rows = list(csv.DictReader(open(path)))
    fps = fps_hint
    for r in rows:
        if float(r["time_sec"]) > 0:
            fps = int(r["frame"]) / float(r["time_sec"])
            break
    keep = {}
    for r in rows:
        if not r["me_x"] or not r["op_x"]:
            continue
        if float(r["confidence"]) < MIN_CONF:
            continue
        keep[int(r["frame"])] = {
            "me": np.array([float(r["me_x"]), float(r["me_z"])]),
            "op": np.array([float(r["op_x"]), float(r["op_z"])]),
            "me_face": float(r["me_facing"]) if r["me_facing"] else None,
            "op_face": float(r["op_facing"]) if r["op_facing"] else None,
            "me_rel": float(r.get("me_facing_rel") or 0),
            "op_rel": float(r.get("op_facing_rel") or 0),
            "me_reach": float(r["me_reach"]) if r["me_reach"] else None,
            "op_reach": float(r["op_reach"]) if r["op_reach"] else None,
            "d": float(r["distance_m"]),
        }
    return keep, fps, len(rows)


def velocities(data, frames, fps):
    """중앙차분으로 속도를 구한다. 사람이 낼 수 없는 값은 버린다.

    속도는 위치의 미분이라 잡음이 증폭된다. 1단계에서 최대 20m/s 같은 값이
    나왔는데, 이는 추적이 튄 것이지 움직임이 아니다.
    """
    idx = {f: i for i, f in enumerate(frames)}
    out = {}
    for f in frames:
        i = idx[f]
        lo, hi = i, i
        while lo > 0 and f - frames[lo - 1] <= VEL_HALF:
            lo -= 1
        while hi + 1 < len(frames) and frames[hi + 1] - f <= VEL_HALF:
            hi += 1
        dt = (frames[hi] - frames[lo]) / fps
        if dt <= 0:
            continue
        v = {}
        ok = True
        for who in ("me", "op"):
            vec = (data[frames[hi]][who] - data[frames[lo]][who]) / dt
            if np.linalg.norm(vec) > MAX_SPEED:
                ok = False
                break
            v[who] = vec
        if ok:
            out[f] = v
    return out


def wrap(deg):
    return (deg + 180) % 360 - 180


def facing_error(pos_self, pos_other, facing_deg):
    """정면이 상대 쪽에서 얼마나 벗어났나 (도, 0~180)."""
    d = pos_other - pos_self
    want = math.degrees(math.atan2(d[1], d[0]))
    return abs(wrap(facing_deg - want))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--label", default=None, help="리포트 제목에 쓸 이름")
    args = ap.parse_args()

    data, fps, n_all = load(args.positions)
    frames = sorted(data)
    if len(frames) < 100:
        print(f"쓸 만한 프레임이 너무 적다 ({len(frames)}).")
        return
    dur = len(frames) / fps
    S = {}          # 리포트에 쓸 수치 모음
    L = []          # 마크다운 줄

    label = args.label or args.positions
    L.append(f"# 공간 스탯 — {label}\n")
    L.append(f"- 입력: `{args.positions}`")
    L.append(f"- 쓸 만한 프레임: {len(frames)} / {n_all} "
             f"({100*len(frames)/max(n_all,1):.1f}%, 약 {dur:.0f}초)")
    L.append(f"- 신뢰도 {MIN_CONF} 미만은 제외\n")

    # ---------------- 거리 ----------------
    d = np.array([data[f]["d"] for f in frames])
    S["distance"] = {"median": float(np.median(d)), "p25": float(np.percentile(d, 25)),
                     "p75": float(np.percentile(d, 75)), "min": float(d.min()),
                     "max": float(d.max()), "std": float(d.std())}
    L.append("## 거리\n")
    L.append("| 항목 | 값 |")
    L.append("|---|---:|")
    L.append(f"| 중앙값 | {np.median(d):.2f} m |")
    L.append(f"| 25~75% | {np.percentile(d,25):.2f} ~ {np.percentile(d,75):.2f} m |")
    L.append(f"| 최소~최대 | {d.min():.2f} ~ {d.max():.2f} m |")
    L.append(f"| 표준편차 | {d.std():.2f} m |\n")

    # 사거리 관계 — 원 두 개가 상대 '몸' 에 닿는지로 나눈다.
    # 주의: 지금 재는 사거리는 발을 안 옮기고 손만 뻗은 거리다. 실제 펀치는
    # 스텝을 밟으므로 이보다 멀리 닿는다. 한 발 사거리가 없어서 단정할 수 없고,
    # 대신 여유값을 바꿔가며 감도를 보여준다.
    mr = data[frames[0]]["me_reach"]
    orr = data[frames[0]]["op_reach"]
    if mr and orr:
        L.append("### 사거리 관계\n")
        L.append(f"정지 사거리: 나 {mr:.2f} m, 상대 {orr:.2f} m (몸 반경 {BODY_R} m 포함)\n")
        L.append("**아직 확정할 수 없는 값이다.** 지금 사거리는 발을 안 옮기고 잰 것이고,")
        L.append("실제 펀치는 스텝을 밟아 더 멀리 닿는다. '한 발 사거리' 가 없으므로")
        L.append("여유값을 바꿔가며 결과가 얼마나 달라지는지만 보여준다.\n")
        L.append("| 한 발 여유 | 동시 위험 | 나만 닿음 | 상대만 닿음 | 둘 다 못 닿음 |")
        L.append("|---:|---:|---:|---:|---:|")
        sens = {}
        n = len(d)
        for step in STEP_ALLOWANCES:
            me_hits = d <= mr + BODY_R + step
            op_hits = d <= orr + BODY_R + step
            both = int((me_hits & op_hits).sum())
            only_me = int((me_hits & ~op_hits).sum())
            only_op = int((~me_hits & op_hits).sum())
            neither = int((~me_hits & ~op_hits).sum())
            sens[f"{step:.1f}"] = {"both": both, "only_me": only_me,
                                   "only_op": only_op, "neither": neither}
            L.append(f"| +{step:.1f} m | {100*both/n:.0f}% | {100*only_me/n:.0f}% "
                     f"| {100*only_op/n:.0f}% | {100*neither/n:.0f}% |")
        S["range_relation"] = {"me_reach": mr, "op_reach": orr, "sensitivity": sens}
        L.append("")
        L.append("> 여유 0 이면 거의 전부 '못 닿음' 으로 나온다. 스파링에서 그럴 리 없으므로,")
        L.append("> 이것 자체가 **한 발 사거리가 반드시 필요하다는 증거**다.")
        L.append("> 제대로 정하려면 2단계(펀치 이벤트)에서 실제로 닿은 거리를 봐야 한다.\n")
    else:
        L.append("### 사거리 관계\n\n사거리 값이 없어 계산하지 못했다.\n")

    # ---------------- 움직임 ----------------
    vel = velocities(data, frames, fps)
    vf = sorted(vel)
    L.append("## 움직임\n")
    if len(vf) < 100:
        L.append("속도를 낼 수 있는 프레임이 부족하다.\n")
    else:
        my_close, op_close, dd = [], [], []
        for f in vf:
            p_me, p_op = data[f]["me"], data[f]["op"]
            u = p_op - p_me
            n = np.linalg.norm(u)
            if n < 1e-6:
                continue
            u = u / n
            a = float(u @ vel[f]["me"])       # 내가 상대 쪽으로 가는 속도
            b = float(-u @ vel[f]["op"])      # 상대가 내 쪽으로 오는 속도
            my_close.append(a)
            op_close.append(b)
            dd.append(-(a + b))               # 거리 변화율
        my_close = np.array(my_close)
        op_close = np.array(op_close)
        dd = np.array(dd)

        closing = dd < -0.05                  # 거리가 줄어드는 순간
        opening = dd > 0.05                   # 거리가 벌어지는 순간
        # 줄어드는 순간 누가 더 기여했나 / 벌어지는 순간 누가 더 물러났나
        me_led_close = int((my_close[closing] > op_close[closing]).sum())
        me_led_open = int(((-my_close[opening]) > (-op_close[opening])).sum())
        fwd = float((my_close > 0.05).mean())
        speed = np.linalg.norm(np.array([vel[f]["me"] for f in vf]), axis=1)
        op_speed = np.linalg.norm(np.array([vel[f]["op"] for f in vf]), axis=1)

        S["movement"] = {
            "closing_frames": int(closing.sum()), "opening_frames": int(opening.sum()),
            "me_led_closing_pct": 100 * me_led_close / max(int(closing.sum()), 1),
            "me_led_opening_pct": 100 * me_led_open / max(int(opening.sum()), 1),
            "me_forward_pct": 100 * fwd,
            "me_speed_median": float(np.median(speed)),
            "op_speed_median": float(np.median(op_speed)),
        }
        L.append("| 항목 | 값 | 뜻 |")
        L.append("|---|---:|---|")
        L.append(f"| 거리 좁혀진 시간 | {closing.sum()/fps:.0f}초 | |")
        L.append(f"| 거리 벌어진 시간 | {opening.sum()/fps:.0f}초 | |")
        L.append(f"| **좁힐 때 내가 주도** | **{100*me_led_close/max(int(closing.sum()),1):.0f}%** "
                 f"| 높으면 내가 들어가는 쪽 |")
        L.append(f"| **벌릴 때 내가 주도** | **{100*me_led_open/max(int(opening.sum()),1):.0f}%** "
                 f"| 높으면 내가 빠지는 쪽 |")
        L.append(f"| 내가 전진 중인 시간 | {100*fwd:.0f}% | |")
        L.append(f"| 내 이동속도 중앙값 | {np.median(speed):.2f} m/s | |")
        L.append(f"| 상대 이동속도 중앙값 | {np.median(op_speed):.2f} m/s | |")
        L.append("")
        L.append("> 주도권은 두 사람의 속도를 서로를 잇는 축에 투영해 나눈 것이다.")
        L.append("> 거리 변화 = (내가 다가간 양) + (상대가 다가온 양) 으로 분해된다.\n")

    # ---------------- 각도 ----------------
    L.append("## 각도\n")
    pairs = [f for f in frames if data[f]["me_face"] is not None
             and data[f]["op_face"] is not None
             and data[f]["me_rel"] >= FACING_REL_MIN
             and data[f]["op_rel"] >= FACING_REL_MIN]
    L.append(f"> **신뢰도 {FACING_REL_MIN} 이상인 프레임만 썼다** — "
             f"{len(pairs)} 개 ({100*len(pairs)/len(frames):.0f}%).")
    L.append("> 정면 방향은 어깨선과 발선(스탠스) 두 단서를 합쳐 구한다.")
    L.append("> 신뢰도는 두 단서가 얼마나 일치하는지다. 어긋나는 순간은 걸러진다.\n")
    if len(pairs) < 100:
        L.append("신뢰할 만한 정면 방향 프레임이 부족하다.\n")
    else:
        me_err, op_err = [], []
        for f in pairs:
            me_err.append(facing_error(data[f]["me"], data[f]["op"], data[f]["me_face"]))
            op_err.append(facing_error(data[f]["op"], data[f]["me"], data[f]["op_face"]))
        me_err, op_err = np.array(me_err), np.array(op_err)
        adv = float((me_err < op_err).mean())
        S["angle"] = {"me_err_median": float(np.median(me_err)),
                      "op_err_median": float(np.median(op_err)),
                      "me_advantage_pct": 100 * adv}
        L.append("| 항목 | 나 | 상대 |")
        L.append("|---|---:|---:|")
        L.append(f"| 정면이 상대에서 벗어난 각 (중앙) | {np.median(me_err):.0f}도 | {np.median(op_err):.0f}도 |")
        L.append(f"| 45도 이내로 마주 본 시간 | {100*(me_err<45).mean():.0f}% | {100*(op_err<45).mean():.0f}% |")
        L.append("")
        L.append(f"**각도 우위 시간: {100*adv:.0f}%** — 내가 상대보다 더 정면으로 "
                 f"보고 있던 시간 비율. 50% 면 대등하다.\n")
        L.append("> 복서는 몸을 비스듬히 서므로 20~30도가 정상이다. 0도가 나오면 오히려 이상하다.\n")

        # 피벗 — 짧은 창 동안 한 방향으로 꾸준히 돈 양으로 센다.
        # 순간 각속도로 세면 잡음이 그대로 카운트된다 (초당 1000회가 나왔다).
        win = max(2, int(PIVOT_WIN * fps))
        refract = max(1, int(PIVOT_REFRACT * fps))
        piv = {}
        for who, key, rkey in (("me", "me_face", "me_rel"), ("op", "op_face", "op_rel")):
            seq = [(f, data[f][key]) for f in frames
                   if data[f][key] is not None and data[f][rkey] >= FACING_REL_MIN]
            cnt, last = 0, -10 ** 9
            for i in range(len(seq)):
                j = i
                while j + 1 < len(seq) and seq[j + 1][0] - seq[i][0] <= win:
                    j += 1
                if j == i or seq[j][0] - seq[i][0] < win * 0.6:
                    continue
                turn = wrap(seq[j][1] - seq[i][1])       # 창 양끝의 순 회전량
                if abs(turn) >= PIVOT_TURN and seq[i][0] - last >= refract:
                    cnt += 1
                    last = seq[i][0]
            piv[who] = cnt
        S["pivot"] = {"me_per_min": piv["me"] / (dur / 60), "op_per_min": piv["op"] / (dur / 60)}
        L.append(f"| 피벗 ({PIVOT_WIN:.1f}초 안에 {PIVOT_TURN:.0f}도 이상 회전) | "
                 f"분당 {piv['me']/(dur/60):.0f}회 | 분당 {piv['op']/(dur/60):.0f}회 |\n")

    # ---------------- 못 한 것 ----------------
    L.append("## 아직 계산하지 못하는 것\n")
    L.append("| 스탯 | 이유 |")
    L.append("|---|---|")
    L.append("| 거리 5단계 체류 시간 | 경계값이 보정 변경으로 무효가 됐고, '한 발 사거리' 원이 아직 없다 |")
    L.append("| 중앙 점유율 · 코너 압박 | 링 경계가 영상마다 다르게 측정된다 (`docs/spatial-model.md`) |")
    L.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(L))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(S, f, indent=2, ensure_ascii=False)
    print("\n".join(L))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
