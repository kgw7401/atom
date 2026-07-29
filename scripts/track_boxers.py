"""스파링 영상에서 링 안 두 선수의 위치를 프레임마다 뽑는다.

1단계 산출물: 프레임별 두 사람의 발 위치 (화면 좌표) + 신뢰도.
미터 변환은 별도 단계에서 붙인다.

사람 구분 방식:
  YOLO 추적 ID는 쓰지 않는다. 실제 영상에서 두 선수 모두 트랙이 중간에 끊겨
  새 ID를 받는 것을 확인했다 (1->14, 3->47).

  대신 상의 밝기로 구분한다. 한 명은 검정 상의, 한 명은 흰 상의라 평소에는
  아주 잘 갈린다 (L* 8~26 vs 143~206). 두 사람이 겹쳐서 색을 못 믿는 순간에만
  직전 프레임 위치로 이어붙이고, 그 구간은 신뢰도를 낮춰 표시한다.

사용법:
  python scripts/track_boxers.py --config configs/IMG_0711.json --out out/IMG_0711.tracks.csv
"""

import argparse
import csv
import json
import math
import os

import cv2
import numpy as np
from ultralytics import YOLO

# COCO 자세 키포인트 번호
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
NOSE = 0
L_ANKLE, R_ANKLE = 15, 16

KP_CONF_MIN = 0.5          # 키포인트를 믿을 최소 점수
COLOR_GAP_MIN = 60.0       # 두 사람 밝기 차가 이만큼은 나야 서로 비교해 구분
MIN_TORSO_PX = 200         # 몸통 표본이 이보다 적으면 색을 안 믿는다

# 한 명만 보일 때 쓰는 절대 기준. 실측한 값은 흰 상의 143~206, 검정 상의 8~26이라
# 그 사이를 넓게 비워둔다. 애매하면 판정하지 않고 직전 위치로 넘긴다.
LIGHT_L_MIN = 110.0
DARK_L_MAX = 45.0

# 순간이동 차단. 사람은 1/60초에 화면을 가로지를 수 없다.
# 링 옆에 서 있던 다른 관원이 선수로 잘못 이어붙은 사고가 있었기 때문에 넣었다.
# 45px/프레임 은 이 영상 배율에서 대략 6m/s 로, 복서 스텝보다 충분히 넉넉하다.
MAX_JUMP_PER_FRAME = 45.0
MAX_JUMP_FLOOR = 120.0     # 검출 흔들림을 감안한 최소 허용치
MAX_JUMP_CAP = 600.0       # 오래 안 보였어도 이 이상은 같은 사람으로 안 본다


def jump_limit(frames_elapsed):
    """직전에 본 지 몇 프레임 지났는지에 따라 허용 이동 거리를 정한다."""
    return min(MAX_JUMP_CAP, max(MAX_JUMP_FLOOR, MAX_JUMP_PER_FRAME * frames_elapsed))


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg["video"] = os.path.expanduser(cfg["video"])
    return cfg


def torso_rect(box, kxy, kcf):
    """상의 색을 잴 몸통 영역. 어깨-엉덩이 키포인트가 있으면 그걸 쓴다."""
    pts = [kxy[i] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP) if kcf[i] >= KP_CONF_MIN]
    if len(pts) >= 3:
        p = np.array(pts)
        x1, y1 = p.min(0)
        x2, y2 = p.max(0)
        return x1, y1, x2, y2
    # 키포인트가 부족하면 박스 상단부로 대체
    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    return bx1 + 0.25 * bw, by1 + 0.25 * bh, bx2 - 0.25 * bw, by1 + 0.50 * bh


def torso_brightness(img, box, kxy, kcf, other_box):
    """몸통의 median 밝기(L*). 상대 박스와 겹치는 픽셀은 빼고 잰다.

    반환: (L값 or None, 표본 픽셀 수)
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = torso_rect(box, kxy, kcf)
    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(min(w - 1, x2)), int(min(h - 1, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, 0

    patch = img[y1:y2, x1:x2]
    mask = np.ones(patch.shape[:2], bool)

    # 상대방 박스와 겹치는 부분은 상대 옷일 수 있으니 제외
    if other_box is not None:
        ox1, oy1, ox2, oy2 = other_box
        ix1, iy1 = int(max(x1, ox1)), int(max(y1, oy1))
        ix2, iy2 = int(min(x2, ox2)), int(min(y2, oy2))
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = False

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[:, :, 0]
    vals = lab[mask]
    if vals.size < MIN_TORSO_PX:
        return None, int(vals.size)
    return float(np.median(vals)), int(vals.size)


def foot_point(box, kxy, kcf, frame_h):
    """발 위치와 근거.

    clipped=True 는 발 위치를 믿기 어렵다는 뜻이다. 박스 아랫변이 화면 끝에
    닿았다고 무조건 잘린 건 아니다 — 발목 키포인트가 화면 안에서 제대로 잡히면
    그 값은 쓸 수 있다. 그래서 실제로 값이 화면 끝에 붙은 경우만 잘림으로 본다.
    """
    x1, y1, x2, y2 = box
    good = [i for i in (L_ANKLE, R_ANKLE) if kcf[i] >= KP_CONF_MIN]
    if good:
        fx = float(np.mean([kxy[i][0] for i in good]))
        fy = float(np.mean([kxy[i][1] for i in good]))
        src = "ankle2" if len(good) == 2 else "ankle1"
    else:
        fx, fy, src = (x1 + x2) / 2.0, float(y2), "bbox"
    clipped = fy >= frame_h - 5
    return fx, fy, src, clipped


def kp_fields(kxy, kcf):
    """어깨·손목 좌표를 CSV 칸으로 만든다.

    사거리 반경(어깨~손목 최대 거리)과 정면 방향(어깨선의 수직)을 나중에
    여기서 구한다. 신뢰도는 좌우 중 낮은 쪽을 쓴다 — 한쪽만 보이면 방향을
    정할 수 없기 때문이다.
    """
    r = {}
    for tag, li, ri in (("sh", L_SHOULDER, R_SHOULDER), ("wr", L_WRIST, R_WRIST),
                        ("hip", L_HIP, R_HIP), ("ank", L_ANKLE, R_ANKLE)):
        pre = {"sh": ("lsh", "rsh"), "wr": ("lwr", "rwr"),
               "hip": ("lhip", "rhip"), "ank": ("lank", "rank")}[tag]
        for name, idx in zip(pre, (li, ri)):
            r[f"{name}_x"] = round(float(kxy[idx][0]), 1)
            r[f"{name}_y"] = round(float(kxy[idx][1]), 1)
        r[f"{tag}_conf"] = round(float(min(kcf[li], kcf[ri])), 3)
    return r


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-frames", type=int, default=0, help="0이면 전체")
    args = ap.parse_args()

    cfg = load_config(args.config)
    video = cfg["video"]
    ring = np.array(cfg["ring_polygon"], dtype=np.int32)

    cap = cv2.VideoCapture(video)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    model_path = args.model or os.path.expanduser(
        "~/.local/share/atom-experiments/2026-07-16/models/yolo26n-pose.pt"
    )
    model = YOLO(model_path)
    print(f"video : {video}\nframes: {n_total} @ {fps:.2f}fps\nmodel : {model_path}\n")

    # 어느 상의가 나인지. 설정에 없으면 light/dark 그대로 둔다.
    me_shirt = cfg.get("me_shirt")
    role_of = {}
    if me_shirt in ("light", "dark"):
        role_of = {me_shirt: "me", ("dark" if me_shirt == "light" else "light"): "opponent"}
        print(f"나 = {me_shirt} 상의\n")

    fields = [
        "frame", "time_sec", "fighter", "role", "det_conf",
        "x1", "y1", "x2", "y2", "box_h",
        "foot_x", "foot_y", "foot_src", "foot_clipped",
        # 어깨와 손목. 사거리 반경과 정면 방향을 여기서 구한다.
        "lsh_x", "lsh_y", "rsh_x", "rsh_y", "sh_conf",
        "lwr_x", "lwr_y", "rwr_x", "rwr_y", "wr_conf",
        # 엉덩이와 좌우 발목. 정면 방향을 어깨 하나에만 의존하지 않기 위해 같이 뽑는다.
        # 특히 발목은 바닥 평면 위의 점이라 깊이가 정확하게 나온다.
        "lhip_x", "lhip_y", "rhip_x", "rhip_y", "hip_conf",
        "lank_x", "lank_y", "rank_x", "rank_y", "ank_conf",
        "torso_L", "id_method", "box_iou",
    ]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=fields)
    writer.writeheader()

    prev = {}          # fighter -> (foot_x, foot_y)  직전 프레임 위치
    stats = {"frames": 0, "two": 0, "one": 0, "zero": 0, "by_color": 0,
             "by_color_abs": 0, "by_prev": 0, "rejected_jump": 0, "clipped": 0}

    stream = model.predict(
        source=video, stream=True, classes=[0], conf=args.conf,
        imgsz=args.imgsz, device=args.device, verbose=False,
    )

    for fi, res in enumerate(stream):
        if args.max_frames and fi >= args.max_frames:
            break
        stats["frames"] += 1

        boxes = res.boxes
        cands = []
        if boxes is not None and len(boxes) and res.keypoints is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            kxy = res.keypoints.xy.cpu().numpy()
            kcf = res.keypoints.conf.cpu().numpy()
            for k in range(len(xyxy)):
                fx, fy, src, clip = foot_point(xyxy[k], kxy[k], kcf[k], frame_h)
                # 발이 링 폴리곤 밖이면 링 밖 사람으로 보고 버린다
                if cv2.pointPolygonTest(ring, (float(fx), float(fy)), False) < 0:
                    continue
                cands.append({
                    "box": xyxy[k], "conf": float(confs[k]),
                    "kxy": kxy[k], "kcf": kcf[k],
                    "fx": fx, "fy": fy, "src": src, "clip": clip,
                    "h": float(xyxy[k][3] - xyxy[k][1]),
                })

        # 링 안에 3명 이상 잡히면 큰 사람 2명만 남긴다 (선수가 카메라에 더 가깝다)
        cands.sort(key=lambda c: -c["h"])
        cands = cands[:2]

        if len(cands) == 0:
            stats["zero"] += 1
            continue

        # --- 몸통 밝기 측정 (상대 박스 영역 제외) ---
        iou = box_iou(cands[0]["box"], cands[1]["box"]) if len(cands) == 2 else 0.0
        for i, c in enumerate(cands):
            other = cands[1 - i]["box"] if len(cands) == 2 else None
            c["L"], c["npx"] = torso_brightness(res.orig_img, c["box"], c["kxy"], c["kcf"], other)

        # --- 정체 배정 ---
        # 색을 최우선으로 쓴다. 위치로 이어붙이는 방식은 한 번 틀리면 계속 틀리기
        # 때문에, 색을 조금이라도 믿을 수 있으면 색이 이긴다.
        # 배정 방법은 선수별로 따로 기록한다. 프레임 하나에 대해 한 값만 적으면
        # "색으로 판정" 이라고 써놓고 실제로는 위치로 때운 선수가 섞여 들어간다.
        assign, how = {}, {}
        Ls = [c["L"] for c in cands]

        if len(cands) == 2 and all(v is not None for v in Ls) and abs(Ls[0] - Ls[1]) >= COLOR_GAP_MIN:
            # 두 명이 다 보이고 밝기 차가 뚜렷하다 -> 밝은 쪽이 흰 상의
            bright = 0 if Ls[0] > Ls[1] else 1
            assign = {"light": cands[bright], "dark": cands[1 - bright]}
            how = {"light": "color", "dark": "color"}
            stats["by_color"] += 2
        else:
            # 절대 밝기로 확실히 판정되는 것부터 먼저 배정한다
            taken = set()
            for j, L in enumerate(Ls):
                if L is None:
                    continue
                name = "light" if L > LIGHT_L_MIN else "dark" if L < DARK_L_MAX else None
                if name and name not in assign:
                    assign[name], how[name] = cands[j], "color_abs"
                    taken.add(j)
                    stats["by_color_abs"] += 1

            # 남은 후보는 직전 위치에서 가장 가까운 쪽으로 이어붙인다.
            # 단 사람이 갈 수 없는 거리면 이어붙이지 않고 버린다 — 그 프레임은
            # 그 선수를 못 찾은 것으로 남긴다. 없는 값을 지어내는 것보다 낫다.
            left = [j for j in range(len(cands)) if j not in taken]
            for name in ("light", "dark"):
                if name in assign or name not in prev or not left:
                    continue
                pf, px, py = prev[name]
                limit = jump_limit(max(1, fi - pf))
                best, bd = None, 1e18
                for j in left:
                    d = math.hypot(cands[j]["fx"] - px, cands[j]["fy"] - py)
                    if d < bd:
                        best, bd = j, d
                if best is None:
                    continue
                if bd > limit:
                    stats["rejected_jump"] += 1
                    continue
                assign[name], how[name] = cands[best], "prev"
                left.remove(best)
                stats["by_prev"] += 1

        # 커버리지는 후보 수가 아니라 실제로 배정된 수로 센다.
        # 순간이동 차단으로 버린 후보가 있으면 후보 수는 2여도 배정은 1이다.
        stats[("zero", "one", "two")[len(assign)]] += 1

        for name, c in assign.items():
            if c["clip"]:
                stats["clipped"] += 1
            writer.writerow({
                "frame": fi,
                "time_sec": round(fi / fps, 3),
                "fighter": name,
                "role": role_of.get(name, name),
                "det_conf": round(c["conf"], 4),
                "x1": round(float(c["box"][0]), 1), "y1": round(float(c["box"][1]), 1),
                "x2": round(float(c["box"][2]), 1), "y2": round(float(c["box"][3]), 1),
                "box_h": round(c["h"], 1),
                "foot_x": round(c["fx"], 1), "foot_y": round(c["fy"], 1),
                "foot_src": c["src"], "foot_clipped": int(c["clip"]),
                **kp_fields(c["kxy"], c["kcf"]),
                "torso_L": "" if c["L"] is None else round(c["L"], 1),
                "id_method": how[name],
                "box_iou": round(iou, 3),
            })
            prev[name] = (fi, c["fx"], c["fy"])

        if fi % 600 == 0:
            print(f"  frame {fi}/{n_total}")

    fout.close()

    n = max(stats["frames"], 1)
    rows_out = stats["by_color"] + stats["by_color_abs"] + stats["by_prev"]
    m = max(rows_out, 1)
    print(f"\n저장: {args.out}")
    print(f"\n[프레임 기준] 전체 {stats['frames']}")
    print(f"  두 명 다 찾음   : {stats['two']} ({100*stats['two']/n:.1f}%)")
    print(f"  한 명만 찾음    : {stats['one']} ({100*stats['one']/n:.1f}%)")
    print(f"  아무도 못 찾음  : {stats['zero']} ({100*stats['zero']/n:.1f}%)")
    print(f"\n[선수 기준] 전체 {rows_out}건")
    print(f"  색 비교로 구분     : {stats['by_color']} ({100*stats['by_color']/m:.1f}%)")
    print(f"  색 절대값으로 구분 : {stats['by_color_abs']} ({100*stats['by_color_abs']/m:.1f}%)")
    print(f"  직전 위치로 이어붙임: {stats['by_prev']} ({100*stats['by_prev']/m:.1f}%)")
    print(f"  발이 화면 밖(잘림) : {stats['clipped']} ({100*stats['clipped']/m:.1f}%)")
    print(f"\n  순간이동이라 거부한 후보: {stats['rejected_jump']}건")
    print("  (다른 사람을 선수로 잘못 이어붙이는 것을 막은 횟수. 그 프레임은 빈칸으로 남는다)")


if __name__ == "__main__":
    main()
