# atom — 복싱 스파링 영상 데이터화

내 스파링 영상을 넣으면 그 안에서 벌어진 일이 숫자로 나오게 만든다.
목표와 범위는 [docs/scope.md](docs/scope.md) 참고.

지금 단계: **1단계 — 두 선수의 위치와 거리 뽑기**

## 환경

예전 실험 환경을 그대로 쓴다. 새로 설치할 게 없다.

```
~/.local/share/atom-experiments/2026-07-16/venv/bin/python
```

Python 3.10 / ultralytics 8.4 / torch 2.13 / opencv 5.0.
모델도 같은 폴더의 `models/` 에 있다 (`yolo26n-pose.pt` 등).

편하게 쓰려면:

```bash
export ATOM_PY=~/.local/share/atom-experiments/2026-07-16/venv/bin/python
```

## 실행 순서

### 1. 위치 뽑기

```bash
$ATOM_PY scripts/track_boxers.py --config configs/IMG_0711.json --out out/IMG_0711.tracks.csv
```

프레임마다 두 선수의 발 위치를 CSV로 뽑는다. 2분 영상에 몇 분 걸린다.

### 2. 눈으로 확인할 영상 만들기

```bash
$ATOM_PY scripts/make_overlay.py --config configs/IMG_0711.json \
    --csv out/IMG_0711.tracks.csv --out out/overlay.mp4 --start 40 --duration 30
```

원본 위에 위치 점과 거리를 그린다. **이 영상을 직접 보는 것이 이 단계의 검증 방법이다.**
빨간 테두리가 뜬 순간은 신뢰도가 낮은 프레임이다.

### 3. 품질 재기

```bash
$ATOM_PY scripts/report_quality.py --csv out/IMG_0711.tracks.csv --out out/quality.md
```

두 명을 다 잡은 비율, 문제 구간의 시간대를 뽑는다.

### 4. 미터로 바꾸는 보정값 구하기

```bash
$ATOM_PY scripts/calibrate_ground.py --csv out/IMG_0711.tracks.csv \
    --person-height 1.73 --out configs/IMG_0711.ground.json
```

화면 픽셀을 실제 미터로 바꾸는 보정값을 구한다. 선수 키가 필요하다.
링 한 변 실측값이 있으면 `--ring-side 5.0` 을 붙여 정확도를 높일 수 있다.

### 5. 최종 산출물 만들기

```bash
$ATOM_PY scripts/make_positions.py --tracks out/IMG_0711.tracks.csv \
    --ground configs/IMG_0711.ground.json --out out/IMG_0711.positions.csv
```

프레임당 두 사람의 바닥 위치(m)와 거리(m), 신뢰도를 담은 CSV. **이게 1단계 결과물이다.**

```
frame,time_sec,me_x,me_z,op_x,op_z,distance_m,confidence
0,0.0,1.079,2.622,-0.516,3.683,1.916,0.6
```

거리를 미터로 표시한 확인 영상을 원하면 3번에 `--positions out/IMG_0711.positions.csv` 를 붙인다.

## 파일 구조

```
configs/    영상별 설정 (링 영역, 실측값, 보정값)
scripts/    실행 스크립트
  ranges.py   거리대 구간 정의 + 한글 글자 그리기 (여러 스크립트가 공유)
out/        결과물 (CSV, 리포트, 확인용 영상) — git에 넣지 않는다
docs/       스코프와 설계 문서
```

## 거리대 구간

| 구간 | 경계 |
|---|---|
| 밀착 (클린치) | ~0.65m |
| 근거리 (훅·어퍼 거리) | 0.65~1.3m |
| 중거리 (잽·스트레이트 거리) | 1.3~1.9m |
| 원거리 (사거리 밖) | 1.9m~ |

`scripts/ranges.py` 한 곳에만 정의한다. 리포트와 확인 영상이 서로 다른 기준을 쓰면
영상을 보며 검증하는 의미가 없어지기 때문이다. 바꾸려면 그 파일만 고치면 된다.

경계값은 체격과 리치에 따라 다르다. 계산으로 정할 수 없으므로 실제 장면을 보고 정한다.

```bash
$ATOM_PY scripts/make_distance_sheet.py --positions out/IMG_0711.positions.csv \
    --config configs/IMG_0711.json --out out/distance_scale.jpg
```

거리별 실제 장면을 격자로 붙인 이미지가 나온다. 그걸 보고 `ranges.py` 를 고치면 된다.

## 설계상 알아둘 것

**YOLO 추적 ID를 쓰지 않는다.** 실제 영상에서 두 선수 모두 트랙이 중간에 끊겨
새 ID를 받는 것을 확인했다. 대신 **상의 밝기**로 구분한다 (한 명 검정, 한 명 흰색).
겹쳐서 색을 못 볼 때만 직전 위치로 이어붙이고, 그 프레임은 표시해둔다.

**링 안쪽만 본다.** 체육관이라 링 밖에서 다른 회원들이 운동 중이다.
발 위치가 `configs/*.json` 의 `ring_polygon` 안에 있는 사람만 선수 후보로 본다.
실제로 링 옆에 서 있던 관원이 선수로 잡힌 사고가 있었다 (`docs/step1-results.md`).

**없는 값을 지어내지 않는다.** 사람이 1/60초에 갈 수 없는 거리면 이어붙이지 않고
그 프레임은 빈칸으로 남긴다. 커버리지 숫자가 조금 낮아져도 그게 맞다.

**정답 라벨을 만들지 않는다.** 내 눈이 정답지다. 확인용 영상을 보고 판단한다.
