# Audio Immersion Design: "체육관에 있는 느낌"

> Status: DRAFT (v2)
> Created: 2026-03-23
> Last Updated: 2026-03-23

## 핵심 원칙

이 앱의 오디오는 **정보 전달이 아니라 공간 창조**다.
사용자가 거실에서 훈련하더라도, 귀에 들리는 건 체육관이어야 한다.

---

## 1. 오디오 레이어 아키텍처

클라이언트가 4개의 독립적인 오디오 레이어를 동시에 재생한다.
각 레이어는 독립적으로 실패할 수 있으며, **L4(코치 음성)만 필수**다.
나머지는 부가 레이어 — 없어도 세션은 진행된다.

```
Layer 4: Coach Voice     ████░░░░████░░░░████░░    (세그먼트별 재생, 필수)
Layer 3: Impact SFX      ░░██░░░░░░██░░░░░░██░░    (실행 구간 내 리듬 가이드)
Layer 2: Structure       █░░░░░░░░░░░░░░░░░░░░█    (벨, 타이머)
Layer 1: Ambient         ████████████████████████   (연속 루프)
──────────────────────────────────────────────────
                    Time →
```

| 레이어 | 역할 | 재생 방식 | 기본 볼륨 | 필수 |
|--------|------|-----------|-----------|------|
| L1: Ambient | 공간감 — "여기는 체육관" | 연속 루프 | 15-25% | X |
| L2: Structure | 시간 구조 — "라운드 시작/끝" | 이벤트 트리거 | 80-100% | X |
| L3: Impact SFX | 리듬 가이드 — "이 타이밍에 쳐라" | 실행 구간 내 | 50-70% | X |
| L4: Coach Voice | 지시 — "뭘 하라" | 세그먼트별 (기존 chunk 시스템) | 100% | O |

---

## 2. Layer 1: Ambient (공간)

사용자가 이어폰을 끼는 순간, 거실이 사라지고 체육관이 시작된다.

### 에셋

| 파일 | 내용 | 길이 | 용도 |
|------|------|------|------|
| `ambient_gym_low.mp3` | 먼 거리 샌드백 소리, 약한 웅성거림, 환풍기 | 60s loop | R1 (적응) |
| `ambient_gym_mid.mp3` | 가까운 샌드백, 줄넘기, 발소리 | 60s loop | R2 (적용) |
| `ambient_gym_high.mp3` | 활발한 체육관, 다수 훈련 소리, 에너지 | 60s loop | R3 (몰입) |
| `ambient_gym_finisher.mp3` | 최대 에너지, 환호/응원 섞임 | 60s loop | Finisher |

### 재생 규칙

```
세션 시작 → ambient_gym_low 페이드인 (2초)
R1 → ambient_gym_low (vol 15%)
R1 종료 → 볼륨 10%로 낮춤 (공간감 유지, 앰비언트 트랙은 유지)
R2 시작 → crossfade to ambient_gym_mid (vol 20%, 전환 1초)
R2 종료 → 볼륨 10%로 낮춤
R3 시작 → crossfade to ambient_gym_high (vol 25%)
R3 종료 → 볼륨 10%로 낮춤
Finisher 시작 → crossfade to ambient_gym_finisher (vol 25%)
세션 종료 → 페이드아웃 (3초)
```

앰비언트 트랙 전환은 **라운드 시작 시점**에 발생한다 (휴식 중 아님).
휴식 구간에서는 트랙을 유지하고 볼륨만 낮춘다.

### 왜 라운드별로 다른 앰비언트인가

R1은 조용한 아침 체육관. R3은 사람 많은 저녁 체육관.
소리만으로 에너지가 올라간다. 사용자가 의식하지 못해도 몸이 반응한다.

---

## 3. Layer 2: Structure (벨, 타이머)

복싱에서 벨은 파블로프의 종이다. 벨이 울리면 몸이 움직인다.

### 에셋

| 파일 | 내용 | 용도 |
|------|------|------|
| `bell_round_start.mp3` | 복싱 벨 3타 (딩딩딩) | 라운드 시작 |
| `bell_round_end.mp3` | 복싱 벨 1타 (딩) | 라운드 종료 |
| `bell_session_end.mp3` | 복싱 벨 연타 (딩딩딩딩딩) | 세션 완전 종료 |
| `beep_10sec.mp3` | 짧은 비프음 | 라운드 종료 10초 전 |
| `beep_countdown.mp3` | 3-2-1 카운트다운 비프 | 휴식 끝, 다음 라운드 준비 |
| `whistle_finisher.mp3` | 짧은 호각 | 피니셔 구간 시작 |

### 타이밍

```
[Intro 20초]
  0s    bell_round_start        ← 세션 시작을 알림
  0.5s  Coach: "1라운드 시작합니다"

[Round 1 - 120초]
  110s  beep_10sec              ← "10초 남았다" 경고
  120s  bell_round_end          ← 라운드 종료

[Rest 20초]
  0s    Coach: "좋아, 쉬어"
  17s   beep_countdown (3-2-1)  ← 다음 라운드 준비
  20s   bell_round_start        ← 라운드 시작

[... Round 2, 3 ...]

[Finisher 60초]
  0s    whistle_finisher        ← 피니셔 돌입
  0.3s  Coach: "마지막이야!"
  50s   beep_10sec
  60s   bell_session_end        ← 세션 완전 종료
  60.5s Coach: "끝! 수고했어"
```

### 핵심

벨 소리는 **절대 생략하지 않는다.** 이건 기능이 아니라 의식(ritual)이다.
사용자가 벨 소리를 듣는 순간 "시작"이라는 조건반사가 만들어진다.

---

## 4. Layer 3: Impact SFX (패드워크 리듬)

### 무엇인가

임팩트 SFX는 **사용자의 실제 펀치 소리가 아니다.**
코치가 미트(패드)를 들고 "여기에, 이 타이밍에 쳐라"고 보여주는 **리듬 가이드**다.

체육관 패드워크를 떠올리면 된다:
1. 코치가 "원투 훅!" 을 외친다
2. 코치가 미트를 내민다 — 잽 위치, 크로스 위치, 훅 위치 순서대로
3. 선수가 미트 타이밍에 맞춰 친다
4. "착착착" 소리가 난다

임팩트 SFX가 바로 이 3번의 미트 타격음이다.
**실시간 감지 없이도 "때리는 리듬"을 제공**하는 것이 핵심 가치.

### 에셋

| 파일 | 매핑 액션 | 소리 특성 | 변형 수 |
|------|-----------|-----------|---------|
| `impact_jab_N.mp3` | J (잽) | 가볍고 빠른 스냅 | 3 |
| `impact_cross_N.mp3` | C (투/크로스) | 묵직한 타격 | 3 |
| `impact_hook_N.mp3` | LH, RH (훅) | 깊고 둔탁한 충격 | 3 |
| `impact_uppercut_N.mp3` | LU, RU (어퍼컷) | 상승 임팩트 | 3 |
| `impact_body_N.mp3` | LB, RB (바디) | 낮고 둔한 바디 타격 | 3 |
| `bag_chain_rattle.mp3` | (자동) | 강한 타격 후 체인 흔들림 | 2 |

방어 액션:

| 파일 | 매핑 액션 | 소리 특성 |
|------|-----------|-----------|
| `whoosh_slip.mp3` | SL (슥) | 공기 가르는 소리 |
| `whoosh_duck.mp3` | DK (덕킹) | 빠른 하강 바람 |
| `whoosh_weave.mp3` | WV (위빙) | 좌우 바람 |
| `step_back.mp3` | BS (백) | 발 스텝 소리 |
| `step_side.mp3` | SS (사이드스텝) | 발 스텝 소리 |

**총: 22개 에셋** (임팩트 15 + 체인 래틀 2 + 방어 5)

### 핵심 규칙: 임팩트는 실행 구간 안에서 재생된다

임팩트 SFX는 세그먼트에 시간을 **추가하지 않는다.**
기존 시스템의 "코치 콜 → pause → 다음 세그먼트"에서,
pause(실행 구간) 안에 임팩트가 리듬 마커로 들어간다.

```
기존 (임팩트 없이):
[Coach: "원투 훅"] ──── pause (사용자가 펀치) ──── [다음 세그먼트]
                       ↑ 침묵

임팩트 추가 후:
[Coach: "원투 훅"] ── [착] [착] [쿵] ── gap ── [다음 세그먼트]
                      ↑ 실행 구간 내 리듬 마커
```

**사용자는 임팩트 소리에 맞춰 펀치를 던진다.**
침묵이었던 실행 구간에 리듬이 생기는 것이지, 세그먼트가 길어지는 게 아니다.

### 타이밍 계산

```
실행 구간 = max(기존 pause, 임팩트 시퀀스 길이 + 200ms)

임팩트 시퀀스 길이 = (액션 수 × ~200ms) + ((액션 수-1) × 250ms)

예시: "원투 훅" (seq: [J, C, LH])
  = 3 × 200ms + 2 × 250ms = 1100ms

기존 pause (chunk_duration + 300ms) = ~1300ms
→ 1300ms > 1100ms → 기존 pause 유지. 시간 추가 없음.

예시: "원투 양훅 슥 투" (seq: [J, C, LH, RH, SL, C])
  = 6 × 200ms + 5 × 250ms = 2450ms

기존 pause = ~1300ms
→ 2450ms > 1300ms → 실행 구간을 2650ms로 확장 (유일하게 시간이 늘어나는 케이스)
```

긴 콤보(6+ 액션)에서만 약간의 시간 추가가 발생한다.
beginner/intermediate의 대부분(2-4액션)은 기존 pause 안에 수용된다.

### 멀티 chunk 콤보의 재생 흐름

chunk가 여러 개인 콤보("원투 슥 투훅투")는 chunk별로 인터리브한다.
코치가 chunk를 부르고, 해당 chunk의 액션만 임팩트로 재생된다.

```
콤보: "원투 슥 투훅투"
chunks: ["원투", "슥", "투훅투"]
seq:    ["J", "C",  "SL",  "C", "LH", "C"]

재생:
[Coach: "원투"]
  → 300ms → [impact_jab] → 250ms → [impact_cross] → 200ms gap
[Coach: "슥"]
  → 300ms → [whoosh_slip] → 200ms gap
[Coach: "투훅투"]
  → 300ms → [impact_cross] → 250ms → [impact_hook] → 250ms → [impact_cross]
  → 300ms gap
[다음 세그먼트]
```

**매핑 규칙:** 각 chunk의 텍스트를 combo_dictionary에서 찾아
해당 chunk에 속하는 seq 액션만 임팩트로 재생한다.
combo_assembly 매핑(session-template-spec Section 2.4)이 이미
"원투 슥 투훅투" → ["원투", "슥", "투훅투"]로 분해하므로,
각 chunk 단위로 seq를 분할하면 된다.

### 변형(Variant) 선택

같은 잽이라도 매번 같은 소리면 기계적이다.
3개 변형 중 랜덤으로 선택하되, 직전과 동일한 변형은 피한다.

```typescript
function pickImpactVariant(action: string, lastVariant: number): string {
  const variants = [1, 2, 3].filter(v => v !== lastVariant);
  const pick = variants[Math.floor(Math.random() * variants.length)];
  return `impact_${ACTION_MAP[action]}_${pick}.mp3`;
}
```

### 체인 래틀 규칙

세그먼트의 **마지막 액션이 훅 또는 어퍼컷**(LH, RH, LU, RU)일 때만
`bag_chain_rattle` 추가.

잽(J), 크로스(C), 바디(LB, RB)에서는 생략.
크로스로 끝나는 콤보가 매우 많아서("원투", "잽잽 투" 등),
크로스를 포함하면 체인 래틀이 세그먼트의 70%+에서 울려 특별함이 사라진다.
훅/어퍼컷은 스윙이 큰 파워샷이므로 체인이 흔들리는 게 자연스럽다.

---

## 5. Layer 4: Coach Voice (기존 시스템 확장)

기존 chunk 시스템은 유지하되, **감정 에너지**를 추가한다.

### 현재 (session-template-spec)

```
27 chunks + 9 cues + 12 round intros = ~50 recordings
```

### 추가 녹음: 감정/구조 클립

| 카테고리 | 예시 | 용도 | 수량 |
|----------|------|------|------|
| 라운드 인트로 | "자 가자!", "집중!", "준비됐지?" | 라운드 시작 직후 | 6 |
| 라운드 종료 격려 | "좋아!", "잘했어!", "한 라운드 끝!" | 라운드 종료 직후 | 6 |
| 푸시 | "더!", "멈추지 마!", "쏟아내!" | R3, Finisher 시작부 | 6 |
| 휴식 | "쉬어", "호흡 가다듬어", "물 마셔" | 휴식 구간 | 4 |
| 종료 | "끝!", "수고했어!", "오늘도 해냈다!" | 세션 종료 | 4 |
| 카운트다운 | "마지막 10초!", "5초!", "3, 2, 1!" | 라운드 종료 전 | 4 |

**추가 녹음: ~30개 클립**

### 격려 클립 배치: 라운드 경계에 집중

격려 클립을 세그먼트 사이에 랜덤 삽입하면 위험하다.
사용자가 힘들어서 대충 했는데 "좋아!"가 나오면 오히려 가짜 코치 느낌이 난다.
실시간 감지가 없으므로 수행 품질을 알 수 없기 때문이다.

**안전한 배치: 라운드를 완료한 것 자체는 사실이므로, 경계 시점에 배치한다.**

```
라운드 시작:    "자 가자!" / "집중!" / "준비됐지?"    ← 항상 사실
라운드 종료:    "잘했어!" / "한 라운드 끝!"           ← 라운드를 끝낸 건 사실
R3 시작:       "더!" / "멈추지 마!"                  ← 푸시, 마지막이니까
Finisher 시작: "쏟아내!" / "마지막이야!"              ← 푸시
세션 종료:      "수고했어!" / "오늘도 해냈다!"         ← 완료한 건 사실
```

세그먼트 사이에는 기존 cue("가드!", "턱 당겨!", "스텝!")만 유지.
이것들은 **기술 리마인더**이지 칭찬이 아니므로, 언제 나와도 부자연스럽지 않다.

### 코치 에너지 아크

같은 "원투 훅"이라도 R1에서와 R3에서 느낌이 달라야 한다.
하지만 chunk를 라운드별로 다르게 녹음하는 건 비현실적이다.

**에너지는 다른 레이어들이 만든다:**

```
R1 (적응):
  앰비언트: low (15%), 임팩트 볼륨: 50%
  라운드 시작 클립: "집중" (차분)

R2 (적용):
  앰비언트: mid (20%), 임팩트 볼륨: 60%
  라운드 시작 클립: "자 가자!" (에너지 상승)

R3 (몰입):
  앰비언트: high (25%), 임팩트 볼륨: 70%
  라운드 시작 클립: "더!" (푸시)

Finisher (폭발):
  앰비언트: finisher (25%), 임팩트 볼륨: 80%
  시작 클립: "마지막이야! 쏟아내!" (절정)
```

코치 음성 chunk 자체는 변하지 않지만,
**주변 레이어(앰비언트 볼륨 + 임팩트 볼륨 + 시작/종료 클립)가 에너지를 만든다.**

---

## 6. 타임라인 시뮬레이션

10분 세션의 한 라운드(R2, 120초)를 시뮬레이션한다.

```
0:00  [L2] bell_round_start (딩딩딩)
0:01  [L1] crossfade → ambient_gym_mid (vol 20%)
0:02  [L4] Coach: "자 가자!"         ← 라운드 인트로 클립

0:04  [L4] Coach: "원투"
      [L3] → 300ms → jab_1 → 250ms → cross_2
0:07  -- gap --

0:09  [L4] Coach: "원투 훅"
      [L3] → 300ms → jab_3 → 250ms → cross_1 → 250ms → hook_2 → chain_rattle
0:13  -- gap --

0:15  [L4] Coach: "가드!"            ← 큐 (임팩트 없음, 기술 리마인더)
0:16  -- gap --

0:18  [L4] Coach: "잽잽 투"
      [L3] → 300ms → jab_2 → 200ms → jab_1 → 250ms → cross_3
0:22  -- gap --

0:24  [L4] Coach: "원투 덕킹 투"
      [L4] chunk: "원투"
      [L3] → jab_1 → cross_2
      [L4] chunk: "덕킹"
      [L3] → whoosh_duck
      [L4] chunk: "투"
      [L3] → cross_3
0:29  -- gap --

      ... (세그먼트 계속) ...

1:50  [L2] beep_10sec               ← 10초 경고
      [L4] Coach: "마지막 10초!"

1:53  [L4] Coach: "원투 훅 투"      ← 마지막 콤보
      [L3] → jab → cross → hook → cross → chain_rattle

1:58  [L4] Coach: "3, 2, 1!"
2:00  [L2] bell_round_end (딩)
      [L4] Coach: "잘했어!"          ← 라운드 종료 격려 (사실 기반)
      [L1] ambient vol → 10%
```

---

## 7. 기술 구현

### 7.1 에셋 구조

```
data/audio/
  chunks/           ← 기존 코치 음성 chunk (현행 유지)
    원투_1.mp3
    원투_2.mp3
    ...
  sfx/              ← NEW
    ambient/
      gym_low.mp3
      gym_mid.mp3
      gym_high.mp3
      gym_finisher.mp3
    bell/
      round_start.mp3
      round_end.mp3
      session_end.mp3
    timer/
      beep_10sec.mp3
      beep_countdown.mp3
      whistle_finisher.mp3
    impact/
      jab_1.mp3, jab_2.mp3, jab_3.mp3
      cross_1.mp3, cross_2.mp3, cross_3.mp3
      hook_1.mp3, hook_2.mp3, hook_3.mp3
      uppercut_1.mp3, uppercut_2.mp3, uppercut_3.mp3
      body_1.mp3, body_2.mp3, body_3.mp3
      bag_chain_1.mp3, bag_chain_2.mp3
    defense/
      whoosh_slip.mp3
      whoosh_duck.mp3
      whoosh_weave.mp3
      step_back.mp3
      step_side.mp3
  coach/             ← NEW (감정/구조 클립)
    round_intro/
      자가자.mp3, 집중.mp3, 준비됐지.mp3, ...
    round_end/
      좋아.mp3, 잘했어.mp3, 한라운드끝.mp3, ...
    push/
      더.mp3, 멈추지마.mp3, 쏟아내.mp3, ...
    rest/
      쉬어.mp3, 호흡.mp3, 물마셔.mp3, ...
    ending/
      끝.mp3, 수고했어.mp3, 오늘도해냈다.mp3, ...
    countdown/
      마지막10초.mp3, 5초.mp3, 321.mp3, ...
```

### 7.2 클라이언트 오디오 엔진

```
┌──────────────────────────────────────────────────────┐
│                AudioSessionEngine                     │
│                                                       │
│  ┌──────────┐  독립 재생, 개별 볼륨 제어               │
│  │ Ambient  │  expo-av Audio instance #1               │
│  │ Player   │  → loop=true, vol=user_sfx_vol * 0.2    │
│  └──────────┘                                          │
│                                                        │
│  ┌──────────┐                                          │
│  │ SFX      │  expo-av Audio instance #2               │
│  │ Player   │  → 이벤트 트리거 (벨, 비프)               │
│  └──────────┘                                          │
│                                                        │
│  ┌──────────┐                                          │
│  │ Impact   │  expo-av Audio instance #3               │
│  │ Player   │  → chunk별 인터리브 재생                  │
│  └──────────┘                                          │
│                                                        │
│  ┌──────────┐                                          │
│  │ Coach    │  기존 chunk playback 시스템               │
│  │ Player   │  → 세그먼트별 chunk 재생                  │
│  └──────────┘                                          │
│                                                        │
│  ┌───────────────────────────────────────────────┐     │
│  │ Sequencer (세션 타임라인 관리)                   │     │
│  │                                               │     │
│  │ onRoundStart(roundNum) {                      │     │
│  │   sfxPlayer.play('bell_round_start')          │     │
│  │   ambientPlayer.crossfade(roundLevel)         │     │
│  │   coachPlayer.play(roundIntroClip)            │     │
│  │   startSegmentLoop(segments)                  │     │
│  │ }                                             │     │
│  │                                               │     │
│  │ onChunkPlay(chunk, chunkActions) {            │     │
│  │   coachPlayer.play(chunk)                     │     │
│  │   → onChunkEnd:                               │     │
│  │     impactPlayer.playSequence(chunkActions)   │     │
│  │ }                                             │     │
│  │                                               │     │
│  │ onRoundEnd() {                                │     │
│  │   sfxPlayer.play('bell_round_end')            │     │
│  │   coachPlayer.play(roundEndClip)              │     │
│  │   ambientPlayer.setVolume(0.1)                │     │
│  │ }                                             │     │
│  │                                               │     │
│  │ // --- 일시정지/재개 ---                        │     │
│  │ onPause() {                                   │     │
│  │   ambientPlayer.pause()                       │     │
│  │   coachPlayer.pause()                         │     │
│  │   impactPlayer.cancelPending()                │     │
│  │   sfxPlayer.pause()                           │     │
│  │   sequencer.pauseTimer()                      │     │
│  │ }                                             │     │
│  │                                               │     │
│  │ onResume() {                                  │     │
│  │   sequencer.resumeTimer()                     │     │
│  │   ambientPlayer.resume()                      │     │
│  │   // Coach/Impact: 현재 세그먼트 처음부터 재시작 │     │
│  │   // (중간 재개하면 리듬이 어긋남)               │     │
│  │   restartCurrentSegment()                     │     │
│  │ }                                             │     │
│  └───────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────┘
```

### 7.3 세그먼트 재생 시퀀스 (상세)

하나의 세그먼트가 재생되는 전체 흐름.
**chunk별로 인터리브** — 각 chunk 재생 후, 그 chunk에 해당하는 임팩트가 실행 구간 내에서 재생된다.

```
예시: "원투 슥 투훅투" (chunks: ["원투", "슥", "투훅투"])

1. Chunk "원투" 재생 (~0.5s)
   → 300ms gap
   → impact_jab (~0.2s) → 250ms → impact_cross (~0.2s)
   → 200ms gap

2. Chunk "슥" 재생 (~0.3s)
   → 300ms gap
   → whoosh_slip (~0.3s)
   → 200ms gap

3. Chunk "투훅투" 재생 (~0.6s)
   → 300ms gap
   → impact_cross → 250ms → impact_hook → 250ms → impact_cross
   → 300ms gap (세그먼트 종료)

4. 다음 세그먼트로 이동
```

단일 chunk 콤보(대부분):
```
예시: "원투 훅" (chunks: ["원투 훅"], seq: [J, C, LH])

1. Chunk "원투 훅" 재생 (~0.7s)
   → 300ms gap
   → impact_jab → 250ms → impact_cross → 250ms → impact_hook
   → chain_rattle (마지막이 훅이므로)
   → 300ms gap

2. 다음 세그먼트로 이동
```

### 7.4 API 확장

기존 세그먼트 응답에 `impact_actions` 추가:

```python
class SegmentResponse(BaseModel):
    text: str                       # "원투 훅"
    chunks: list[ChunkResponse]     # 코치 음성 chunks (기존)
    impact_actions: list[str]       # ["J", "C", "LH"] (NEW)
    # combo_dictionary.json의 seq에서 자동 추출
    # cue 세그먼트("가드!" 등)는 빈 배열
```

chunk별 액션 분할은 클라이언트에서 수행:
combo_assembly 매핑으로 chunk 경계를 알고 있으므로,
seq 배열을 chunk 단위로 분할할 수 있다.

SFX 에셋은 앱 번들에 포함 (서버에서 다운로드 불필요).

### 7.5 에셋 소싱

| 카테고리 | 방법 | 비용 |
|----------|------|------|
| 코치 음성 (chunks) | 자체 녹음 (기존) | $0 |
| 코치 감정 클립 | 자체 녹음 (추가 ~30개) | $0 |
| 벨, 비프, 호각 | 무료 SFX 라이브러리 (Freesound, Pixabay) | $0 |
| 임팩트 사운드 | 무료 SFX 라이브러리 + 가공 | $0 |
| 방어 효과음 | 무료 SFX 라이브러리 | $0 |
| 앰비언트 | 무료 + 직접 믹싱 (여러 소스 레이어링) | $0 |

**에셋 사양:**
- 포맷: MP3 (128kbps) or AAC
- 샘플레이트: 44100Hz
- 채널: Mono (SFX, Impact), Stereo (Ambient)
- 총 앱 번들 크기 추가: ~5MB

```
크기 산출:
  앰비언트 4개 × 60s × 128kbps ≈ 3.8MB
  임팩트+방어 22개 × ~0.3s      ≈ 0.1MB
  벨/타이머 6개 × ~2s            ≈ 0.2MB
  코치 감정 30개 × ~1s           ≈ 0.5MB
  합계: ~4.6MB → 여유 포함 ~5MB
```

---

## 8. 감정 곡선 (Emotional Arc)

전체 세션을 하나의 감정 여정으로 설계한다.

```
에너지
  ▲
  │                                           ████
  │                                          █    █
  │                                ████     █      █
  │                               █    █   █        █
  │                    ████      █      █ █          █
  │                   █    █   █        █            █
  │        ████      █      █ █                      █
  │       █    █   █        █                        █
  │      █      █ █                                  █
  │     █        █                                    █
  │    █                                               █
  │   █                                                 ███
  └──────────────────────────────────────────────────────────▶ Time
     Intro   R1      Rest    R2      Rest    R3    Finisher  End

     긴장    적응    회복    빌드업   준비    몰입   폭발     성취
```

| 구간 | 앰비언트 | 임팩트 볼륨 | 코치 경계 클립 | 느낌 |
|------|----------|-------------|----------------|------|
| Intro | low (15%) | - | - | "자, 시작하자" |
| R1 | low (15%) | 50% | 시작: "집중" / 종료: "좋아!" | "천천히, 정확하게" |
| Rest 1 | low (10%) | - | "쉬어", "호흡 가다듬어" | 회복 |
| R2 | mid (20%) | 60% | 시작: "자 가자!" / 종료: "잘했어!" | "리듬 타자" |
| Rest 2 | mid (10%) | - | "다음 라운드 준비" | 빌드업 |
| R3 | high (25%) | 70% | 시작: "더!" / 종료: "한 라운드 끝!" | "지금이야, 몰입!" |
| Finisher | finisher (25%) | 80% | 시작: "마지막이야! 쏟아내!" | "다 쏟아!" |
| End | fade out | - | "수고했어!", "오늘도 해냈다!" | "해냈다" |

---

## 9. 사용자 설정

### SFX 볼륨 슬라이더

사용자마다 선호가 다르다. 4개 레이어를 개별 제어하는 건 과하지만,
**코치 음성 vs 나머지**로 2그룹 제어는 필요하다.

```
설정 화면:
┌──────────────────────────────┐
│ 코치 음성 볼륨   ████████░░ 80% │  ← L4 (Coach Voice)
│ 효과음 볼륨     ██████░░░░ 60% │  ← L1 + L2 + L3 (Ambient + Structure + Impact)
│ 효과음 ON/OFF   [■ ON]         │  ← 전체 SFX 끄기 (코치만 남김)
└──────────────────────────────┘
```

효과음 OFF 시: 기존 chunk-only 재생으로 fallback. 기능적으로 완전 동일.

---

## 10. 에러 처리 및 Fallback

각 레이어는 독립적으로 실패할 수 있다. **세션은 절대 중단되지 않는다.**

| 상황 | 동작 | 사용자 영향 |
|------|------|-------------|
| 앰비언트 파일 로드 실패 | 해당 레이어 비활성화, 로그 기록 | 배경음 없이 진행 (기능 정상) |
| 벨/타이머 재생 실패 | 해당 이벤트 건너뜀, 로그 기록 | 벨 없이 진행 (코치 음성이 구조 안내) |
| 임팩트 SFX 파일 누락 | 해당 액션 임팩트 건너뜀 | 일부 펀치 소리 없이 진행 |
| 임팩트 Player 초기화 실패 | 전체 Impact 레이어 비활성화 | 코치 음성만으로 진행 (기존 동작) |
| expo-av 인스턴스 부족 (저사양) | Ambient 레이어 먼저 포기 → Impact 포기 | 벨 + 코치만 남음 |
| 코치 음성 chunk 실패 | **세션 중단 가능** — 유일한 필수 레이어 | 기존 fallback 규칙 따름 |

**우선순위 (메모리 부족 시 해제 순서):**
1. Ambient (가장 먼저 포기)
2. Impact SFX
3. Structure (벨/타이머)
4. Coach Voice (절대 포기 안 함)

---

## 11. 에셋 수량 요약

| 카테고리 | 수량 | 녹음/소싱 |
|----------|------|-----------|
| 코치 chunks (기존) | ~50 | 자체 녹음 (완료/진행중) |
| 코치 감정/구조 클립 | ~30 | 자체 녹음 (추가) |
| 벨/타이머/호각 | 6 | 무료 SFX |
| 임팩트 (5종 x 3변형) | 15 | 무료 SFX |
| 체인 래틀 | 2 | 무료 SFX |
| 방어 효과음 | 5 | 무료 SFX |
| 앰비언트 루프 (4종) | 4 | 무료 + 믹싱 |
| **총 추가 에셋** | **~62** | **~5MB** |

---

## 12. 구현 우선순위

기존 chunk 시스템에 점진적으로 레이어를 추가한다.

### Phase 1: 벨 + 앰비언트 (최소 몰입)
- 라운드 시작/종료 벨 추가
- 앰비언트 루프 1개 (전 라운드 공통, Phase 3에서 4종으로 확장)
- 10초 경고 비프
- SFX 볼륨 설정 + ON/OFF 토글
- **효과: 이것만으로 "체육관 느낌"의 60%**
- **에셋: 벨 3개 + 비프 2개 + 호각 1개 + 앰비언트 1개 = 7개**

### Phase 2: 임팩트 SFX
- 5종 펀치 임팩트 + 방어 효과음
- combo_dictionary.json의 seq 기반 자동 매핑
- chunk별 인터리브 재생 로직
- 체인 래틀 규칙 (훅/어퍼컷만)
- API에 `impact_actions` 필드 추가
- **효과: "리듬 가이드" — 패드워크 느낌 추가**
- **에셋: 임팩트 15 + 체인 2 + 방어 5 = 22개**

### Phase 3: 감정 코칭 + 에너지 아크
- 라운드 인트로/종료 클립 녹음 (~30개)
- 라운드별 앰비언트 4종으로 확장
- 라운드별 임팩트 볼륨 스케일링
- **효과: 감정 곡선 완성 — 적응 → 몰입 → 폭발 → 성취**

---

## 13. 핵심 제약

- **레이턴시:** 임팩트 SFX는 chunk 종료 후 300ms 이내 재생 시작 (지연되면 리듬 어긋남)
- **메모리:** expo-av 인스턴스 4개 동시 운영 → 저사양 디바이스에서 Section 10 해제 순서 적용
- **볼륨 밸런스:** 코치 음성이 항상 최우선. 임팩트/앰비언트가 코치를 묻으면 안 됨
- **SFX는 앱 번들 포함:** 서버 다운로드 X. 즉시 재생 가능해야 함 (~5MB 추가)
- **일시정지 시:** 전체 레이어 동시 정지. 재개 시 현재 세그먼트 처음부터 재시작 (리듬 보장)

---

## Changelog

| 날짜 | 변경 | 사유 |
|------|------|------|
| 2026-03-23 | 초안 작성 | 체육관 몰입 오디오 설계 |
| 2026-03-23 | v2: 자체 리뷰 기반 9개 문제 수정 | 아래 상세 |

### v2 수정 내역

| # | 문제 | 수정 |
|---|------|------|
| 1 | 임팩트 SFX가 "내 펀치"인지 "가이드"인지 불명확 | "패드워크 리듬 가이드"로 재정의 (Section 4 도입부) |
| 2 | 임팩트 추가 시 세그먼트 시간 2.5배 팽창 | 임팩트가 기존 실행 구간 안에서 재생되도록 변경 (Section 4 타이밍 계산) |
| 3 | Section 4 (chunk별 인터리브) vs Section 7.3 (전체 후 재생) 모순 | chunk별 인터리브로 통일 (Section 4, 7.3 모두) |
| 4 | 사용자 볼륨 제어 없음 | Section 9 추가: 코치/SFX 2그룹 + ON/OFF |
| 5 | 일시정지/재개 동기화 누락 | Section 7.2에 onPause/onResume 추가 |
| 6 | 에러 fallback 전략 없음 | Section 10 추가: 레이어별 fallback + 해제 우선순위 |
| 7 | 체인 래틀이 크로스에도 울려 70%+ 세그먼트에서 발생 | 훅/어퍼컷(LH, RH, LU, RU)으로만 제한 |
| 8 | 앱 번들 크기 15-20MB 과대추정 | 실제 계산: ~5MB (Section 7.5) |
| 9 | 격려 클립이 랜덤 삽입되면 "가짜 코치" 느낌 | 라운드 경계 시점에만 배치 (Section 5) |
