# UI/UX Spec: Atom Phase 2 Redesign

> Status: APPROVED
> Created: 2026-03-11
> Based on: specs/mlp-prd.md (APPROVED)

---

## 1. Design Principles

1. **앱이 코칭한다, 캐릭터가 말하지 않는다** — LLM 메시지는 정보 카드로 표시. 1인칭 대화 형식 사용하지 않음.
2. **존댓말 통일** — 모든 UI 텍스트와 LLM 생성 메시지는 존댓말.
3. **하나의 루프를 깊게** — 새 기능 추가보다 기존 세션 루프의 감각적 경험을 강화.
4. **절제된 톤** — 과한 축하, 이모지 남용, 게이미피케이션 피로감 방지.

---

## 2. Design System

### 2.1 Colors

| Token | Hex | Usage |
|-------|-----|-------|
| BG | `#0a0a0a` | 앱 전체 배경 |
| SURFACE | `#141414` | 카드, 입력 필드 배경 |
| RED | `#e63946` | Primary accent, CTA, 코치 카드 테두리 |
| RED_BG | `#1a0608` | 선택 상태 배경 |
| GREEN | `#4caf50` | 완료 상태 |
| ORANGE | `#ff9800` | 중단 상태 |
| GOLD | `#ffd700` | 마일스톤 달성 |
| TEXT_1 | `#f0f0f0` | Primary text |
| TEXT_2 | `#999` | Secondary text |
| TEXT_3 | `#555` | Tertiary text, placeholder |
| TEXT_GHOST | `#333` | Ghost text, 비활성 |
| BORDER | `#242424` | 카드 테두리 |

### 2.2 Typography

| Size | Weight | Usage |
|------|--------|-------|
| 96pt | 200 | ActiveSession 타이머 |
| 52pt | 700 | ActiveSession 콤보 이름 |
| 42pt | 900 | 스트릭 숫자 (ProfileScreen) |
| 40pt | 900 | 앱 타이틀 "ATOM" |
| 32pt | 700 | SessionEnd 상태 타이틀 |
| 28pt | 700 | HomeScreen 인사 |
| 22pt | 700 | 플랜 포커스, 큰 스탯 |
| 18pt | 400 | 코치 카드 본문 (lineHeight 28) |
| 17pt | 600 | 카드 타이틀, 본문 |
| 13pt | 400 | 설명, 메타 정보 |
| 11pt | 700 | SectionLabel (uppercase, letterSpacing 1.5) |
| 10pt | 400 | 날짜, 마이크로 라벨 |

### 2.3 Spacing

| Token | Value | Usage |
|-------|-------|-------|
| PADDING_SCREEN | 24px | 화면 좌우 패딩 |
| PADDING_CARD | 16px | 카드 내부 패딩 |
| GAP_SECTION | 24px | 섹션 간 간격 |
| GAP_ITEM | 8px | 아이템 간 간격 |
| RADIUS_CARD | 12px | 카드 모서리 |
| RADIUS_BADGE | 20px | 배지, 필 모서리 |

### 2.4 Haptic Feedback

| Type | Trigger |
|------|---------|
| `impactLight` | 템플릿 카드 탭, 콤보 딜리버리, 온보딩 선택 |
| `impactMedium` | PrimaryButton 탭, 휴식 진입 |
| `impactHeavy` | 세션 완료 전환 (ActiveSession → SessionEnd) |
| `notificationSuccess` | 마일스톤 배지 reveal |

### 2.5 Animations

| Animation | Duration | Usage |
|-----------|----------|-------|
| FadeIn | 200ms | 화면 마운트 |
| FadeInDown | 120ms | 콤보 이름 전환 (crossfade) |
| ZoomIn (spring) | 400ms | 마일스톤 배지, 스트릭 카드 |
| Scale pulse | 1.0 → 1.015 → 1.0 (매초) | ActiveSession 타이머 |

기술: `react-native-reanimated` (Expo SDK 내장)

---

## 3. Shared Components

### 3.1 CoachCard

LLM이 생성한 메시지를 표시하는 정보 카드.

```
사용처: PlanPreviewScreen, SessionEndScreen (이 2곳만)

┌───────────────────────────────┐
║  오늘은 잽-크로스 위주로       │  ← 빨간 왼쪽 테두리 (3px, RED)
║  기초를 다져보겠습니다.        │    배경 SURFACE
║  총 3라운드입니다.             │    본문 18pt, TEXT_1, lineHeight 28
└───────────────────────────────┘

Props:
  message: string    — LLM 생성 메시지
```

위 화면에서 CoachCard 상단에 SectionLabel로 라벨 표시:
- PlanPreview: "코치 코멘트"
- SessionEnd: "세션 리뷰"

### 3.2 SectionLabel

```
콤보 숙련도                        ← TEXT_2, 11pt, weight 700
                                     uppercase, letterSpacing 1.5

Props:
  text: string
  color?: string (default: TEXT_2)
```

### 3.3 PrimaryButton

```
┌─────────────────────────────────┐
│           세션 시작              │  ← bg RED, radius 12, padding 18
└─────────────────────────────────┘    text: #fff, 17pt, weight 700
                                       haptic: impactMedium on press

Props:
  label: string
  onPress: () => void
  disabled?: boolean (opacity 0.3)
```

### 3.4 SecondaryButton

```
┌─────────────────────────────────┐
│           홈으로                │  ← bg transparent, border BORDER
└─────────────────────────────────┘    text: TEXT_2, 16pt
                                       radius 10, padding 16

Props:
  label: string
  onPress: () => void
```

### 3.5 StatCard

```
┌──────────┐
│    23     │  ← TEXT_1, 22pt, weight 700
│  총 세션  │  ← TEXT_3, 11pt
└──────────┘    bg SURFACE, radius 10, border BORDER

Props:
  value: string
  label: string
```

### 3.6 MilestoneBadge

```
달성:                              미달성:
┌──────────────┐                   ┌──────────────┐
│ ✓ 첫 세션    │  ← GOLD text     │ ○ 10회 달성  │  ← TEXT_GHOST text
│   2025/11/03 │  ← TEXT_3        │              │
└──────────────┘  bg GOLD_BG       └──────────────┘  bg #151515
                  border GOLD                         border BORDER

Props:
  label: string
  achieved: boolean
  date?: string
```

### 3.7 ComboProgressBar

```
┌───────────────────────────────┐
│ 잽-크로스               숙련  │  ← name: TEXT_1 15pt, level: mastery color
│ ████████████░░░░░░  22/31    │  ← 4px bar, fill color = mastery color
└───────────────────────────────┘    count: TEXT_3

Mastery levels:
  새싹  (1-5):   #888
  성장  (6-15):  #4caf50
  숙련  (16-30): #e63946
  마스터 (31+):  #ffd700

Bar fill = count / nextLevelThreshold

Props:
  name: string
  count: number
```

---

## 4. Navigation

```
App Launch
    │
    ├─ [AsyncStorage: atom_onboarding_complete?]
    │       │ NO                          │ YES
    │       ▼                             ▼
    │  OnboardingStack (modal)       TabNavigator
    │  ┌─────────────────┐          ├── HomeTab (🥊)
    │  │ O1: Experience  │          │   Home
    │  │ O2: Goal        │          │    → SessionSetup
    │  └───────┬─────────┘          │    → PlanPreview
    │          │ fade               │    → ActiveSession
    │          └─────────────────▶  │    → SessionEnd
    │                               │    → Settings
    │                               │
    │                               ├── HistoryTab (📋)
    │                               │   HistoryScreen
    │                               │
    │                               └── ProfileTab (👤)
    │                                   ProfileScreen
    │                                    → Settings
```

**Screen Transitions:**

| Transition | Type |
|-----------|------|
| Stack push (기본) | slide right (native default) |
| → ActiveSession | fade (몰입 진입) |
| → SessionEnd | fade (결과 공개) |
| Onboarding 내 | slide horizontal |
| Onboarding → 앱 | fade (전환) |

---

## 5. Screens

### 5.1 Onboarding: ExperienceScreen (신규)

첫 실행 시 표시. 복싱 경험 레벨 수집.

```
┌─────────────────────────────────┐
│                                 │
│  ATOM                           │  42pt RED, weight 900, spacing 6
│  AI BOXING COACH                │  11pt TEXT_3, spacing 3
│                                 │
│  ─────────────────────────────  │  BORDER, 1px
│                                 │
│  복싱 경험                      │  SectionLabel
│                                 │
│  ┌───────────────────────────┐  │
│  │ [●] 처음                  │  selected: bg RED_BG, border RED
│  │     복싱을 배운 적 없음   │  desc: TEXT_3, 13pt
│  ├───────────────────────────┤  │
│  │ [ ] 초급                  │  unselected: bg SURFACE, border BORDER
│  │     기본 펀치를 알고 있음 │
│  ├───────────────────────────┤  │
│  │ [ ] 중급                  │
│  │     콤비네이션 가능       │
│  ├───────────────────────────┤  │
│  │ [ ] 상급                  │
│  │     스파링 경험 있음      │
│  └───────────────────────────┘  │
│                                 │
│  ┌─────────────────────────────┐│
│  │           다음              ││  PrimaryButton (선택 전 disabled)
│  └─────────────────────────────┘│
└─────────────────────────────────┘
```

| 선택지 | experience_level |
|--------|-----------------|
| 처음 | beginner |
| 초급 | novice |
| 중급 | intermediate |
| 상급 | advanced |

Haptic: `impactLight` on selection

### 5.2 Onboarding: GoalScreen (신규)

훈련 목표 수집. 선택지 또는 자유 입력.

```
┌─────────────────────────────────┐
│ ←                               │
│                                 │
│  훈련 목표                      │  SectionLabel
│                                 │
│  ┌───────────────────────────┐  │
│  │ [ ] 체력 향상              │  선택 카드 (ExperienceScreen과 동일 스타일)
│  ├───────────────────────────┤  │
│  │ [ ] 다이어트              │
│  ├───────────────────────────┤  │
│  │ [ ] 실전 기술 연습        │
│  ├───────────────────────────┤  │
│  │ [ ] 스트레스 해소         │
│  └───────────────────────────┘  │
│                                 │
│  직접 입력 (선택)               │  TEXT_3, 11pt
│  ┌───────────────────────────┐  │
│  │                           │  TextInput, bg SURFACE, border BORDER
│  └───────────────────────────┘  │
│                                 │
│  ┌─────────────────────────────┐│
│  │         시작하기            ││  PrimaryButton
│  └─────────────────────────────┘│
└─────────────────────────────────┘
```

**동작:**
1. "시작하기" 누르면:
   - `AsyncStorage.setItem('atom_onboarding_complete', 'true')`
   - `updateProfile({ experience_level, goal })` API 호출
   - fade 전환으로 TabNavigator 진입

### 5.3 HomeScreen (리디자인)

현재: 중앙 정렬, 타이틀이 hero (56pt), 인사가 보조.
변경: Strava Stats Dashboard 패턴. 최대 정보 밀도, 스크롤 없이 한 화면 완결.

```
┌─────────────────────────────────┐
│  ATOM                 🔥 7      │  32pt RED left + streak pill right
│                                 │
│  ┌──────┬──────┬──────┬──────┐  │
│  │  23  │ 180분│ 3.2회│ 초급 │  4-column compact stats
│  │ 세션 │ 훈련 │ 주당 │ 레벨 │  value: 20pt TEXT_1, weight 700
│  └──────┴──────┴──────┴──────┘  label: 10pt TEXT_3
│                                 │
│  이번 주                        │  SectionLabel
│  ▁ ▃ ▅ █ ░ ░ ░                 │  7-bar sparkline (분 단위)
│  월 화 수 목 금 토 일           │  bar: RED, empty: #1a1a1a
│                                 │  labels: 10pt TEXT_3
│  ─────────────────────────────  │  BORDER, 1px
│                                 │
│  최근: 기본기 · 3R · 12분      │  compact 1-line session summary
│  다음: 10회 달성까지 2회        │  compact 1-line milestone hint
│                                 │  둘 다 14pt, TEXT_2
│                                 │
│                                 │
│                                 │
│  ┌─────────────────────────────┐│
│  │         세션 시작            ││  PrimaryButton, 하단 고정
│  └─────────────────────────────┘│
│         ⚙ 서버 설정             │  TEXT_GHOST, 12pt, 텍스트 링크
└─────────────────────────────────┘
```

**Streak Pill (조건: streak >= 2):**
```
🔥 7      ← bg RED_BG, border RED, radius 16
              RED text, 14pt, weight 700
              streak < 2이면 pill 숨김
```

**4-Column Stats:**
```
┌──────┬──────┬──────┬──────┐
│  23  │ 180분│ 3.2회│ 초급 │   bg SURFACE, border BORDER, radius 10
│ 세션 │ 훈련 │ 주당 │ 레벨 │   flex: 1 each, gap 8
└──────┴──────┴──────┴──────┘
세션 0이면: 모두 "–" 표시
```

**7-Bar Sparkline:**
```
각 요일의 훈련 시간(분)을 bar 높이로 표현.
max height: 40px, min (>0): 8px
bar width: flex, gap: 6px, border-radius: 3px top
color: minutes > 0 ? RED : #1a1a1a
아래 요일 라벨: 10pt TEXT_3, 오늘 요일만 TEXT_1
```

**최근 세션 / 마일스톤 라인:**
```
최근 세션 있을 때:  "최근: {template} · {rounds}R · {duration}분"
세션 없을 때:       "첫 세션을 시작해보세요."

다음 마일스톤:      "다음: {milestone_label}까지 {N}회"
전부 달성:          표시 안 함
```

**인사 로직 없음** — Compression 디자인은 데이터 기반. 텍스트 인사 대신 sparkline과 stats가 상태를 전달.

**첫 사용자 (total_sessions === 0):**
```
┌─────────────────────────────────┐
│  ATOM                           │  32pt RED
│                                 │
│  ┌──────┬──────┬──────┬──────┐  │
│  │  –   │  –   │  –   │  –   │  stats 모두 "–"
│  │ 세션 │ 훈련 │ 주당 │ 레벨 │
│  └──────┴──────┴──────┴──────┘  │
│                                 │
│  이번 주                        │
│  ░ ░ ░ ░ ░ ░ ░                 │  모두 빈 bar
│  월 화 수 목 금 토 일           │
│  ─────────────────────────────  │
│                                 │
│  첫 세션을 시작해보세요.        │  14pt TEXT_2
│                                 │
│                                 │
│                                 │
│  ┌─────────────────────────────┐│
│  │         세션 시작            ││
│  └─────────────────────────────┘│
│         ⚙ 서버 설정             │
└─────────────────────────────────┘
```

### 5.4 SessionSetupScreen (리디자인)

현재: "코치에게 요청" 프레이밍, 버튼에 인라인 로딩.
변경: 깔끔한 라벨, 전체 화면 로딩 상태.

```
기본 상태:
┌─────────────────────────────────┐
│ ← 세션 설정                     │
│                                 │
│  템플릿                         │  SectionLabel
│  ┌───────────────────────────┐  │
│  │ ██ 기본기                 │  selected: bg RED_BG, border RED
│  │    잽·스트레이트·훅·어퍼 │  desc: TEXT_3, 13pt
│  ├───────────────────────────┤  │
│  │ ░░ 콤비네이션             │  unselected: bg SURFACE, border BORDER
│  │    연속 콤보 드릴         │
│  ├───────────────────────────┤  │
│  │ ░░ 종합                   │
│  │    템플릿 혼합 훈련       │
│  └───────────────────────────┘  │
│  ─────────────────────────────  │
│                                 │
│  요청사항 (선택)                │  SectionLabel
│  ┌───────────────────────────┐  │
│  │ 예: 잽 크로스 위주로,     │  placeholder TEXT_GHOST
│  │ 오늘 어깨가 좀 아파서     │  bg SURFACE, border BORDER
│  │ 가볍게 해주세요           │
│  └───────────────────────────┘  │
│                                 │
│  ┌─────────────────────────────┐│
│  │         플랜 생성           ││  PrimaryButton
│  └─────────────────────────────┘│
└─────────────────────────────────┘

로딩 상태 (전체 화면 교체):
┌─────────────────────────────────┐
│                                 │
│                                 │
│                                 │
│         ●  ●  ●                 │  3 dots, RED
│                                 │  scale 0.6→1.0→0.6, 600ms stagger
│  플랜을 생성하고 있습니다...     │  TEXT_2, 16pt, centered
│                                 │
│                                 │
└─────────────────────────────────┘
```

Haptic: `impactLight` on template selection

### 5.5 PlanPreviewScreen (리디자인)

현재: 코치 메시지가 일반 카드, CTA가 스크롤 하단.
변경: "코치 코멘트" 라벨의 CoachCard, CTA 하단 고정.

```
┌─────────────────────────────────┐
│ ← 플랜 미리보기                 │
│                                 │
│  코치 코멘트                    │  SectionLabel (color: RED)
│  ┌───────────────────────────┐  │
│  ║ 오늘은 잽-크로스 위주로    │  CoachCard
│  ║ 기초를 다져보겠습니다.     │  LLM 생성, 존댓말
│  ║ 총 3라운드입니다.          │
│  └───────────────────────────┘  │
│                                 │
│  잽-크로스 기초 훈련            │  plan.focus, 22pt TEXT_1, weight 700
│  3라운드 · 약 12분              │  meta, 13pt TEXT_3
│  ─────────────────────────────  │
│                                 │
│  ROUND 1                 180초  │  RED, 11pt uppercase
│    잽-크로스        1 → 2       │  name: TEXT_1, actions: TEXT_3
│    원-투-훅      1 → 2 → 3     │
│    잽-크로스-훅  1 → 2 → 3     │
│                                 │
│  ROUND 2                 180초  │
│    ...                          │
│                                 │
│  (bottom padding 100px)         │  CTA에 가리지 않도록
│                                 │
├─────────────────────────[fixed]─┤
│  ┌─────────────────────────────┐│  position: absolute, bottom: 0
│  │       세션 시작 →           ││  PrimaryButton
│  └─────────────────────────────┘│  bg gradient: BG → transparent (위쪽)
└─────────────────────────────────┘
```

coach_message가 비어있으면 (fallback plan) CoachCard 섹션 숨김.

### 5.6 ActiveSessionScreen (리디자인)

핵심 몰입 화면. 최소한의 요소만 표시.

```
ROUND:
╔═════════════════════════════════╗
║                                 ║
║  ROUND 2 / 3             [#333]║  11pt TEXT_GHOST, uppercase
║                                 ║
║                                 ║
║                                 ║
║           02:47                 ║  96pt TEXT_1, weight 200, spacing 8
║                                 ║  매초 scale pulse: 1.0 → 1.015 → 1.0
║                                 ║
║  ─────────────────────────────  ║  BORDER, 1px
║                                 ║
║                                 ║
║         잽-크로스               ║  52pt TEXT_1, weight 700, centered
║                                 ║  FadeInDown 120ms on change
║          1 → 2                  ║  18pt TEXT_3, centered
║                                 ║
║                                 ║
║                                 ║
║         ● ● ○                   ║  round dots, 10px, centered
║                                 ║  done=RED, active=RED(0.5), pending=#333
║  중단                           ║  TEXT_GHOST, 16pt, bottom-left
╚═════════════════════════════════╝

REST:
╔═════════════════════════════════╗
║                                 ║
║  REST                    [#333] ║
║                                 ║
║                                 ║
║                                 ║
║           00:45                 ║  96pt TEXT_GHOST (#444), weight 200
║                                 ║
║                                 ║
║  ─────────────────────────────  ║
║                                 ║
║         잠시 쉬세요.            ║  24pt TEXT_GHOST (#444)
║                                 ║
║  다음: 원-투-훅                 ║  13pt TEXT_3
║                                 ║
║                                 ║
║         ● ● ○                   ║
║  중단                           ║
╚═════════════════════════════════╝
```

**변경 사항:**

| 요소 | Before | After |
|------|--------|-------|
| 타이머 | 72pt | 96pt + 매초 scale pulse |
| 콤보 전환 | 즉시 교체 | FadeInDown + FadeOutUp crossfade |
| 콤보 딜리버리 | 시각+음성 | + impactLight 햅틱 |
| 휴식 표시 | "잠시 쉬세요" | + "다음: {next combo}" 미리보기 |
| 중단 버튼 | 보통 | ghost화 (#333) |
| 준비 상태 | "준비..." | 동일 유지 |

### 5.7 SessionEndScreen (리디자인)

현재: '!' / '...' 텍스트 아이콘, 스탯이 행 나열.
변경: ✓/– 아이콘, 3-column 스탯, ZoomIn 애니메이션.

```
COMPLETED:
┌─────────────────────────────────┐
│                                 │
│             ○                   │  80px circle
│             ✓                   │  bg #0a1f0a, border GREEN 2px
│                                 │  "✓" 36pt GREEN
│       수고하셨습니다.           │  ZoomIn spring on mount + impactHeavy
│                                 │  32pt TEXT_1, weight 700
│                                 │
│  세션 리뷰                      │  SectionLabel (color: RED)
│  ┌───────────────────────────┐  │
│  ║ [LLM session_review       │  CoachCard
│  ║  .completed, 존댓말]      │
│  └───────────────────────────┘  │
│                                 │
│  ┌──────────┬──────────┬──────┐ │
│  │    3     │    18    │12:03 │ │  3-column StatCard
│  │  라운드  │   콤보   │ 시간 │ │
│  └──────────┴──────────┴──────┘ │
│                                 │
│  ┌───────────────────────────┐  │  streak >= 2일 때만 표시
│  │  🔥  7일 연속             │  │  bg RED_BG, border RED
│  └───────────────────────────┘  │  ZoomIn spring + impactMedium
│                                 │
│  [milestone badges]             │  newMilestones 있을 때만
│                                 │  stagger ZoomIn 200ms + notificationSuccess
│                                 │
│  ┌─────────────────────────────┐│
│  │         다시 시작           ││  PrimaryButton
│  └─────────────────────────────┘│
│  ┌─────────────────────────────┐│
│  │           홈으로            ││  SecondaryButton
│  └─────────────────────────────┘│
└─────────────────────────────────┘

ABANDONED:
  Circle: bg #2e2a1a, border ORANGE, "–" 36pt ORANGE
  Title: "중단되었습니다."
  Stats: rounds "2/4" 형식
  No streak card
  Primary CTA: "다시 시도"
  Secondary: "홈으로"
```

### 5.8 HistoryScreen (리디자인)

현재: 영문 날짜, 텍스트 status badge.
변경: 상대 날짜, 주간 요약, 색상 스트라이프.

```
┌─────────────────────────────────┐
│ 훈련 기록                       │
│                                 │
│  이번 주 3회 훈련               │  TEXT_2, 13pt
│                                 │
│  ┌───────────────────────────┐  │
│  ║ 기본기             ●완료  │  왼쪽 3px GREEN 스트라이프
│  │ 어제 · 오전 10:32         │  ● = 8px GREEN dot, right
│  │ 3라운드 · 18콤보 · 12분  │  date: TEXT_3, stats: TEXT_3
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  ║ 콤비네이션         ●중단  │  왼쪽 3px ORANGE 스트라이프
│  │ 3일 전                    │  ● = 8px ORANGE dot
│  │ 2/4라운드 · 8콤보 · 6분  │
│  └───────────────────────────┘  │
│                                 │
└─────────────────────────────────┘
```

**상대 날짜 형식:**

| 조건 | 표시 |
|------|------|
| 오늘 | 오늘 HH:MM |
| 어제 | 어제 HH:MM |
| 2-6일 전 | N일 전 |
| 7일+ | M월 D일 |

**빈 상태:**
```
┌─────────────────────────────────┐
│ 훈련 기록                       │
│                                 │
│                                 │
│           🥊                    │  48pt centered
│                                 │
│  아직 훈련 기록이 없습니다.     │  TEXT_3, 16pt, centered
│                                 │
│  ┌─────────────────────────────┐│
│  │         세션 시작           ││  PrimaryButton → HomeTab SessionSetup
│  └─────────────────────────────┘│
└─────────────────────────────────┘
```

### 5.9 ProfileScreen (리디자인)

현재: 텍스트 "서버 설정" 링크, 콤보는 텍스트만, 마일스톤 1열.
변경: 기어 아이콘, 콤보 프로그레스 바, 2열 마일스톤.

```
┌─────────────────────────────────┐
│ 프로필                    [⚙]  │  기어 아이콘 → Settings
│                                 │
│  ┌───────────────────────────┐  │
│  │  🔥 7            초급     │  streak left (RED, 42pt, weight 900)
│  │     일 연속              │  + "일 연속" (RED, 16pt)
│  │     최고 14일            │  + 최고 기록 (TEXT_3, 12pt)
│  │                   ┌────┐ │  level badge right (bg #2a2a2a, radius 8)
│  │                   │초급│ │
│  │                   └────┘ │
│  └───────────────────────────┘  │  bg SURFACE, radius 12, border BORDER
│                                 │
│  ┌───────┬──────────┬────────┐  │
│  │ 23    │  180분   │  3.2회 │  │  3-column StatCard
│  │총 세션│훈련 시간 │주당 빈도│  │
│  └───────┴──────────┴────────┘  │
│                                 │
│  훈련 기록 (12주)               │  SectionLabel
│  ┌───────────────────────────┐  │
│  │ [heatmap grid]            │  │  12px cells (10→12), 3px gap
│  └───────────────────────────┘  │  bg #111, radius 8, padding 10
│  ░ ░ ░ ░ ░  적음 → 많음        │  legend right-aligned
│                                 │
│  콤보 숙련도                    │  SectionLabel
│  ┌───────────────────────────┐  │
│  │ ComboProgressBar          │  │  (see component spec 3.7)
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │ ComboProgressBar          │  │
│  └───────────────────────────┘  │
│                                 │
│  마일스톤                       │  SectionLabel
│  ┌──────────────┬─────────────┐ │
│  │ MilestoneBadge             │ │  2-column grid, 8px gap
│  ├──────────────┼─────────────┤ │
│  │              │             │ │
│  └──────────────┴─────────────┘ │
│                                 │
│  목표                           │  SectionLabel
│  체력 향상          [수정]      │  TEXT_1 16pt + "수정" 텍스트 링크 TEXT_3
└─────────────────────────────────┘
```

streak가 0이면: "오늘 훈련하고 스트릭을 시작하세요." (TEXT_3)

### 5.10 SettingsScreen (리디자인)

현재: 영문 라벨, Alert으로 상태 표시.
변경: 한국어 라벨, 연결 테스트 버튼, 인라인 상태.

```
┌─────────────────────────────────┐
│ ← 서버 설정                     │
│                                 │
│  백엔드 서버 주소               │  SectionLabel
│  ┌───────────────────────────┐  │
│  │ http://192.168.1.42:8000  │  │  TextInput, bg SURFACE, border BORDER
│  └───────────────────────────┘  │
│  컴퓨터에서 atom serve 실행 후  │  TEXT_3, 12pt
│  로컬 IP를 입력하세요.          │
│  예: http://192.168.x.x:8000   │
│                                 │
│  ┌─────────────────────────────┐│
│  │           저장              ││  PrimaryButton
│  └─────────────────────────────┘│
│  ┌─────────────────────────────┐│
│  │         연결 테스트         ││  SecondaryButton → GET /api/profile
│  └─────────────────────────────┘│
│                                 │
│  ✓ 연결됨                       │  GREEN, 14pt (성공 시)
│  ✗ 연결 실패                    │  RED, 14pt (실패 시)
└─────────────────────────────────┘
```

---

## 6. Backend Change

### LLM 톤 변경: 반말 → 존댓말

**File:** `src/atom/services/session_service.py`

SYSTEM_PROMPT에서 코치 성격 지시를 수정:
- Before: "Speak casually in Korean (반말), like a friendly but knowledgeable coach"
- After: "Speak politely in Korean (존댓말). Warm but professional tone."

`coach_message` 및 `session_review` 출력이 존댓말로 생성되도록 변경.

`_fallback_plan()` 기본 메시지도 존댓말로:
- `coach_message`: "준비되셨으면 시작하겠습니다."
- `session_review.completed`: "수고하셨습니다."
- `session_review.abandoned`: "다음에 다시 해보세요."

---

## 7. CoachCard 사용 범위

| Screen | 사용 여부 | 내용 |
|--------|----------|------|
| PlanPreview | ✓ | "코치 코멘트" + plan.coach_message (LLM) |
| SessionEnd | ✓ | "세션 리뷰" + session_review (LLM) |
| Home | ✗ | — |
| SessionSetup | ✗ | — |
| ActiveSession | ✗ | — |
| History | ✗ | — |
| Profile | ✗ | — |
| Settings | ✗ | — |
| Onboarding | ✗ | — |

---

## 8. Files

**Create (11):**
```
mobile/src/theme.ts
mobile/src/components/CoachCard.tsx
mobile/src/components/PrimaryButton.tsx
mobile/src/components/SecondaryButton.tsx
mobile/src/components/SectionLabel.tsx
mobile/src/components/StatCard.tsx
mobile/src/components/MilestoneBadge.tsx
mobile/src/components/ComboProgressBar.tsx
mobile/src/hooks/useOnboarding.ts
mobile/src/screens/onboarding/ExperienceScreen.tsx
mobile/src/screens/onboarding/GoalScreen.tsx
```

**Modify (10):**
```
mobile/App.tsx
mobile/src/screens/HomeScreen.tsx
mobile/src/screens/SessionSetupScreen.tsx
mobile/src/screens/PlanPreviewScreen.tsx
mobile/src/screens/ActiveSessionScreen.tsx
mobile/src/screens/SessionEndScreen.tsx
mobile/src/screens/HistoryScreen.tsx
mobile/src/screens/ProfileScreen.tsx
mobile/src/screens/SettingsScreen.tsx
src/atom/services/session_service.py
```

---

## Changelog

| Date | Change | Reason |
|------|--------|--------|
| 2026-03-11 | Initial spec | Based on approved PRD + design review |
| 2026-03-11 | HomeScreen → Compression 디자인 | Strava Stats Dashboard 패턴: 4-col stats + 7-bar sparkline + compact summary. 인사 텍스트 제거, 데이터 기반 대시보드로 전환. |
