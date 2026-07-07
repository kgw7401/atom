# Atom MVP — Unified Spec

**Status:** CONVERGED
**Source documents:** final-prd.md, ui-ux-design.md, training-program-design.md, audio-immersion-design.md
**Generated:** 2026-03-23
**Convergence round:** 1

## 1. Why

Atom is a "daily 10-minute boxing routine" app. The core loop is:
Open app → See today's training → One-tap start → 10-min session → Completion + streak → Tomorrow.

The current codebase has a working session engine but lacks the **program system**, **streak**, **fixed 10-min structure**, and **PRD-aligned UX**. This build transforms the prototype into the PRD vision.

## 2. Requirements

### MUST (Core — This Build)

1. **7-Day Program System** — Predetermined daily themes with round-specific segments
2. **Streak Tracking** — Consecutive training days with grace period
3. **Fixed 10-min Session** — Intro(20s) → R1(3m) → Rest(15s) → R2(3m) → Rest(15s) → R3(3m, includes Finisher) → Outro(10s)
4. **Round-based Templates** — R1 easy combos, R2 medium, R3 hard, Finisher explosive
5. **Home Screen Redesign** — Streak + Day progress + Today's theme + One-tap start
6. **Session End Redesign** — Coach comment + Streak + Day progress + Next preview
7. **Simplified Onboarding** — 2 steps (experience level + training preference)
8. **Phase-aware Session UI** — Different colors/labels per phase
9. **Today API** — Single endpoint returning everything the home screen needs

### SHOULD (Next Sprint)

10. Tier system (Rookie → Champion)
11. Audio immersion Layer 1-3 (ambient, bells, impact SFX)
12. Share card generation
13. Streak recovery/grace period UI

### Out of Scope

- Video analysis (Track B)
- BPM metronome
- Community features
- Custom drill creation

## 3. Architecture

### System Overview

```
┌─────────────────────────────────────────────────┐
│  Mobile (Expo/React Native)                      │
│                                                  │
│  Onboarding → Home → ActiveSession → SessionEnd  │
│       ↕            ↕           ↕          ↕      │
│  ┌──────────────────────────────────────────┐    │
│  │ AsyncStorage (onboarding, settings)      │    │
│  └──────────────────────────────────────────┘    │
└──────────────────┬───────────────────────────────┘
                   │ HTTP/JSON
┌──────────────────▼───────────────────────────────┐
│  Backend (FastAPI)                                │
│                                                  │
│  GET /api/today     → Home screen data           │
│  POST /api/sessions/plan → Generate from program │
│  POST /api/sessions/log  → Log + update streak   │
│  GET /api/sessions  → History                    │
│  GET /api/profile   → Profile + streak           │
│  PUT /api/profile   → Update profile             │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │ SQLite (programs, templates, sessions,     │  │
│  │         profiles, streaks)                 │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### Key Decisions

| Decision | Chosen | Reasoning |
|----------|--------|-----------|
| Program data | Hardcoded seed (JSON) | training-program-design.md has exact sequences |
| Session timing | Fixed 10min | PRD mandate, no user config |
| Streak storage | UserProfile fields | Simple, single-user app |
| Program progress | New ProgramProgress table | Track day, started/completed |
| Finisher | Part of template JSON | Different types per day |

## 4. Implementation Plan

### Task 1: Backend — Program Data Model
- **Scope:** `src/atom/models/tables.py`, new migration
- **Depends on:** nothing
- **Work:**
  - Add to UserProfile: `current_streak`, `longest_streak`, `last_training_date`, `tier`
  - Add ProgramProgress table: `id`, `level`, `week`, `current_day`, `started_at`, `completed_at`
  - Add ProgramDayTemplate table: `id`, `level`, `week`, `day_number`, `theme`, `theme_description`, `coach_comment`, `r1_segments_json`, `r2_segments_json`, `r3_segments_json`, `finisher_json`
- **Acceptance:** Models importable, tables create successfully

### Task 2: Backend — Seed Beginner Week 1
- **Scope:** `src/atom/seed_programs.py` (new), `src/atom/seed.py`
- **Depends on:** Task 1
- **Work:**
  - Create seed data for Beginner Week 1 (7 days) from training-program-design.md
  - Each day has: theme, description, coach_comment, r1/r2/r3 segments with cues inline, finisher with type
  - Seed on startup if no programs exist
- **Acceptance:** 7 ProgramDayTemplate rows seeded for beginner/week1

### Task 3: Backend — New APIs (today + program-based plan)
- **Scope:** `src/atom/api/routers/sessions.py`, `src/atom/api/routers/today.py` (new), schemas
- **Depends on:** Task 2
- **Work:**
  - GET /api/today: Returns { streak, day_number, day_total, theme, theme_description, coach_comment, level, week, next_day_preview }
  - Modify POST /api/sessions/plan: Accept `program_day_id` or auto-detect from ProgramProgress; build round-specific segments
  - Modify POST /api/sessions/log: Update streak (current_streak++, last_training_date), advance ProgramProgress.current_day
  - New session generation: Build plan with R1/R2/R3/Finisher segments from ProgramDayTemplate, resolve audio chunks per segment
- **Acceptance:** /api/today returns correct data; /api/sessions/plan generates round-based session

### Task 4: Frontend — Theme + Navigation Overhaul
- **Scope:** `mobile/src/theme.ts`, `mobile/App.tsx` or navigation setup
- **Depends on:** nothing
- **Work:**
  - Update colors to match UI spec (#0A0A0A bg, #E5383B red, #FF6B35 orange, #FFD166 gold, #4EA8DE blue)
  - Add tab navigation: 훈련(Home), 기록(History), 나(Profile)
  - Remove SessionSetup and PlanPreview from main flow
- **Acceptance:** App has 3 tabs, colors match spec

### Task 5: Frontend — Simplified Onboarding
- **Scope:** `mobile/src/screens/onboarding/`
- **Depends on:** Task 4
- **Work:**
  - Screen 1: Welcome ("ATOM / 매일 10분, 복싱 루틴 / 시작하기")
  - Screen 2: Experience level ("처음이야" / "조금 해봤어" / "꽤 쳤어")
  - Screen 3: Training preference ("기본기" / "체력" / "스피드" / "다 좋아")
  - On completion: POST profile update + start first program + navigate to Home
  - Remove old 5-screen onboarding
- **Acceptance:** Onboarding completes in 3 taps, first program auto-created

### Task 6: Frontend — Home Screen Redesign
- **Scope:** `mobile/src/screens/HomeScreen.tsx`
- **Depends on:** Task 3, Task 4
- **Work:**
  - Fetch GET /api/today on mount
  - Display: streak (🔥 N일 연속), program progress (● ● ● ○ ○ ○ ○), day theme + description, coach comment
  - Big red "시작" button (85% width, fixed bottom)
  - One tap → POST /api/sessions/plan → navigate to ActiveSession (skip setup/preview)
  - State variants: first day, mid-program, last day (Day 7), program complete
- **Acceptance:** Home shows today's info, one-tap starts session

### Task 7: Frontend — Session Screen Redesign (Phase-Aware)
- **Scope:** `mobile/src/screens/ActiveSessionScreen.tsx`
- **Depends on:** Task 3
- **Work:**
  - Fixed 10-min timeline: Intro(20s) → R1(180s) → Rest(15s) → R2(180s) → Rest(15s) → R3(180s, includes Finisher) → Outro(10s)
  - Phase labels: "INTRO", "ROUND 1 · 적응", "REST", "ROUND 2 · 적용", "ROUND 3 · 몰입", "FINISHER", "COOL DOWN"
  - Background tints: intro(#0A0A0A), round(#0A140A), rest(#080D18), finisher(#1A0A00), outro(#0A0A0A)
  - Timer shows phase countdown + total session time
  - Round progress dots (3 + finisher)
  - Haptic on segment transitions
  - Countdown 3-2-1 before session starts
- **Acceptance:** Session plays through all phases with correct timing and UI

### Task 8: Frontend — Session End Redesign
- **Scope:** `mobile/src/screens/SessionEndScreen.tsx`
- **Depends on:** Task 3, Task 6
- **Work:**
  - Checkmark animation (Victory Gold)
  - "완료!" title
  - Coach comment from API response
  - Streak display with +1 animation
  - Day progress dots (updated)
  - Next day preview: "내일: Day N — [theme]"
  - Session stats (3R · 10분 · N콤보)
  - "홈으로 돌아가기" button
- **Acceptance:** Completion shows streak, coach comment, next day preview

## 5. Session Timeline Detail

```
Phase       Duration  Cumulative  Content
─────────────────────────────────────────────────────
Intro       20s       0:20        Coach intro
Round 1     180s      3:20        R1 segments (easy, slow pace)
Rest 1      15s       3:35        "쉬어" + breathing
Round 2     180s      6:35        R2 segments (medium pace)
Rest 2      15s       6:50        "다음 라운드 준비"
Round 3     180s      9:50        R3 segments + Finisher (fast → max intensity)
Outro       10s       10:00       Cool down + completion
─────────────────────────────────────────────────────
Total                 10:00
```

## 6. Streak Rules

- Complete a session → streak +1, update last_training_date
- Miss 1 day → streak preserved (grace period, hidden from user)
- Miss 2+ days → streak resets to 0
- Coach comment adapts: Day 1 / streak milestone (3,7,14,30) / return after break

## 7. Boundaries

- ✅ Always: Coach voice chunks play for every segment
- ✅ Always: Session is exactly 10 minutes
- ✅ Always: Home screen loads in <1s with cached data
- 🚫 Never: User configures rounds/duration (removed)
- 🚫 Never: Random template selection (program-based)
- 🚫 Never: Complex onboarding (max 3 screens)

## Changelog

| Date | Round | Change | Reason |
|------|-------|--------|--------|
| 2026-03-23 | R1 | Initial unified spec | Merged 4 source specs |
