# Atom v2 — Roadmap & Technical Spec

> Status: DRAFT
> Created: 2026-03-14
> Based on: specs/prd-final.md + specs/ui-ux-spec-final.md
> Supersedes: specs/phase2-spec.md

---

## 1. 현재 상태 (As-Is)

### 이미 구현된 것

| 레이어 | 구현 완료 |
|--------|----------|
| **DB 스키마** | Action, Combination, SessionTemplate, DrillPlan, SessionLog, UserProfile, CustomSession |
| **Backend API** | sessions, combos, history, profile, custom_sessions 라우터 |
| **Mobile 화면** | Home, Train, ActiveSession, SessionEnd, History, Profile, Settings, Onboarding (5단계) |
| **핵심 기능** | LLM 세션 플래닝(1 call/session), TTS 콤보 전달, 기본 스트릭, 주간 스파클라인 |

### 갭 (Gap) — 새 PRD 대비 없는 것

| 신규 요구사항 | 우선순위 |
|-------------|---------|
| 홈/체육관 모드 토글 (메인 UX 진입점) | P0 |
| 의도 입력 화면 (자연어/태그 → AI 브리핑) | P0 |
| 훈련 중 컬러 UX (그린/레드/블루) | P0 |
| 리듬 TTS (잔여 시간에 따른 긴박감 변화) | P0 |
| 성장 카드 (세션 종료 후 공유 가능한 이미지) | P1 |
| AI Insight (세션 종료 후 데이터 기반 한줄 총평) | P1 |
| 성장 지표 그래프 (스피드/파워/지구력) | P1 |
| 무료/Pro 요금제 페이월 | P1 |
| Google 로그인 | P2 |
| 소셜 피드 (성장 카드 공유, 랭킹) | P2 |

---

## 2. 아키텍처 원칙 (변경 없음)

- **LLM 1 call/session** 원칙 유지
- **Data-first**: 모든 신규 기능에 Data Contract 필수
- **Track A 독립성**: Track B 없이 동작
- **Append-only SessionLog**: 덮어쓰기 금지

---

## 3. 마일스톤 로드맵

### Phase 1 — Core UX Redesign (MVP 완성) `P0`

**목표:** 새 PRD의 핵심 훈련 루프를 완성한다. 홈/체육관 모드, 의도 입력, 몰입 훈련 UX.

#### M1-A: 홈 대시보드 개편

**변경 사항:**
- 현재: 추천 템플릿 카드 1개 + 최근 세션 리스트
- 변경: **[🥊 체육관 모드]** + **[🏠 홈 모드]** 두 개의 거대 버튼을 메인에 배치
- 성장 지표 placeholder (스피드/파워/지구력 — Phase 2에서 실제 계산)
- 스트릭 pill은 유지, 주간 스파클라인은 유지

**영향 파일:**
- `mobile/src/screens/HomeScreen.tsx` — 버튼 레이아웃 개편
- `mobile/src/api/session.ts` — `mode: 'home' | 'gym'` 파라미터 추가

**Data Contract:**
```
Input:  training_environment: 'home' | 'gym'  (UserProfile에 이미 존재)
Output: generatePlan({ mode: 'home' | 'gym', template?, user_prompt? })
```

---

#### M1-B: 의도 입력 화면 (Intent Screen)

**새 화면:** `IntentScreen` — 세션 설정과 ActiveSession 사이에 삽입

```
[AI 코치 멘트] "오늘 특별히 신경 쓰고 싶은 게 있나요?"
  ↓
[텍스트 입력창 or 음성 버튼]
  ↓
[추천 태그] #카운터 #체력지옥 #위빙중점 #가볍게
  ↓
[AI 브리핑] "카운터 위주 6라운드 구성할게요. 글러브 끼고 시작하세요."
  ↓
[훈련 시작] 버튼
```

**영향 파일:**
- `mobile/src/screens/IntentScreen.tsx` — 신규 생성
- `mobile/App.tsx` — 네비게이션 스택에 IntentScreen 추가
- `src/atom/api/routers/sessions.py` — plan 생성 시 mode + intent 반영
- `src/atom/services/session_service.py` — LLM 프롬프트에 mode/intent 컨텍스트 추가

**Data Contract:**
```
Input:
  - mode: 'home' | 'gym'
  - user_prompt: str (의도 자유 입력, optional)
  - intent_tags: list[str] (선택된 키워드 태그, optional)
  - user_profile: { recent_7d_sessions, identified_weak_areas }

Output:
  - DrillPlan.plan_json에 { mode, intent_summary, ai_briefing_text } 추가
  - Storage: drill_plans 테이블 (기존 plan_json에 필드 추가)
```

**LLM 프롬프트 변경 (session_service.py):**
```python
# 기존: template + history context
# 신규: template + history context + mode + intent
system_prompt = f"""
모드: {'체육관 (워밍업 없이, 짧고 강렬한 명령형)' if mode == 'gym' else '홈 (워밍업 포함, 리듬감 있고 친절한)'}
유저 의도: {user_prompt or '없음'}
선택 태그: {', '.join(intent_tags) or '없음'}
최근 7일 데이터: {history_summary}
"""
```

---

#### M1-C: 훈련 중 UX — 컬러 + 리듬 TTS

**현재 ActiveSessionScreen 변경:**
- 타이머 영역 = 화면의 40% (현재보다 대형화)
- 배경색 상태 머신:
  - 운동 중: `#1A2E1A` (그린 계열)
  - 잔여 30초: `#2E1A1A` (레드 점멸, 애니메이션)
  - 휴식 중: `#1A1A2E` (블루 계열)
- 현재 지시어 (콤보 이름) 대문자 표시, 화면 중앙
- [⏸ 일시정지] / [⏭ 다음] 50:50 분할 거대 버튼

**리듬 TTS 로직:**
```
round_remaining > 60s  → 보통 속도, 친절한 톤 ("잽! 크로스! 좋아요!")
round_remaining 30-60s → 빠른 속도, 격려 ("페이스 유지!")
round_remaining < 30s  → 빠른 비트 효과 + 긴박한 명령 ("연타! 더 빠르게!")
rest                   → 낮고 부드러운 톤 ("잘 했어요. 숨 고르세요.")
```

**영향 파일:**
- `mobile/src/screens/ActiveSessionScreen.tsx`
- `mobile/src/hooks/useTTS.ts` — 신규 (TTS 속도/톤 제어 hook)

**Data Contract:**
```
Input:  DrillPlan.plan_json + round_remaining_sec (real-time, local state)
Output: background_color (local UI state), TTS utterance (ephemeral audio)
Storage: 없음 (ephemeral, SessionLog에 영향 없음)
```

---

### Phase 2 — Growth & Insight `P1`

**목표:** "보이지 않는 성장을 데이터로 증명" — 세션 종료 후 AI 인사이트와 공유 가능한 성장 카드.

#### M2-A: AI Insight & 성장 카드

**세션 종료 화면 (SessionEndScreen) 개편:**

```
[AI 오디오 총평]
"수고하셨어요! 오늘 카운터 연습으로 콤보 속도가 역대 최고였습니다!"

[성장 카드 비주얼]
  ┌─────────────────────────┐
  │    ATOM                 │
  │  오늘의 훈련             │
  │  🥊 체육관 모드 · 6라운드 │
  │  1,230 스트라이크        │
  │  ⚡ TOP SPEED 배지       │
  │  🔥 12일 연속 스트릭     │
  └─────────────────────────┘

[📸 성장 카드 공유하기]
```

**AI Insight 생성 방식:**
- 세션 종료 후 별도 LLM call (1 call/session plan + 1 call/insight = 2 calls/session max)
- 또는: 세션 종료 시 deterministic 로직으로 insight 생성 (비용 0)
  - 예: 콤보 수 > 평균 → "오늘 콤보 수 역대 최고!"
  - 예: 새 콤보 처음 드릴 → "처음 도전한 콤보!"
  - **권장: deterministic 먼저, LLM은 Pro 유저에만**

**영향 파일:**
- `mobile/src/screens/SessionEndScreen.tsx` — 성장 카드 UI + 공유 버튼
- `mobile/src/services/insightEngine.ts` — 신규 (deterministic insight 계산)
- `src/atom/api/routers/sessions.py` — `/sessions/{id}/insight` 엔드포인트 추가
- `src/atom/services/session_service.py` — insight 생성 로직

**Data Contract:**
```
Input:
  - SessionLog: { combos_delivered, rounds_completed, total_duration_sec }
  - UserProfile: { combo_mastery_json, total_sessions, current_streak }
  - session_history_avg: { avg_combos_per_session, avg_rounds }

Output:
  - insight_text: str  (예: "오늘 콤보 수 역대 최고! 1,230개")
  - badges: list[str]  (예: ["TOP_SPEED", "NEW_COMBO", "STREAK_12"])
  - growth_card_data: { mode, rounds, strikes, streak, badges, insight }

Storage:
  - SessionLog.combo_feedback_json에 { insight, badges } 추가 (mutable on first write only)
  - 성장 카드 이미지: 클라이언트 사이드 생성 (react-native-view-shot)
  - Retention: 세션별 영구 보존
```

**성장 카드 구현 (클라이언트):**
```typescript
// react-native-view-shot으로 View를 PNG로 캡처
// expo-sharing으로 인스타그램 공유
import ViewShot from 'react-native-view-shot';
import * as Sharing from 'expo-sharing';
```

---

#### M2-B: 성장 지표 그래프 (홈 대시보드)

**3가지 지표 정의:**

| 지표 | 계산 방식 | 기반 데이터 |
|-----|----------|-----------|
| **스피드** | 라운드당 콤보 수 / 라운드 시간 | SessionLog.combos_delivered / total_duration_sec |
| **파워** | 하드 드릴(power/conditioning 템플릿) 비율 | SessionLog.template_name 분포 |
| **지구력** | 평균 라운드 완료율 | rounds_completed / rounds_total 평균 |

**영향 파일:**
- `src/atom/api/routers/history.py` — `/history/growth-metrics` 엔드포인트 추가
- `src/atom/services/profile_service.py` — 지표 계산 로직
- `mobile/src/screens/HomeScreen.tsx` — 지표 그래프 컴포넌트 추가
- `mobile/src/components/GrowthGraph.tsx` — 신규 (라인 차트, react-native-svg)

**Data Contract:**
```
Input:  SessionLog 최근 30개 (user_id 기준)
Output:
  - speed_trend: list[{ date, value }]   (0-100 정규화)
  - power_trend:  list[{ date, value }]
  - endurance_trend: list[{ date, value }]
Storage: 없음 (computed on-demand, cached 1시간)
```

---

#### M2-C: 홈/체육관 모드 특화 콘텐츠

**홈 모드 DrillPlan 특성:**
- 워밍업 라운드 포함 (1라운드 = 가벼운 스트레칭 콤보)
- 쉐도우/맨몸 드릴 위주 태그 (`shadow`, `bodyweight`)
- TTS 톤: 친절하고 리드미컬, 설명형
- 기본 5-6라운드, 라운드당 2분

**체육관 모드 DrillPlan 특성:**
- 워밍업 생략
- 샌드백/미트 타격 위주 태그 (`bag`, `mitts`, `power`)
- TTS 톤: 짧고 강렬한 명령형
- 기본 6-8라운드, 라운드당 3분

**영향 파일:**
- `src/atom/seed.py` — 홈/체육관별 SessionTemplate 추가
- `src/atom/services/session_service.py` — mode별 프롬프트 분기

---

### Phase 3 — Monetization & Social `P2`

**목표:** 무료/Pro 요금제 도입, 소셜 공유, Google 로그인.

#### M3-A: 요금제 페이월

**Free tier:**
- 고정 기본 루틴 (프리셋 3가지)
- 기본 TTS (톤 변화 없음)
- 기록: 세션 수, 라운드 수, 스트릭

**Pro tier (is_premium = true):**
- 의도 입력 + AI 세션 생성 (무제한)
- 리듬 TTS (톤 변화)
- 체육관 모드 특화 드릴
- 성장 그래프 + AI Insight
- 성장 카드 공유

**구현:**
- `UserProfile.is_premium` 필드 이미 존재
- `src/atom/api/routers/sessions.py` — is_premium 체크 미들웨어
- `mobile/src/components/ProBadge.tsx` — 이미 존재, 페이월 모달에 활용
- 결제: RevenueCat SDK (Expo 호환, AppStore/PlayStore 내결제)

**Data Contract:**
```
Input:  UserProfile.is_premium (boolean)
Output: feature_flags: { ai_session, rhythm_tts, gym_mode, growth_graph, share_card }
Storage: UserProfile (기존 필드)
```

---

#### M3-B: Google 로그인

**구현:**
- `expo-auth-session` + Google OAuth
- 백엔드: JWT 토큰 검증, `users` 테이블 신규 추가
- 현재: 단일 유저 가정 → multi-user 마이그레이션

**신규 DB 테이블:**
```sql
users (
  id          TEXT PRIMARY KEY,
  google_sub  TEXT UNIQUE,
  email       TEXT,
  display_name TEXT,
  created_at  DATETIME
)
-- 기존 테이블에 user_id FK 추가 (Alembic migration)
```

---

#### M3-C: 피드 & 소셜 (Strava형)

**Feed Screen 개편:**
- 내 성장 카드 타임라인
- 친구/팔로워 훈련 기록 (Phase 3 후반)
- 챌린지: "이번 주 5회 훈련" 등 공개 챌린지

---

## 4. 우선순위 요약

```
Phase 1 (즉시 착수)          Phase 2 (Phase 1 완료 후)     Phase 3 (선택)
─────────────────────        ──────────────────────────    ─────────────────
M1-A: 홈 대시보드 개편        M2-A: AI Insight + 성장 카드  M3-A: 페이월
M1-B: 의도 입력 화면          M2-B: 성장 지표 그래프        M3-B: Google 로그인
M1-C: 훈련 중 컬러 + TTS     M2-C: 모드별 특화 콘텐츠      M3-C: 소셜 피드
```

---

## 5. 의존성 맵

```
M1-A (홈 모드 버튼)
  └──▶ M1-B (의도 입력) → M1-C (훈련 중 UX)
            │
            ▼
       M2-A (AI Insight + 카드)
            │
            ▼
       M2-B (성장 그래프) ←── M2-C (모드 특화)
            │
            ▼
       M3-A (페이월) → M3-B (구글 로그인) → M3-C (소셜)
```

---

## 6. 신규 화면 목록

| 화면 | 상태 | 위치 |
|-----|------|------|
| `IntentScreen` | 신규 생성 | SessionSetup → Intent → PlanPreview |
| `GrowthCardScreen` | SessionEndScreen 개편 | 세션 완료 후 |
| `ProPaywallModal` | 신규 생성 | Pro 기능 진입 시 |
| `LoginScreen` | 신규 생성 (Phase 3) | 앱 최초 실행 |

---

## 7. 신규 API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/sessions/plan` | mode + intent_tags 파라미터 추가 |
| `GET` | `/sessions/{id}/insight` | AI Insight + 배지 반환 |
| `GET` | `/history/growth-metrics` | 스피드/파워/지구력 트렌드 |
| `POST` | `/auth/google` | Google OAuth (Phase 3) |

---

## 8. 오픈 퀘스천

- [ ] **리듬 TTS 엔진**: expo-speech의 rate/pitch 파라미터로 충분한가, 아니면 ElevenLabs 같은 클라우드 TTS가 필요한가?
- [ ] **성장 카드 공유**: react-native-view-shot + expo-sharing 조합, 인스타그램 스토리 직접 연동 여부
- [ ] **AI Insight**: deterministic 로직만으로 충분한가, Pro 유저에게만 LLM insight 제공?
- [ ] **페이월 결제 시스템**: RevenueCat vs 직접 구현

## Changelog

| Date | Change |
|------|--------|
| 2026-03-14 | 초안 작성 — 새 PRD/UI UX 기반 v2 로드맵 |
