# Spec: Phase 2 — Drill Quality, Custom Sessions & Data Enrichment

> Status: DRAFT
> Created: 2026-03-11
> Last Updated: 2026-03-11
> Depends on: Phase 1 (MLP v1) — complete

## 1. Objective & Scope

### Why

Phase 1 delivered a working drill coach with LLM session planning, streaks, milestones, and a profile dashboard. Three foundational gaps remain:

1. **데이터 빈약** — 수집 데이터가 훈련의 양(세션 수, 콤보 전달 횟수)에 한정. 질(난이도, 숙련도, 컨디션)에 대한 신호가 없어 LLM 코칭이 generic해질 수밖에 없다.
2. **드릴 다양성 부족** — 템플릿 3개, 시드 콤보 12개로는 변주 폭이 좁다. 2주 사용 후 "매번 같은 느낌"이 된다.
3. **무료 가치 부족** — LLM 기능을 유료화하면 무료 유저에게 남는 가치가 없다. 무료에서도 충분히 쓸만한 제품이어야 유료 전환 동기가 생긨다.

### What

Phase 2는 4개 워크스트림으로 구성된다:

| Workstream | Tier | Summary |
|---|---|---|
| **A. Drill Foundation** | Free | 템플릿 확장, 콤보 확장, 태그 시스템, 커스텀 세션 |
| **B. Data Enrichment** | Free | 라운드 난이도 피드백, 세션 RPE, 콤보 자기평가 |
| **C. Coach Enhancement** | Paid | LLM 프롬프트 강화, 코치 스타일, 피처 게이팅 |
| **D. Infrastructure** | — | DB 변경, API 추가, 프로필 집계 개선 |

### Session Creation 3-Tier Model

Phase 2 완료 후 사용자가 세션을 시작하는 3가지 경로:

```
┌─ FREE ─────────────────────────────────────────────┐
│                                                     │
│  1. 기본 세션 (Default)                              │
│     템플릿 선택 → 개선된 자동 생성 (deterministic)    │
│     - 8개 템플릿 중 선택                              │
│     - 태그 기반 콤보 필터링                           │
│     - Progressive difficulty, 반복 방지               │
│                                                     │
│  2. 커스텀 세션 (Custom)                              │
│     직접 구성 → 저장 → 재사용                         │
│     - 라운드/시간/휴식/페이스 설정                    │
│     - 콤보 직접 선택                                  │
│     - 이름 붙여서 저장                                │
│                                                     │
└─────────────────────────────────────────────────────┘

┌─ PAID (LLM) ───────────────────────────────────────┐
│                                                     │
│  3. AI 코치 세션 (Coach)                             │
│     템플릿 + 자연어 요청 → LLM 맞춤 생성             │
│     - 히스토리 + 난이도 데이터 기반                   │
│     - 코치 메시지 + 세션 리뷰                        │
│     - 코치 스타일 선택                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 2. Free / Paid Boundary

### Free Tier

| Feature | Description |
|---|---|
| 8개 템플릿 | 시스템 프리셋 세션 (deterministic 생성) |
| 25+ 시드 콤보 | 확장된 콤보 라이브러리 |
| 커스텀 콤보 생성 | 직접 콤보 만들기 (기존 기능) |
| 커스텀 세션 빌더 | 라운드/시간/콤보 직접 구성 + 저장 |
| 마이크로 피드백 | 라운드 난이도, 세션 RPE, 콤보 평가 |
| 프로필 & 통계 | 히트맵, 마일스톤, 스트릭 (기존 기능) |
| TTS 음성 안내 | 세션 중 음성 콤보 호출 (기존 기능) |

### Paid Tier (LLM)

| Feature | Description |
|---|---|
| AI 코치 세션 | LLM 기반 맞춤 세션 생성 |
| 코치 메시지 | 세션 전 코치 인사 + 세션 후 리뷰 |
| 코치 스타일 | 엄격/친근/분석적 중 선택 |
| 자연어 요청 | "오늘 디펜스 위주로" 같은 자유 입력 |
| 데이터 기반 코칭 | 난이도 히스토리, RPE 트렌드 참조 |

### Monetization (결정 필요)

수익 모델은 구독 또는 크레딧 중 미정. 이 spec에서는 feature gating 로직만 정의한다. 결제 시스템은 별도 spec.

---

## 3. Workstream A: Drill Foundation

### 3.1 Template Diversification (3 → 8)

기존 3개 템플릿에 5개 추가.

#### Length vs Difficulty — 콤보 속성 분리

기존 `complexity = len(actions)` 방식을 폐기하고 2가지 독립 축으로 분리:

| 축 | 필드 | 소속 | 의미 | 예시 |
|---|---|---|---|---|
| **Length** | `length` | Combination | 콤보의 액션 수 (자동 계산) | 원투=2, 원투훅=3 |
| **Difficulty** | `difficulty` | Combination | 기술 난이도 (1-5, 수동 부여) | 더블잽=1, 슬립크로스=3, 원투슬립크로스=4 |

**왜 분리하는가:**
- `jab, jab` (더블잽): length=2, difficulty=**1** — 같은 동작 반복, 쉬움
- `slip, cross` (슬립크로스): length=2, difficulty=**3** — 방어→공격 전환, 타이밍 필요
- 기존 complexity=len(actions)로는 이 둘이 같은 난이도로 취급됨

#### Weight — 템플릿이 정의하는 맥락적 가중치

**Weight는 콤보의 고정 속성이 아니라 템플릿이 정의한다.** 같은 콤보라도 템플릿에 따라 선택 빈도가 달라야 한다:

- **잽** → 기본기 템플릿에서는 핵심 (weight=3), 파워 템플릿에서는 보조 (weight=1)
- **덕킹어퍼** → 디펜스 템플릿에서는 핵심 (weight=3), 스피드 템플릿에서는 부적합 (필터링됨)

**구현:** `SessionTemplate`에 `tag_weights` 컬럼 추가 (JSON: `{tag: weight}`).
콤보의 유효 weight는 세션 생성 시 동적으로 계산:

```python
def effective_weight(combo: Combination, template: SessionTemplate) -> int:
    """콤보의 태그 중 template.tag_weights에 매칭되는 최대 weight 반환. 매칭 없으면 기본값 1."""
    if not template.tag_weights:
        return 1
    matched = [template.tag_weights[t] for t in combo.tags if t in template.tag_weights]
    return max(matched) if matched else 1
```

**템플릿별 tag_weights 예시:**

| Template | tag_weights | 효과 |
|---|---|---|
| 기본기 | `{"fundamental": 3}` | 기본 콤보 자주, 나머지 가끔 |
| 스피드 | `{"speed": 3, "fundamental": 2}` | 짧고 빠른 콤보 위주 |
| 파워 | `{"power": 3, "body": 2}` | 강타 + 바디 콤보 위주 |
| 디펜스 | `{"defense": 3, "counter": 3}` | 방어/카운터 콤보 위주 |
| 컨디셔닝 | `{"fundamental": 2, "speed": 2}` | 고르게, 빠른 호출 |
| 테크니컬 | `{"fundamental": 3}` | 기본기 깊이 반복 |
| 콤비네이션 | `{"fundamental": 2}` | 다양한 조합, 기본기에 약간 가중 |
| 종합 | `{}` | 모든 콤보 균등 (weight=1) |

#### Template 정의

`combo_include_defense` boolean과 `combo_complexity_range`를 폐기하고 **태그 기반 필터링**으로 통합:

| # | name | display_name | Rounds | Duration | Rest | Pace | required_tags | difficulty_range | tag_weights | 설명 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | fundamentals | 기본기 | 3 | 120s | 45s | 1-3s | fundamental | [1, 2] | `{"fundamental": 3}` | *기존, 마이그레이션* |
| 2 | combos | 콤비네이션 | 4 | 150s | 60s | 1-3s | — | [2, 4] | `{"fundamental": 2}` | *기존, 마이그레이션* |
| 3 | mixed | 종합 | 5 | 180s | 60s | 1-2s | — | [1, 5] | `{}` | *기존, 마이그레이션* |
| 4 | **speed** | **스피드** | 4 | 90s | 30s | **0.5-1.5s** | speed | [1, 3] | `{"speed": 3, "fundamental": 2}` | 짧은 간격, 빠른 반응 |
| 5 | **power** | **파워** | 3 | 120s | 60s | **3-5s** | power | [2, 4] | `{"power": 3, "body": 2}` | 느린 페이스, 강타 집중 |
| 6 | **defense** | **디펜스** | 4 | 120s | 45s | 2-3s | defense | [2, 4] | `{"defense": 3, "counter": 3}` | 방어 + 카운터 패턴 |
| 7 | **conditioning** | **컨디셔닝** | 6 | **60s** | **20s** | **0.5-1.5s** | — | [1, 3] | `{"fundamental": 2, "speed": 2}` | 짧은 라운드, 짧은 휴식, 고밀도 |
| 8 | **technical** | **테크니컬** | 3 | 180s | 60s | **4-6s** | fundamental | [1, 2] | `{"fundamental": 3}` | 하나의 콤보 깊이 반복 |

**Schema change**: `SessionTemplate` 컬럼 변경:
- `combo_complexity_range` → `difficulty_range` (JSON [min, max]) — difficulty 기반 필터링
- `combo_include_defense` → **삭제** — `required_tags`로 대체
- `required_tags` (JSON array, nullable) — null이면 태그 필터링 없음
- `tag_weights` (JSON dict, default={}) — **NEW** — 태그별 선택 빈도 가중치

**콤보 필터링 + 가중치 로직:**
```python
# Step 1: 태그 + 난이도 필터링
eligible_combos = all_combos
if template.required_tags:
    eligible_combos = [c for c in combos if any(t in c.tags for t in template.required_tags)]
eligible_combos = [c for c in eligible_combos if template.difficulty_range[0] <= c.difficulty <= template.difficulty_range[1]]

# Step 2: 가중치 계산
for combo in eligible_combos:
    combo.effective_weight = effective_weight(combo, template)  # 1-3

# Step 3: 가중치 기반 확률 선택
selected = random.choices(eligible_combos, weights=[c.effective_weight for c in eligible_combos])
```

#### Deterministic plan 개선 (free tier)

현재 `_fallback_plan()`은 콤보를 무작위로 선택한다. 다음 규칙으로 개선:
1. `required_tags` + `difficulty_range`로 후보 콤보 필터링
2. 라운드별 progressive difficulty (라운드 1은 낮은 difficulty, 마지막 라운드는 높은 difficulty)
3. **tag_weights 기반 확률 선택** — `effective_weight(combo, template)` 계산 후 가중치 반영
4. 연속 동일 콤보 방지 (직전 콤보와 다른 콤보 선택)
5. combo_exposure 참조: under-practiced 콤보에 추가 가중치

### 3.2 Seed Combo Expansion (12 → 28)

기존 12개 + 신규 16개:

**기존 (1-12) — length, difficulty, tags 재부여:**

| # | display_name | actions | length | difficulty | tags |
|---|---|---|---|---|---|
| 1 | 잽 | [jab] | 1 | 1 | fundamental |
| 2 | 크로스 | [cross] | 1 | 1 | fundamental |
| 3 | 원투 | [jab, cross] | 2 | 1 | fundamental |
| 4 | 더블잽 | [jab, jab] | 2 | 1 | fundamental, speed |
| 5 | 리드훅바디 | [lead_hook, rear_bodyshot] | 2 | 3 | power, body |
| 6 | 원투훅 | [jab, cross, lead_hook] | 3 | 2 | fundamental |
| 7 | 원투바디 | [jab, cross, rear_bodyshot] | 3 | 2 | body |
| 8 | 잽잽크로스 | [jab, jab, cross] | 3 | 1 | speed |
| 9 | 슬립원투 | [slip, jab, cross] | 3 | 3 | defense, counter |
| 10 | 덕킹원투 | [duck, jab, cross] | 3 | 3 | defense, counter |
| 11 | 원투쓰리투 | [jab, cross, lead_hook, cross] | 4 | 3 | fundamental |
| 12 | 원투바디훅 | [jab, cross, rear_bodyshot, lead_hook] | 4 | 3 | body |

**신규 (13-28):**

| # | display_name | actions | length | difficulty | tags |
|---|---|---|---|---|---|
| 13 | 리드어퍼 | [lead_uppercut] | 1 | 2 | power |
| 14 | 리어어퍼 | [rear_uppercut] | 1 | 2 | power |
| 15 | 원투어퍼 | [jab, cross, rear_uppercut] | 3 | 3 | power |
| 16 | 잽어퍼크로스 | [jab, lead_uppercut, cross] | 3 | 3 | power |
| 17 | 바디바디헤드 | [rear_bodyshot, lead_bodyshot, lead_hook] | 3 | 3 | body, power |
| 18 | 슬립크로스 | [slip, cross] | 2 | 3 | defense, counter, speed |
| 19 | 슬립크로스훅 | [slip, cross, lead_hook] | 3 | 3 | defense, counter |
| 20 | 덕킹어퍼 | [duck, rear_uppercut] | 2 | 3 | defense, counter, power |
| 21 | 백스텝원투 | [backstep, jab, cross] | 3 | 3 | defense, counter |
| 22 | 원투슬립크로스 | [jab, cross, slip, cross] | 4 | 4 | defense, counter |
| 23 | 원투훅어퍼 | [jab, cross, lead_hook, rear_uppercut] | 4 | 4 | power |
| 24 | 잽바디크로스 | [jab, rear_bodyshot, cross] | 3 | 2 | body |
| 25 | 더블잽크로스훅 | [jab, jab, cross, lead_hook] | 4 | 2 | speed |
| 26 | 원투바디바디 | [jab, cross, rear_bodyshot, lead_bodyshot] | 4 | 3 | body |
| 27 | 원투훅바디크로스 | [jab, cross, lead_hook, rear_bodyshot, cross] | 5 | 4 | body, power |
| 28 | 슬립슬립백스텝 | [slip, slip, backstep] | 3 | 2 | defense |

**Difficulty 부여 기준:**

| Difficulty | 기준 | 예시 |
|---|---|---|
| 1 | 단일 기본 동작 또는 같은 동작 반복 | 잽, 더블잽, 잽잽크로스 |
| 2 | 기본 콤비네이션, 자연스러운 연결 | 원투훅, 원투바디, 리드어퍼 |
| 3 | 방어↔공격 전환, 비직관적 연결, 바디워크 | 슬립크로스, 덕킹어퍼, 원투쓰리투 |
| 4 | 긴 시퀀스 + 전환 포함, 고급 패턴 | 원투슬립크로스, 원투훅어퍼 |
| 5 | 5+ 액션, 다중 전환, 최상급 | 원투훅바디크로스 |

### 3.3 Combo Tag System + Schema Changes

`Combination` 테이블 변경:

```python
# Combination 테이블 — 변경/추가 컬럼
complexity → 삭제 (length로 대체, 자동 계산)
length: Mapped[int] = mapped_column(Integer, nullable=False)       # len(actions), 자동 계산
difficulty: Mapped[int] = mapped_column(Integer, nullable=False)   # 1-5, 수동 부여
tags: Mapped[dict] = mapped_column(JSON, default=list)             # list[str]
# weight는 Combination에 없음 — SessionTemplate.tag_weights에서 맥락적으로 결정
```

**Migration:** 기존 `complexity` 값은 `length`로 복사. `difficulty`는 시드 데이터 기반으로 부여. 사용자 생성 콤보는 `difficulty = length` (heuristic).

**Tags:**

| Tag | 설명 | 예시 콤보 |
|---|---|---|
| `fundamental` | 기본기 빌딩블록 | 잽, 원투, 원투훅 |
| `power` | 강타 위주 (훅, 어퍼, 바디) | 리드어퍼, 원투어퍼 |
| `speed` | 짧고 빠른 시퀀스 | 더블잽, 잽잽크로스 |
| `defense` | 방어 동작 포함 | 슬립원투, 덕킹어퍼 |
| `counter` | 방어 후 공격 패턴 | 슬립크로스, 백스텝원투 |
| `body` | 바디샷 포함 | 원투바디, 바디바디헤드 |

**사용처:**
- `SessionTemplate.required_tags`에 태그 목록 지정 → 해당 태그의 콤보만 세션에 포함
- `SessionTemplate.difficulty_range`로 난이도 범위 필터링
- `weight` 기반 확률적 콤보 선택 (deterministic plan, LLM 참고 정보)
- LLM 프롬프트에 태그 + difficulty + weight 정보 포함
- 커스텀 세션 빌더에서 태그 기반 필터링 UI

### 3.4 Custom Session Builder

사용자가 직접 세션을 구성하고 저장/재사용하는 기능.

#### 3.4.1 Custom Session Schema

```python
class CustomSession(Base):
    __tablename__ = "custom_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    rounds: Mapped[int] = mapped_column(Integer, nullable=False)          # 1-12
    round_duration_sec: Mapped[int] = mapped_column(Integer, nullable=False)  # 60-180
    rest_sec: Mapped[int] = mapped_column(Integer, nullable=False)        # 15-60
    pace: Mapped[str] = mapped_column(String(20), nullable=False)         # slow/normal/fast
    combo_ids: Mapped[dict] = mapped_column(JSON, nullable=False)         # list[str] combo display_names
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
```

**Pace 매핑:**

| Pace | Interval (sec) | 설명 |
|---|---|---|
| slow | [4, 6] | 충분한 시간, 폼 집중 |
| normal | [2, 4] | 기본 페이스 |
| fast | [0.5, 1.5] | 반응 속도 훈련 |

#### 3.4.2 Custom Session → Plan 변환

커스텀 세션은 LLM 없이 deterministic하게 PlanDetail로 변환된다:

```
CustomSession config
  → 라운드별 콤보 배분 (선택된 콤보를 라운드에 균등 분포, 연속 반복 방지)
  → pace 설정에 따른 interval 계산
  → PlanDetail JSON (ActiveSessionScreen과 동일 포맷)
```

**변환 규칙:**
1. 선택된 콤보를 shuffle하여 라운드에 균등 분배
2. 라운드 내에서 같은 콤보 연속 배치 금지
3. `coach_message`, `session_review`는 빈 문자열 (무료 — 코치 기능 없음)
4. `session_type`은 `"custom"`, `template`은 `"custom"`
5. 반환 포맷이 기존 PlanDetail과 동일하므로 ActiveSessionScreen 변경 불필요

#### 3.4.3 API

```
GET    /api/custom-sessions           → list[CustomSessionResponse]
GET    /api/custom-sessions/{id}      → CustomSessionResponse
POST   /api/custom-sessions           → CustomSessionResponse (201)
PUT    /api/custom-sessions/{id}      → CustomSessionResponse
DELETE /api/custom-sessions/{id}      → 204
POST   /api/custom-sessions/{id}/plan → PlanResponse (plan 생성 + 실행)
```

`POST /api/custom-sessions/{id}/plan`:
- 커스텀 세션 config를 읽어서 deterministic plan 생성
- DrillPlan 테이블에 저장 (template_id = custom_session.id, llm_model = "deterministic")
- 기존 PlanResponse 형식으로 반환
- ActiveSessionScreen에서 동일하게 실행 가능

#### 3.4.4 Mobile UI: Custom Session Builder

**SessionSetupScreen 개편:**

현재: 3개 템플릿 선택 + 프롬프트 입력
변경: 3가지 모드 선택

```
┌─────────────────────────────────────────┐
│  세션 시작                               │
│                                         │
│  ┌─ 기본 세션 ─────────────────────┐    │
│  │  시스템 템플릿으로 빠르게 시작    │    │
│  │  [기본기] [스피드] [파워] ...    │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─ 커스텀 세션 ───────────────────┐    │
│  │  직접 구성하거나 저장된 세션 선택  │    │
│  │  [+ 새로 만들기]                 │    │
│  │  [저장된 세션 1] [저장된 세션 2]  │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─ AI 코치 세션 ──── PRO ─────────┐    │
│  │  AI가 맞춤 세션을 생성합니다      │    │
│  │  [템플릿 선택] + [요청 입력]     │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**Custom Session Builder (새로 만들기):**

```
┌─────────────────────────────────────────┐
│  커스텀 세션 만들기                       │
│                                         │
│  세션 이름                               │
│  [___________________________]          │
│                                         │
│  ── ROUNDS ──                           │
│  라운드 수     [  3  ] [-] [+]          │
│  라운드 시간   [ 120 ]초 [-] [+]        │
│  휴식 시간     [  45 ]초 [-] [+]        │
│                                         │
│  ── PACE ──                             │
│  [ 느림 ]  [ 보통 ✓]  [ 빠름 ]          │
│                                         │
│  ── COMBOS ──                           │
│  태그 필터: [전체] [기본] [파워] [방어]   │
│                                         │
│  ☑ 잽                    (1)            │
│  ☑ 원투                   (2)            │
│  ☐ 더블잽                 (2)            │
│  ☑ 원투훅                 (3)            │
│  ☐ 슬립원투              (3)            │
│  ...                                    │
│                                         │
│  선택됨: 5개 콤보                        │
│                                         │
│  [저장 & 시작]                           │
└─────────────────────────────────────────┘
```

---

## 4. Workstream B: Data Enrichment

### 4.1 Round Difficulty Rating

**What:** 각 라운드 종료 후 1탭 난이도 평가.

**UX:**
```
┌─────────────────────────────────────┐
│                                     │
│        라운드 2 완료                 │
│                                     │
│   [ 쉬움 ]  [ 적절 ]  [ 힘듦 ]      │
│                                     │
│        3초 후 자동 닫힘              │
│                                     │
└─────────────────────────────────────┘
```

- ActiveSessionScreen에서 라운드 종료 → 휴식 시작 사이에 오버레이 표시
- 3개 버튼: `easy` / `good` / `hard`
- 3초 타임아웃, 미응답 시 null (피드백 강제 아님)
- 마지막 라운드 후에도 표시 (세션 종료 직전)

**Storage:**
`delivery_log_json` 이벤트에 추가:
```json
{
  "type": "round_feedback",
  "ts": 1234567890.0,
  "round": 2,
  "difficulty": "hard"   // "easy" | "good" | "hard" | null
}
```

기존 `delivery_log_json` 포맷에 새 이벤트 타입을 추가하는 것이므로 스키마 변경 없음.

### 4.2 Session RPE (운동 자각도)

**What:** 세션 완료 후 전체 운동 강도 평가.

**UX:**
SessionEndScreen에서 stats 표시 전에 RPE 입력:

```
┌─────────────────────────────────────┐
│                                     │
│           ✓                         │
│     수고하셨습니다.                  │
│                                     │
│     오늘 운동 강도는 어땠나요?       │
│                                     │
│   [ 가벼움 ]  [ 적당 ]  [ 힘들었음 ] │
│                                     │
│            건너뛰기                  │
│                                     │
└─────────────────────────────────────┘
```

- 3단계: `light` (1-3) / `moderate` (4-6) / `hard` (7-10)
- "건너뛰기" 옵션 (nullable)
- 선택 후 기존 SessionEndScreen 통계 화면으로 전환

**Storage:**
`SessionLog`에 새 컬럼 추가:
```python
rpe: Mapped[str | None] = mapped_column(String(20), default=None)  # light/moderate/hard
```

`SessionLogRequest`에 `rpe` 필드 추가 (optional).

### 4.3 Combo Self-Assessment (콤보 자기평가)

**What:** 세션 종료 후 수행한 콤보에 대한 자기평가.

**UX:**
SessionEndScreen에서 RPE 입력 후, stats 화면에 평가 섹션 추가:

```
┌─────────────────────────────────────┐
│  ── 콤보 평가 (선택) ──              │
│                                     │
│  이번 세션에서 수행한 콤보:           │
│                                     │
│  원투         [ 👍 ] [ 👎 ]         │
│  원투훅       [ 👍 ] [ 👎 ]         │
│  슬립원투     [ 👍 ] [ 👎 ]         │
│  원투바디훅   [ 👍 ] [ 👎 ]         │
│                                     │
│  [제출]  또는  [건너뛰기]            │
└─────────────────────────────────────┘
```

- 세션에서 사용된 **고유 콤보** 목록 표시 (중복 제거)
- 각 콤보에 👍 (nailed) / 👎 (struggled) 2택
- 미평가 콤보는 null (부분 평가 허용)
- "건너뛰기"로 전체 스킵 가능

**Storage:**
`SessionLogRequest`에 `combo_feedback` 필드 추가:
```json
{
  "combo_feedback": {
    "원투": "nailed",
    "원투훅": "struggled",
    "슬립원투": null
  }
}
```

`SessionLog`에 새 컬럼:
```python
combo_feedback_json: Mapped[dict | None] = mapped_column(JSON, default=None)
```

**Profile 집계에 미치는 영향:**
현재 `combo_exposure_json`은 `{콤보명: 횟수}` 형태.
이를 확장하여 mastery 신호를 포함:

```json
{
  "원투": {
    "count": 45,
    "nailed": 30,
    "struggled": 5,
    "mastery_rate": 0.86
  }
}
```

`mastery_rate = nailed / (nailed + struggled)` (평가 안 한 세션은 제외)

---

## 5. Workstream C: Coach Enhancement

### 5.1 Enhanced LLM Context

LLM 프롬프트에 다음 데이터를 추가:

**난이도 히스토리 (최근 5세션):**
```
## Recent Difficulty Feedback
- Session 2026-03-10 (combos): Round 1 easy, Round 2 good, Round 3 hard, Round 4 hard. RPE: hard
- Session 2026-03-09 (fundamentals): Round 1 easy, Round 2 easy, Round 3 good. RPE: light
→ 사용자가 combos 템플릿에서 후반 라운드를 어려워함. 난이도 조절 필요.
```

**콤보 숙련도 (mastery_rate 기반):**
```
## Combo Mastery
- Struggling combos (mastery < 50%): 슬립원투 (35%), 원투바디훅 (40%)
- Strong combos (mastery > 80%): 원투 (92%), 잽잽크로스 (85%)
→ 약한 콤보를 적절히 포함하되, 한 세션에 2개 이하로 제한.
```

**RPE 트렌드:**
```
## Training Load
- Last 5 sessions RPE: light, moderate, hard, hard, moderate
- Average: moderate-hard
→ 오늘은 moderate 이하로 조절 권장.
```

### 5.2 Coach Styles

사용자가 코치 스타일을 선택하면 `SYSTEM_PROMPT`의 성격 섹션이 변경됨.

| Style | name | 말투 | 특징 |
|---|---|---|---|
| **엄격한 코치** | strict | 단호하고 직접적 | 약점 집중, 칭찬 절제, 목표 지향 |
| **친근한 코치** | friendly | 따뜻하고 격려적 | 작은 성장도 칭찬, 감정적 지지, 재미 강조 |
| **분석적 코치** | analytical | 데이터 중심, 객관적 | 숫자 인용, 비교 분석, 개선점 구체적 제시 |

**Storage:**
`UserProfile`에 새 컬럼:
```python
coach_style: Mapped[str] = mapped_column(String(20), default="friendly")
```

**프롬프트 예시 (strict):**
```
당신은 엄격하지만 실력 있는 복싱 코치입니다. 쓸데없는 칭찬은 하지 않습니다.
사용자의 약점을 정확히 짚어주고, 구체적인 개선 방향을 제시합니다.
"잘했다"보다는 "이건 부족하다, 이렇게 고쳐라"를 선호합니다.
단, 진짜 잘한 부분에 대해서는 인정합니다.
```

**프롬프트 예시 (friendly):**
```
당신은 따뜻하고 격려적인 복싱 코치입니다. 사용자의 작은 성장도 잘 발견해서 칭찬합니다.
훈련이 즐거운 경험이 되도록 분위기를 만듭니다.
어려운 부분은 부담 없이 "천천히 해보자"라는 톤으로 전달합니다.
사용자가 꾸준히 돌아오고 싶게 만드는 것이 목표입니다.
```

**프롬프트 예시 (analytical):**
```
당신은 데이터 중심의 분석적 복싱 코치입니다. 감정보다 숫자로 말합니다.
"지난 5세션 대비 훅 콤보 mastery가 40%에서 65%로 상승했습니다" 같은 구체적 피드백을 선호합니다.
개선 포인트는 데이터 근거와 함께 제시합니다.
```

### 5.3 Feature Gating

**구현 방식:** 서버 사이드 간단한 boolean 체크.

```python
# UserProfile에 추가
is_premium: Mapped[bool] = mapped_column(Boolean, default=False)
```

**Gating 규칙:**

| Endpoint | Free | Paid |
|---|---|---|
| `POST /api/sessions/plan` | `llm_model="deterministic"` (fallback only) | LLM 호출 허용 |
| `POST /api/custom-sessions/{id}/plan` | 허용 (항상 deterministic) | 허용 |
| Coach message in plan | 빈 문자열 | LLM 생성 |
| Session review in plan | 빈 문자열 | LLM 생성 |
| Coach style selection | 표시 안 함 | 선택 가능 |
| User prompt input | 표시 안 함 | 입력 가능 |

**Free tier `POST /api/sessions/plan`:**
- LLM을 호출하지 않고 개선된 deterministic 알고리즘 사용
- 태그 필터링, progressive difficulty, 반복 방지 적용
- `coach_message`와 `session_review`는 빈 문자열

---

## 6. Data Contracts

### 6.1 Modified Tables

#### SessionLog (modified)

| Field | Type | Change | Mutable |
|---|---|---|---|
| rpe | String(20), nullable | **NEW** | No (append-only) |
| combo_feedback_json | JSON, nullable | **NEW** | No (append-only) |

#### Combination (modified)

| Field | Type | Change | Mutable |
|---|---|---|---|
| complexity | — | **REMOVED** (length로 대체) | — |
| length | Integer, not null | **NEW** (= len(actions), 자동 계산) | No |
| difficulty | Integer, not null | **NEW** (1-5, 기술 난이도) | Yes |
| tags | JSON, default=[] | **NEW** | Yes |

#### SessionTemplate (modified)

| Field | Type | Change | Mutable |
|---|---|---|---|
| combo_complexity_range | — | **REMOVED** (difficulty_range로 대체) | — |
| combo_include_defense | — | **REMOVED** (required_tags로 대체) | — |
| difficulty_range | JSON [min, max], not null | **NEW** | No (system-defined) |
| required_tags | JSON array, nullable | **NEW** (null = 태그 필터 없음) | No (system-defined) |
| tag_weights | JSON dict, default={} | **NEW** (태그별 선택 빈도 가중치) | No (system-defined) |

#### UserProfile (modified)

| Field | Type | Change | Mutable |
|---|---|---|---|
| coach_style | String(20), default="friendly" | **NEW** | Yes |
| is_premium | Boolean, default=False | **NEW** | Yes |
| combo_mastery_json | JSON, default={} | **NEW** (replaces combo_exposure_json logic) | Derived |

### 6.2 New Tables

#### CustomSession

| Field | Type | Required | Mutable |
|---|---|---|---|
| id | String(36), PK | auto | No |
| display_name | String(200) | Yes | Yes |
| rounds | Integer | Yes | Yes |
| round_duration_sec | Integer | Yes | Yes |
| rest_sec | Integer | Yes | Yes |
| pace | String(20) | Yes | Yes |
| combo_ids | JSON (list[str]) | Yes | Yes |
| created_at | DateTime(tz) | auto | No |
| updated_at | DateTime(tz) | auto | auto |

### 6.3 Profile Aggregation Changes

`combo_mastery_json` 구조 변경:
```json
// Before (combo_exposure_json)
{"원투": 45, "원투훅": 20}

// After (combo_mastery_json)
{
  "원투": {"count": 45, "nailed": 30, "struggled": 5, "mastery_rate": 0.86},
  "원투훅": {"count": 20, "nailed": 8, "struggled": 6, "mastery_rate": 0.57}
}
```

`combo_exposure_json`은 하위 호환을 위해 유지하되, 새로운 `combo_mastery_json`을 primary로 사용. `mastery_rate`는 `combo_feedback_json`이 있는 세션에서만 계산.

**난이도 히스토리** — 별도 저장하지 않고 `delivery_log_json`의 `round_feedback` 이벤트에서 on-demand로 추출. 프로필에 집계 필드 추가하지 않음 (LLM 프롬프트 빌드 시 최근 세션에서 직접 추출).

---

## 7. API Changes

### 7.1 New Endpoints

```
# Custom Sessions CRUD
GET    /api/custom-sessions              → list[CustomSessionResponse]
GET    /api/custom-sessions/{id}         → CustomSessionResponse
POST   /api/custom-sessions              → CustomSessionResponse (201)
PUT    /api/custom-sessions/{id}         → CustomSessionResponse
DELETE /api/custom-sessions/{id}         → 204

# Custom Session Plan Generation
POST   /api/custom-sessions/{id}/plan    → PlanResponse
```

### 7.2 Modified Endpoints

```
# SessionLog — add rpe, combo_feedback
POST   /api/sessions/log
  Request body adds: rpe? (string), combo_feedback? (dict)

# Plan Generation — gating
POST   /api/sessions/plan
  Free: deterministic plan (improved fallback)
  Paid: LLM plan (enhanced context)

# Profile — add new fields
GET    /api/profile
  Response adds: coach_style, is_premium, combo_mastery_json

PUT    /api/profile
  Body adds: coach_style? (strict|friendly|analytical)
```

### 7.3 New Schemas

```python
class CustomSessionCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=200)
    rounds: int = Field(ge=1, le=12)
    round_duration_sec: int = Field(ge=60, le=180)
    rest_sec: int = Field(ge=15, le=60)
    pace: Literal["slow", "normal", "fast"]
    combo_ids: list[str] = Field(min_length=1)  # combo display_names

class CustomSessionResponse(BaseModel):
    id: str
    display_name: str
    rounds: int
    round_duration_sec: int
    rest_sec: int
    pace: str
    combo_ids: list[str]
    created_at: datetime
    updated_at: datetime

class SessionLogRequest(BaseModel):  # modified
    # ... existing fields ...
    rpe: str | None = None                    # NEW: light/moderate/hard
    combo_feedback: dict[str, str] | None = None  # NEW: {combo: nailed/struggled}
```

---

## 8. Mobile UI Changes

### 8.1 SessionSetupScreen (Major Rewrite)

현재: 단일 모드 (템플릿 선택 + 프롬프트)
변경: 3-모드 탭 (기본 세션 / 커스텀 세션 / AI 코치)

**기본 세션 탭:**
- 8개 템플릿 카드 (2열 그리드)
- 선택 → 즉시 plan 생성 (deterministic) → PlanPreview

**커스텀 세션 탭:**
- 저장된 커스텀 세션 리스트
- "새로 만들기" 버튼 → CustomSessionBuilder (새 화면 or 모달)
- 선택 → plan 생성 → PlanPreview

**AI 코치 탭 (PRO 뱃지):**
- 기존 템플릿 선택 + 프롬프트 입력 UI
- 미결제 시 잠금 상태 + 안내 메시지

### 8.2 ActiveSessionScreen (Minor Addition)

**라운드 피드백 오버레이 추가:**
- 라운드 종료 시 3초간 난이도 선택 오버레이 표시
- 3개 버튼 (쉬움/적절/힘듦)
- 선택 or 타임아웃 후 휴식 화면으로 전환
- 피드백은 로컬에 저장, 세션 종료 시 `delivery_log_json`에 포함하여 전송

### 8.3 SessionEndScreen (Addition)

**RPE 입력 스텝 추가:**
- 기존 통계 화면 전에 RPE 선택 화면 삽입
- 3개 버튼 (가벼움/적당/힘들었음) + 건너뛰기
- 선택 후 기존 통계 화면으로

**콤보 평가 섹션 추가:**
- 통계 카드 아래에 콤보 평가 섹션
- 세션에서 사용된 고유 콤보 리스트
- 각 콤보에 👍/👎 토글
- 제출 or 건너뛰기

### 8.4 New Screens

**CustomSessionBuilderScreen:**
- 세션 이름 입력
- 라운드 설정 (수, 시간, 휴식) — 스텝퍼 UI
- 페이스 선택 (3버튼)
- 콤보 선택 (태그 필터 + 체크박스 리스트)
- 선택된 콤보 수 표시
- "저장 & 시작" 버튼

### 8.5 Navigation Changes

```
HomeTab Stack:
  Home
  SessionSetup (rewritten: 3-mode)
  CustomSessionBuilder (NEW)
  PlanPreview
  ActiveSession (round feedback added)
  SessionEnd (RPE + combo assessment added)
  Settings
```

---

## 9. Implementation Tasks

의존 관계 순서로 정렬. 각 태스크는 독립적으로 테스트 가능.

| # | Task | Size | Depends | Description |
|---|---|---|---|---|
| 1 | DB Migration: schema changes + new table | M | — | Combination: complexity→삭제, length/difficulty/tags 추가. SessionTemplate: combo_complexity_range→difficulty_range, combo_include_defense→삭제, required_tags/tag_weights 추가. SessionLog: rpe, combo_feedback_json 추가. UserProfile: coach_style, is_premium, combo_mastery_json 추가. CustomSession 테이블 생성. |
| 2 | Seed data expansion | S | 1 | 기존 3개 템플릿 마이그레이션 (difficulty_range, required_tags, tag_weights). 5개 신규 템플릿 추가. 기존 12개 콤보에 difficulty/tags 부여. 16개 신규 콤보 추가. |
| 3 | Custom session service + API | M | 1 | CustomSession CRUD 서비스. Deterministic plan 생성 로직. API 엔드포인트 구현. |
| 4 | Improved deterministic plan | M | 1, 2 | `_fallback_plan()` 개선: required_tags + difficulty_range 필터링, progressive difficulty, tag_weights 기반 확률 선택, 반복 방지, exposure 가중치. Free tier default 세션의 품질 향상. |
| 5 | Session feedback API | S | 1 | SessionLogRequest에 rpe, combo_feedback 필드 추가. SessionLog 저장 시 새 컬럼 반영. |
| 6 | Profile aggregation update | M | 1, 5 | combo_mastery_json 집계 로직 (nailed/struggled/mastery_rate). combo_exposure_json 하위 호환 유지. |
| 7 | LLM prompt enhancement | M | 5, 6 | 난이도 히스토리, 콤보 숙련도(mastery_rate), RPE 트렌드, difficulty/weight 정보를 프롬프트에 추가. 코치 스타일별 SYSTEM_PROMPT 분기. |
| 8 | Feature gating | S | 7 | is_premium 체크. Free/Paid 분기 로직. Plan generation 게이팅. |
| 9 | Mobile: SessionSetup rewrite | L | 3, 4 | 3-모드 탭 UI. 기본 세션 (8 템플릿). 커스텀 세션 리스트. AI 코치 (잠금 상태 포함). |
| 10 | Mobile: CustomSessionBuilder | L | 3, 9 | 새 화면. 라운드/시간/페이스 설정. 태그 필터 + 콤보 선택 (difficulty 표시). 저장 & 실행. |
| 11 | Mobile: Round feedback | M | 5 | ActiveSessionScreen에 라운드 종료 오버레이 추가. 3초 타임아웃. 로컬 저장. |
| 12 | Mobile: RPE + combo assessment | M | 5, 11 | SessionEndScreen에 RPE 스텝 추가. 콤보 평가 섹션 추가. logSession 요청에 새 필드 포함. |

**총 예상 규모:** S×3 + M×7 + L×2

---

## 10. Migration from Phase 1

### DB Migration

하위 호환 보장:
- 모든 새 컬럼은 `nullable` 또는 `default` 값 보유
- 기존 SessionLog 데이터는 `rpe=null`, `combo_feedback_json=null`
- `combo_exposure_json`은 유지, `combo_mastery_json`은 새로 집계 시작

**Combination 마이그레이션:**
- `complexity` 컬럼 삭제, `length` 컬럼 추가 (기존 complexity 값 복사)
- `difficulty` 컬럼 추가: 시드 콤보는 spec 테이블 값 부여, 사용자 생성 콤보는 `difficulty = length` (heuristic)
- `tags` 컬럼 추가: 시드 콤보는 spec 테이블 값 부여, 사용자 생성 콤보는 `tags = []`

**SessionTemplate 마이그레이션:**
- `combo_complexity_range` → `difficulty_range`로 변환 (기존 [1,2] → [1,2] 등 값 매핑)
- `combo_include_defense` 삭제 → `required_tags`로 대체 (defense=True였던 mixed는 `required_tags=null`로, 나머지도 null)
- `tag_weights` 추가 (기존 3개 템플릿에 spec 테이블 값 부여)
- 기존 3개 템플릿은 새 컬럼으로 마이그레이션, 5개 신규 템플릿 INSERT

### Seed Data

- 기존 3개 템플릿 마이그레이션 (difficulty_range, required_tags 부여)
- 기존 12개 콤보에 length, difficulty, weight, tags 부여 (display_name, actions 변경 없음)
- 5개 신규 템플릿 + 16개 신규 콤보 INSERT

### API 호환

- 기존 API 응답에 새 필드 추가 (additive change)
- 기존 클라이언트가 새 필드를 무시하더라도 동작에 문제 없음
- `rpe`, `combo_feedback`은 optional이므로 기존 요청 포맷도 유효

---

## Changelog

| Date | Change | Reason |
|---|---|---|
| 2026-03-11 | Initial draft | Product discussion: data enrichment, drill quality, custom sessions, free/paid structure |
| 2026-03-12 | complexity→length/difficulty 분리, combo_include_defense→required_tags 전환, tag_weights를 Template로 이동 | Review R1: 콤보 길이 ≠ 기술 난이도, 태그 기반 필터링 통합 |
| 2026-03-12 | weight를 Combination 고정 속성에서 SessionTemplate.tag_weights 맥락적 가중치로 변경 | Review R2: 같은 콤보도 템플릿에 따라 선택 빈도가 달라야 함 |
