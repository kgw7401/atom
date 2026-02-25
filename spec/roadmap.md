# Implementation Roadmap

Version: v0.1
Last Updated: 2026-02-25
Depends on: spec/overview.md, spec/state-vector.md, spec/runtime.md

---

## Current Status

| Phase | Status | Summary |
|-------|--------|---------|
| Phase 1 | **DONE** | LSTM 9-class classifier, 99.7% accuracy, 623 segments |
| Phase 2a | **DONE** | State engine core: vector, observation, EMA update, confidence, DB schema |
| Phase 2b | **DONE** | Vision → State pipeline: keypoint extraction, action classification, end-to-end pipeline |
| Phase 2c+ | NOT STARTED | Policy engine, API, coaching, mobile |

---

## Phase 2 Breakdown

Phase 2 is decomposed into 6 sub-phases.
Each sub-phase has a clear deliverable, verification criteria, and zero external dependencies.

```
2a ──▶ 2b ──▶ 2c ──▶ 2d ──▶ 2e ──▶ 2f
Core   Pipe   Policy  API    Coach  Mobile
```

Strict rule: **다음 phase는 이전 phase의 verification을 통과한 후에만 시작한다.**

---

## Phase 2a: State Engine Core

**Goal:** State vector의 수학적 정의를 코드로 구현하고, 합성 데이터로 검증한다.

### Scope

| Build | NOT Build |
|-------|-----------|
| Observation function (18 dims) | Vision pipeline integration |
| EMA state update | API endpoints |
| Confidence model | Report generation |
| DB schema (user_state, state_transitions) | Mobile app |
| Unit tests with synthetic inputs | |

### Deliverables

```
src/state/
  vector.py           # StateVector dataclass, dimension definitions
  observation.py      # f(segments, keypoints) → ObservationVector
  update.py           # ema_update(S_t, O_t, α) → S_{t+1}
  confidence.py       # confidence_update(n, mask) → C_{t+1}
  constants.py        # All hyperparameters from state-vector.md §9

server/models/
  db.py               # user_state, sessions, state_transitions tables

tests/state/
  test_observation.py
  test_update.py
  test_confidence.py
  test_invariants.py
```

### Verification Criteria

| # | Test | Pass Condition |
|---|------|----------------|
| 1 | Observation from synthetic segments | 18 dims computed, all ∈ [0,1] |
| 2 | Observation with missing data | Partial mask produced, unobserved dims = None |
| 3 | EMA update preserves bounds | S_{t+1} ∈ [0,1]^18 for all valid inputs |
| 4 | EMA update idempotent skip | Unobserved dims: S_{t+1,i} == S_{t,i} |
| 5 | Confidence monotonic | C_{t+1,i} >= C_{t,i} for all i |
| 6 | Initialization from first observation | S_0 = O_0 (observed), S_0 = 0.5 (unobserved) |
| 7 | Determinism | Same input → same output, 100 random seeds |
| 8 | Known-answer test | Hand-computed S_{t+1} matches code output for 3 cases |
| 9 | DB schema migration | Tables created on empty database, no errors |
| 10 | Round-trip persistence | Write S_t → read → compare, zero floating point drift |

### Key Decisions

- **Segment format:** `list[ActionSegment]` where `ActionSegment = (class_id: int, t_start: float, t_end: float)`
- **Keypoint format:** `np.ndarray` of shape `(T, 11, 2)` — T frames, 11 keypoints, (x, y)
- **Storage format:** JSON array in SQLite, `FLOAT[]` in PostgreSQL
- All constants are module-level in `constants.py`, not scattered across functions

---

## Phase 2b: Vision → State Pipeline

**Goal:** 실제 비디오 파일에서 State Vector 업데이트까지 end-to-end 파이프라인을 연결한다.

### Scope

| Build | NOT Build |
|-------|-----------|
| MediaPipe → keypoint extraction | Policy engine |
| LSTM → action segments | Session generation |
| Pipeline orchestrator | Mobile app |
| Integration tests with real video | LLM coaching |

### Deliverables

```
src/vision/
  keypoint_extractor.py    # MediaPipe wrapper → keypoint trajectories
  action_classifier.py     # LSTM inference → action segments

server/services/
  analysis_pipeline.py     # video → keypoints → segments → O_t → S_t update
                           # (refactor existing to integrate state engine)

tests/integration/
  test_pipeline_e2e.py     # Real video → state update
  fixtures/
    shadow_30s.mp4         # Test video: shadow boxing, 30s
    heavybag_60s.mp4       # Test video: heavy bag, 60s
```

### Verification Criteria

| # | Test | Pass Condition |
|---|------|----------------|
| 1 | Keypoint extraction | Output shape (T, 11, 2), no NaN values |
| 2 | Action classification | Segments match expected actions in test video |
| 3 | Full pipeline: shadow 30s | O_t computed, at least 10 dims observed |
| 4 | Full pipeline: heavy bag 60s | O_t computed, conditioning dims observed (≥90s 아니므로 ∅ 예상) |
| 5 | State update from pipeline | S_1 written to DB, audit log present |
| 6 | Two consecutive sessions | S_2 = EMA(S_1, O_2), delta computed |
| 7 | Pipeline failure recovery | Corrupt video → FAILED status, state unchanged |

### Key Decisions

- `keypoint_extractor.py`는 `src/inference/utils.py`의 기존 코드를 래핑
- 비디오 fixtures는 git-lfs 또는 별도 storage (repo에 직접 커밋하지 않음)
- Pipeline은 동기 실행 (MVP). 비동기 task queue는 Phase 2d에서 도입

---

## Phase 2c: Policy Engine

**Goal:** State Vector로부터 약점을 탐지하고, 약점 기반 훈련 세션을 자동 생성한다.

### Scope

| Build | NOT Build |
|-------|-----------|
| Weakness detection | LLM coaching text |
| Priority scoring | Mobile UI |
| Drill selection rules | Audio generation |
| Session plan structure | |
| Training focus mapping | |

### Deliverables

```
src/policy/
  weakness.py          # detect_weaknesses(S_t, C_t) → list[Weakness]
  priority.py          # score_priorities(weaknesses) → ranked list
  session_planner.py   # plan_session(priorities, drill_library) → SessionPlan
  drill_library.py     # Drill definitions loaded from YAML

configs/
  drills.yaml          # Drill catalog: name, type, targets (which dims)
  thresholds.yaml      # τ_i and w_group values from state-vector.md §8

tests/policy/
  test_weakness.py
  test_priority.py
  test_session_planner.py
```

### Verification Criteria

| # | Test | Pass Condition |
|---|------|----------------|
| 1 | Weakness detection | Low-dim state → correct weaknesses identified |
| 2 | Confidence filtering | Low-confidence dims excluded from weaknesses |
| 3 | Priority ordering | Defense weakness ranked above repertoire weakness |
| 4 | Session plan structure | Valid plan: has rounds, drills, durations |
| 5 | Targeted session | State with weak hooks → plan includes hook drills |
| 6 | Balanced state | No weaknesses → maintenance/general session generated |
| 7 | Determinism | Same S_t → same session plan |

### Key Decisions

- Policy는 100% rule-based (no ML, no LLM)
- `drills.yaml`의 각 drill은 `target_dims: [5, 6]` 처럼 어떤 state 차원을 훈련하는지 명시
- Session plan은 JSON 구조체 (아직 오디오 아님)

---

## Phase 2d: API & Session Runtime

**Goal:** 모바일 클라이언트가 사용할 수 있는 완전한 REST API를 구현한다.

### Scope

| Build | NOT Build |
|-------|-----------|
| All REST endpoints | React Native app |
| Session lifecycle (runtime.md) | Audio generation |
| Async task processing | LLM integration |
| Idempotency guards | Push notifications |
| Transaction model | |

### Deliverables

```
server/routers/
  sessions.py    # POST create, POST upload, GET status, GET report
  users.py       # POST create, GET state, GET history
  training.py    # POST generate-plan, GET plan/:id

server/services/
  session_manager.py    # Session state machine transitions
  state_service.py      # State read/write with transaction safety

tests/api/
  test_session_lifecycle.py
  test_idempotency.py
  test_concurrent_updates.py
```

### API Endpoints

```
POST   /api/v1/users                    Create user
GET    /api/v1/users/:id/state          Get current S_t, C_t
GET    /api/v1/users/:id/history        Get state transition history

POST   /api/v1/sessions                 Create session
POST   /api/v1/sessions/:id/upload      Upload video
GET    /api/v1/sessions/:id/status      Get session status + pipeline progress
GET    /api/v1/sessions/:id/report      Get analysis report

POST   /api/v1/training/generate        Generate training plan from current state
GET    /api/v1/training/plans/:id       Get training plan details
```

### Verification Criteria

| # | Test | Pass Condition |
|---|------|----------------|
| 1 | Session lifecycle | CREATED → UPLOADING → PROCESSING → COMPLETED |
| 2 | Upload + auto-process | Video upload triggers pipeline automatically |
| 3 | Status polling | Progress updates reflected in GET status |
| 4 | Idempotency | Duplicate upload-complete → no double state update |
| 5 | Concurrent sessions | Two sessions, same user → serialized state updates |
| 6 | State endpoint | GET state returns 18-dim vector + confidence |
| 7 | History endpoint | Returns chronological state transitions with deltas |
| 8 | Error propagation | Pipeline failure → session FAILED with error detail |
| 9 | Training plan generation | POST generate → plan based on current state weaknesses |

### Key Decisions

- Video processing은 background task (asyncio 또는 simple thread pool)
- 파일 저장은 로컬 filesystem (MVP), S3 호환 스토리지는 배포 시 전환
- Response format은 JSON, Pydantic schema로 type-safe

---

## Phase 2e: Coaching & Reports

**Goal:** State delta와 약점을 사람이 읽을 수 있는 코칭 언어로 변환한다.

### Scope

| Build | NOT Build |
|-------|-----------|
| LLM prompt templates | Mobile app |
| Report generation from ΔS_t | Audio TTS |
| Progress visualization data | Real-time feedback |
| Coaching feedback structure | |

### Deliverables

```
server/services/
  report_generator.py    # (refactor) ΔS_t → structured report + coaching text
  coaching_llm.py        # LLM adapter: state summary → coaching language

configs/
  prompts/
    session_report.txt   # Prompt template for session coaching
    progress_report.txt  # Prompt template for multi-session progress

tests/coaching/
  test_report_generator.py
  test_llm_fallback.py
```

### Report Structure

```json
{
  "session_summary": {
    "duration": 180,
    "total_punches": 142,
    "mode": "shadow"
  },
  "state_delta": {
    "improved": [{"dim": "tech_hook", "delta": +0.08, "now": 0.62}],
    "regressed": [{"dim": "guard_endurance", "delta": -0.04, "now": 0.51}],
    "unchanged": [...]
  },
  "weaknesses": [
    {"dim": "tech_uppercut", "value": 0.31, "confidence": 0.72}
  ],
  "coaching": {
    "summary": "훅 테크닉이 눈에 띄게 향상되었습니다. ...",
    "focus_areas": ["어퍼컷 연습 시 가드 유지에 집중하세요."],
    "next_session_hint": "다음 세션에서는 어퍼컷 드릴을 중심으로 ..."
  }
}
```

### Verification Criteria

| # | Test | Pass Condition |
|---|------|----------------|
| 1 | Report from state delta | Valid JSON report with all sections |
| 2 | LLM coaching text | Korean coaching sentences, no hallucinated metrics |
| 3 | LLM disabled fallback | System works without LLM (template-based fallback) |
| 4 | Progress over 5 sessions | Multi-session trend correctly summarized |
| 5 | Prompt template determinism | Same ΔS_t → same prompt (LLM output may vary) |

### Key Decisions

- LLM은 text-only API, 절대 metrics 계산 안 함 (overview.md invariant)
- LLM 없이도 동작: template-based fallback이 항상 존재
- Prompt에는 ΔS_t의 숫자 요약만 전달, raw vector는 전달하지 않음

---

## Phase 2f: Mobile App MVP

**Goal:** React Native 앱에서 비디오 촬영/업로드, 리포트 확인, AI 세션 실행이 가능하다.

### Scope

| Build | NOT Build |
|-------|-----------|
| Video capture + upload | On-device inference |
| Session status + report view | Real-time pose overlay |
| State visualization (radar chart) | Social features |
| AI session playback (audio commands) | Payment/subscription |
| Training plan view | |

### Screens

```
1. Home           → Current state radar chart + recent sessions
2. Record         → Camera view + start/stop recording
3. Processing     → Upload progress + pipeline status polling
4. Report         → Session report + coaching text + delta visualization
5. AI Session     → Audio-guided drill execution
6. History        → State evolution over time (line chart per dim group)
```

### Verification Criteria

| # | Test | Pass Condition |
|---|------|----------------|
| 1 | Video capture | Record 60s video, file saved locally |
| 2 | Upload | Video uploaded to server, session created |
| 3 | Status polling | Processing progress displayed, auto-refresh |
| 4 | Report display | Report rendered with coaching text and delta |
| 5 | Radar chart | 5 group scores displayed, updates after new session |
| 6 | AI session | Audio commands play, session recorded |
| 7 | History chart | Multi-session trend visible per dimension group |
| 8 | Offline handling | No crash on network failure, retry prompt shown |

---

## Phase 3: Future (Post-MVP)

Not planned in detail. Directional only.

| Feature | Description |
|---------|-------------|
| On-device inference | LSTM → CoreML/TFLite, real-time classification |
| Adaptive α | Per-dimension smoothing based on observation variance |
| Population norms | Reference values from user cohort instead of fixed constants |
| Sparring analysis | Two-person interaction modeling, counter-punch detection |
| Footwork dimensions | Expand state vector: stance, movement, weight transfer |
| Audio TTS session | Server-generated audio files for AI session mode |
| Learned policy | Gradient-based policy optimization from user outcome data |

---

## Dependency Graph

```
Phase 1 (DONE)
  └── LSTM classifier

Phase 2a: State Engine Core
  ├── state-vector.md formulas → code
  └── DB schema

Phase 2b: Vision → State Pipeline
  ├── depends on: 2a (observation function)
  └── depends on: Phase 1 (LSTM model)

Phase 2c: Policy Engine
  └── depends on: 2a (state vector definition)

Phase 2d: API & Session Runtime
  ├── depends on: 2a (state service)
  ├── depends on: 2b (analysis pipeline)
  ├── depends on: 2c (training plan generation)
  └── depends on: runtime.md (transaction model)

Phase 2e: Coaching & Reports
  ├── depends on: 2a (state delta)
  └── depends on: 2c (weakness detection)

Phase 2f: Mobile App
  └── depends on: 2d (API endpoints)
```

Note: 2c can run in parallel with 2b — both depend only on 2a.

```
        2a
       / \
     2b   2c
       \ / \
       2d   2e
        |
       2f
```

---

## Estimation Guide

No time estimates. Instead, scope indicators:

| Phase | Files to Create/Modify | New Tests |
|-------|----------------------|-----------|
| 2a | ~6 source, ~4 test | ~30 test cases |
| 2b | ~3 source, ~2 test | ~10 test cases |
| 2c | ~4 source, ~3 test | ~15 test cases |
| 2d | ~5 source, ~3 test | ~20 test cases |
| 2e | ~3 source, ~2 test | ~10 test cases |
| 2f | ~10 RN screens/components | Manual + E2E |

---

# End of Roadmap
