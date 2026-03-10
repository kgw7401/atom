# Spec: Track A — AI Boxing Coach (Product Track)

> Status: APPROVED
> Created: 2026-03-07
> Last Updated: 2026-03-07
> Parent: [atom-project.md](./atom-project.md)

## 1. Objective

- **What:** A timer-based boxing drill coach that delivers personalized combo instructions via voice (TTS) and terminal UI, powered by LLM session planning with preset templates and natural language customization.
- **Why:** Shadow boxing without guidance is repetitive and unstructured. A drill coach that knows your training history and adapts session plans bridges the gap between having a coach and training alone.
- **Who:** Solo developer / boxing learner who wants structured, varied shadow boxing sessions with personalized drill delivery.
- **Success Criteria:**
  - [ ] User can create, edit, delete, and list custom combinations with gym-specific naming (dual: display_name + action list)
  - [ ] 11-action taxonomy seeded on first run (jab, cross, lead_hook, rear_hook, lead_uppercut, rear_uppercut, lead_bodyshot, rear_bodyshot, slip, duck, backstep)
  - [ ] User can start a drill session by selecting a template (Fundamentals / Combos / Mixed) with optional natural language tweaks
  - [ ] LLM generates a structured session plan (rounds, combos, timing) from template + user history + NL input — one API call per session
  - [ ] CLI session runner delivers combos via terminal output + macOS TTS in real-time with round/rest timing
  - [ ] Every session is logged (delivery log: combos called, timestamps, round structure, duration) — append-only, immutable
  - [ ] User profile aggregates session history to inform future LLM planning (combo exposure frequency, session patterns)

## 2. Technical Design

### 2.1 Architecture

```
┌──────────────────────┐   ┌──────────────────────────────┐
│    CLI Interface      │   │    FastAPI Server (optional)  │
│  atom combo ...       │   │  /api/combos, /api/sessions  │
│  atom session start   │   │  (for future frontends)      │
│  atom profile ...     │   │                              │
└──────────┬───────────┘   └──────────┬───────────────────┘
           │                          │
           ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Service Layer (shared)                  │
│  ComboService │ SessionService │ LLMPlanner │ ProfileSvc │
├──────────────────────────────────────────────────────────┤
│                  SQLAlchemy Async + SQLite                │
│  Actions | Combinations | DrillPlans | SessionLogs       │
│  UserProfiles | SessionTemplates                         │
└─────────────────────────────────────────────────────────┘
```

**Key architectural decision:** CLI and FastAPI both import the same service layer.
CLI calls services directly (no HTTP required). FastAPI wraps services as REST endpoints for future frontends. No server process needed for CLI-only usage.

### 2.2 Data Model

#### Actions (reference table, system-seeded)

| Field | Type | Notes |
|-------|------|-------|
| id | str (UUID) | PK |
| name | str | Canonical English name, unique (e.g., `jab`) |
| display_name | str | User-facing name (e.g., `잽`, `Jab`) |
| category | str | `offense` / `defense` / `movement` |
| description | str | Optional short description |
| sort_order | int | Display ordering |

**Seed data (11 actions):**

| name | display_name | category |
|------|-------------|----------|
| jab | 잽 | offense |
| cross | 크로스 | offense |
| lead_hook | 리드훅 | offense |
| rear_hook | 리어훅 | offense |
| lead_uppercut | 리드어퍼컷 | offense |
| rear_uppercut | 리어어퍼컷 | offense |
| lead_bodyshot | 리드바디 | offense |
| rear_bodyshot | 리어바디 | offense |
| slip | 슬립 | defense |
| duck | 덕킹 | defense |
| backstep | 백스텝 | movement |

#### Combinations (user-scoped)

| Field | Type | Notes |
|-------|------|-------|
| id | str (UUID) | PK |
| display_name | str | User-facing name, unique (e.g., `원투훅`). Primary identifier for CLI and LLM. |
| actions | JSON list[str] | Ordered action names (e.g., `["jab", "cross", "lead_hook"]`) |
| complexity | int | Auto-computed: len(actions). Used for template filtering |
| is_system | bool | True for seeded combos, False for user-created |
| created_at | datetime | |
| updated_at | datetime | |

**Seed combos (examples):**

| display_name | actions | complexity |
|-------------|---------|------------|
| 잽 | `["jab"]` | 1 |
| 원투 | `["jab", "cross"]` | 2 |
| 원투훅 | `["jab", "cross", "lead_hook"]` | 3 |
| 원투바디 | `["jab", "cross", "rear_bodyshot"]` | 3 |
| 원투쓰리투 | `["jab", "cross", "lead_hook", "cross"]` | 4 |
| 슬립원투 | `["slip", "jab", "cross"]` | 3 |
| 덕킹원투 | `["duck", "jab", "cross"]` | 3 |
| 잽잽크로스 | `["jab", "jab", "cross"]` | 3 |
| 더블잽 | `["jab", "jab"]` | 2 |
| 리드훅바디 | `["lead_hook", "rear_bodyshot"]` | 2 |

#### SessionTemplates (system-defined presets)

| Field | Type | Notes |
|-------|------|-------|
| id | str (UUID) | PK |
| name | str | `fundamentals` / `combos` / `mixed` |
| display_name | str | User-facing (e.g., `기본기`, `콤비네이션`, `종합`) |
| description | str | What this template focuses on |
| default_rounds | int | Default round count |
| default_round_duration_sec | int | Default round length |
| default_rest_sec | int | Default rest between rounds |
| combo_complexity_range | JSON [int, int] | Min/max combo complexity to include |
| combo_include_defense | bool | Whether to include defense-action combos (false = offense only) |
| pace_interval_sec | JSON [int, int] | Min/max seconds between combo calls |

**3 preset templates:**

| Template | Rounds | Round Duration | Rest | Complexity | Pace (sec) | Focus |
|----------|--------|---------------|------|------------|------------|-------|
| **Fundamentals** | 3 | 120s (2min) | 45s | 1-2 | 8-12 | Singles & doubles, slow pace, form focus |
| **Combos** | 4 | 150s (2.5min) | 60s | 3-4 | 6-10 | Multi-action sequences, medium pace |
| **Mixed** | 5 | 180s (3min) | 60s | 1-4 | 5-8 | Offense + defense, varied complexity, high volume |

#### DrillPlans (LLM-generated, per-session)

| Field | Type | Notes |
|-------|------|-------|
| id | str (UUID) | PK |
| template_id | str (UUID) | FK to SessionTemplate used as base |
| user_prompt | str or null | User's natural language customization input |
| llm_model | str | Model used (e.g., `claude-3-haiku`) |
| plan_json | JSON | Full structured plan (see schema below) |
| created_at | datetime | |

**DrillPlan JSON schema:**

```json
{
  "session_type": "drill",
  "template": "fundamentals",
  "focus": "기본 원투 콤보",
  "total_duration_minutes": 8,
  "rounds": [
    {
      "round_number": 1,
      "duration_seconds": 120,
      "rest_after_seconds": 45,
      "instructions": [
        {
          "timestamp_offset": 5.0,
          "combo_display_name": "원투",
          "actions": ["jab", "cross"]
        }
      ]
    }
  ]
}
```

#### SessionLogs (append-only, immutable)

| Field | Type | Notes |
|-------|------|-------|
| id | str (UUID) | PK |
| drill_plan_id | str (UUID) | FK to DrillPlan |
| template_name | str | Denormalized for fast queries |
| started_at | datetime | Session start timestamp |
| completed_at | datetime or null | Null if abandoned mid-session |
| total_duration_sec | float | Actual elapsed time |
| rounds_completed | int | How many rounds finished |
| rounds_total | int | Total rounds in plan |
| combos_delivered | int | Total combo instructions called |
| delivery_log_json | JSON | Full delivery record (see below) |
| status | str | `completed` / `abandoned` / `error` |
| created_at | datetime | |

**delivery_log_json schema:**

```json
{
  "events": [
    {"type": "round_start", "round": 1, "ts": 0.0},
    {"type": "combo_called", "round": 1, "ts": 5.0, "combo_display_name": "원투", "actions": ["jab", "cross"]},
    {"type": "combo_called", "round": 1, "ts": 15.2, "combo_display_name": "원투훅", "actions": ["jab", "cross", "lead_hook"]},
    {"type": "round_end", "round": 1, "ts": 120.0},
    {"type": "rest_start", "round": 1, "ts": 120.0},
    {"type": "rest_end", "round": 1, "ts": 165.0},
    {"type": "session_end", "ts": 480.0, "reason": "completed"}
  ]
}
```

#### UserProfiles (derived, re-computable from SessionLogs)

| Field | Type | Notes |
|-------|------|-------|
| id | str (UUID) | PK |
| experience_level | str | `beginner` / `intermediate` / `advanced` — user-set initially, updated by system |
| goal | str | User's training goal (free text) |
| total_sessions | int | Count of completed sessions |
| total_training_minutes | float | Sum of session durations |
| last_session_at | datetime or null | |
| combo_exposure_json | JSON | `{"combo_display_name": count}` — how many times each combo was drilled |
| template_preference_json | JSON | `{"fundamentals": 5, "combos": 12, ...}` — template usage counts |
| session_frequency | float | Avg sessions per week (rolling 4-week window) |
| created_at | datetime | |
| updated_at | datetime | |

**Aggregation timing:** Profile is re-computed immediately after each SessionLog is saved (on session end). This ensures the LLM planner always sees up-to-date data for the next session.

### 2.3 API Design

#### Combo Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/combos` | List all combos (filterable by complexity) |
| GET | `/api/combos/{id}` | Get combo by ID |
| POST | `/api/combos` | Create combo (validates actions exist) |
| PUT | `/api/combos/{id}` | Update combo (system combos are immutable) |
| DELETE | `/api/combos/{id}` | Delete combo (system combos cannot be deleted) |

#### Session Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/templates` | List available session templates |
| POST | `/api/sessions/plan` | Generate drill plan (template + optional NL prompt) |
| POST | `/api/sessions/{plan_id}/start` | Mark session as started |
| POST | `/api/sessions/{plan_id}/log` | Submit delivery log (on session end) |
| GET | `/api/sessions` | List past sessions (paginated) |
| GET | `/api/sessions/{id}` | Get session detail + delivery log |

#### Profile Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/profile` | Get aggregated user profile |
| PUT | `/api/profile` | Update user-set fields (experience_level, goal) |

### 2.4 CLI Commands

```bash
# Combo management
atom combo list [--complexity 1-4]
atom combo add "원투훅" jab cross lead_hook
atom combo show <id-or-name>
atom combo edit <id-or-name> --display-name "원투쓰리"
atom combo delete <id-or-name>

# Session
atom session start [--template fundamentals|combos|mixed] [--prompt "오늘은 방어 위주로"]
atom session history [--limit 10]
atom session show <id>

# Profile
atom profile show
atom profile set --experience intermediate --goal "스파링 준비"

# System
atom init          # Initialize DB, seed actions + combos + templates
atom server start  # Start FastAPI server (optional — for external API access / future frontend)
```

### 2.5 LLM Integration (A2)

**Provider:** Anthropic Claude API (configurable via env var)

**Call pattern:** One call per session, at plan generation time.

**Prompt structure:**
```
System: Boxing coach system prompt (action list, template constraints, output format)

User:
# Template: {template_name}
# Constraints: {round count, duration, rest, complexity range, pace}

# Available Combos (filtered by template complexity range)
- 원투: ["jab", "cross"]
- 더블잽: ["jab", "jab"]
- 원투훅: ["jab", "cross", "lead_hook"]
- ... (all combos matching complexity range)

# User Profile
Experience: {level}
Total sessions: {n}
Training frequency: {x}/week

# Combo Exposure (top 5 most/least drilled)
Most drilled: 원투 (42x), 잽 (38x), ...
Least drilled: 덕킹원투 (2x), 리드훅바디 (0x), ...

# User Request (if provided)
"{user's natural language input}"

Generate a drill session plan using ONLY combos from the Available Combos list above.
```

**Post-processing:** After LLM returns the plan JSON, the system resolves each `combo_display_name` to its database record. Unknown display_names are rejected and the plan is re-generated or falls back.

**Warmup handling:** The LLM system prompt instructs the model to start each session with simpler combos in round 1 and progressively increase complexity. No separate warmup round type needed.

**Fallback:** If LLM call fails, generate a deterministic plan from the template defaults using random combo selection from the eligible pool.

**Cost:** ~$0.002/session with Claude Haiku. Bounded by one-call-per-session design.

### 2.6 Session Engine State Machine (A2)

```
[IDLE] ──(start)──> [ROUND_ACTIVE] ──(timer)──> [COMBO_CALL]
                         ^                           │
                         │                      (pace timer)
                         │                           │
                    [REST_PERIOD] <──(round ends)─────┘
                         │
                    (rest timer)
                         │
                    [ROUND_ACTIVE] ... or [SESSION_END]
```

States:
- **IDLE**: Waiting for session start
- **ROUND_ACTIVE**: Round timer running, cycling through combo calls
- **COMBO_CALL**: Delivering a combo instruction (TTS + terminal output)
- **REST_PERIOD**: Rest timer between rounds
- **SESSION_END**: All rounds complete, log session

Unlike phase1's 6-state machine (which waited for pose detection), this is a pure timer-based flow. No WAITING/EXECUTING/GUARD_WATCH states.

### 2.7 Dependencies

| Dependency | Purpose | Version |
|-----------|---------|---------|
| Python | Runtime | 3.11+ |
| FastAPI | REST API | latest |
| uvicorn | ASGI server | latest |
| SQLAlchemy | ORM (async) | 2.0+ |
| aiosqlite | SQLite async driver | latest |
| alembic | DB migrations | latest |
| anthropic | Claude API client | latest |
| click | CLI framework | latest |
| httpx | HTTP client (for LLM fallback / future use) | latest |
| pydantic | Schema validation | 2.0+ |

## 3. Implementation Plan

### Task 1: Project Scaffolding & Database Setup (A1)

- **Scope:**
  - Initialize Python project (pyproject.toml, src layout)
  - SQLAlchemy async engine + model definitions (all 6 tables)
  - Alembic migration setup
  - `atom init` CLI command (create DB, run migrations)
  - Seed data script (11 actions, 10+ default combos, 3 templates)
- **Verification:**
  - `atom init` creates SQLite DB with all tables
  - Seed data present: `SELECT count(*) FROM actions` = 11
  - Alembic migration runs cleanly
- **Complexity:** M

### Task 2: Combo CRUD — API + CLI (A1)

- **Scope:**
  - FastAPI endpoints: GET/POST/PUT/DELETE `/api/combos`
  - Pydantic request/response schemas
  - Validation: actions must exist in Actions table
  - System combos are immutable (no edit/delete)
  - CLI commands: `atom combo list/add/show/edit/delete`
  - CLI calls service layer directly (no HTTP)
- **Verification:**
  - `atom combo add "테스트콤보" jab cross` → creates combo, returns ID
  - `atom combo list` → shows all combos with display names
  - `atom combo delete <system-combo>` → returns error
  - API tests: create, read, update, delete, validation errors
- **Complexity:** M

### Task 3: LLM Drill Planner (A2)

- **Scope:**
  - LLM client abstraction (Anthropic Claude, configurable)
  - Session plan generation: template + user profile + NL input → structured JSON
  - Plan validation (required fields, valid combo references, timing constraints)
  - Deterministic fallback plan generator
  - POST `/api/sessions/plan` endpoint
  - `atom session start` CLI flow (select template → optional prompt → generate plan → confirm)
- **Verification:**
  - Generate plan with each template type → valid JSON, correct round structure
  - Plan respects template constraints (complexity range, pace, round count)
  - Fallback plan works when LLM is unavailable (no API key set)
  - Combos in plan all exist in the database
- **Complexity:** L

### Task 4: CLI Session Runner (A2)

- **Scope:**
  - Session engine state machine (IDLE → ROUND_ACTIVE → COMBO_CALL → REST_PERIOD → SESSION_END)
  - Real-time timer management (round duration, rest, combo pacing)
  - Terminal output: current round, combo instruction, countdown, session progress
  - macOS TTS integration (`say` command, non-blocking)
  - Delivery log recording (all events with timestamps)
  - Session log persistence (save via service layer on session end)
  - Graceful abort (Ctrl+C saves partial log with `abandoned` status)
- **Verification:**
  - Run a full session → terminal displays combos at correct intervals
  - TTS speaks combo display_names
  - Session log written to DB with correct event count and timestamps
  - Ctrl+C mid-session → log saved with `abandoned` status
  - Round/rest transitions at correct times
- **Complexity:** L

### Task 5: Session History & Profile Endpoints (A1 + A4)

- **Scope:**
  - GET `/api/sessions` (paginated list with summary stats)
  - GET `/api/sessions/{id}` (full detail + delivery log)
  - GET `/api/profile` (aggregated user profile)
  - PUT `/api/profile` (update experience_level, goal)
  - Profile aggregation logic: compute combo exposure, template preference, session frequency from SessionLogs
  - CLI commands: `atom session history`, `atom session show <id>`, `atom profile show/set`
- **Verification:**
  - After 3 sessions: `atom profile show` → displays correct total_sessions, combo exposure counts
  - `atom session history` → lists sessions with date, template, duration, status
  - Profile combo_exposure accurately reflects delivery logs
- **Complexity:** M

### Task 6: Voice + Visual Output Design (A3 — Design Only)

- **Scope:**
  - Evaluate TTS options beyond macOS `say` (latency, quality, cross-platform)
  - Design the interface contract between Session Engine and output layer
  - Define output event protocol (so future frontends can subscribe)
  - Document frontend platform decision criteria
  - **No implementation** — this task produces a design document that informs A3 implementation
- **Verification:**
  - Design doc written to `specs/a3-output-design.md`
  - Interface contract defined (event types, callback signatures)
  - At least 2 TTS options evaluated with pros/cons
- **Complexity:** S

## 4. Boundaries

- **Always:**
  - Every combo must be data-driven (DB-stored, not hardcoded)
  - Session logs are append-only and immutable — never overwrite or delete
  - LLM is called once per session (at plan generation), never during session execution
  - CLI and API share the same service layer — CLI calls services directly, API wraps them as HTTP endpoints
  - All actions in a combo must reference valid entries in the Actions table
  - System-seeded combos and actions are immutable

- **Ask first:**
  - Adding new actions beyond the initial 11
  - Changing the LLM provider or model
  - Any schema migration that alters existing tables (vs. adding new columns)
  - Frontend platform selection for A3

- **Never:**
  - Depend on Track B (pose detection) for any Track A feature
  - Make the LLM call blocking during session execution
  - Store user performance/accuracy data (no verification = no accuracy tracking)
  - Delete or mutate SessionLog records after creation
  - Hardcode combo definitions outside the database

## 5. Testing Strategy

- **Unit:**
  - Data model: combo CRUD operations, action validation, seed data loading
  - Session engine: state transitions, timer accuracy, event logging
  - LLM planner: prompt construction, plan validation, fallback generation
  - Profile aggregation: combo exposure counting, frequency calculation

- **Integration:**
  - Full flow: `atom init` → `atom combo add` → `atom session start --template fundamentals` → session runs → `atom session history` → `atom profile show`
  - API tests: all endpoints with valid/invalid payloads
  - LLM integration: mock LLM client for deterministic tests, one live smoke test

- **Conformance:**
  - Input: `atom combo add "원투훅" jab cross lead_hook`
  - Expected: Combo created with display_name="원투훅", actions=["jab", "cross", "lead_hook"], complexity=3

  - Input: `atom combo add "잘못된콤보" jab nonexistent_action`
  - Expected: Error — "Action 'nonexistent_action' not found"

  - Input: `atom session start --template fundamentals`
  - Expected: LLM generates plan with 3 rounds, 120s each, 45s rest, combos of complexity 1-2 only

  - Input: `atom session start --template combos --prompt "훅 위주로 연습하고 싶어"`
  - Expected: LLM plan includes higher proportion of hook-containing combos

  - Input: User completes 10 sessions, 8 using "combos" template
  - Expected: `atom profile show` → template_preference shows combos=8, session_frequency calculated correctly

  - Input: `atom combo delete <system-combo-id>`
  - Expected: Error — "System combos cannot be deleted"

  - Input: Ctrl+C during active session at round 2 of 4
  - Expected: SessionLog saved with status="abandoned", rounds_completed=2, partial delivery_log preserved

## 6. Data Contracts

### A1: Foundation

#### Inputs
| Field | Type | Source | Required |
|-------|------|--------|----------|
| action_name | str | System seed | yes |
| combo_display_name | str | User input | yes |
| combo_actions | list[str] | User input | yes |

#### Outputs
| Field | Type | Consumer | Storage | Mutable |
|-------|------|----------|---------|---------|
| Action rows | DB records | A2 (plan generation), Combo validation | SQLite | no (seed only) |
| Combination rows | DB records | A2 (LLM context), A4 (exposure tracking) | SQLite | yes (user combos) |
| UserProfile row | DB record | A2 (LLM context) | SQLite | yes (re-computed) |
| SessionTemplate rows | DB records | A2 (plan generation) | SQLite | no (seed only) |

### A2: LLM Drill Engine

#### Inputs
| Field | Type | Source | Required |
|-------|------|--------|----------|
| template_name | str | User selection (CLI) | yes |
| user_prompt | str | User NL input (CLI) | no |
| user_profile | UserProfile | A1 DB | yes |
| combinations | list[Combination] | A1 DB | yes |

#### Outputs
| Field | Type | Consumer | Storage | Mutable |
|-------|------|----------|---------|---------|
| DrillPlan | JSON + DB record | Session Engine (runtime) | SQLite | no |
| SessionLog | JSON + DB record | A4 (aggregation), User review | SQLite | no (append-only) |

### A4: User Profile Aggregation

#### Inputs
| Field | Type | Source | Required |
|-------|------|--------|----------|
| SessionLog records | DB records | A2 output | yes |

#### Outputs
| Field | Type | Consumer | Storage | Mutable |
|-------|------|----------|---------|---------|
| combo_exposure_json | JSON | A2 LLM context | SQLite (UserProfile) | yes (re-computed) |
| template_preference_json | JSON | A2 LLM context | SQLite (UserProfile) | yes (re-computed) |
| session_frequency | float | A2 LLM context | SQLite (UserProfile) | yes (re-computed) |

### Schema Version
- Current: v1
- Migration strategy: Alembic migrations. All schema changes are additive (new columns/tables). Existing data is never deleted by migrations.

### Volume & Retention
- Expected volume: ~1 session log/day, ~10 combo calls per session
- Retention policy: Keep forever (training history is the product's value)
- DB size estimate: <10MB/year at daily usage

## 7. Open Questions

- [ ] **LLM model selection:** Claude Haiku for cost efficiency, or Claude Sonnet for better plan quality? Evaluate during Task 3 implementation.
- [ ] **TTS voice selection:** macOS `say` has limited Korean voice quality. Evaluate alternatives (Google Cloud TTS, edge models) during Task 6.
- [ ] **Combo numbering convention:** Should we support traditional boxing numbering (1=jab, 2=cross, 3=lead hook...) as aliases? Decide before Task 2 if requested.
- [ ] **Session pause/resume:** Should users be able to pause mid-round and resume? Current design doesn't support this. Add if requested during Task 4.
- [ ] **Multi-user support:** Current design is single-user (no auth). If needed later, add user_id FK to all user-scoped tables.

## Changelog

| Date | Change | Reason |
|------|--------|--------|
| 2026-03-07 | Initial draft | Phase 2 complete |
| 2026-03-07 | Shared service layer: CLI calls services directly, no server required | Review R1: CLI-server coupling friction |
| 2026-03-07 | Dropped `name` field from Combinations; `display_name` is primary identifier | Review R1: redundant field, LLM can't generate UUIDs |
| 2026-03-07 | Added plan post-processing (resolve display_name → DB record) | Review R1: LLM output needs validation |
| 2026-03-07 | Profile aggregation runs on session end; warmup via LLM prompt | Review R1: timing undefined, warmup missing |
| 2026-03-07 | CLI calls service layer directly (not API via httpx) in Tasks 2 and 4 | Review R2: consistency with shared service layer |
| 2026-03-07 | Added Available Combos section to LLM prompt (prevents hallucinated names) | Review R2: LLM context gap |
| 2026-03-07 | Dropped `category` field from Combinations (template uses complexity, not category) | Review R2: unused field |
| 2026-03-07 | Status → APPROVED | User approved after 2 review rounds |
