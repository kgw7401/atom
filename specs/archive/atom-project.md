# Spec: Atom — AI Boxing Coach & Analysis Platform

> Status: APPROVED
> Created: 2026-03-06
> Last Updated: 2026-03-07

## 1. Objective

- **What:** An AI-powered boxing training and analysis platform with two parallel tracks — (A) a product track delivering an LLM-powered personalized drill coach, and (B) a research track exploring video understanding and pose-based form verification.
- **Why:** Learning boxing is hard without consistent, personalized feedback. Coaches can't always be present, shadow boxing becomes repetitive without guidance, and sparring feels overwhelming without tactical preparation. Atom bridges this gap by being a data-driven coach that knows you.
- **Who:** The primary user is the developer (solo project), a boxing learner who wants structured, varied training and deeper fight understanding. Secondary goal: learn ML through the research track.
- **Success Criteria:**
  - [ ] A working drill session that delivers voice + visual combination instructions in real-time
  - [ ] Users can define custom combinations with gym-specific naming
  - [ ] LLM-powered session planning that adapts based on user history and natural language preferences
  - [ ] Session history is tracked, stored, and used to personalize future drills
  - [ ] A video analysis pipeline that extracts tactical insights from boxing footage (research track)
  - [ ] Every feature produces and consumes well-defined data — no feature exists without a data contract

## 2. Project Architecture

### 2.1 Two Parallel Tracks

```
┌──────────────────────────────────────────────────────────────┐
│                        ATOM PLATFORM                          │
├────────────────────────────┬─────────────────────────────────┤
│  Track A: PRODUCT (Coach)  │  Track B: RESEARCH (Vision/ML)  │
│                            │                                 │
│  A1. Combo Registry        │  B1. Pose Estimation Pipeline   │
│  A2. LLM Drill Engine      │  B2. Action Recognition / TAD  │
│  A3. Voice + Visual UI     │  B3. VLM Tactical Analysis      │
│  A4. User Profile & Data   │  B4. Data Bridge → Track A      │
│                            │                                 │
│  🎯 Shippable product      │  🔬 Learning + experimentation  │
│  ⏱️ Build first             │  ⏱️ Build in parallel            │
├────────────────────────────┴─────────────────────────────────┤
│                    SHARED DATA LAYER                          │
│                                                              │
│  Action Vocabulary    Combination Templates                   │
│  User Session History Fighter Pattern Database                │
│  Data Pipeline & Versioning                                   │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
[Track A: Coach Flow]
User Profile + Session History + Natural Language Input
  → LLM Drill Planner (generates session plan from data + user request)
    → Session Engine (round timer, rest, combo delivery sequence)
      → Voice (TTS) + Visual (screen) output
        → Session Log (combos drilled, timestamps, round data)
          → feeds back into User Profile

[Track B: Research Flow]
Boxing Video Input
  → Frame Extraction + Pose Estimation (MediaPipe BlazePose)
    → Action Recognition (classify each action)
      → Temporal Action Detection (locate actions in untrimmed video)
        → Structured Action Timeline (who, what, when)
          → VLM Tactical Narrative (Gemini — post-session analysis)
            → Tactical Insights (patterns, habits, vulnerabilities)

[Data Bridge: B → A]
Structured Action Timeline + Tactical Insights
  → Export as Drill Templates → Coach System
  e.g., "Fighter drops guard after right cross"
    → Drill: "practice exploiting dropped guard after cross"
```

### 2.3 Core Data Model

Every entity defines: what produces it, what consumes it, and how it's stored.

| Entity | Producer | Consumer | Storage |
|---|---|---|---|
| **Action** | System seed + user customization | Combination, Session Engine, Action Recognition | DB (reference table) |
| **Combination** | User CRUD + LLM generation | Drill Planner, Session Engine | DB (user-scoped) |
| **DrillPlan** | LLM Drill Planner | Session Engine | DB (per-session) |
| **SessionLog** | Session Engine (real-time) | User Profile, LLM Planner, Analytics | DB (append-only, immutable) |
| **UserProfile** | Aggregated from SessionLogs | LLM Drill Planner, Adaptive logic | DB (user-scoped) |
| **PoseFrame** | MediaPipe pipeline | Action Recognition model | File/object store (ephemeral or archived) |
| **ActionTimeline** | TAD pipeline | VLM Analysis, Pattern Detection | DB + file (structured JSON) |
| **TacticalInsight** | VLM + pattern analysis | User review, Drill Template export | DB (linked to video source) |

### 2.4 Data Lifecycle Principles

These apply to **every milestone**:

1. **Define before build:** Each feature spec must include a "Data Contract" section (see template below)
2. **Append-only where possible:** Session logs and action timelines are immutable records. Never overwrite training data.
3. **Version everything:** Model artifacts, training data splits, and schemas must be versioned
4. **Pipeline-first:** Data transformations must be reproducible scripts, not manual steps
5. **Local-first storage:** SQLite + filesystem for development, with clear migration path to PostgreSQL + object store

#### Data Contract Template

Every feature-level spec must include this section:

```markdown
## Data Contract

### Inputs
| Field | Type | Source | Required |
|-------|------|--------|----------|
| (field name) | (type) | (which milestone/entity produces it) | yes/no |

### Outputs
| Field | Type | Consumer | Storage | Mutable |
|-------|------|----------|---------|---------|
| (field name) | (type) | (which milestone/entity consumes it) | DB/file/ephemeral | yes/no |

### Schema Version
- Current: v1
- Migration strategy: (how to handle schema changes without breaking existing data)

### Volume & Retention
- Expected volume: (e.g., ~1 session log/day, ~30 frames/sec for video)
- Retention policy: (keep forever / TTL / archive after N days)
```

### 2.5 Tech Stack

| Layer | Choice | Rationale | Evidence |
|---|---|---|---|
| Language | Python 3.11+ | Fresh start, modern syntax, ML ecosystem | User decision: build from scratch |
| Backend | FastAPI | Async, lightweight, OpenAPI docs | Standard for ML-serving APIs |
| Database | SQLite (dev) → PostgreSQL | Local-first, upgrade path | Previous project lesson |
| LLM | OpenAI / Claude API | Drill planning, session coaching | GPTCoach (CHI 2025) validates pattern |
| TTS | TBD | Needs low latency for real-time drill calls | Open question |
| Frontend | TBD | Voice-primary, visual secondary | Open question |
| ML: Pose | MediaPipe BlazePose (IMAGE mode) | 33 keypoints, 30+ FPS on mobile, single-person | SOTA for sports coaching 2025 |
| ML: Actions | Pose features → classifier (XGBoost / lightweight) | Proven for boxing at small data scale | PLOS ONE 2025: 91% accuracy |
| ML: Video | Gemini 2.5 Pro (1-4 FPS post-session) | 1M token context, tactical narrative | Not for real-time detection (1 FPS sampling misses punches) |

**Key ML architecture insight (from research):**
- Pose classifier for punch **detection** (real-time, accurate)
- VLM for post-session **narrative/insight** (not real-time, but rich understanding)
- Video foundation models (VideoMAE etc.) **overfit on <500 clips** — not viable at current data scale

## 3. Milestone Breakdown

Each milestone becomes its own feature-level spec (`specs/<milestone>.md`).
Each milestone spec **must include a Data Contract section**.

---

### Track A: Product (Coach) — Build First, Ship Incrementally

#### A1: Foundation — Data Model, Combo Registry & User Profile Schema
> **Goal:** Define and persist all core data structures — Actions, Combinations, UserProfile schema, and SessionLog schema. Users can create, edit, and organize custom combinations with gym-specific names. The UserProfile and SessionLog schemas are defined here so downstream features (A2, A4) have a stable contract to build on.

- Scope: Full data model (Action, Combination, UserProfile, SessionLog, DrillPlan), Combination CRUD, CLI or simple API
- Data contract: Produces Action + Combination + UserProfile + SessionLog tables; consumed by all downstream features
- Key decisions: Database schema, action taxonomy, combo format, profile schema (what fields define a user's training state), session log schema (what gets recorded per session)
- Complexity: **M** (expanded from S — now includes profile + session schemas)
- Depends on: Nothing

#### A2: LLM-Powered Drill Session Engine
> **Goal:** An LLM-driven session engine that generates drill plans from user data, manages round/rest timing, and delivers combo sequences. Users can customize sessions via natural language.

- Scope: LLM integration, prompt engineering with user data context, session state machine, round timer, session logging
- Data contract: Consumes UserProfile + SessionHistory + Combinations (schemas from A1) → Produces DrillPlan + SessionLog (writes to schemas from A1)
- Key decisions: LLM provider, prompt structure, how much session logic is LLM vs deterministic
- LLM cost control: LLM is called **once per session** (to generate the drill plan), not per-combo. Session execution is deterministic — the engine follows the plan without additional LLM calls. This bounds cost to ~1 API call/day for daily training.
- Complexity: **M**
- Depends on: A1 (data model + schemas)
- Reference: GPTCoach (CHI 2025) — open source, validated pattern

#### A3: Voice + Visual Output
> **Goal:** Deliver drill instructions via TTS voice and on-screen visual cues simultaneously.

- Scope: TTS integration, visual display layer, synchronization between voice and visual
- Data contract: Consumes DrillPlan (combo sequence + timing) → Produces audio + visual events
- Key decisions: TTS engine (latency target), visual UI framework, device target
- Complexity: **M**
- Depends on: A2

#### A4: User Profile Aggregation & Adaptive History
> **Goal:** Build the aggregation logic that computes UserProfile from raw SessionLogs. This is the intelligence layer that summarizes training patterns — session frequency, combo coverage, identified weak areas — and prepares context for the LLM.

- Scope: Profile aggregation pipelines, history analytics, LLM context summarization (fit history into token budget)
- Data contract: Consumes SessionLog (append-only, schema from A1) → Produces UserProfile fields (derived, re-computable from logs)
- Key decisions: Aggregation frequency (real-time vs batch), how to summarize history for LLM context window, what metrics define "weakness"
- Complexity: **M**
- Depends on: A2 (needs session logs to exist), A1 (profile schema defined there)

---

### Track B: Research (Vision/ML) — Parallel, Learning-Oriented

#### B1: Pose Estimation Pipeline
> **Goal:** Build a reproducible pipeline that takes video frames and outputs 33-keypoint pose data using MediaPipe BlazePose.

- Scope: Video ingestion, frame extraction, MediaPipe integration, keypoint output format
- Data contract: Consumes video file → Produces PoseFrame sequence (JSON/parquet per frame)
- Key decisions: Frame rate extraction, output format, storage strategy (keep all frames vs sample)
- Complexity: **S**
- Depends on: Nothing
- Note: IMAGE mode only (VIDEO mode has timestamp errors — validated lesson)

#### B2: Action Recognition & Temporal Action Detection
> **Goal:** Classify individual boxing actions from pose data, then locate action instances in untrimmed video with start/end timestamps.

- Scope: Feature engineering from keypoints, classifier training, sliding window TAD, evaluation
- Data contract: Consumes PoseFrame sequence → Produces ActionTimeline (structured: [start, end, class, confidence])
- Key decisions: Feature set (reuse 68-feature approach or redesign), classifier architecture, TAD window strategy
- Complexity: **L**
- Depends on: B1
- ML terminology: Action Recognition (trimmed clip → class) + TAD (untrimmed video → [start, end, class])

#### B3: VLM Tactical Analysis
> **Goal:** Use a Video Language Model to generate tactical narrative insights from boxing footage, grounded in structured action data.

- Scope: VLM API integration (Gemini 2.5 Pro), prompt engineering with ActionTimeline context, insight structuring
- Data contract: Consumes ActionTimeline + raw video → Produces TacticalInsight (natural language + structured patterns)
- Key decisions: FPS sampling rate (cost vs accuracy tradeoff), prompt structure, insight schema
- Complexity: **L**
- Depends on: B2 (ActionTimeline provides grounding data for VLM)
- Limitation: VLM at 1 FPS misses individual punches — use ActionTimeline for detection, VLM for narrative

#### B4: Data Bridge — Research → Product
> **Goal:** Export tactical insights and fighter patterns from Track B into drill templates usable by Track A's coach system.

- Scope: Insight → Drill Template converter, pattern-to-combo mapping, import into Combo Registry
- Data contract: Consumes TacticalInsight + FighterPattern → Produces Combination templates tagged with source
- Key decisions: Auto-generation vs user-curated export, quality threshold for auto-import
- Complexity: **S**
- Depends on: B3 + A1

---

### Milestone Dependency Map

```
TRACK A (Product)                    TRACK B (Research)
═══════════════                      ══════════════════

A1 (Data Model + Combo Registry)     B1 (Pose Pipeline)
  │                                    │
  ▼                                    ▼
A2 (LLM Drill Engine)               B2 (Action Recognition + TAD)
  │                                    │
  ├──▶ A3 (Voice + Visual)            ▼
  │                                  B3 (VLM Tactical Analysis)
  ▼                                    │
A4 (User Profile)                      ▼
                                     B4 (Data Bridge) ──▶ A1
                                         (exports drill templates)

Parallel: A1..A4 and B1..B3 can proceed simultaneously
Bridge:   B4 connects research outputs back to the product
```

## 4. Boundaries

- **Always:**
  - **Data-first:** Every feature must define its data contract before implementation begins
  - **Append-only logs:** Session logs and action timelines are immutable — never overwrite
  - Custom combo naming must be supported from day one
  - Build incrementally — each milestone must be usable standalone
  - Track A is the priority — it ships a useful product without Track B
  - Prefer simple, working solutions over architecturally perfect but unfinished ones

- **Ask first:**
  - TTS engine selection (cost, latency, quality tradeoffs)
  - Frontend platform choice (mobile app vs web app vs CLI)
  - LLM provider selection (OpenAI, Claude, local) and cost implications
  - Whether to use cloud APIs or local-only ML inference
  - Video analysis training data sourcing (legal/ethical considerations)

- **Never:**
  - Build a feature without a data contract
  - Hardcode combinations — all combos must be data-driven and user-customizable
  - Use VLMs for real-time punch detection (1 FPS sampling misses punches)
  - Use video foundation models (VideoMAE) on <500 training clips (overfitting risk)
  - Build a multi-user platform prematurely — this is a personal tool first
  - Mix Track A and Track B dependencies — Track A must work without Track B

## 5. Testing Strategy

- **Unit:** Each data model operation (combo CRUD, session state transitions, data pipeline steps)
- **Integration:** Full drill session flow — create combo → LLM generates plan → session delivers instructions → log stored → profile updated
- **Data Pipeline:** Each pipeline step tested independently with fixture data
- **Conformance:**
  - Input: User creates combo "Combo 3" = [jab, cross, lead hook]
  - Expected: LLM drill planner can reference "Combo 3" and include it in session plans
  - Input: User says "focus on defense today, I want to practice slipping"
  - Expected: LLM generates a drill plan weighted toward defensive combos with slips
  - Input: User completes 10 sessions heavily drilling hooks
  - Expected: User profile reflects hook overrepresentation; LLM adjusts future plans
  - Input: 3-minute sparring video processed through B1 → B2 pipeline
  - Expected: ActionTimeline with timestamped, classified actions for both fighters
  - Input: ActionTimeline fed to VLM (B3)
  - Expected: Tactical narrative identifying patterns (e.g., "guard drops after right cross")

## 6. Open Questions

### Resolve per-milestone (not blockers for project start)
- [ ] **Frontend platform:** Mobile app (React Native / Flutter), web app, or start with CLI/terminal? → Resolve before A3
- [ ] **TTS engine:** System TTS, cloud API (Google/AWS), or edge model? Latency target? → Resolve before A3
- [ ] **LLM provider:** OpenAI, Claude, or local model? → Evaluate during A2 spec. Cost bounded by 1-call-per-session design.
- [ ] **Deployment target:** Local-only for now, or plan for mobile/cloud from the start? → Resolve before A3

### Resolve when Track B starts
- [ ] **Video source for research:** Where to get labeled boxing video data? (YouTube? Personal footage? Existing datasets like BoxingVI?)
- [ ] **Data bridge automation:** Should Track B insights auto-generate drill templates, or should the user manually curate them?

### Resolved
- [x] ~~**Session data format:**~~ → Now part of A1 scope (define schemas upfront)
- [x] ~~**LLM cost:**~~ → Bounded by 1 LLM call per session, deterministic execution

## 7. Post-Track A Roadmap: Connecting Product and Research

Track A and Track B are **independently buildable but intentionally connected**. Here's the plan for after Track A ships:

### Phase 1: Track A Complete (A1→A4 shipped)
At this point you have:
- A working drill coach with LLM-powered session planning
- Custom combos, voice + visual output, user history
- Rich session log data accumulating over time

### Phase 2: Begin Track B (B1→B2) using Track A's data
- **B1 (Pose Pipeline)** can start anytime, but gains value when you have your own sparring footage to process
- **B2 (Action Recognition)** benefits from Track A's Action vocabulary — same taxonomy, same labels
- Training data for B2 can come from: (1) existing datasets like BoxingVI, (2) your own footage labeled with Track A's action vocabulary

### Phase 3: Close the Loop (B3→B4)
- **B3 (VLM Analysis)** produces tactical insights from video
- **B4 (Data Bridge)** converts those insights into Combination templates that flow back into Track A
- At this point, the system becomes self-improving: train → record → analyze → generate better drills → train again

### The Connection Points
```
Track A data that Track B uses:
  - Action vocabulary (shared taxonomy)
  - Combination format (so B4 can export compatible templates)
  - SessionLog schema (if recording pose data during drill sessions)

Track B data that Track A uses:
  - Drill Templates from B4 (new combos based on fight analysis)
  - TacticalInsight summaries (LLM can reference in drill planning)
```

### Key Principle
Track B is research — it may take longer, change direction, or produce unexpected results. That's fine. Track A must **never wait** on Track B. But every Track B milestone should ask: "how does this data flow back to the coach?"

## 8. Research References

These papers/tools were validated during the interview phase and inform the technical decisions:

| Reference | Relevance |
|---|---|
| [GPTCoach (CHI 2025)](https://github.com/StanfordHCI/GPTCoach-CHI2025) | LLM coaching pattern — open source, 93% MI-consistent |
| [LLM-SPTRec (Nature 2026)](https://www.nature.com/articles/s41598-026-37075-z) | Knowledge-graph grounded LLM sports recommendations |
| [BoxingVI Benchmark (Nov 2024)](https://arxiv.org/html/2511.16524v1) | 6,915 labeled punch clips, 6 types — potential training data |
| [Active Learning Boxing (PLOS ONE 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12061147/) | 91% accuracy with pose-based classifier, 15% labeling effort |
| [Gemini 2.5 Video Understanding](https://developers.googleblog.com/en/gemini-2-5-video-understanding/) | 1M+ token context for video, but 1 FPS default sampling |
| [MMA Analysis with Gemini (Feb 2026)](https://pub.towardsai.net/structured-video-captioning-with-gemini-an-mma-analysis-use-case-bfbb8fd91a26) | Multi-agent VLM analysis pattern validated for combat sports |

## Changelog

| Date | Change | Reason |
|---|---|---|
| 2026-03-06 | Initial draft | Phase 2 complete |
| 2026-03-07 | Restructured: two parallel tracks (Product vs Research) | Review R1: group product and research separately |
| 2026-03-07 | Added LLM to drill engine (A2), removed hand-coded selection | Review R1: LLM enriches sessions, simplifies customization |
| 2026-03-07 | Fixed ML terminology: Action Recognition + TAD as subtasks of video understanding | Review R1: "action detection" and "video analysis" were falsely separated |
| 2026-03-07 | Added Data Contract to every milestone, Data Lifecycle Principles section | Review R1: data is first-class citizen across all features |
| 2026-03-07 | Added Research References section with validated papers | Review R1: ML stack must be evidence-based, not assumed |
| 2026-03-07 | Updated tech stack with evidence column | Review R1: don't rely on previous project context |
| 2026-03-07 | Expanded A1 to include UserProfile + SessionLog schemas | Review R2: profile schema needed before A2 builds on it |
| 2026-03-07 | Added LLM cost control to A2: 1 call per session, deterministic execution | Review R2: bound LLM cost explicitly |
| 2026-03-07 | Added Data Contract Template to section 2.4 | Review R2: enforce standard format for all feature specs |
| 2026-03-07 | Refined A4 to aggregation-only (schema lives in A1) | Review R2: separate schema definition from computation |
| 2026-03-07 | Reorganized Open Questions: per-milestone vs per-track, marked resolved items | Review R3: LLM provider deferred to A2 spec |
| 2026-03-07 | Added Section 7: Post-Track A Roadmap | Review R3: clarify how Track B connects back after Track A ships |
| 2026-03-07 | Status → APPROVED | User approved after 3 review rounds |
