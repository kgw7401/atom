# Spec: Atom Track B — Boxing Action Detection Model

> Status: APPROVED
> Created: 2026-03-07
> Last Updated: 2026-03-07

## 1. Objective

- **What:** Build a pose-based boxing action detection model that identifies what a boxer does (punches, defense, movement) from video. The model directly powers Track A's feedback loop (detecting whether the user executed the instructed combo) and extends to sparring/fight analysis for tactical insights.
- **Why:** Track A without video analysis is a TTS drill timer. With this model, Track A becomes a data-driven coach: it sees what you did, compares it to what it asked, and gives specific feedback. The same model extends to analyzing sparring and fights — understanding real boxing patterns informs better training.
- **Who:** Primary consumer: Track A's drill verification system. Secondary: the developer learning ML. Tertiary: fight analysis pipeline (B4 extension).
- **Success Criteria:**
  - [ ] Model detects 6 punch types from pose data with ≥80% accuracy (BoxingVI test set)
  - [ ] Model locates punch events in untrimmed solo training video (≥70% correctness, spot-check)
  - [ ] Track A can verify: "user was instructed jab-cross-hook → user actually did jab-cross" (session matching works)
  - [ ] Pipeline handles three scenarios: solo training (single-person), sparring (two people), fight footage (two people)
  - [ ] Full pipeline runs end-to-end: video → pose → actions → ActionTimeline
  - [ ] Learning goal: hands-on experience with pose estimation, feature engineering, ML training, evaluation

## 2. Technical Design

### 2.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRACK B: ACTION DETECTION MODEL               │
│                                                                  │
│  B1: Pose Estimation Pipeline                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  [Training mode] BoxingVI clips → all frames → MediaPipe │   │
│  │  [Analysis mode] Any video → configurable FPS → MediaPipe│   │
│  │    Single-person: MediaPipe directly on frame             │   │
│  │    Multi-person:  YOLO detect + track → ROI crop          │   │
│  │                   → MediaPipe per fighter                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│       │                                                          │
│       ▼                                                          │
│  B2: Action Detection Model                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Pose features → XGBoost classifier (trained on BoxingVI) │   │
│  │  Sliding window TAD → ActionTimeline per fighter           │   │
│  │  [start_time, end_time, action_class, confidence]         │   │
│  └──────────────────────────────────────────────────────────┘   │
│       │                                                          │
│       ├──────────────────────┐                                   │
│       ▼                      ▼                                   │
│  B3: Track A Integration    B4: Fight Analysis (extension)       │
│  ┌───────────────────┐   ┌──────────────────────────────┐      │
│  │ Combo grouping     │   │ Both fighters' ActionTimelines│      │
│  │ Session matching   │   │ + video → Gemini 2.5 Pro     │      │
│  │ Drill feedback     │   │ → Situational tactics         │      │
│  │ → Track A consumes │   │ → DrillTemplate export        │      │
│  └───────────────────┘   └──────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Scenarios

| Scenario | Who's in video | B1 mode | B2 output | Downstream |
|---|---|---|---|---|
| **BoxingVI training data** | Single person per clip | Training mode (all frames, no detection) | Used for classifier training only | B2 Task 6 |
| **Solo training** | User alone (drill or shadow boxing) | Analysis mode (single-person, MediaPipe directly) | ActionTimeline × 1 fighter | B3: Drill verification → Track A feedback |
| **Sparring** | User + partner | Analysis mode (multi-person, YOLO + MediaPipe) | ActionTimeline × 2 fighters | B3 (own actions) + B4 (interaction analysis) |
| **Fight footage** | Two external fighters (YouTube) | Analysis mode (multi-person, YOLO + MediaPipe) | ActionTimeline × 2 fighters | B4: Tactical analysis → DrillTemplate export |

### 2.3 Data Model

**PoseFrame** (B1 output)

| Field | Type | Description |
|---|---|---|
| video_id | string | Source video identifier |
| fighter_id | string | "user" (single-person analysis), "fighter_a"/"fighter_b" (multi-person analysis), null (training mode) |
| frame_number | int | Sequential frame index |
| timestamp | float | Seconds from video start |
| keypoints | float[33][4] | 33 keypoints × (x, y, z, visibility) |
| confidence | float | Overall pose detection confidence |

Storage: Parquet. One file per video per fighter.

**ActionTimeline** (B2 output)

| Field | Type | Description |
|---|---|---|
| video_id | string | Source video |
| fighter_id | string | Which fighter |
| actions | array | Detected action instances |
| actions[].start_time | float | Action start (seconds) |
| actions[].end_time | float | Action end (seconds) |
| actions[].action_class | string | Punch/defense type |
| actions[].confidence | float | Classifier confidence |

Action classes (v1, from BoxingVI):
`jab`, `cross`, `lead_hook`, `rear_hook`, `lead_uppercut`, `rear_uppercut`

Future v2 additions: `body_shot`, `slip`, `duck`, `block`, `idle`

**TTSInstructionLog** (B3 input — from Track A session engine)

| Field | Type | Description |
|---|---|---|
| session_id | string | Track A session ID |
| instructions | array | Ordered list of TTS combo calls |
| instructions[].timestamp | float | When TTS called the combo (seconds from session start) |
| instructions[].combo_name | string | Human-readable: "jab-cross-hook" |
| instructions[].combo_actions | string[] | Machine-readable: ["jab", "cross", "lead_hook"] |

This log is produced by Track A's session engine (A2) during drill sessions. Track A writes it as part of its SessionLog. B3 reads it to compare instructions vs detected actions. Until A2 exists, B3 tests against mock instruction logs.

**ComboSequence** (B3 — grouped actions)

| Field | Type | Description |
|---|---|---|
| video_id | string | Source video |
| fighter_id | string | Which fighter |
| combos | array | Detected combo sequences |
| combos[].actions | string[] | e.g., ["jab", "cross", "lead_hook"] |
| combos[].start_time | float | First action start |
| combos[].end_time | float | Last action end |

Grouping rule: actions within 0.8s gap = same combo. Gap ≥0.8s = new combo.

**SessionMatch** (B3 — instruction vs execution)

| Field | Type | Description |
|---|---|---|
| session_id | string | Track A session ID |
| matches | array | Per-instruction comparison |
| matches[].instruction_time | float | When TTS called the combo |
| matches[].instructed_combo | string[] | What was called: ["jab", "cross", "lead_hook"] |
| matches[].detected_combo | string[] | What user actually did: ["jab", "cross"] |
| matches[].result | string | "success" / "partial" / "miss" |
| matches[].detail | string | "lead_hook missing" |

Matching window: 3 seconds after TTS instruction timestamp.

**DrillFeedback** (B3 → Track A)

| Field | Type | Description |
|---|---|---|
| session_id | string | Track A session ID |
| combo_stats | object | Per-combo: {attempts, successes, partials, misses, success_rate} |
| overall_accuracy | float | Total successes / total instructions |
| missed_actions | string[] | Actions frequently missed (e.g., "lead_hook" missed 7 times) |
| notes | string[] | Actionable observations for LLM context |

This is the data that feeds back into Track A's UserProfile and LLM session planning.

**SituationalTactic** (B4 — fight analysis output)

| Field | Type | Description |
|---|---|---|
| video_id | string | Source fight video |
| situation | string | "Opponent throws jab-cross" |
| effective_response | string | "Slip outside, counter with cross-hook" |
| frequency | int | Times observed |
| success_rate | float | VLM-estimated (approximate) |
| evidence | array | [{timestamp, outcome}] |

> Note: success_rate and outcome are VLM-estimated at 2-4 FPS. Approximate, not ground truth.

**DrillTemplate** (B4 → Track A)

| Field | Type | Description |
|---|---|---|
| source | string | "track_b_analysis" |
| source_video_id | string | Which fight |
| scenario | string | TTS-ready: "Opponent jab-cross!" |
| target_combo | string[] | ["slip", "cross", "lead_hook"] |
| combo_name | string | "Slip-Cross-Hook Counter" |
| context_note | string | "Effective counter to jab-cross (observed 4/5 times)" |

### 2.4 Dependencies

| Dependency | Used by | Purpose |
|---|---|---|
| MediaPipe Python SDK | B1 | Pose estimation (33 keypoints, IMAGE mode) |
| ultralytics (YOLOv8) | B1 analysis mode (multi-person) | Person detection for multi-person video |
| scikit-learn / XGBoost | B2 | Action classification |
| Gemini 2.5 Pro API | B4 | Tactical analysis (fight footage) |
| BoxingVI dataset | B2 | Primary training data (6,915 labeled punch clips, 6 punch types) |
| Olympic Boxing Kaggle dataset | B2 | Secondary training data (312K frames, 8 classes incl. body punches + blocks) |
| yt-dlp | B1 | YouTube video download |
| pandas / pyarrow | B1, B2, B3 | Data handling, Parquet I/O |

### 2.5 Infrastructure

- **Local-first.** Training and inference on local machine (Apple Silicon MPS or CPU). Cloud GPU if needed.
- **Storage:** Parquet for pose data, JSON for timelines, SQLite for metadata index.
- **No raw YouTube video storage.** Extract keypoints, delete source video (copyright).

### 2.6 Decisions Made

| Decision | Choice | Reasoning |
|---|---|---|
| B3 before B4 | Drill verification before fight analysis | User stated: "the prerequisite is making model for track A." Track A integration is higher priority than fight analysis. |
| XGBoost over deep learning | XGBoost on pose features | Literature validates 91% accuracy with pose-based classifier. No GPU needed for inference. Lightweight, explainable. Deep learning (VideoMAE) overfits on <500 clips. |
| MediaPipe over MoveNet | BlazePose IMAGE mode | 33 keypoints (vs 17 for MoveNet). Better body coverage for punch classification. Validated in 2025 sports coaching literature. |
| Two B1 modes | Training, Analysis | Different scenarios need different ingestion. Training mode for labeled data (BoxingVI clips, all frames). Analysis mode for full videos (configurable FPS, optional multi-person via YOLO). |
| Combo grouping: rule-based | Time-gap heuristic (0.8s) | Actions within 0.8s = same combo. Simple, no ML needed. Validated approach from boxing training domain. |
| Session matching: 3s window | Compare within 3s of TTS call | User needs ~1-2s reaction time after hearing instruction. 3s window is generous but avoids false negatives. |
| VLM for outcome estimation | Not ground truth | No pipeline component can reliably detect if a punch "landed." VLM gives approximate visual judgment. Marked as estimated. |
| Dual dataset strategy | BoxingVI + Olympic Boxing Kaggle | BoxingVI: best for pose-conditioned 6-class classification. Olympic Boxing: adds body punches, blocks, dense frame-level labels for TAD. Combined coverage fills each other's gaps. |
| v2 classifier upgrade | AcT (Action Transformer) | When data exceeds ~10K clips, AcT runs at ~12ms on MPS (real-time capable), 88-93% accuracy in literature. Compatible with MediaPipe 2D input. Validated upgrade from XGBoost. |
| VLM → own analysis model | Progression v1→v3 | BoxMind (Paris 2024) proved own tactical model is feasible (69.8% accuracy) but needs 10.9K events + 119 hours footage. VLM bootstraps until enough interaction data accumulates. |
| Architecture validation | BoxMind convergence | BoxMind (deployed Paris 2024 Olympics) independently uses same pipeline: pose → lightweight detector → action timeline → tactical reasoning. Validates Track B's architecture. |

## 3. Implementation Plan

### B1: Pose Estimation Pipeline (Complexity: M)

- [ ] **Task 1: Video ingestion**
  - Scope: Unified video loader with two modes:
    - **Training mode:** Local video file (short trimmed clips). Extract all frames. No fighter detection needed.
    - **Analysis mode:** Local file or YouTube URL (via yt-dlp). Configurable FPS (default 30). Support time-range trimming.
  - Verification: Training mode: 2-second BoxingVI clip → all frames with timestamps. Analysis mode: 3-min video at 30fps → 5,400 frames, correctly timestamped.
  - Complexity: S

- [ ] **Task 2: MediaPipe BlazePose integration**
  - Scope: Per-frame pose estimation in IMAGE mode. 33 keypoints × (x, y, z, visibility). Works on full frames and cropped ROIs (after Task 4).
  - Verification: Frame with visible boxer → 33 keypoints with valid coordinates (0-1 range) and visibility scores.
  - Complexity: S

- [ ] **Task 3: Quality filtering, normalization & storage**
  - Scope: Drop frames with confidence <0.5. Normalize keypoints relative to body bounding box (scale invariance). Normalize frame rate to 30fps. Serialize to Parquet. End-to-end pipeline for training mode and single-person analysis mode.
  - Verification: Mixed-confidence input → filtered Parquet with only high-confidence frames. Keypoints scale-invariant across resolutions. File loadable by pandas with expected schema.
  - Complexity: S

- [ ] **Task 4: Multi-person detection + tracking (analysis mode)**
  - Scope: YOLOv8 person detection per frame. IoU-based tracker to assign consistent fighter_id across frames. Crop ROI per fighter. Run MediaPipe per crop (reuse Task 2). Output: separate PoseFrame Parquet per fighter.
  - Verification: 3-min sparring video with 2 fighters → two PoseFrame files (fighter_a, fighter_b). Spot-check 10 random frames for identity consistency.
  - Complexity: M
  - Note: Not needed for B2 training (BoxingVI uses training mode) or Track A solo analysis. Required for sparring/fight analysis.

### B2: Action Detection Model (Complexity: L)

- [ ] **Task 5: Dataset preparation (BoxingVI + Olympic Boxing)**
  - Scope: Two datasets:
    - **BoxingVI** (6,915 clips, 6 punch types): Download, process through B1 training mode → PoseFrame Parquet per clip. Train/val/test split (70/15/15).
    - **Olympic Boxing Kaggle** (312K frames, 8 classes including body punches + blocks): Download, extract punch frames, process through B1 training mode. Adds body_punch and block classes absent from BoxingVI.
    - Merge strategy: BoxingVI for 6-class punch-type classification (primary). Olympic Boxing for frame-level detection pretraining and future class expansion (body punches, blocks). Document class distribution for both, check for imbalance.
  - Verification: Both datasets processed. Splits documented. Class distribution visualized per dataset. Merged label mapping documented.
  - Complexity: M

- [ ] **Task 6: Feature engineering from keypoints**
  - Scope: Design feature set from 33 keypoints per 30-frame window (1 second):
    - Joint angles: elbow, shoulder, hip
    - Angular velocities: change in angles across frames
    - Relative positions: wrist-to-shoulder, wrist-to-hip distance
    - Temporal derivatives: acceleration of key joints
  - Verification: 30-frame PoseFrame window → fixed-dimension feature vector. Feature importance analysis shows discriminative features for each punch type.
  - Complexity: M

- [ ] **Task 7: Action classifier training & evaluation**
  - Scope: Train XGBoost on engineered features. Hyperparameter tuning (cross-validation). Evaluate: per-class precision/recall/F1, confusion matrix, overall accuracy. Baseline comparison (random forest, SVM). Version model artifact.
  - Verification: ≥80% accuracy on BoxingVI test set. Confusion matrix documented. Model saved with version tag.
  - Complexity: M

- [ ] **Task 8: Sliding window TAD**
  - Scope: Slide classification window over continuous PoseFrame sequences. Window: 30 frames, stride: 5 frames. Non-maximum suppression for overlapping detections. Confidence threshold ≥0.7. Output: ActionTimeline.
  - Verification: 3-min untrimmed solo training video → ActionTimeline. Manual spot-check of 20 random predictions shows ≥70% correctness.
  - Complexity: L

- [ ] **Task 9: Model iteration & upgrade path (ongoing)**
  - Scope: Two tracks of iteration:
    - **Data iteration:** Process own footage through B1. Label actions with annotation tool. Retrain on BoxingVI + Olympic Boxing + self-labeled data. Focus on domain adaptation and adding defensive actions.
    - **Architecture upgrade (v2):** When combined training data exceeds ~10K clips, evaluate AcT (Action Transformer) as classifier upgrade. AcT is compatible with MediaPipe 2D keypoint input, runs at ~12ms on Apple Silicon MPS (below 33ms real-time threshold), and achieves 88-93% accuracy in literature. Compare against XGBoost v1 on same test set before switching.
  - Verification: Each model version ≥ previous version accuracy on BoxingVI test set. AcT upgrade: side-by-side comparison documented.
  - Complexity: M (ongoing, not blocking B3)

### B3: Track A Integration — Drill Verification (Complexity: M)

- [ ] **Task 10: Combo sequence recognition**
  - Scope: Group ActionTimeline entries into ComboSequences using time-gap heuristic. Actions within 0.8s = same combo. Gap ≥0.8s or idle ≥0.5s = combo boundary.
  - Verification: ActionTimeline with [jab@12.3, cross@12.5, lead_hook@12.9, (gap), jab@15.1, cross@15.3] → two combos: [jab, cross, lead_hook] and [jab, cross].
  - Complexity: S

- [ ] **Task 11: Session matching — instruction vs execution**
  - Scope: Compare TTS instruction log (from Track A session engine) with detected ComboSequences. For each instruction: find detected combo within 3s matching window. Classify: success (exact match), partial (subset match), miss (no match or wrong combo).
  - Verification: TTS log [{t=12.0, combo="jab-cross-hook"}] + detected combo [jab, cross] at t=12.3-12.9 → result: "partial", detail: "lead_hook missing".
  - Complexity: M

- [ ] **Task 12: DrillFeedback generation**
  - Scope: Aggregate SessionMatch results into DrillFeedback. Per-combo stats (attempts, success rate). Identify frequently missed actions. Generate actionable notes for Track A's LLM context (e.g., "user misses lead_hook 70% of the time in 3-punch combos").
  - Verification: Session with 20 instructions → DrillFeedback with per-combo breakdown, overall accuracy, and ≥2 actionable notes. Notes are specific (not generic).
  - Complexity: M

- [ ] **Task 13: Track A data integration**
  - Scope: DrillFeedback feeds into Track A's UserProfile (combo_mastery updates) and LLM session planner (context for next session). Define the interface: what fields Track A reads, how DrillFeedback maps to UserProfile updates.
  - Verification: DrillFeedback with "jab-cross success rate 90%, jab-cross-hook success rate 30%" → UserProfile combo_mastery updated accordingly. LLM receives this in session context.
  - Complexity: S
  - **Dependency:** Tasks 10-12 can be built and tested with mock TTS logs and mock Track A schemas. Task 13 requires A1's data model to be finalized before wiring. Design the interface now; connect when A1 ships.

### B4: Fight Analysis Extension (Complexity: M, optional)

> **Analysis model progression:**
> - **v1 (now):** Gemini 2.5 Pro VLM for tactical analysis — pragmatic given limited interaction data.
> - **v2:** As B1-B3 accumulate detected interaction data, start labeling interaction patterns (attack → response → outcome).
> - **v3:** Train own tactical model (graph-based or sequence model on interaction sequences). Requires ~5K+ annotated interaction sequences. BoxMind (Paris 2024) proved this is feasible with sufficient data (10.9K events, 119 hours footage → 69.8% match prediction accuracy).
> VLM is not a permanent dependency — it bootstraps tactical analysis until enough structured data exists to train a dedicated model.

- [ ] **Task 14: Gemini 2.5 Pro integration for fight footage**
  - Scope: API client. Send both fighters' ActionTimelines + sampled video frames (2-4 FPS) to Gemini. VLM correlates interactions: who attacked → who responded → estimated outcome. Structured output parsing.
  - Verification: Fight video + two ActionTimelines → parseable SituationalTactic JSON. Cost per 3-min video documented.
  - Complexity: S

- [ ] **Task 15: Situational tactic extraction — prompt engineering**
  - Scope: Prompts that produce actionable sparring tips: "when you see X, try Y." VLM must correlate both fighters' actions, identify recurring patterns, and estimate outcomes. Focus on actionable advice, not generic narrative.
  - Verification: 3-round fight video → ≥3 distinct situational tactics, each with concrete situation + response + evidence timestamps.
  - Complexity: M

- [ ] **Task 16: DrillTemplate export to Track A**
  - Scope: Convert SituationalTactics → DrillTemplates compatible with Track A's Combo Registry. Handle unknown actions (flag for review). User-curated import (no auto-import v1).
  - Verification: SituationalTactic → valid DrillTemplate with actions from Track A vocabulary. Unknown actions produce warning + partial template.
  - Complexity: S

## 4. Boundaries

- ✅ **Always:**
  - Every pipeline step independently testable with fixture data
  - Version all model artifacts, training data splits, and evaluation metrics
  - Pipeline reproducible: same input → same output (except VLM non-determinism)
  - Copyright: store only keypoints from YouTube videos, delete raw video after extraction
  - Append-only for PoseFrame and ActionTimeline (immutable records)
  - Local-first: runs on local machine unless explicitly moved to cloud
  - B3 (Track A integration) is higher priority than B4 (fight analysis)
  - Track A still works without Track B (audio-only mode) — Track B adds video verification

- ⚠️ **Ask first:**
  - Training data sourcing beyond BoxingVI (legal/ethical)
  - Cloud GPU usage (cost)
  - Adding defensive action classes (slip, duck, block) — requires new training data
  - Changes to Track A's data model for B3 integration
  - Auto-import of DrillTemplates (currently user-curated only)

- 🚫 **Never:**
  - Use VLMs for real-time punch detection (1 FPS misses punches)
  - Use VideoMAE / video foundation models on <500 clips (overfitting)
  - Store raw YouTube video files (copyright)
  - Ship a model without documented evaluation metrics
  - Auto-import drill templates without user review (v1)
  - Make Track A fail if Track B is unavailable — audio-only mode must always work

## 5. Testing Strategy

- **Unit:**
  - B1: Single frame with visible boxer → 33 keypoints, valid coordinates
  - B1 analysis mode (multi-person): Sparring frame with 2 people → 2 separate ROI detections
  - B2: Known jab PoseFrame sequence → classifier outputs "jab" with ≥0.8 confidence
  - B3: Known ActionTimeline + TTS log → correct SessionMatch results

- **Integration:**
  - Solo pipeline: user training video → B1 (analysis, single-person) → B2 → B3 → DrillFeedback
  - Fight pipeline: YouTube fight → B1 (analysis, multi-person) → B2 → B4 → SituationalTactic
  - Determinism: same video re-run → identical PoseFrame and ActionTimeline

- **Data Pipeline:**
  - Each stage tested with saved fixture data (sample Parquet, JSON)
  - Fixture files committed to repo for reproducibility

- **ML Evaluation:**
  - B2: confusion matrix, per-class precision/recall/F1, overall accuracy
  - B2 TAD: spot-check protocol (20 random predictions, verify correctness)
  - Metrics tracked per model version in evaluation log

- **Conformance:**
  - Input: User's 3-min solo training video (shadow boxing with TTS drill)
  - Expected: ActionTimeline with ≥10 detected actions, each with class + timestamps + confidence ≥0.7
  - Input: ActionTimeline + TTS instruction log [{t=12.0, "jab-cross-hook"}, {t=25.0, "jab-cross"}...]
  - Expected: SessionMatch with per-instruction results (success/partial/miss) + DrillFeedback summary
  - Input: 3-min sparring video through full fight pipeline
  - Expected: Two ActionTimelines (one per fighter), VLM produces ≥2 situational tactics

## 6. Open Questions

- [ ] **BoxingVI access:** Verify dataset is freely available for research. Download procedure.
- [ ] **Annotation tool for self-labeling:** Build custom or use existing (CVAT, Label Studio)?
- [ ] **Defensive action taxonomy:** When to add slip/duck/block? Needs separate training data not in BoxingVI. Likely after v1 model ships.
- [ ] **Own sparring footage:** When to start processing personal sparring? After B2 is validated on solo footage?
- [ ] **VLM cost:** Benchmark Gemini 2.5 Pro cost per video before committing to B4 at scale.
- [ ] **B3 ↔ A1 interface:** Exact contract between DrillFeedback and Track A's UserProfile/LLM context. Resolve when A1 spec is drafted.
- [ ] **Session recording setup:** What camera angle/distance/position works best for B1 analysis mode (single-person)? Needs testing with real footage.
- [ ] **Olympic Boxing Kaggle license:** Non-commercial only. Verify compatibility with project's intended use before investing in processing.
- [ ] **AcT upgrade timing:** When does combined dataset cross ~10K clips threshold? Track data accumulation rate to plan v2 transition.
- [ ] **Own tactical model (v3):** What interaction annotation schema to use when labeling attack→response→outcome patterns? Design early so B1-B3 data is labeled consistently for future model training.
- [ ] **BoxMAC re-release:** Monitor arXiv for re-submission — 13 classes including 7 defensive actions would be the most complete boxing dataset. Withdrawn Feb 2025.
- [ ] **ShadowPunch dataset size:** Exact clip count undisclosed. Evaluate when available — CC BY 4.0 license is most permissive.

## Data Contract

### Inputs
| Field | Type | Source | Required |
|-------|------|--------|----------|
| video_file | mp4/webm | Local recording or YouTube (yt-dlp) | yes |
| mode | string | "training" (trimmed clips) / "analysis" (full videos) | yes |
| multi_person | boolean | Whether to run YOLO person detection (analysis mode only) | no (default: false) |
| tts_instruction_log | JSON | Track A session engine — TTSInstructionLog format (see Data Model) | B3 only |

### Outputs
| Field | Type | Consumer | Storage | Mutable |
|-------|------|----------|---------|---------|
| PoseFrame | Parquet | B2 classifier | File (per-video, per-fighter) | no |
| ActionTimeline | JSON | B3, B4 | File + SQLite index | no |
| ComboSequence | JSON | B3 session matching | Ephemeral (derived) | no |
| SessionMatch | JSON | B3 feedback generation | SQLite | no |
| DrillFeedback | JSON | Track A (UserProfile, LLM context) | SQLite | no |
| SituationalTactic | JSON | B4 drill export, user review | SQLite | no |
| DrillTemplate | JSON | Track A Combo Registry (A1) | DB (via A1 import) | yes |

### Schema Version
- Current: v1
- Migration: Append new fields, never remove. PoseFrame and ActionTimeline are immutable.

### Volume & Retention
- Solo training: ~1 video/day during active training. ~10MB PoseFrame per 3-min video.
- Fight analysis: ~2-5 videos/week during research. Same size per video.
- All derived data (ActionTimeline, DrillFeedback): <100KB per session. Keep indefinitely.
- Raw video: Deleted after pose extraction for YouTube sources. User's own training videos: user decides.

## Changelog

| Date | Change | Reason |
|---|---|---|
| 2026-03-07 | Complete rewrite: Track B refocused as action detection model serving Track A | User clarified: Track B's purpose is building the model that detects user actions for drill feedback, not just analyzing external fights |
| 2026-03-07 | Added B3 (Track A Integration) before B4 (Fight Analysis) | Model must serve Track A first. Fight analysis is an extension. |
| 2026-03-07 | Added two B1 modes (training, analysis) | Model must support solo training, sparring, and fight footage |
| 2026-03-07 | Added ComboSequence, SessionMatch, DrillFeedback data models | B3 needs these to verify drill execution and generate feedback for Track A |
| 2026-03-07 | Merged clip/solo/fight into training mode + analysis mode (single/multi-person) | Review R1: fewer code paths, cleaner abstraction |
| 2026-03-07 | Added TTSInstructionLog data model with explicit format | Review R1: B3 needs a defined interface for Track A's instruction data |
| 2026-03-07 | Task 13: explicit dependency on A1, Tasks 10-12 testable with mocks | Review R1: clarify what blocks on Track A vs what can be built independently |
| 2026-03-07 | Renumbered tasks to sequential 1-16 (was 1-17 with gaps), fixed all mode naming references | Review R2: consistency pass after R1 mode merge |
| 2026-03-07 | Added Olympic Boxing Kaggle as second dataset (Task 5), AcT upgrade path (Task 9), VLM→own model progression (B4), BoxMind architecture validation | Review R3: user requested dataset research, model alternatives, and own analysis model consideration |
| 2026-03-07 | Added References section with all papers and datasets cited | Review R3: user requested all references attached to spec |
| 2026-03-07 | Status: DRAFT → APPROVED | User approved after R3 review. 12 open questions remain (none blocking). |

## References

### Boxing-Specific Datasets

| # | Name | Citation | URL |
|---|---|---|---|
| D1 | BoxingVI | arXiv 2511.16524, Nov 2024. 6,915 clips, 6 punch types, 18 athletes | [Paper](https://arxiv.org/html/2511.16524v1) / [GitHub](https://github.com/Bikudebug/BoxingVI) |
| D2 | Olympic Boxing Punch Classification | Stefanski et al., MDPI Entropy 2024. 312K frames, 8 classes, referee-annotated | [Kaggle](https://www.kaggle.com/datasets/piotrstefaskiue/olympic-boxing-punch-classification-video-dataset) / [Paper](https://www.mdpi.com/1099-4300/26/8/617) |
| D3 | BoxMAC (withdrawn) | arXiv 2412.18204, Dec 2024. 60K frames, 13 classes (6 punches + 7 actions). Withdrawn Feb 2025 | [arXiv](https://arxiv.org/abs/2412.18204) |
| D4 | ShadowPunch | ICLR 2025 submission. High-fps shadowboxing, frame-level event spotting, CC BY 4.0 | [OpenReview](https://openreview.net/forum?id=Jq8HYNZG9s) |
| D5 | FACTS Boxing | arXiv 2412.16454, Dec 2024. 8,000 clips, 8 classes, transformer model 83.25% | [Paper](https://arxiv.org/html/2412.16454v1) |
| D6 | Punch_DL | 240 clips, 7 punch types + strong/weak variants, video + IMU | [GitHub](https://github.com/balezz/Punch_DL) |
| D7 | BoxingPro Multimodal | MDPI Electronics Oct 2025. 200fps video + 9-axis IMU. Not yet released | [Paper](https://www.mdpi.com/2079-9292/14/21/4155) |
| D8 | PhysPose Boxing | arXiv 2504.08175, Apr 2025. Multi-camera 3D pose, elite sparring. Not yet released | [Paper](https://arxiv.org/html/2504.08175v1) / [Project](https://hosseinfeiz.github.io/physpose/) |
| D9 | KTH (boxing class) | ~100 clips, single "boxing" class, 160×120px. Historical baseline only | [Dataset](https://www.csc.kth.se/cvap/actions/) |

### Transfer Learning Datasets

| # | Name | Size | Relevant Classes | URL |
|---|---|---|---|---|
| T1 | Kinetics-400 | 300K clips, 400 classes | boxing, kickboxing, punching person, martial art (~4K+ relevant) | [GitHub](https://github.com/cvdfoundation/kinetics-dataset) |
| T2 | HMDB51 | 6,766 clips, 51 classes | punch (~1,060 clips), kick_person | [Dataset](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) |
| T3 | UCF101 | 13,320 clips, 101 classes | Boxing Punching Bag, Boxing Speed Bag (~200 clips) | [Dataset](https://www.crcv.ucf.edu/data/UCF101.php) |
| T4 | NTU RGB+D 120 | 114,480 videos, 120 classes | A50 punch/slap, A51 kicking (3D skeleton) | [Dataset](https://rose1.ntu.edu.sg/dataset/actionRecognition/) |
| T5 | MADS | 53K frames | Karate sequences (MoCap ground truth) | [Dataset](http://visal.cs.cityu.edu.hk/downloads/mads-data-download/) |
| T6 | TUHAD | 1,936 samples, 8 taekwondo techniques | CC BY | [Paper](https://www.mdpi.com/1424-8220/20/17/4871) |
| T7 | AVA | 1.62M annotations | fight/hit a person, martial art. CC BY 4.0 | [Dataset](https://research.google.com/ava/) |
| T8 | FineGym | Large-scale gymnastics | Hierarchical annotation architecture template | [Project](https://sdolivia.github.io/FineGym/) |
| T9 | FineSports | 10K NBA videos, 52 classes | Multi-person spatio-temporal methodology | [GitHub](https://github.com/PKU-ICST-MIPL/FineSports_CVPR2024) |

### Action Recognition & Classification Papers

| # | Paper | Key Finding | URL |
|---|---|---|---|
| P1 | Active Learning Boxing (PLOS ONE 2025) | 91% accuracy, pose-based classifier, 15% labeling effort | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12061147/) |
| P2 | BoxMind (arXiv Jan 2026) | Full boxing AI deployed at Paris 2024 Olympics. Punch F1=0.783, match prediction 69.8% | [Paper](https://arxiv.org/html/2601.11492) |
| P3 | AcT — Action Transformer (Pattern Recognition 2022) | Lightweight pure-attention on 2D pose, MPOSE2021 benchmark | [Paper](https://arxiv.org/abs/2107.00606) |
| P4 | SkateFormer (ECCV 2024) | 92.6-97.0% on NTU RGB+D, skeletal-temporal attention | [Paper](https://arxiv.org/abs/2403.09508) |
| P5 | Contrastive Learning + CBAM for Boxing (Springer 2025) | Contrastive learning for boxing video classification | [Paper](https://link.springer.com/article/10.1007/s11760-025-05018-2) |
| P6 | Multimodal Boxing (Nature Scientific Reports 2025) | 3D-ResNet + BERT for boxing action + psychology | [Paper](https://www.nature.com/articles/s41598-025-34771-0) |
| P7 | BoxNet (IEEE CCDC 2023) | GCN for skeleton-based boxing technique recognition | [Paper](https://ieeexplore.ieee.org/document/10327379/) |
| P8 | Hierarchical Punch Pipeline (Springer CVIP 2024) | Two-stage hierarchical approach to punch classification | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-93688-3_19) |

### Temporal Action Detection Papers

| # | Paper | Key Finding | URL |
|---|---|---|---|
| P9 | TriDet (CVPR 2023) | 69.3% mAP on THUMOS-14, trident-based TAD | [Paper](https://arxiv.org/abs/2303.07347) |
| P10 | SoccerNet 2024 Challenges | Ball Action Spotting benchmark, E2E-Spot baseline | [Paper](https://arxiv.org/html/2409.10587v1) |
| P11 | Temporal Action Detection Overview (Springer AI Review 2023) | Canonical TAD survey, still definitive | [Paper](https://link.springer.com/article/10.1007/s10462-023-10650-w) |
| P12 | Adaptive Temporal Action Localization (MDPI Electronics 2025) | Adaptive TAL methods | [Paper](https://www.mdpi.com/2079-9292/14/13/2645) |

### Pose Estimation & Video Understanding

| # | Paper / Resource | Key Finding | URL |
|---|---|---|---|
| P13 | MediaPipe BlazePose | 33 keypoints, IMAGE mode, single-person standard | [Google AI](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) |
| P14 | Roboflow: Best Pose Models 2025 | Comparison of BlazePose, MoveNet, YOLOv11 Pose | [Blog](https://blog.roboflow.com/best-pose-estimation-models/) |
| P15 | ML Pose Estimation Narrative Review (PMC 2024) | Survey of pose estimation for sports applications | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11566680/) |
| P16 | VideoMAE Overfitting on Small Data | GitHub issue confirming overfitting on <500 clips | [GitHub](https://github.com/MCG-NJU/VideoMAE/issues/129) |
| P17 | InternVideo2 (arXiv 2024) | SOTA across 60+ video/audio tasks | [Paper](https://arxiv.org/abs/2403.15377) |
| P18 | MLX Benchmarking on Apple Silicon (arXiv 2025) | M1/M2 inference latency baselines | [Paper](https://arxiv.org/html/2510.18921v1) |

### VLM & LLM for Coaching

| # | Paper / Resource | Key Finding | URL |
|---|---|---|---|
| P19 | Gemini 2.5 Video Understanding | 1M+ token context, 1 FPS default sampling | [Blog](https://developers.googleblog.com/en/gemini-2-5-video-understanding/) |
| P20 | MMA Analysis with Gemini (Towards AI, Feb 2026) | Structured video captioning for combat sports | [Article](https://pub.towardsai.net/structured-video-captioning-with-gemini-an-mma-analysis-use-case-bfbb8fd91a26) |
| P21 | GameRun AI: VLMs for Sports | Spatio-temporal VLM architecture for sports | [Blog](https://gamerun.ai/blog/the-spatio-temporal-frontier-architecting-vlms-for-the-unstructured-realism-of-sports) |
| P22 | GPTCoach (CHI 2025, Stanford HCI) | LLM coaching with MI strategies, 93% MI-consistent | [Paper](https://arxiv.org/abs/2405.06061) / [GitHub](https://github.com/StanfordHCI/GPTCoach-CHI2025) |
| P23 | LLM-SPTRec (Nature Scientific Reports 2026) | Knowledge-graph augmented sports training plans | [Paper](https://www.nature.com/articles/s41598-026-37075-z) |
| P24 | JMIR: LLM Exercise Coaching Evaluation (2025) | Scoping review of LLM coaching strategies | [Paper](https://www.jmir.org/2025/1/e79217) |
| P25 | AI in Martial Arts Survey (SAGE 2025) | Comprehensive survey of AI applications in martial arts | [Paper](https://journals.sagepub.com/doi/10.1177/17543371241273827) |
