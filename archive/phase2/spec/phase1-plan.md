# Atom Phase 1: Data Engine Plan

## Philosophy

The data engine is the product. Everything else is a wrapper.

```
Data Engine (core)
├── CTR-GCN Model — "이 사람이 뭘 쳤는지 알아내기"
├── Analysis Pipeline — "영상 → 구조화된 데이터"
├── Session Generator — "다음에 뭘 시킬지 결정하기"
└── Feedback Generator — "뭐가 달라졌는지 보여주기"

Wrappers (later)
├── Backend API — thin REST layer over the engine
└── Mobile App — thin UI that triggers engine + displays results
```

**Priority order:**
1. Data collection + CTR-GCN model (the brain)
2. Analysis pipeline (the nervous system)
3. LLM integration: session generation + feedback (the voice)
4. Backend API + Mobile app (the skin) — deferred, trivial once engine works

---

## Part 1: CTR-GCN Model

### 1.1 Data Collection

Two parallel tracks to gather ~4000+ labeled keypoint sequences.

**Track A — Self-filming (~2000 clips)**

| Class | Target | Notes |
|-------|--------|-------|
| jab | 200+ | Vary stance (orthodox/southpaw) |
| cross | 200+ | |
| lead_hook | 200+ | Head-level |
| rear_hook | 200+ | Head-level |
| lead_uppercut | 200+ | |
| rear_uppercut | 200+ | |
| body_shot | 200+ | Critical: must be distinguishable from head shots by Y-coordinate |
| idle / guard | 100+ | Standing guard, footwork without punches |
| slip | 50+ | Phase 2 priority, minimal for now |
| duck | 50+ | Phase 2 priority |
| backstep | 50+ | Phase 2 priority |

Variation matrix per punch:
- Camera angle: front, 45-left, 45-right
- Distance: 2m, 3m, 4m
- Lighting: bright, dim
- Clothing: tank top, t-shirt, hoodie

**Track B — YouTube pipeline (~2000-3000 clips)**

```
1. Search: "boxing combination drill", "boxing pad work tutorial"
   → Pro coach channels, clear single-person shots

2. Audio labeling (Whisper STT):
   "Jab! Cross! Hook!" → timestamp + action label
   → Only keep segments where audio label is clear

3. Keypoint extraction (MediaPipe):
   → 33 landmarks per frame, VIDEO mode
   → Store keypoints only (not video — copyright)

4. Quality filter:
   → Average visibility > 0.5
   → Single person in frame
   → Sample-verify 100 clips: >95% label accuracy
```

**Output format per sample:**
```python
{
    "label": "cross",                    # action class
    "keypoints": np.array (T, 33, 3),   # x, y, z normalized
    "fps": 30,
    "source": "self" | "youtube",
    "metadata": {...}
}
```

**Tools to build:**
- `ml/scripts/extract_keypoints.py` — MediaPipe batch extraction from video files
- `ml/scripts/collect_youtube.py` — YouTube download + Whisper + auto-label + extract
- `ml/scripts/visualize.py` — Overlay skeleton on frames for spot-checking

**Verification:**
- ≥200 samples per punch class, ≥100 idle, ≥50 each defensive
- Skeleton visualization spot-check: 20 random samples look correct
- Label accuracy: sample 100 → >95% correct

---

### 1.2 Graph Topology

MediaPipe outputs 33 landmarks. CTR-GCN needs a graph (adjacency matrix). We use a subset optimized for upper-body boxing actions.

**Selected joints (15):**

```
Joint Index → Name
  0  → nose
  7  → left_ear
  8  → right_ear
 11  → left_shoulder
 12  → right_shoulder
 13  → left_elbow
 14  → right_elbow
 15  → left_wrist
 16  → right_wrist
 23  → left_hip
 24  → right_hip
 25  → left_knee
 26  → right_knee
 27  → left_ankle
 28  → right_ankle
```

**Why 15, not 11 (old LSTM) or 33 (full MediaPipe):**
- GCN needs graph structure — more nodes = richer spatial relationships
- Upper body is critical for punch recognition, but hips/knees/ankles capture weight transfer and stance
- 33 is overkill: hand fingers, face mesh, toe landmarks add noise not signal
- 15 keeps the graph manageable while capturing the full kinetic chain (fist → elbow → shoulder → hip → knee → ankle)

**Adjacency matrix edges (natural skeleton):**
```
nose ↔ left_ear, right_ear
nose ↔ left_shoulder, right_shoulder
left_shoulder ↔ left_elbow ↔ left_wrist
right_shoulder ↔ right_elbow ↔ right_wrist
left_shoulder ↔ left_hip
right_shoulder ↔ right_hip
left_hip ↔ right_hip
left_hip ↔ left_knee ↔ left_ankle
right_hip ↔ right_knee ↔ right_ankle
```

**File:** `ml/graph/boxing_graph.py`
- `num_node = 15`
- `self_link`, `inward`, `outward` edge lists
- `get_adjacency_matrix()` → normalized A (identity + inward + outward)

**Verification:**
- Adjacency matrix shape: `(3, 15, 15)` — 3 subsets (identity, centripetal, centrifugal)
- Symmetric: `A[i][j] == A[j][i]` for each subset
- Graph visualization: plot nodes + edges, verify skeleton makes sense

---

### 1.3 CTR-GCN Architecture

Adapted from [official CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) for our 15-joint boxing skeleton.

**Model structure:**

```
Input: (N, 3, 30, 15, 1)
  N = batch, C = 3 (xyz), T = 30 frames (1s), V = 15 joints, M = 1 person

→ BatchNorm (data normalization)
→ 10× CTRGC Block:
    ├── Spatial: Channel-wise Topology Refinement Graph Conv
    │   - Learn channel-specific graph topologies
    │   - Refine shared topology per channel group
    └── Temporal: Multi-Scale Temporal Conv (MS-TCN)
        - Dilated convolutions at multiple scales
        - Captures both fast jabs and slow hooks
→ Global Average Pool (T, V dimensions)
→ FC → 11 classes
```

**Key modules:**

| Module | File | Purpose |
|--------|------|---------|
| `CTRGC` | `ml/model/modules.py` | Channel-wise topology refinement graph conv |
| `MSTCN` | `ml/model/modules.py` | Multi-scale temporal convolution |
| `CTRGCBlock` | `ml/model/modules.py` | CTRGC + MSTCN + residual |
| `CTRGCN` | `ml/model/ctrgcn.py` | Full model: 10 blocks + GAP + FC |

**Channel progression:** `3 → 64 → 64 → 64 → 64 → 128 → 128 → 128 → 256 → 256 → 256`

**Data augmentation (in feeder):**

| Augmentation | Description | Why |
|-------------|-------------|-----|
| Horizontal flip | Mirror left↔right | Stance invariance (orthodox/southpaw) |
| Random scale | 0.9–1.1× | Person size variation |
| Temporal crop | Random start within window | Temporal invariance |
| Gaussian noise | σ=0.01 on coordinates | Robustness to pose jitter |
| Joint dropout | Zero out random joints (p=0.05) | Robustness to occlusion |

**File:** `ml/feeders/boxing_feeder.py`
- PyTorch Dataset
- Loads `.npy` keypoint files + labels
- Applies augmentation pipeline
- Outputs `(C, T, V, M)` tensors

**Verification:**
- Forward pass: `(16, 3, 30, 15, 1)` → `(16, 11)` output
- Parameter count: ~1-2M (should be lightweight for server inference)
- Single forward pass: <50ms on CPU

---

### 1.4 Training & Evaluation

**Training config (`ml/configs/ctrgcn.yaml`):**

```yaml
model:
  num_classes: 11
  num_joints: 15
  in_channels: 3
  num_layers: 10

training:
  epochs: 80
  batch_size: 32
  optimizer: SGD
  lr: 0.1
  momentum: 0.9
  nesterov: true
  weight_decay: 0.0004
  lr_schedule:
    type: step
    milestones: [40, 60]
    gamma: 0.1

data:
  window_size: 30      # frames (1 second at 30fps)
  stride: 5            # sliding window stride
  split: [0.8, 0.1, 0.1]  # train/val/test
  stratified: true

augmentation:
  flip: true
  scale_range: [0.9, 1.1]
  noise_std: 0.01
  joint_dropout: 0.05
```

**Training script:** `ml/scripts/train.py`
- Stratified 80/10/10 split
- Track: loss, accuracy, per-class accuracy per epoch
- Save best model (val accuracy) + latest model
- TensorBoard or Weights & Biases logging
- Early stopping: patience 10 on val accuracy

**Evaluation script:** `ml/scripts/evaluate.py`
- Confusion matrix (11×11)
- Per-class precision, recall, F1
- Overall accuracy
- Specific analysis: body_shot vs head attacks confusion
- Inference speed benchmark

**Export script:** `ml/scripts/export_model.py`
- TorchScript export (`.pt`)
- Bundle: model + preprocessing config (joint indices, normalization params)
- Validate: exported model matches training model on test samples

**Target metrics:**

| Metric | Target | Hard requirement |
|--------|--------|------------------|
| Overall test accuracy | ≥90% | Yes |
| Per-class accuracy (7 punches + idle) | ≥80% each | Yes |
| body_shot vs head confusion | ≤15% | Yes |
| Inference per window (CPU) | <50ms | Yes |
| Defensive classes (slip/duck/backstep) | ≥70% | No (Phase 2 focus) |

**Fallback if <90%:**
1. Reduce to 8 classes: merge defensive into "other"
2. Raise confidence threshold to 0.8
3. Collect more data for confused classes
4. Try different window sizes (45 frames = 1.5s)

**Verification:**
- Confusion matrix shows no systematic error >15%
- TorchScript model reproduces training model outputs exactly
- End-to-end: video file → MediaPipe → CTR-GCN → action labels (manual inspection)

---

## Part 2: Analysis Pipeline

The pipeline transforms a raw session video into structured drill analysis data.

```
Video (mp4)
  → Stage 1: Pose Extraction (MediaPipe)
    → Stage 2: Action Classification (CTR-GCN)
      → Stage 3: Sequence Recognition (rule-based)
        → Stage 4: Session Matching (rule-based)
          → DrillAnalysis (structured JSON)
```

### 2.1 Stage 1 — Pose Extraction

**Input:** Video file (mp4)
**Output:** `List[KeypointFrame]` — per-frame (timestamp_ms, keypoints[33][4])

```python
@dataclass
class KeypointFrame:
    timestamp_ms: float
    keypoints: np.ndarray    # (33, 4) — x, y, z, visibility
    avg_visibility: float
```

**Process:**
1. Open video with OpenCV, get fps and total frames
2. Run MediaPipe PoseLandmarker in VIDEO mode
3. For each frame:
   - Extract 33 landmarks (x, y, z, visibility)
   - Compute avg_visibility
   - Filter: avg_visibility < 0.5 → drop frame
4. Normalize coordinates:
   - Center: translate so hip_center = (0, 0, 0)
   - Scale: divide by shoulder_width
5. Resample to 30fps if source fps differs
6. Select 15-joint subset (indices from graph topology)

**File:** `ml/pipeline/pose_extractor.py`

**Verification:**
- 12-min video at 30fps → ~21,600 frames → processes in <30s
- Dropped frames logged with timestamps
- Skeleton visualization on 10 random frames matches video

---

### 2.2 Stage 2 — Action Classification

**Input:** Keypoint sequence from Stage 1
**Output:** `List[DetectedAction]`

```python
@dataclass
class DetectedAction:
    timestamp: float         # seconds (center of window)
    action: str              # e.g. "jab", "cross"
    confidence: float        # 0.0–1.0
    window_start: float      # seconds
    window_end: float        # seconds
```

**Process:**
1. Sliding window: size=30 frames (1s), stride=5 frames (~0.167s)
2. For each window:
   - Extract (3, 30, 15, 1) tensor
   - CTR-GCN forward pass → softmax → (11,) probabilities
   - Top class + confidence
3. Filter: confidence < 0.7 → discard
4. Filter: action == "idle" → discard (not informative)
5. Non-maximum suppression:
   - Overlapping windows with same action: keep highest confidence
   - Merge adjacent same-action windows within 0.2s

**File:** `ml/pipeline/action_classifier.py`

**Verification:**
- Known test video with clear jab-cross → outputs [jab, cross] with confidence >0.8
- NMS correctly merges duplicate detections
- Processing speed: <50ms per window → 12-min session (~2100 windows) in <2min

---

### 2.3 Stage 3 — Sequence Recognition

**Input:** `List[DetectedAction]` from Stage 2
**Output:** `List[DetectedCombo]`

```python
@dataclass
class DetectedCombo:
    start_time: float       # seconds
    end_time: float         # seconds
    actions: List[str]      # e.g. ["jab", "cross", "lead_hook"]
```

**Rules (from TECHSPEC §2.1):**
```
1. Sort DetectedActions by timestamp
2. Initialize current_combo = [first_action]
3. For each subsequent action:
   - gap = action.timestamp - previous_action.timestamp
   - if gap < 0.8s → add to current_combo
   - if gap ≥ 0.8s → close current_combo, start new one
4. Single-action "combos" are kept (single punches are valid)
```

**File:** `ml/pipeline/sequence_recognizer.py`

**Verification:**
- Input: `[jab@1.0, cross@1.3, lead_hook@1.5, jab@3.0, cross@3.2]`
- Output: `[[jab,cross,lead_hook]@1.0-1.5, [jab,cross]@3.0-3.2]`
- Edge case: no actions → empty list
- Edge case: all actions within 0.8s → single long combo

---

### 2.4 Stage 4 — Session Matching

**Input:**
- TTS instruction log: `List[TTSInstruction]` (from session)
- Detected combos from Stage 3

```python
@dataclass
class TTSInstruction:
    timestamp: float        # seconds (when TTS played)
    combo_name: str         # user's name, e.g. "잽잽양훅" (display only)
    expected_actions: List[str]  # canonical, e.g. ["jab", "cross", "lead_hook", "rear_hook"]

@dataclass
class Match:
    instruction_idx: int
    combo_idx: Optional[int]
    result: str             # "success" | "partial" | "miss"
    detail: str             # human-readable explanation
    expected: List[str]
    actual: Optional[List[str]]
```

**Rules (from TECHSPEC §2.1):**
```
1. For each TTS instruction at time T:
   - Search window: [T, T + 3.0s]
   - Find all DetectedCombos overlapping this window
   - If no combo found → result = "miss"
   - If combo found:
     - Compare instruction.expected_actions vs combo.actions
     - Exact match → "success"
     - Partial overlap (≥50% actions match) → "partial"
     - No overlap → "miss"
   - If multiple combos in window, pick best match
2. Aggregate into ComboStats:
   - Per action sequence key (e.g. "jab-cross-lead_hook"): {attempts, successes, partials, misses, success_rate}
```

**No combo library lookup needed:**
- TTS log stores both user's name AND action sequence at recording time
- Matching uses action sequences directly (universal, not gym-specific)
- User's combo name is passed through for display in feedback only

**File:** `ml/pipeline/session_matcher.py`

**Output:**
```python
@dataclass
class DrillResult:
    matches: List[Match]
    combo_stats: Dict[str, ComboStat]  # per-combo aggregates
    total_instructions: int
    total_success: int
    total_partial: int
    total_miss: int
    overall_success_rate: float
```

**Verification:**
- Known TTS log + known detections → expected matches
- Edge cases: overlapping instructions, combo in gap between instructions
- Partial match: "원-투-바디" instructed, detected [jab, cross, lead_hook] → partial (바디 missing)

---

### 2.5 Pipeline Orchestrator

**File:** `ml/pipeline/pipeline.py`

```python
class AnalysisPipeline:
    def __init__(self, model_path: str, config: BoxingConfig):
        self.pose_extractor = PoseExtractor()
        self.classifier = ActionClassifier(model_path, config)
        self.recognizer = SequenceRecognizer(config)
        self.matcher = SessionMatcher(config)

    def analyze(self, video_path: str, tts_log: List[TTSInstruction]) -> DrillResult:
        # Stage 1
        keypoints = self.pose_extractor.extract(video_path)
        # Stage 2
        actions = self.classifier.classify(keypoints)
        # Stage 3
        combos = self.recognizer.recognize(actions)
        # Stage 4
        result = self.matcher.match(tts_log, combos)
        return result
```

**Error handling:**
- Stage 1 fails (no poses): return empty result with error flag
- Stage 2 all low confidence: return result with 0 detections, all instructions "miss"
- Each stage logs intermediate results for debugging

**Performance target:** <2 minutes for a 12-minute session video

**Verification:**
- End-to-end test with a real boxing video + mock TTS log
- Pipeline returns valid DrillResult
- Intermediate results (keypoints, actions, combos) are inspectable

---

## Part 3: LLM Integration

### 3.1 Session Generation

**Purpose:** Given user history, generate the next drill session plan.

**3-Layer Context (from TECHSPEC §2.2):**

| Layer | Content | When | ~Tokens |
|-------|---------|------|---------|
| Layer 1: User Profile | experience, combo_mastery, strengths, weaknesses | Always | ~500 |
| Layer 2: Recent Sessions | Last 3 sessions: date, type, key results, feedback | Always | ~500 |
| Layer 3: Long-term Trends | Weekly summaries, trend description | Project transitions only | ~300 |

**Prompt template:** `ml/prompts/session_drill.txt`

```
[System]
You are a boxing coach creating a personalized drill session.
Output valid JSON matching the schema below. No explanation outside JSON.

[Schema]
{
  "session_type": "drill",
  "total_rounds": 3-5,
  "round_duration": 120-180 (seconds),
  "rest_duration": 30-60 (seconds),
  "focus": "string (오늘의 포커스)",
  "focus_message": "string (Korean, 1-2 sentences for user)",
  "rounds": [{
    "round": 1,
    "theme": "string",
    "instructions": [
      {"time_offset": 0, "combo_name": "원-투", "actions": ["jab", "cross"], "repeat": 1-3}
    ],
    "focus_reminders": [
      {"time_offset": int, "message": "string (Korean)"}
    ],
    "motivation": [
      {"time_offset": int, "message": "string (Korean)"}
    ]
  }]
}

[Rules]
- Compose combos freely from the action vocabulary: {actions_list}
- Use the user's combo vocabulary when available; for new combos, output the action sequence
- Mix familiar combos (high success rate) with challenging ones (low success rate)
- Gradually introduce new action sequences at the edges of what user knows
- Gradual progression within rounds: warm up → challenge → cool down
- Focus reminders tied to user's current weakness
- Total session 8-15 minutes

[Action Vocabulary]
{actions_list}  # jab, cross, lead_hook, rear_hook, lead_uppercut, rear_uppercut, body_shot

[User's Combo Vocabulary]
{user_combo_vocab}  # e.g. {"잽잽양훅": ["jab","cross","lead_hook","rear_hook"], ...}

[User Profile]
{layer_1_json}

[Recent Sessions]
{layer_2_json}

[Generate the next drill session]
```

**Cold-start (no history):**
- Only self-assessment available
- LLM generates basic 2-hit combos (jab-cross type)
- Auto-names from default Korean: ["jab", "cross"] → "잽-크로스"
- Hardcoded fallback if LLM fails

**Validation:**
- JSON schema validation
- All actions in instructions are valid action classes (from boxing.yaml)
- time_offsets are sequential and within round_duration
- total_rounds between 3-5
- Retry once on failure, then use fallback

**File:** `ml/services/session_generator.py`

**Verification:**
- Cold-start generates valid beginner session
- Experienced user gets session reflecting their mastery gaps
- Output validates against schema 100% of time (with retry)

---

### 3.2 Feedback Generation

**Purpose:** Transform DrillResult into 3-5 sentence Korean coach feedback.

**Prompt template:** `ml/prompts/feedback_drill.txt`

```
[System]
You are a boxing coach giving feedback after a drill session.
Speak naturally in Korean as a coach would.

Rules:
- 3-5 sentences only
- Specific: use actual numbers from the data
- Change-oriented: compare to previous sessions when available
- Actionable: something they can focus on next time
- Restrained encouragement: no excessive praise

Forbidden:
- Raw decimals (say "10번 중 7번" not "70%" or "0.7")
- Technical terms (entropy, diversity_index, EMA)
- Excessive praise ("대단해요!", "최고예요!")

[User Profile]
{layer_1_json}

[This Session Results]
{drill_result_json}

[Generate feedback in Korean]
```

**File:** `ml/services/feedback_generator.py`

**Verification:**
- Output is Korean, 3-5 sentences
- No forbidden patterns (regex check)
- References actual combo names and counts from the session

---

### 3.3 Combo Mastery Update

Not LLM — pure math. Runs after each session analysis.

**EMA update:**
```python
def update_mastery(current_rate: float, session_rate: float, alpha: float = 0.3) -> float:
    return alpha * session_rate + (1 - alpha) * current_rate
```

**State machine:**
```
new → learning:     first attempt (any session_rate)
learning → proficient:  drill_success_rate ≥ 0.5 AND sessions_attempted ≥ 3
proficient → mastered:  drill_success_rate ≥ 0.8 AND consecutive_above_80 ≥ 3
```

**File:** `ml/services/mastery_updater.py`

**Verification:**
- EMA converges correctly over 10 mock sessions
- State transitions fire at exact thresholds
- New combo in session creates record with status "new"

---

## Part 4: Configuration & Vocab

### Two layers: ML config (fixed) vs User vocab (customizable)

The ML model outputs universal biomechanical classes (jab, cross, lead_hook...).
But how combos are **named** and **called** varies per gym and per user.

```
ML Layer (fixed, boxing.yaml):
  Action classes — what the model detects
  Pipeline params — engineering constants

User Vocab Layer (per-user data):
  Combo names — what the user/gym calls things
  Freely composed — any sequence of actions is a valid combo
  LLM generates new combos dynamically
```

**Why no fixed combo library:**
- A fixed list limits what the LLM can generate
- Real boxing has infinite combo variations
- Gym terminology varies: "잽잽양훅" vs "원투양훅" for the same [jab, cross, lead_hook, rear_hook]
- Combos are just ordered action sequences — the actions are fixed, the sequences are not

### `ml/configs/boxing.yaml` (ML config only)

```yaml
domain: boxing

actions:
  classes:
    - jab
    - cross
    - lead_hook
    - rear_hook
    - lead_uppercut
    - rear_uppercut
    - body_shot
    - slip
    - duck
    - backstep
    - idle
  # Default Korean names for individual actions (not combos)
  # Users can override these during onboarding
  default_korean:
    jab: "잽"
    cross: "크로스"
    lead_hook: "훅"
    rear_hook: "백훅"
    lead_uppercut: "어퍼컷"
    rear_uppercut: "백어퍼컷"
    body_shot: "바디"
    slip: "슬립"
    duck: "덕킹"
    backstep: "백스텝"

keypoints:
  source: mediapipe_pose
  subset_indices: [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
  subset_names:
    - nose
    - left_ear
    - right_ear
    - left_shoulder
    - right_shoulder
    - left_elbow
    - right_elbow
    - left_wrist
    - right_wrist
    - left_hip
    - right_hip
    - left_knee
    - right_knee
    - left_ankle
    - right_ankle

pipeline:
  fps: 30
  window_size: 30
  stride: 5
  action_confidence_threshold: 0.7
  pose_visibility_threshold: 0.5
  combo_gap_threshold: 0.8
  session_match_window: 3.0
  nms_overlap_threshold: 0.2

mastery:
  ema_alpha: 0.3
  states:
    learning:
      min_success_rate: 0.0
    proficient:
      min_success_rate: 0.5
      min_sessions: 3
    mastered:
      min_success_rate: 0.8
      min_consecutive: 3
```

### User Combo Vocab (per-user data)

Stored as part of user profile. Set during onboarding, editable anytime.

```python
# Example: user's combo vocabulary
user.combo_vocab = {
    "잽잽양훅": ["jab", "cross", "lead_hook", "rear_hook"],
    "원투바디": ["jab", "cross", "body_shot"],
    "원투": ["jab", "cross"],
    "원투쓰리": ["jab", "cross", "lead_hook"],
    # ... user adds more over time
}
```

**How it flows through the system:**

| Stage | Uses vocab? | How |
|-------|------------|-----|
| CTR-GCN model | No | Outputs universal action classes |
| Pipeline Stages 1-3 | No | Works with action classes only |
| Pipeline Stage 4 (matching) | Yes | TTS log has both user name + action sequence; matches on action sequences |
| Session generator (LLM) | Yes | Gets user's vocab; uses their names in session plan; can also invent new combos as action sequences |
| Feedback generator (LLM) | Yes | Uses user's combo names in coach feedback |
| TTS | Yes | Speaks user's combo names |
| Mastery tracking | No | Tracks by action sequence string (e.g. "jab-cross-lead_hook"), not by user name |

**Combo identity for mastery:**
- Key = canonical action sequence string: `"jab-cross-lead_hook"`
- User name is display-only, doesn't affect tracking
- Two users can have different names for the same combo; mastery tracks the same underlying sequence

**LLM can create new combos:**
- Session generator can output action sequences the user hasn't named yet
- System auto-generates a display name from action Korean defaults: `["jab", "cross", "body_shot"]` → "잽-크로스-바디"
- User can rename it later in their vocab

---

## Directory Structure

```
/ml/
  configs/
    boxing.yaml                  # Action classes, keypoints, pipeline params (NO combo library)
    ctrgcn.yaml                  # Model hyperparameters
  graph/
    boxing_graph.py              # 15-joint adjacency matrix
  model/
    ctrgcn.py                    # Full CTR-GCN model
    modules.py                   # CTRGC, MSTCN, CTRGCBlock
  feeders/
    boxing_feeder.py             # PyTorch Dataset + augmentation
  pipeline/
    __init__.py
    pipeline.py                  # Orchestrator
    pose_extractor.py            # Stage 1: MediaPipe
    action_classifier.py         # Stage 2: CTR-GCN
    sequence_recognizer.py       # Stage 3: rule-based
    session_matcher.py           # Stage 4: rule-based
  services/
    session_generator.py         # LLM session plan generation
    feedback_generator.py        # LLM feedback generation
    mastery_updater.py           # EMA + state machine
    llm_client.py                # Async LLM abstraction
  prompts/
    session_drill.txt
    feedback_drill.txt
  scripts/
    extract_keypoints.py         # Batch MediaPipe extraction
    collect_youtube.py           # YouTube data pipeline
    visualize.py                 # Skeleton overlay visualization
    train.py                     # CTR-GCN training
    evaluate.py                  # Evaluation + confusion matrix
    export_model.py              # TorchScript export
  data/                          # gitignored
    raw/                         # Video files
    keypoints/                   # Extracted .npy files
    labels/                      # Action labels
    splits/                      # Train/val/test splits
  models/                        # gitignored, saved checkpoints
/tests/
  model/
    test_graph.py                # Adjacency matrix tests
    test_ctrgcn.py               # Forward pass, parameter count
    test_feeder.py               # Dataset loading, augmentation
  pipeline/
    test_pose_extractor.py
    test_action_classifier.py
    test_sequence_recognizer.py
    test_session_matcher.py
    test_pipeline_e2e.py         # Full pipeline integration
  services/
    test_session_generator.py
    test_feedback_generator.py
    test_mastery_updater.py
```

---

## Implementation Order

```
Step 1: boxing.yaml (ML config) + graph topology + tests
Step 2: CTR-GCN model architecture + forward pass tests
Step 3: Data collection scripts (MediaPipe extraction + YouTube pipeline)
Step 4: Feeder (Dataset) + augmentation
Step 5: Training script + train model
Step 6: Evaluation + export
Step 7: Pipeline Stage 1 (pose extractor)
Step 8: Pipeline Stage 2 (action classifier with trained model)
Step 9: Pipeline Stage 3 (sequence recognizer)
Step 10: Pipeline Stage 4 (session matcher)
Step 11: Pipeline orchestrator + E2E test
Step 12: LLM session generator
Step 13: LLM feedback generator
Step 14: Mastery updater
Step 15: Full integration test (video → analysis → feedback → next session)
```

**Steps 1-2** can start immediately (no data needed).
**Step 3** runs in parallel with 1-2 (data collection is independent work).
**Steps 4-6** require data from Step 3.
**Steps 7-11** require trained model from Step 6.
**Steps 12-14** can run in parallel with 7-11 (LLM work is independent of pipeline).
**Step 15** requires everything.

---

## Verification: End-to-End

```
1. Record a 3-minute boxing video (jab-cross combos)
2. Create mock TTS log: [{"t": 5.0, "combo_name": "원투", "actions": ["jab","cross"]}, {"t": 15.0, "combo_name": "원투쓰리", "actions": ["jab","cross","lead_hook"]}, ...]
3. Run pipeline.analyze(video, tts_log)
4. Inspect DrillResult:
   - detected_actions: should find jabs and crosses
   - detected_combos: should group into [jab, cross] sequences
   - matches: should have success/partial/miss for each instruction
   - combo_stats: reasonable success rates
5. Feed DrillResult to feedback_generator → Korean coach feedback
6. Feed combo_stats to mastery_updater → ComboMastery records
7. Feed updated mastery to session_generator → next session plan
8. Verify next session plan adjusts based on performance
```
