# Spec: Session Template System

> Status: APPROVED
> Created: 2026-03-21
> Last Updated: 2026-03-21

## Overview Flow

**Fig: How a session plays (end to end)**

```
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐    ┌──────────┐
│ Pick Template│───▶│ Shuffle Segs  │───▶│ Resolve Audio    │───▶│ Play     │
│ (rotate by   │    │ (intro/outro  │    │ (clip URL per    │    │ (client  │
│  recent 3)   │    │  pinned)      │    │  segment)        │    │  concat) │
└──────────────┘    └───────────────┘    └──────────────────┘    └──────────┘

Audio Chunk Pipeline (one-time recording):
┌────────────┐    ┌───────────┐    ┌─────────────────────────┐
│ Combo Dict │───▶│ 27 chunks │───▶│ Self-record (~50 takes) │
│ decompose  │    │ + 9 cues  │    │ → data/audio/chunks/    │
└────────────┘    └───────────┘    └─────────────────────────┘
```

**Fig: Client playback (per round)**

```
[intro clip] → pause → [seg clip] → pause → [seg clip] → pause → ... → [outro clip]
     ↑ pinned           ↑ shuffled order                                    ↑ pinned
```

## TL;DR

- **Goal:** Replace current 4-template system with ~20 themed preset sessions per level, using gym-specific combo naming and per-segment audio clips
- **Approach:** Combo dictionary (68 combos) → decomposed into 27 reusable audio chunks (~50 recordings with variant takes) → client assembles chunks per combo at runtime
- **Scope:** 7 tasks, estimated complexity M overall
- **Key risk:** Recording ~89 clips is manual effort; consistency across clips needs care
- **Decisions needing input:** Combo vocabulary mapping (user provides gym-specific names)

## 1. Objective

- **What:** A preset session template system with ~60 total templates (20 per level), gym-specific combo naming, per-segment audio clips, and runtime segment shuffling
- **Why:** Remove per-session LLM cost, guarantee coaching quality through hand-curated templates, provide variety through shuffle + rotation
- **Who:** Solo boxing trainers using Atom app
- **Success Criteria:**
  - 20 templates per level (beginner, intermediate, advanced) with distinct topics
  - Combo dictionary with gym-specific sequence names mapped by user
  - Per-segment audio clips generated via batch TTS (one clip per unique text)
  - Client plays shuffled segments using individual clips + pauses
  - Same template feels different on each play (segment reordering)

## 2. Technical Design

### 2.1 Combo Dictionary

A JSON config file that maps action sequences to spoken text. This is the **authoring** layer — templates reference the spoken text directly, but the dictionary provides structure for consistency.

**File:** `src/atom/data/combo_dictionary.json`

```json
{
  "meta": {
    "gym": "user_default",
    "stance": "orthodox",
    "updated": "2026-03-21"
  },
  "actions": {
    "J": "잽",
    "C": "투",
    "LH": "레프트 훅",
    "RH": "라이트 훅",
    "LU": "레프트 어퍼컷",
    "RU": "라이트 어퍼컷",
    "LB": "레프트 바디",
    "RB": "라이트 바디"
  },
  "defense": {
    "SL": "슥",
    "DK": "덕킹",
    "WV": "위빙",
    "BS": "백",
    "SS": "사이드스텝"
  },
  "combos": {
    "basic": [
      {"id": "jab",          "seq": ["J"],         "call": "잽"},
      {"id": "one_two",      "seq": ["J", "C"],    "call": "원투"},
      {"id": "jab_jab",      "seq": ["J", "J"],    "call": "잽잽"},
      {"id": "yang_hook",    "seq": ["LH", "RH"],  "call": "양훅"},
      {"id": "yang_upper",   "seq": ["LU", "RU"],  "call": "양어퍼"},
      {"id": "hook_two",     "seq": ["LH", "C"],   "call": "훅 투"},
      {"id": "upper_two",    "seq": ["LU", "C"],   "call": "어퍼 투"},
      {"id": "body_hook",    "seq": ["LB", "LH"],  "call": "바디 훅"},
      {"id": "jab_hook",     "seq": ["J", "LH"],   "call": "잽 훅"},
      {"id": "two_hook",     "seq": ["C", "LH"],   "call": "투 훅"},
      {"id": "two_body",     "seq": ["C", "LB"],   "call": "투 바디"},
      {"id": "body_two",     "seq": ["LB", "C"],   "call": "바디 투"}
    ],
    "intermediate": [
      {"id": "one_two_hook",    "seq": ["J", "C", "LH"],       "call": "원투 훅"},
      {"id": "one_two_body",    "seq": ["J", "C", "LB"],       "call": "원투 바디"},
      {"id": "one_two_upper",   "seq": ["J", "C", "LU"],       "call": "원투 어퍼"},
      {"id": "jabjab_two",      "seq": ["J", "J", "C"],        "call": "잽잽 투"},
      {"id": "yang_upper_hook", "seq": ["LU", "RU", "LH"],     "call": "양어퍼 훅"},
      {"id": "hook_two_hook",   "seq": ["LH", "C", "LH"],      "call": "훅 투 훅"},
      {"id": "upper_two_hook",  "seq": ["LU", "C", "LH"],      "call": "어퍼 투 훅"},
      {"id": "body_body_hook",  "seq": ["LB", "RB", "LH"],     "call": "바디 바디 훅"},
      {"id": "two_hook_two",    "seq": ["C", "LH", "C"],       "call": "투 훅 투"},
      {"id": "one_two_duck",    "seq": ["J", "C", "DK"],       "call": "원투 덕킹"},
      {"id": "one_two_back",    "seq": ["J", "C", "BS"],       "call": "원투 백"},
      {"id": "slip_two",        "seq": ["SL", "C"],             "call": "슥 투"},
      {"id": "duck_two",        "seq": ["DK", "C"],             "call": "덕킹 투"},
      {"id": "weave_hook",      "seq": ["WV", "LH"],            "call": "위빙 훅"},
      {"id": "slip_one_two",    "seq": ["SL", "J", "C"],        "call": "슥 원투"},
      {"id": "back_one_two",    "seq": ["BS", "J", "C"],        "call": "백 원투"},
      {"id": "slip_upper",      "seq": ["SL", "LU"],            "call": "슥 어퍼"},
      {"id": "duck_hook",       "seq": ["DK", "LH"],            "call": "덕킹 훅"},
      {"id": "weave_one_two",   "seq": ["WV", "J", "C"],        "call": "위빙 원투"},
      {"id": "one_two_hook_two","seq": ["J", "C", "LH", "C"],   "call": "원투 훅 투"},
      {"id": "one_two_hook_body","seq": ["J", "C", "LH", "LB"], "call": "원투 훅 바디"},
      {"id": "one_two_body_hook","seq": ["J", "C", "LB", "LH"], "call": "원투 바디 훅"},
      {"id": "jabjab_two_hook", "seq": ["J", "J", "C", "LH"],   "call": "잽잽 투 훅"},
      {"id": "one_two_yang_hook","seq": ["J", "C", "LH", "RH"], "call": "원투 양훅"},
      {"id": "one_two_yang_upper","seq": ["J", "C", "LU", "RU"],"call": "원투 양어퍼"},
      {"id": "two_hook_two_hook","seq": ["C", "LH", "C", "LH"], "call": "투 훅 투 훅"},
      {"id": "one_two_upper_two","seq": ["J", "C", "LU", "C"],  "call": "원투 어퍼 투"},
      {"id": "body_hook_two",   "seq": ["LB", "LH", "C"],       "call": "바디 훅 투"},
      {"id": "yanghook_yangup", "seq": ["LH","RH","LU","RU"],   "call": "양훅 양어퍼"}
    ],
    "advanced": [
      {"id": "12_slip_2",            "seq": ["J","C","SL","C"],                     "call": "원투 슥 투"},
      {"id": "12_duck_2",            "seq": ["J","C","DK","C"],                     "call": "원투 덕킹 투"},
      {"id": "12_slip_2h2",          "seq": ["J","C","SL","C","LH","C"],            "call": "원투 슥 투훅투"},
      {"id": "12_slip_2h2_12",       "seq": ["J","C","SL","C","LH","C","J","C"],    "call": "원투 슥 투훅투 원투"},
      {"id": "12_duck_2h2",          "seq": ["J","C","DK","C","LH","C"],            "call": "원투 덕킹 투훅투"},
      {"id": "12_hook_back_12",      "seq": ["J","C","LH","BS","J","C"],            "call": "원투 훅 백 원투"},
      {"id": "12_back_12_hook",      "seq": ["J","C","BS","J","C","LH"],            "call": "원투 백 원투 훅"},
      {"id": "jj2h_back",            "seq": ["J","J","C","LH","BS"],                "call": "잽잽 투 훅 백"},
      {"id": "12_yanghook_2",        "seq": ["J","C","LH","RH","C"],                "call": "원투 양훅 투"},
      {"id": "12_weave_h2",          "seq": ["J","C","WV","LH","C"],                "call": "원투 위빙 훅투"},
      {"id": "12_weave_12",          "seq": ["J","C","WV","J","C"],                 "call": "원투 위빙 원투"},
      {"id": "12_body_hook_back_12", "seq": ["J","C","LB","LH","BS","J","C"],       "call": "원투 바디 훅 백 원투"},
      {"id": "j_slip_2h",            "seq": ["J","SL","C","LH"],                    "call": "잽 슥 투훅"},
      {"id": "12_yangup_h2",         "seq": ["J","C","LU","RU","LH","C"],           "call": "원투 양어퍼 훅투"},
      {"id": "12_yanghook_back_12",  "seq": ["J","C","LH","RH","BS","J","C"],       "call": "원투 양훅 백 원투"},
      {"id": "12_hook_slip_2",       "seq": ["J","C","LH","SL","C"],                "call": "원투 훅 슥 투"},
      {"id": "12_slip_12",           "seq": ["J","C","SL","J","C"],                 "call": "원투 슥 원투"},
      {"id": "slip_2h2",             "seq": ["SL","C","LH","C"],                    "call": "슥 투훅투"},
      {"id": "duck_h2h",             "seq": ["DK","LH","C","LH"],                   "call": "덕킹 훅투훅"},
      {"id": "weave_h2h",            "seq": ["WV","LH","C","LH"],                   "call": "위빙 훅투훅"},
      {"id": "12_12_hook",           "seq": ["J","C","J","C","LH"],                 "call": "원투 원투 훅"},
      {"id": "12_body_hook_2",       "seq": ["J","C","LB","LH","C"],                "call": "원투 바디 훅 투"},
      {"id": "12_yanghook_slip_2",   "seq": ["J","C","LH","RH","SL","C"],           "call": "원투 양훅 슥 투"},
      {"id": "jj2_slip_2h",          "seq": ["J","J","C","SL","C","LH"],            "call": "잽잽 투 슥 투훅"},
      {"id": "12_duck_h2",           "seq": ["J","C","DK","LH","C"],                "call": "원투 덕킹 훅투"},
      {"id": "12_slip_2_back",       "seq": ["J","C","SL","C","BS"],                "call": "원투 슥 투 백"},
      {"id": "j_slip_12_hook",       "seq": ["J","SL","J","C","LH"],                "call": "잽 슥 원투 훅"}
    ]
  },
  "cues": [
    {"id": "guard",    "call": "가드!"},
    {"id": "nice",     "call": "좋아!"},
    {"id": "faster",   "call": "더 빠르게!"},
    {"id": "footwork", "call": "스텝!"},
    {"id": "breathe",  "call": "호흡!"},
    {"id": "chin",     "call": "턱 당겨!"},
    {"id": "power",    "call": "강하게!"},
    {"id": "relax",    "call": "힘 빼!"},
    {"id": "eyes",     "call": "눈!"}
  ]
}
```

**User workflow:** I provide this base dictionary → user edits `call` values to match their gym → templates use the `call` text.

### 2.2 Template Schema Changes

**Fig: Template data model**

```
SessionTemplate
├── id (UUID)
├── name (str)           e.g. "beginner_07"
├── level (str)          beginner | intermediate | advanced
├── topic (str) [NEW]    e.g. "원투 기초 반복"
├── segments_json [RENAMED from blocks_json]
│   ├── intro: [text, ...]          ← pinned first
│   ├── segments: [text, ...]       ← shuffled
│   └── outro: [text, ...]          ← pinned last
└── created_at

AudioChunk [NEW]
├── id (UUID)
├── text (str)               e.g. "원투"
├── variant (int)            e.g. 1, 2, 3 (multiple takes)
├── audio_path (str)         e.g. "chunks/원투_1.mp3"
├── duration_ms (int)
└── created_at
(unique constraint: text + variant)

ComboAssembly [NEW — config, not DB]
Maps combo text → ordered list of chunk texts
e.g. "원투 슥 투훅투" → ["원투", "슥", "투훅투"]
```

**Key changes:**

- **Remove** `tempo` and `intensity` from segments
- **Remove** 4-block structure (warmup/main/pressure/cooldown) → flat `intro + segments + outro`
- **Add** `topic` field to SessionTemplate
- **Add** `AudioClip` table for pre-generated clips
- **Segment type** implied by position: intro/outro are structural, segments are combos or cues

### 2.3 Segment Structure

```json
{
  "intro": ["자, 시작합니다"],
  "segments": [
    "원투",
    "원투 훅",
    "가드!",
    "원투 슥 투훅투",
    "원투",
    "원투 훅 백 원투"
  ],
  "outro": ["수고했어!"]
}
```

**Ratio guideline:** >=90% of `segments` entries are combos, <=10% are cues.

**Pause rule:** No `pause_sec` in templates. Pause is calculated at runtime:
```
pause = clip.duration_ms + 300ms
```
Longer combos → longer clip → longer pause naturally. No manual tuning needed.

### 2.4 Audio Architecture

**Fig: Audio clip lifecycle**

```
                    ONE-TIME (self-recording)
┌────────────┐    ┌──────────────┐    ┌─────────────────────────┐
│ Combo Dict │───▶│ Extract 27   │───▶│ Record chunks           │
│ + cues     │    │ chunks + 9   │    │ (2-3 takes for top 14)  │
│            │    │ cues         │    │ → data/audio/chunks/    │
└────────────┘    └──────────────┘    └─────────────────────────┘
                                              │
                                              ▼
                                    data/audio/chunks/
                                    ├── 원투_1.mp3, 원투_2.mp3
                                    ├── 슥_1.mp3, 슥_2.mp3
                                    ├── 투훅투_1.mp3
                                    ├── 가드.mp3
                                    └── ...  + 12 round intros

                    PER-SESSION (runtime)
┌────────────┐    ┌──────────────┐    ┌─────────────────────┐
│ Pick       │───▶│ Shuffle      │───▶│ Resolve chunks      │
│ template   │    │ segments     │    │ per combo + concat   │
└────────────┘    └──────────────┘    └─────────────────────┘
                                              │
                                              ▼
                                    API Response per segment:
                                    {text, chunks: [{clip_url, duration_ms}, ...]}
```

**Chunk-based audio:** Instead of recording all 77 unique texts, record 27 reusable
chunks + 9 cues. Combos are assembled from chunks at runtime.

**Assembly examples:**
```
원투 슥 투훅투 원투 = [원투] + [슥] + [투훅투] + [원투]  (4 chunks)
원투 훅 백 원투     = [원투 훅] + [백] + [원투]          (3 chunks)
원투 양훅 투        = [원투] + [양훅] + [투]             (3 chunks)
원투 훅             = [원투 훅]                          (1 chunk)
```

Natural break points: defense actions (슥, 덕킹, 위빙, 백) create natural
speech pauses between chunks. Strike phrases (원투, 투훅투, 양훅) are recorded
as fluid units to avoid robotic concatenation.

**Recording guidelines:**
- Strong, commanding coach tone — like calling combos in the gym
- Short and punchy delivery (no filler words)
- Neutral intonation that works in any position (start/mid/end of combo)
- Consistent volume and mic distance across all clips
- Format: MP3 or WAV, any sample rate (normalized on import)

**Recording inventory:**
- 27 unique chunks (13 strike atoms + 10 strike phrases + 4 defense)
- Top 14 chunks (reuse >= 4): record 2-3 takes each for variety
- Remaining 13 chunks: 1 take each
- 9 cues: 1 take each
- 12 round intros ("1라운드 시작합니다" ~ "12라운드 시작합니다")
- Total: ~50 recordings

**Chunk gap:** ~150ms silence between chunks when assembled. Client inserts gap
during playback.

**Cost:** $0 (self-recorded)

### 2.5 Session Generation Flow

```python
async def generate_plan(level, rounds, round_duration_sec, rest_sec):
    # 1. Pick template (not in last 3)
    template = pick_template(level)

    # 2. For each round, shuffle segments differently
    plan_rounds = []
    for r in range(rounds):
        shuffled = list(template.segments)
        random.shuffle(shuffled)

        # Scale segment count to round duration
        target_count = estimate_segment_count(shuffled, round_duration_sec)
        selected = shuffled[:target_count]

        # Build round: intro + selected + outro
        texts = template.intro + selected + template.outro

        # Resolve audio chunks per combo
        round_segments = []
        for text in texts:
            chunk_texts = combo_assembly[text]  # e.g. ["원투", "슥", "투훅투"]
            chunks = []
            for ct in chunk_texts:
                chunk = pick_random_variant(ct)  # random take for variety
                chunks.append({
                    "text": ct,
                    "clip_url": chunk.audio_path,
                    "duration_ms": chunk.duration_ms,
                })
            round_segments.append({"text": text, "chunks": chunks})

        plan_rounds.append(round_segments)

    # 3. Save DrillPlan, return
    ...
```

**Round number injection:** Intro text "시작합니다" → "{N}라운드 시작합니다" per round. This means intro clips need either:

- Pre-generated per round number ("1라운드 시작합니다" ... "12라운드 시작합니다") — **12 extra clips, recommended**
- Or client uses local TTS for intro only

**Target segments per template (for a 180s round):**
- Beginner: ~50 segments (basic combos only — short, fast cycling)
- Intermediate: ~40 segments (basic + intermediate combos)
- Advanced: ~30 segments (all categories including long sequences)
- Scaled proportionally for other round durations.

**Chunk management:** When combo dictionary changes, run `atom chunks validate` to find missing chunks. Record new chunks if needed, re-import. Old orphaned chunks can be cleaned up manually.

### 2.6 API Response Changes

**Remove:** `tempo`, `intensity` from SegmentResponse
**Add:** `clip_url`, `duration_ms` to SegmentResponse

```python
class ChunkResponse(BaseModel):
    text: str
    clip_url: str          # /audio/chunks/원투_1.mp3
    duration_ms: int

class SegmentResponse(BaseModel):
    text: str              # full combo text e.g. "원투 슥 투훅투"
    chunks: list[ChunkResponse]  # assembled chunks
    # Client playback: chunk1 → 150ms gap → chunk2 → ... → last chunk → 300ms pause

class RoundResponse(BaseModel):
    round: int
    segments: list[SegmentResponse]
    # Remove: script, audio_url, timestamps (no longer needed)

class PlanResponse(BaseModel):
    id: str
    template_name: str     # NEW: which template was used
    template_topic: str    # NEW: session topic
    rounds: int
    round_duration_sec: int
    rest_sec: int
    plan: PlanDetail
    audio_ready: bool
```

### 2.7 Session Topics (Examples)


| Level        | #     | Topic     | Focus                |
| ------------ | ----- | --------- | -------------------- |
| Beginner     | 1-5   | 잽과 크로스 기초 | Jab/cross drilling   |
| Beginner     | 6-10  | 원투 반복     | One-two rhythm       |
| Beginner     | 11-15 | 기본 콤비네이션  | 2-3 punch combos     |
| Beginner     | 16-20 | 가드와 스텝    | Defense + basics     |
| Intermediate | 1-5   | 훅 마스터     | Hook variations      |
| Intermediate | 6-10  | 바디워크      | Body shot combos     |
| Intermediate | 11-15 | 디펜스 콤보    | Slip/duck + counter  |
| Intermediate | 16-20 | 4타 콤비네이션  | 4-punch sequences    |
| Advanced     | 1-5   | 복합 콤비네이션  | 5+ punch combos      |
| Advanced     | 6-10  | 프레셔 파이팅   | High-volume pressure |
| Advanced     | 11-15 | 카운터 어택    | Defense → offense    |
| Advanced     | 16-20 | 풀 라운드 시뮬  | Round simulation     |


Each template: ~30-50 shuffleable segments (varies by level) + 1 intro + 1 outro.

**Combo level mapping:**
- Beginner templates → basic combos only
- Intermediate templates → basic + intermediate combos
- Advanced templates → basic + intermediate + advanced combos

**Decisions Made:**

- Flat segment structure instead of 4-block (warmup/main/pressure/cooldown) — the topic handles the "what" and shuffling makes blocks pointless
- Round number in intro handled by pre-generating "N라운드 시작합니다" clips (1-12)
- Hash-based clip filenames to avoid filesystem issues with Korean text
- No intensity/tempo fields — coach voice settings baked into TTS generation

## 3. Implementation Plan

**Fig: Task dependencies**

```
[T1: Combo Dict] ──▶ [T3: Seed 60 Templates] ──▶ [T5: Audio Clips] ──▶ [T7: Mobile Update]
                                                        ▲
[T2: Schema]  ──────▶ [T4: Service Layer] ─────────────┘
                              │
                              ▼
                      [T6: API Update]  ──────────────▶ [T7: Mobile Update]
```

- **Task 1: Create combo dictionary**
  - Scope: `src/atom/data/combo_dictionary.json`
  - Create base dictionary with actions, combos, cues
  - User maps gym-specific names afterward
  - Verification: JSON validates, all combo IDs unique
  - Complexity: S
- **Task 2: Update data model**
  - Scope: `src/atom/models/tables.py`, `src/atom/api/schemas.py`
  - Add `topic` to SessionTemplate, rename `blocks_json` → `segments_json`
  - Add `AudioChunk` table (text, variant, audio_path, duration_ms)
  - Add combo assembly config (combo text → chunk list mapping)
  - Remove `tempo`/`intensity` from schemas
  - Add `clip_url`/`duration_ms` to SegmentResponse
  - Verification: `alembic` migration runs, models validate
  - Complexity: M
- **Task 3: Create 60 seed templates**
  - Scope: `src/atom/seed_templates.py`
  - 20 templates × 3 levels, each with topic and segments
  - Follow combo dictionary for consistent naming
  - 90%+ combo segments, <=10% cue segments
  - Verification: All templates load, segment ratio check
  - Complexity: L
- **Task 4: Update service layer**
  - Scope: `src/atom/services/template_service.py`, `src/atom/services/session_service.py`
  - New shuffle logic (flat segments, not blocks)
  - Clip resolution (text → AudioClip lookup)
  - Remove tempo/intensity handling
  - Verification: Unit test — shuffle preserves intro/outro, segments change order
  - Complexity: M
- **Task 5: Audio chunk management**
  - Scope: `src/atom/services/audio_service.py`, CLI commands
  - `atom chunks checklist` → generate recording list (27 chunks + 9 cues + 12 intros)
  - `atom chunks import` → normalize audio, register in AudioChunk table
  - `atom chunks validate` → verify every combo can be assembled from available chunks
  - Combo assembly mapping (combo text → chunk sequence) as JSON config
  - Verification: `atom chunks validate` reports 0 missing chunks
  - Complexity: M
- **Task 6: Update API**
  - Scope: `src/atom/api/routers/sessions.py`, `src/atom/api/schemas.py`
  - Response includes `clip_url` + `duration_ms` per segment
  - Add `template_name` + `template_topic` to PlanResponse
  - Remove per-round audio generation (no more concatenation)
  - Verification: API returns valid response with clip URLs
  - Complexity: S
- **Task 7: Update mobile client**
  - Scope: `mobile/src/screens/ActiveSessionScreen.tsx`, audio playback logic
  - Segment-by-segment playback: play clip → wait pause → next clip
  - Download clips on first use, cache locally
  - Verification: Session plays smoothly with correct timing
  - Complexity: M

## 4. Boundaries

- **Always:**
  - Use combo dictionary `call` values for all segment text
  - Maintain >=90% combo ratio in every template
  - Pin intro/outro segments (never shuffle)
  - Strong coach tone in all recorded clips (no soft/slow variations)
- **Ask first:**
  - Adding new combo vocabulary entries
  - Re-recording clips or changing recording style
  - Modifying template count per level
- **Never:**
  - Use LLM to generate session content at runtime
  - Include tempo/intensity in segment data
  - Generate audio via TTS (all clips are self-recorded)
  - Generate per-round concatenated audio (use per-segment clips)

## 5. Testing Strategy

- **Unit:**
  - Shuffle logic: intro/outro stay pinned, middle segments reorder
  - Segment count scaling: matches target round duration
  - Chunk resolution: every combo can be assembled from available AudioChunks
- **Integration:**
  - Full plan generation: pick template → shuffle → resolve clips → API response
  - Chunk validation: all combos in templates assembleable from recorded chunks
- **Conformance:**
  - Input: `POST /api/sessions/plan {level: "beginner", rounds: 3, round_duration_sec: 180, rest_sec: 30}`
  - Expected: Response with 3 rounds, each with segments containing `clip_url` and `duration_ms`, intro first, outro last, middle segments in different order per round

## 6. Open Questions

- User to provide gym-specific combo name mappings for the combo dictionary
- Should clips be downloadable as a bundle (zip) for offline use, or downloaded individually on-demand?
- ~~Maximum segment count per template~~ → 50 segments per template (for 180s round)

## Changelog


| Date       | Change                                         | Reason                                  |
| ---------- | ---------------------------------------------- | --------------------------------------- |
| 2026-03-21 | Initial draft                                  | Phase 2 complete                        |
| 2026-03-21 | Expanded combo dictionary from 14 to 63 combos | User feedback: combos need more variety |
| 2026-03-21 | Applied gym naming: 크로스→투, 더블잽→잽잽, 슬립→슥, 백스텝→백 | User feedback: gym-specific names       |
| 2026-03-21 | Curated for sparring: removed 10 impractical, added 2. 63→40 combos | Biomechanical audit |
| 2026-03-21 | Restructured: basic/intermediate/advanced categories + 15 sparring sequences | User: want real fight combos, not punch-count categories |
| 2026-03-21 | Added 17 combos (5 intermediate, 12 advanced). Pause/segment count by level. Combo-level mapping. | Round 2 review |
| 2026-03-21 | Replaced ElevenLabs TTS with self-recording. Removed all TTS code references. | User: will record clips manually |
| 2026-03-21 | Removed pause_sec from templates. Pause = clip_duration + 300ms at runtime. | User feedback: simplify pause to clip duration + 0.3s |
| 2026-03-21 | Chunk-based audio: 27 chunks (~50 recordings) instead of 89 per-combo clips. 2-3 takes for top 14. | User: minimize recordings, maximize reuse |
| 2026-03-21 | Status → APPROVED | User approved after round 2 review |


