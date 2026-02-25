# Boxing State Vector Specification

Version: v0.1
Last Updated: 2026-02-25
Depends on: spec/overview.md

---

## 1. Purpose

The Boxing State Vector **S_t** is the central mathematical abstraction of the system.
It is a fixed-dimension numerical representation of a boxer's current ability,
updated deterministically after each training session.

```
S_t → Policy(S_t) → A_t → Execute → O_{t+1} → Update(S_t, O_{t+1}) → S_{t+1}
```

The state vector does NOT contain:
- Raw video data
- Natural language descriptions
- Session logs or history

It contains ONLY normalized numerical values in [0, 1].

---

## 2. Notation

| Symbol | Description |
|--------|-------------|
| S_t ∈ [0,1]^d | State vector at time step t (after t-th session) |
| O_t ∈ [0,1]^d | Observation vector from session t (some dims may be ∅) |
| C_t ∈ [0,1]^d | Confidence vector at time step t |
| ΔS_t = S_t − S_{t−1} | State delta (improvement/regression) |
| d = 18 | State vector dimension (MVP) |
| K = 8 | Number of punch types |
| α ∈ (0,1) | Smoothing factor for EMA update |
| τ_i | Policy threshold for dimension i |
| n_i | Cumulative observation count for dimension i |

Punch type index:

| k | Punch Type |
|---|------------|
| 1 | jab |
| 2 | cross |
| 3 | lead_hook |
| 4 | rear_hook |
| 5 | lead_uppercut |
| 6 | rear_uppercut |
| 7 | lead_body |
| 8 | rear_body |

These correspond to the 8 attack classes of the LSTM classifier (the 9th class `guard` is a non-attack state).

---

## 3. State Vector Definition

**S_t = [s_1, s_2, ..., s_18] ∈ [0,1]^18**

Organized into 5 groups:

### 3.1 Offensive Profile (dims 1–4)

Captures WHAT the boxer throws and how varied the selection is.

| Dim | Name | Description |
|-----|------|-------------|
| s_1 | `repertoire_entropy` | Diversity of punch type selection |
| s_2 | `level_change_ratio` | Proportion of body-level punches |
| s_3 | `lead_rear_balance` | Symmetry between lead and rear hand |
| s_4 | `combo_diversity` | Variety of tactical punch sequences |

### 3.2 Technique Quality (dims 5–8)

Captures HOW WELL the boxer executes each punch category.

| Dim | Name | Covers |
|-----|------|--------|
| s_5 | `tech_straight` | jab, cross |
| s_6 | `tech_hook` | lead_hook, rear_hook |
| s_7 | `tech_uppercut` | lead_uppercut, rear_uppercut |
| s_8 | `tech_body` | lead_body, rear_body |

### 3.3 Defense (dims 9–12)

Captures defensive habits and responsiveness.

| Dim | Name | Description |
|-----|------|-------------|
| s_9  | `guard_consistency` | Fraction of time maintaining proper guard |
| s_10 | `guard_recovery` | Speed of returning to guard after punching |
| s_11 | `guard_endurance` | Guard maintenance under fatigue |
| s_12 | `defensive_reaction` | Response speed to defensive stimuli |

### 3.4 Rhythm & Tempo (dims 13–15)

Captures timing, fluidity, and transitions.

| Dim | Name | Description |
|-----|------|-------------|
| s_13 | `work_rate` | Sustained punch output (punches/min) |
| s_14 | `combo_fluency` | Smoothness of multi-punch combinations |
| s_15 | `transition_speed` | Defense-to-offense transition time |

### 3.5 Conditioning (dims 16–18)

Captures performance maintenance under fatigue.

| Dim | Name | Description |
|-----|------|-------------|
| s_16 | `volume_endurance` | Work rate maintenance over session duration |
| s_17 | `technique_endurance` | Form quality maintenance under fatigue |
| s_18 | `rhythm_stability` | Consistency of output pacing |

---

## 4. Observation Function

Each training session produces raw data:

```
RawData = {
  segments: [(class_k, t_start, t_end), ...],   # from LSTM
  keypoints: [(t, K_t), ...],                     # from MediaPipe
  metadata: {duration, mode, script?}             # session info
}
```

The observation function **f : RawData → O_t** computes each dimension deterministically.

### 4.1 Offensive Profile

**o_1: repertoire_entropy**

```
p_k = count(class = k) / N_total    for k ∈ {1, ..., K}

H(p) = −Σ_{k=1}^{K} p_k · ln(p_k)    (convention: 0·ln(0) = 0)

o_1 = H(p) / ln(K)
```

Range: 0 (single punch type only) → 1 (uniform distribution).

---

**o_2: level_change_ratio**

```
N_body = count(class ∈ {lead_body, rear_body})

o_2 = N_body / N_total
```

Range: 0 (no body shots) → 1 (all body shots).

Note: The state stores the raw ratio. The policy engine determines the ideal range (typically 0.2–0.4).

---

**o_3: lead_rear_balance**

```
N_lead = count(class ∈ {jab, lead_hook, lead_uppercut, lead_body})
r_lead = N_lead / N_total

o_3 = 1 − 2 · |r_lead − 0.5|
```

Range: 0 (completely one-sided) → 1 (perfectly balanced).

---

**o_4: combo_diversity**

Extract all consecutive punch subsequences of length 2 and 3 (within a max gap of `t_combo_gap`):

```
seqs = extract_subsequences(segments, max_gap=t_combo_gap)

o_4 = |unique(seqs)| / |seqs|

where t_combo_gap = 1.0 s    (tunable hyperparameter)
```

Range: 0 (identical combos) → 1 (all unique).

Minimum: requires ≥ 3 combo sequences, else o_4 = ∅ (not observed).

---

### 4.2 Technique Quality

For each punch instance j of type k, a quality score **q_j** is computed from keypoint geometry.

**Sub-scores** (per punch instance):

```
q_guard(j)     = 1 − clip(d(wrist_off, nose) / d_ref_guard, 0, 1)
q_extension(j) = clip(θ_elbow(j) / θ_ref_ext, 0, 1)
q_rotation(j)  = clip(Δφ_shoulder(j) / φ_ref_rot, 0, 1)
q_return(j)    = 1 − clip(t_return(j) / t_ref_return, 0, 1)
```

Where:
- `wrist_off` = wrist of non-punching hand at peak extension of punch
- `nose` = nose keypoint (proxy for chin)
- `d_ref_guard` = reference distance for "guard down" (calibrated)
- `θ_elbow(j)` = elbow angle of punching arm at peak extension
- `θ_ref_ext` = full extension reference angle (~170°)
- `Δφ_shoulder(j)` = shoulder line rotation during punch
- `φ_ref_rot` = reference rotation angle (punch-type dependent)
- `t_return(j)` = time from peak extension to guard recovery
- `t_ref_return` = reference return time (~0.3 s)

**Composite quality per punch:**

```
q_j = w_g · q_guard(j) + w_e · q_extension(j) + w_r · q_rotation(j) + w_t · q_return(j)

where w_g + w_e + w_r + w_t = 1
```

Default weights:

| Weight | Value | Rationale |
|--------|-------|-----------|
| w_g | 0.35 | Guard hand is most critical safety factor |
| w_e | 0.25 | Extension determines reach and power |
| w_r | 0.20 | Hip/shoulder rotation contributes power |
| w_t | 0.20 | Return speed affects defense readiness |

**Grouped technique scores:**

```
o_5 = mean(q_j for j where class ∈ {jab, cross})
o_6 = mean(q_j for j where class ∈ {lead_hook, rear_hook})
o_7 = mean(q_j for j where class ∈ {lead_uppercut, rear_uppercut})
o_8 = mean(q_j for j where class ∈ {lead_body, rear_body})
```

Minimum: each group requires ≥ 2 instances, else o_i = ∅.

---

### 4.3 Defense

**o_9: guard_consistency**

Define instantaneous guard quality at frame t:

```
g(t) = max(0, 1 − d_avg_wrist_nose(t) / d_ref_guard)
```

Where `d_avg_wrist_nose(t)` = mean distance of both wrists from nose.

```
F_non_punch = {t : t is not within any punch segment}

o_9 = mean(g(t) for t ∈ F_non_punch)
```

---

**o_10: guard_recovery**

For each punch segment ending at t_end:

```
t_recover(j) = min{t > t_end_j : g(t) > γ_guard} − t_end_j

where γ_guard = 0.7    (guard quality threshold)

o_10 = 1 − clip(mean(t_recover) / t_ref_recover, 0, 1)

where t_ref_recover = 0.5 s
```

Higher = faster recovery.

---

**o_11: guard_endurance**

Split session into thirds by duration:

```
T_1 = [0, T/3),  T_3 = [2T/3, T]

g_first = mean(g(t) for t ∈ T_1 ∩ F_non_punch)
g_last  = mean(g(t) for t ∈ T_3 ∩ F_non_punch)

o_11 = clip(g_last / g_first, 0, 1)
```

1.0 = no fatigue degradation, < 1.0 = guard drops under fatigue.

Minimum: session duration ≥ 90 s, else o_11 = ∅.

---

**o_12: defensive_reaction**

**AI Session mode only.** Measures response time to defensive commands (slip, block, guard):

```
For each defensive command at time t_cmd:
  t_react(j) = t_detected_action − t_cmd

o_12 = 1 − clip(mean(t_react) / t_ref_react, 0, 1)

where t_ref_react = 1.5 s
```

Video analysis mode: o_12 = ∅ (not observed).

---

### 4.4 Rhythm & Tempo

**o_13: work_rate**

```
punch_rate = N_total / (T / 60)    (punches per minute)

o_13 = clip(punch_rate / R_ref, 0, 1)

where R_ref = 80 punches/min    (high-intensity reference)
```

---

**o_14: combo_fluency**

For all identified combo sequences, compute inter-punch intervals:

```
For combo (a_1, a_2, ..., a_n):
  ipi_i = t_start(a_{i+1}) − t_end(a_i)    for i = 1, ..., n−1

IPI_avg = mean(all ipi across all combos)

o_14 = 1 − clip(IPI_avg / IPI_ref, 0, 1)

where IPI_ref = 0.8 s
```

Higher = more fluid combos.

---

**o_15: transition_speed**

Identify defense→offense transitions: a guard/defensive segment followed by an attack segment:

```
For each transition (guard_end, attack_start):
  t_trans = t_start(attack) − t_end(guard)

o_15 = 1 − clip(mean(t_trans) / t_ref_trans, 0, 1)

where t_ref_trans = 1.0 s
```

Minimum: requires ≥ 2 transitions, else o_15 = ∅.

---

### 4.5 Conditioning

**o_16: volume_endurance**

```
rate_first = punch_rate in T_1 = [0, T/3)
rate_last  = punch_rate in T_3 = [2T/3, T]

o_16 = clip(rate_last / rate_first, 0, 1)
```

---

**o_17: technique_endurance**

```
tech_first = mean(q_j for j with t_start ∈ T_1)
tech_last  = mean(q_j for j with t_start ∈ T_3)

o_17 = clip(tech_last / tech_first, 0, 1)
```

---

**o_18: rhythm_stability**

Divide session into 30-second windows. Compute punch rate per window:

```
rates = [rate_w1, rate_w2, ..., rate_wn]
cv = std(rates) / mean(rates)    (coefficient of variation)

o_18 = 1 − clip(cv / CV_ref, 0, 1)

where CV_ref = 1.0
```

Minimum: requires ≥ 3 windows (session ≥ 90 s), else o_18 = ∅.

---

## 5. State Update Rule

### 5.1 Exponential Moving Average (EMA)

For each dimension i where O_{t,i} ≠ ∅:

```
S_{t+1, i} = α · S_{t, i} + (1 − α) · O_{t, i}
```

For unobserved dimensions (O_{t,i} = ∅):

```
S_{t+1, i} = S_{t, i}    (no update)
```

**Default α = 0.7** — prioritizes accumulated history over single-session noise.

### 5.2 Initialization

At t = 0 (first session):

```
S_0,i = O_0,i    for all observed dimensions
S_0,i = 0.5      for unobserved dimensions (neutral prior)
```

### 5.3 Why EMA

| Property | Benefit |
|----------|---------|
| Deterministic | Same inputs always produce same output |
| Single-pass | No need to store session history |
| Recency-weighted | Recent performance weighted more |
| Noise-smoothing | One bad session doesn't destroy state |
| O(1) memory per dim | Only stores current state + confidence |

---

## 6. Confidence Model

Each dimension has a confidence score indicating reliability:

```
C_{t, i} = 1 − exp(−n_i / n_ref)

where:
  n_i = cumulative count of sessions with valid O_{t,i}
  n_ref = 5    (after 5 observations, confidence ≈ 0.63)
```

Properties:
- C → 0 when n = 0 (no data)
- C → 0.63 when n = n_ref
- C → 1.0 asymptotically

**Usage by Policy Engine:**
- Dimensions with C < 0.3 are unreliable — policy should not base decisions on them
- Dimensions with C < 0.3 may trigger "need more data" recommendations

---

## 7. Delta and Progress Tracking

```
ΔS_t = S_t − S_{t−1}
```

Per dimension:
- ΔS_{t,i} > 0 → improvement
- ΔS_{t,i} < 0 → regression
- ΔS_{t,i} ≈ 0 → plateau

**Significant change threshold:**

```
|ΔS_{t,i}| > ε = 0.02    (2% change is considered meaningful)
```

---

## 8. Policy Interface

The Policy Engine receives the tuple **(S_t, C_t)** and produces training plan A_t.

### 8.1 Weakness Detection

```
W = { i : s_i < τ_i  AND  C_{t,i} ≥ 0.3 }
```

Default thresholds τ_i:

| Group | Dims | τ_i | Rationale |
|-------|------|-----|-----------|
| Offensive Profile | 1–4 | 0.4 | Diversity below 40% needs work |
| Technique | 5–8 | 0.5 | Below 50% indicates poor form |
| Defense | 9–12 | 0.5 | Defense is critical for safety |
| Rhythm | 13–15 | 0.4 | Below 40% affects combinations |
| Conditioning | 16–18 | 0.6 | Fatigue resistance is fundamental |

### 8.2 Priority Scoring

For each weakness i ∈ W, compute priority:

```
priority_i = w_group(i) · (τ_i − s_i) · C_{t,i}
```

Where `w_group(i)` is the importance weight of the dimension's group:

| Group | w_group | Rationale |
|-------|---------|-----------|
| Defense | 1.5 | Safety first |
| Technique | 1.2 | Form prevents injury and builds power |
| Conditioning | 1.0 | Base importance |
| Offensive Profile | 0.8 | Variety is secondary to execution |
| Rhythm | 0.8 | Timing improves with other skills |

The policy engine selects the top-N weaknesses by priority for the next training session.

---

## 9. Hyperparameters Summary

All tunable constants in one place:

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Smoothing factor | α | 0.7 | EMA decay rate |
| Confidence reference | n_ref | 5 | Sessions for ~63% confidence |
| Combo gap threshold | t_combo_gap | 1.0 s | Max gap between punches in a combo |
| Guard distance ref | d_ref_guard | calibrated | "Guard down" reference distance |
| Extension angle ref | θ_ref_ext | 170° | Full arm extension reference |
| Rotation angle ref | φ_ref_rot | per-type | Shoulder rotation reference |
| Return time ref | t_ref_return | 0.3 s | Punch return time reference |
| Guard threshold | γ_guard | 0.7 | Guard quality threshold |
| Recovery time ref | t_ref_recover | 0.5 s | Guard recovery reference |
| Reaction time ref | t_ref_react | 1.5 s | Defensive reaction reference |
| Work rate ref | R_ref | 80 p/min | High-intensity reference |
| Inter-punch interval ref | IPI_ref | 0.8 s | Combo fluency reference |
| Transition time ref | t_ref_trans | 1.0 s | Defense→offense reference |
| CV reference | CV_ref | 1.0 | Rhythm stability reference |
| Significant delta | ε | 0.02 | Meaningful change threshold |
| Quality weight: guard | w_g | 0.35 | Punch quality sub-weight |
| Quality weight: extension | w_e | 0.25 | Punch quality sub-weight |
| Quality weight: rotation | w_r | 0.20 | Punch quality sub-weight |
| Quality weight: return | w_t | 0.20 | Punch quality sub-weight |

---

## 10. Observability by Mode

Not all dimensions are observable in every training mode:

| Dim | Shadow Boxing | Heavy Bag | AI Session |
|-----|:---:|:---:|:---:|
| s_1 repertoire_entropy | ✓ | ✓ | ✓ |
| s_2 level_change_ratio | ✓ | ✓ | ✓ |
| s_3 lead_rear_balance | ✓ | ✓ | ✓ |
| s_4 combo_diversity | ✓ | ✓ | ✓ |
| s_5 tech_straight | ✓ | ✓ | ✓ |
| s_6 tech_hook | ✓ | ✓ | ✓ |
| s_7 tech_uppercut | ✓ | ✓ | ✓ |
| s_8 tech_body | ✓ | ✓ | ✓ |
| s_9 guard_consistency | ✓ | ✓ | ✓ |
| s_10 guard_recovery | ✓ | ✓ | ✓ |
| s_11 guard_endurance | ✓* | ✓* | ✓* |
| s_12 defensive_reaction | ✗ | ✗ | ✓ |
| s_13 work_rate | ✓ | ✓ | ✓ |
| s_14 combo_fluency | ✓ | ✓ | ✓ |
| s_15 transition_speed | ✓ | ✓ | ✓ |
| s_16 volume_endurance | ✓* | ✓* | ✓* |
| s_17 technique_endurance | ✓* | ✓* | ✓* |
| s_18 rhythm_stability | ✓* | ✓* | ✓* |

`✓*` = requires session ≥ 90 s.

---

## 11. Data Flow Summary

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  MediaPipe   │────▶│    LSTM      │────▶│   Segments   │
│  Keypoints   │     │  Classifier  │     │ (class, t, t)│
└─────────────┘     └──────────────┘     └──────┬───────┘
       │                                         │
       │              ┌──────────────────────────┘
       │              │
       ▼              ▼
┌────────────────────────┐
│   Observation Function │
│   f(keypoints, segs)   │
│                        │
│  ┌──────────────────┐  │
│  │ O_t ∈ [0,1]^18   │  │
│  │ (some dims = ∅)   │  │
│  └──────────────────┘  │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│   State Update (EMA)   │
│                        │
│  S_{t+1} = α·S_t +    │
│    (1−α)·O_t           │
│                        │
│  C_{t+1} updated       │
│                        │
│  ┌──────────────────┐  │
│  │ S_{t+1}, C_{t+1} │  │
│  └──────────────────┘  │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│    Policy Engine       │
│                        │
│  Weakness detection    │
│  Priority scoring      │
│  Session generation    │
│                        │
│  ┌──────────────────┐  │
│  │ Training Plan A_t │  │
│  └──────────────────┘  │
└────────────────────────┘
```

---

## 12. Design Invariants

These MUST hold at all times:

1. **Fixed dimension.** d = 18 for MVP. No dynamic resizing.
2. **Bounded range.** ∀i: S_{t,i} ∈ [0, 1].
3. **Deterministic.** Same RawData → same O_t → same S_{t+1}. No randomness.
4. **Monotonic confidence.** C_{t,i} is non-decreasing (more data = more confidence).
5. **Graceful missing data.** Unobserved dimensions are preserved, never zeroed.
6. **LLM-independent.** State computation requires zero LLM calls.
7. **Testable.** Every dimension formula is unit-testable with synthetic inputs.

---

## 13. Future Extensions (Post-MVP)

These are NOT part of the current spec but inform design decisions:

- **Per-punch-type technique** (expand dims 5–8 to 8 individual dims)
- **Sparring-specific dims** (opponent interaction, counter-punch rate, defensive success rate)
- **Footwork dims** (stance width, weight transfer, lateral movement)
- **Adaptive α** (per-dimension smoothing factor based on variance)
- **Population normalization** (reference values from user cohort instead of fixed constants)

The fixed-dimension constraint (Invariant 1) may be relaxed in future phases via versioned state schemas with migration rules.

---

# End of State Vector Specification
