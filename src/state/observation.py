"""Observation function: f(segments, keypoints) → ObservationVector.

Computes all 18 dimensions of the observation vector from raw session data.
Each dimension is computed by a deterministic, pure function.

Reference: spec/state-vector.md §4
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from src.state.constants import (
    ALL_PUNCH_CLASSES,
    BODY_CLASSES,
    CLASS_GUARD,
    CV_REF,
    D_REF_GUARD,
    HOOK_CLASSES,
    IPI_REF,
    KP_LEFT_SHOULDER,
    KP_LEFT_WRIST,
    KP_NOSE,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_WRIST,
    LEAD_CLASSES,
    MIN_COMBO_SEQUENCES,
    MIN_SESSION_DURATION,
    MIN_TECHNIQUE_INSTANCES,
    MIN_TRANSITIONS,
    NUM_DIMS,
    NUM_PUNCH_TYPES,
    R_REF,
    RHYTHM_WINDOW,
    STRAIGHT_CLASSES,
    T_COMBO_GAP,
    T_REF_REACT,
    T_REF_RECOVER,
    T_REF_TRANS,
    UPPERCUT_CLASSES,
    W_EXTENSION,
    W_GUARD,
    W_RETURN,
    W_ROTATION,
)
from src.state.types import ActionSegment, DefensiveCommand, KeypointFrame, ObservationVector


def compute_observation(
    segments: list[ActionSegment],
    keypoints: list[KeypointFrame],
    duration: float,
    mode: str = "shadow",
    defensive_commands: list[DefensiveCommand] | None = None,
) -> ObservationVector:
    """Compute observation vector from a single session's raw data.

    Args:
        segments: Action segments from LSTM classifier.
        keypoints: Pose keypoint frames from MediaPipe.
        duration: Total session duration in seconds.
        mode: "shadow", "heavy_bag", or "ai_session".
        defensive_commands: Defensive commands (AI session mode only).

    Returns:
        ObservationVector with NaN for unobserved dimensions.
    """
    values = np.full(NUM_DIMS, np.nan, dtype=np.float64)
    mask = np.zeros(NUM_DIMS, dtype=bool)

    punch_segments = [s for s in segments if s.class_id in ALL_PUNCH_CLASSES]

    if len(punch_segments) == 0:
        return ObservationVector(values=values, mask=mask)

    # --- Group 1: Offensive Profile (dims 0-3) ---
    _set(values, mask, 0, _compute_repertoire_entropy(punch_segments))
    _set(values, mask, 1, _compute_level_change_ratio(punch_segments))
    _set(values, mask, 2, _compute_lead_rear_balance(punch_segments))
    o4 = _compute_combo_diversity(punch_segments)
    if o4 is not None:
        _set(values, mask, 3, o4)

    # --- Group 2: Technique Quality (dims 4-7) ---
    tech_groups = [
        (4, STRAIGHT_CLASSES),
        (5, HOOK_CLASSES),
        (6, UPPERCUT_CLASSES),
        (7, BODY_CLASSES),
    ]
    for dim_idx, class_set in tech_groups:
        score = _compute_technique_group(punch_segments, keypoints, class_set)
        if score is not None:
            _set(values, mask, dim_idx, score)

    # --- Group 3: Defense (dims 8-11) ---
    if keypoints:
        guard_scores = _compute_guard_scores_per_frame(keypoints)

        o9 = _compute_guard_consistency(segments, guard_scores, keypoints)
        if o9 is not None:
            _set(values, mask, 8, o9)

        o10 = _compute_guard_recovery(punch_segments, guard_scores, keypoints)
        if o10 is not None:
            _set(values, mask, 9, o10)

        if duration >= MIN_SESSION_DURATION:
            o11 = _compute_guard_endurance(segments, guard_scores, keypoints, duration)
            if o11 is not None:
                _set(values, mask, 10, o11)

    if mode == "ai_session" and defensive_commands:
        o12 = _compute_defensive_reaction(defensive_commands, segments)
        if o12 is not None:
            _set(values, mask, 11, o12)

    # --- Group 4: Rhythm & Tempo (dims 12-14) ---
    _set(values, mask, 12, _compute_work_rate(punch_segments, duration))

    o14 = _compute_combo_fluency(punch_segments)
    if o14 is not None:
        _set(values, mask, 13, o14)

    o15 = _compute_transition_speed(segments)
    if o15 is not None:
        _set(values, mask, 14, o15)

    # --- Group 5: Conditioning (dims 15-17) ---
    if duration >= MIN_SESSION_DURATION:
        o16 = _compute_volume_endurance(punch_segments, duration)
        if o16 is not None:
            _set(values, mask, 15, o16)

        if keypoints:
            o17 = _compute_technique_endurance(punch_segments, keypoints, duration)
            if o17 is not None:
                _set(values, mask, 16, o17)

        o18 = _compute_rhythm_stability(punch_segments, duration)
        if o18 is not None:
            _set(values, mask, 17, o18)

    return ObservationVector(values=values, mask=mask)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _set(values: np.ndarray, mask: np.ndarray, idx: int, val: float) -> None:
    values[idx] = np.clip(val, 0.0, 1.0)
    mask[idx] = True


# ---------------------------------------------------------------------------
# Group 1: Offensive Profile
# ---------------------------------------------------------------------------

def _compute_repertoire_entropy(punch_segments: list[ActionSegment]) -> float:
    """Shannon entropy of punch type distribution, normalized by ln(K)."""
    counts = Counter(s.class_id for s in punch_segments)
    total = sum(counts.values())
    probs = np.array([counts.get(i, 0) / total for i in range(1, NUM_PUNCH_TYPES + 1)])
    # Filter zero probs for log
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    max_entropy = np.log(NUM_PUNCH_TYPES)
    return float(entropy / max_entropy)


def _compute_level_change_ratio(punch_segments: list[ActionSegment]) -> float:
    """Proportion of body-level punches."""
    n_body = sum(1 for s in punch_segments if s.class_id in BODY_CLASSES)
    return n_body / len(punch_segments)


def _compute_lead_rear_balance(punch_segments: list[ActionSegment]) -> float:
    """Symmetry between lead and rear hand. 1 = balanced, 0 = one-sided."""
    n_lead = sum(1 for s in punch_segments if s.class_id in LEAD_CLASSES)
    r_lead = n_lead / len(punch_segments)
    return 1.0 - 2.0 * abs(r_lead - 0.5)


def _compute_combo_diversity(punch_segments: list[ActionSegment]) -> float | None:
    """Unique combo sequences / total sequences."""
    seqs = _extract_combo_sequences(punch_segments)
    if len(seqs) < MIN_COMBO_SEQUENCES:
        return None
    unique = len(set(seqs))
    return unique / len(seqs)


def _extract_combo_sequences(
    punch_segments: list[ActionSegment],
) -> list[tuple[int, ...]]:
    """Extract consecutive punch subsequences of length 2-3 within T_COMBO_GAP."""
    if len(punch_segments) < 2:
        return []

    sorted_segs = sorted(punch_segments, key=lambda s: s.t_start)
    sequences: list[tuple[int, ...]] = []

    for i in range(len(sorted_segs)):
        seq = [sorted_segs[i].class_id]
        for j in range(i + 1, min(i + 3, len(sorted_segs))):
            gap = sorted_segs[j].t_start - sorted_segs[j - 1].t_end
            if gap > T_COMBO_GAP:
                break
            seq.append(sorted_segs[j].class_id)
            if len(seq) >= 2:
                sequences.append(tuple(seq))

    return sequences


# ---------------------------------------------------------------------------
# Group 2: Technique Quality
# ---------------------------------------------------------------------------

def _compute_technique_group(
    punch_segments: list[ActionSegment],
    keypoints: list[KeypointFrame],
    class_set: set[int],
) -> float | None:
    """Average technique quality for a group of punch types."""
    group_segs = [s for s in punch_segments if s.class_id in class_set]
    if len(group_segs) < MIN_TECHNIQUE_INSTANCES:
        return None

    if not keypoints:
        return None

    scores = [_compute_punch_quality(seg, keypoints) for seg in group_segs]
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def _compute_punch_quality(
    segment: ActionSegment,
    keypoints: list[KeypointFrame],
) -> float | None:
    """Quality score for a single punch instance from keypoint geometry.

    q = w_g * guard_score + w_e * extension_score + w_r * rotation_score + w_t * return_score
    """
    # Get keypoint frames during this punch segment
    seg_frames = _get_frames_in_range(keypoints, segment.t_start, segment.t_end)
    if len(seg_frames) < 2:
        return None

    # Get frames after the punch for return score
    post_frames = _get_frames_in_range(keypoints, segment.t_end, segment.t_end + 0.5)

    # Determine punching side
    is_lead = segment.class_id in LEAD_CLASSES

    # Sub-scores
    q_guard = _guard_hand_score(seg_frames, is_lead)
    q_ext = _extension_score(seg_frames, is_lead)
    q_rot = _rotation_score(seg_frames)
    q_ret = _return_score(seg_frames, post_frames, is_lead)

    return float(
        W_GUARD * q_guard + W_EXTENSION * q_ext + W_ROTATION * q_rot + W_RETURN * q_ret
    )


def _guard_hand_score(frames: list[KeypointFrame], is_lead: bool) -> float:
    """Non-punching hand distance from nose. Closer = better guard."""
    # Off-hand: if lead is punching, off-hand is right (rear), and vice versa
    off_wrist = KP_RIGHT_WRIST if is_lead else KP_LEFT_WRIST

    distances = []
    for f in frames:
        nose = f.keypoints[KP_NOSE]
        wrist = f.keypoints[off_wrist]
        d = np.linalg.norm(wrist - nose)
        distances.append(d)

    avg_dist = np.mean(distances)
    return float(max(0.0, 1.0 - avg_dist / D_REF_GUARD))


def _extension_score(frames: list[KeypointFrame], is_lead: bool) -> float:
    """Arm extension at peak. Uses elbow angle."""
    if is_lead:
        shoulder_idx, elbow_idx, wrist_idx = KP_LEFT_SHOULDER, 5, KP_LEFT_WRIST  # left arm
    else:
        shoulder_idx, elbow_idx, wrist_idx = KP_RIGHT_SHOULDER, 6, KP_RIGHT_WRIST  # right arm

    max_angle = 0.0
    for f in frames:
        angle = _compute_angle(
            f.keypoints[shoulder_idx], f.keypoints[elbow_idx], f.keypoints[wrist_idx]
        )
        max_angle = max(max_angle, angle)

    from src.state.constants import THETA_REF_EXT

    return float(min(max_angle / THETA_REF_EXT, 1.0))


def _rotation_score(frames: list[KeypointFrame]) -> float:
    """Shoulder line rotation during punch."""
    if len(frames) < 2:
        return 0.5

    angles = []
    for f in frames:
        ls = f.keypoints[KP_LEFT_SHOULDER]
        rs = f.keypoints[KP_RIGHT_SHOULDER]
        angle = np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0]))
        angles.append(angle)

    rotation = abs(max(angles) - min(angles))
    # Use a generic reference of 25 degrees
    return float(min(rotation / 25.0, 1.0))


def _return_score(
    seg_frames: list[KeypointFrame],
    post_frames: list[KeypointFrame],
    is_lead: bool,
) -> float:
    """Speed of returning to guard after punch."""
    if not post_frames:
        return 0.3  # no data after punch — conservative estimate

    punch_wrist = KP_LEFT_WRIST if is_lead else KP_RIGHT_WRIST

    # Peak extension position (last frame of punch)
    peak_pos = seg_frames[-1].keypoints[punch_wrist]
    nose = seg_frames[-1].keypoints[KP_NOSE]
    peak_dist = np.linalg.norm(peak_pos - nose)

    # Check how quickly wrist returns to guard distance
    for f in post_frames:
        d = np.linalg.norm(f.keypoints[punch_wrist] - f.keypoints[KP_NOSE])
        if d < D_REF_GUARD * 0.6:  # returned close to guard
            t_elapsed = f.timestamp - seg_frames[-1].timestamp
            from src.state.constants import T_REF_RETURN

            return float(max(0.0, 1.0 - t_elapsed / T_REF_RETURN))

    return 0.2  # did not return to guard in time


# ---------------------------------------------------------------------------
# Group 3: Defense
# ---------------------------------------------------------------------------

def _compute_guard_scores_per_frame(keypoints: list[KeypointFrame]) -> np.ndarray:
    """Precompute guard quality g(t) for every frame.

    g(t) = max(0, 1 - avg_wrist_nose_dist / D_REF_GUARD)
    """
    scores = np.zeros(len(keypoints), dtype=np.float64)
    for i, f in enumerate(keypoints):
        nose = f.keypoints[KP_NOSE]
        lw = f.keypoints[KP_LEFT_WRIST]
        rw = f.keypoints[KP_RIGHT_WRIST]
        avg_dist = (np.linalg.norm(lw - nose) + np.linalg.norm(rw - nose)) / 2.0
        scores[i] = max(0.0, 1.0 - avg_dist / D_REF_GUARD)
    return scores


def _compute_guard_consistency(
    segments: list[ActionSegment],
    guard_scores: np.ndarray,
    keypoints: list[KeypointFrame],
) -> float | None:
    """Average guard quality during non-punching frames."""
    non_punch_mask = _get_non_punch_frame_mask(segments, keypoints)
    if not np.any(non_punch_mask):
        return None
    return float(np.mean(guard_scores[non_punch_mask]))


def _compute_guard_recovery(
    punch_segments: list[ActionSegment],
    guard_scores: np.ndarray,
    keypoints: list[KeypointFrame],
) -> float | None:
    """Average time to return to guard after punching."""
    if not punch_segments or not keypoints:
        return None

    from src.state.constants import GAMMA_GUARD

    timestamps = np.array([f.timestamp for f in keypoints])
    recovery_times = []

    for seg in punch_segments:
        # Find frames after this punch ends
        post_mask = timestamps > seg.t_end
        post_indices = np.where(post_mask)[0]

        for idx in post_indices:
            if guard_scores[idx] >= GAMMA_GUARD:
                recovery_times.append(timestamps[idx] - seg.t_end)
                break

    if not recovery_times:
        return None

    avg_recovery = np.mean(recovery_times)
    return float(max(0.0, 1.0 - avg_recovery / T_REF_RECOVER))


def _compute_guard_endurance(
    segments: list[ActionSegment],
    guard_scores: np.ndarray,
    keypoints: list[KeypointFrame],
    duration: float,
) -> float | None:
    """Guard quality in last third / first third."""
    timestamps = np.array([f.timestamp for f in keypoints])
    non_punch_mask = _get_non_punch_frame_mask(segments, keypoints)

    t_third = duration / 3.0
    first_mask = non_punch_mask & (timestamps < t_third)
    last_mask = non_punch_mask & (timestamps >= 2 * t_third)

    if not np.any(first_mask) or not np.any(last_mask):
        return None

    g_first = np.mean(guard_scores[first_mask])
    g_last = np.mean(guard_scores[last_mask])

    if g_first == 0:
        return None

    return float(np.clip(g_last / g_first, 0.0, 1.0))


def _compute_defensive_reaction(
    commands: list[DefensiveCommand],
    segments: list[ActionSegment],
) -> float | None:
    """Average reaction time to defensive commands (AI session only)."""
    guard_segments = [s for s in segments if s.class_id == CLASS_GUARD]
    if not commands or not guard_segments:
        return None

    reaction_times = []
    for cmd in commands:
        # Find first guard segment after this command
        for seg in guard_segments:
            if seg.t_start >= cmd.timestamp:
                reaction_times.append(seg.t_start - cmd.timestamp)
                break

    if not reaction_times:
        return None

    avg_react = np.mean(reaction_times)
    return float(max(0.0, 1.0 - avg_react / T_REF_REACT))


# ---------------------------------------------------------------------------
# Group 4: Rhythm & Tempo
# ---------------------------------------------------------------------------

def _compute_work_rate(punch_segments: list[ActionSegment], duration: float) -> float:
    """Punches per minute, normalized."""
    if duration <= 0:
        return 0.0
    punch_rate = len(punch_segments) / (duration / 60.0)
    return float(min(punch_rate / R_REF, 1.0))


def _compute_combo_fluency(punch_segments: list[ActionSegment]) -> float | None:
    """Average inter-punch interval within combos."""
    sorted_segs = sorted(punch_segments, key=lambda s: s.t_start)
    ipis = []

    for i in range(len(sorted_segs) - 1):
        gap = sorted_segs[i + 1].t_start - sorted_segs[i].t_end
        if gap <= T_COMBO_GAP:
            ipis.append(gap)

    if not ipis:
        return None

    avg_ipi = np.mean(ipis)
    return float(max(0.0, 1.0 - avg_ipi / IPI_REF))


def _compute_transition_speed(segments: list[ActionSegment]) -> float | None:
    """Average defense-to-offense transition time."""
    sorted_segs = sorted(segments, key=lambda s: s.t_start)
    transitions = []

    for i in range(len(sorted_segs) - 1):
        current = sorted_segs[i]
        nxt = sorted_segs[i + 1]
        if current.class_id == CLASS_GUARD and nxt.class_id in ALL_PUNCH_CLASSES:
            trans_time = nxt.t_start - current.t_end
            if trans_time >= 0:
                transitions.append(trans_time)

    if len(transitions) < MIN_TRANSITIONS:
        return None

    avg_trans = np.mean(transitions)
    return float(max(0.0, 1.0 - avg_trans / T_REF_TRANS))


# ---------------------------------------------------------------------------
# Group 5: Conditioning
# ---------------------------------------------------------------------------

def _compute_volume_endurance(
    punch_segments: list[ActionSegment], duration: float
) -> float | None:
    """Punch rate in last third / first third."""
    t_third = duration / 3.0

    first = [s for s in punch_segments if s.t_start < t_third]
    last = [s for s in punch_segments if s.t_start >= 2 * t_third]

    if not first:
        return None

    rate_first = len(first) / t_third
    rate_last = len(last) / t_third

    if rate_first == 0:
        return None

    return float(np.clip(rate_last / rate_first, 0.0, 1.0))


def _compute_technique_endurance(
    punch_segments: list[ActionSegment],
    keypoints: list[KeypointFrame],
    duration: float,
) -> float | None:
    """Average technique score in last third / first third."""
    t_third = duration / 3.0

    first_segs = [s for s in punch_segments if s.t_start < t_third]
    last_segs = [s for s in punch_segments if s.t_start >= 2 * t_third]

    if len(first_segs) < 2 or len(last_segs) < 2:
        return None

    first_scores = [_compute_punch_quality(s, keypoints) for s in first_segs]
    last_scores = [_compute_punch_quality(s, keypoints) for s in last_segs]

    first_valid = [s for s in first_scores if s is not None]
    last_valid = [s for s in last_scores if s is not None]

    if not first_valid or not last_valid:
        return None

    tech_first = np.mean(first_valid)
    tech_last = np.mean(last_valid)

    if tech_first == 0:
        return None

    return float(np.clip(tech_last / tech_first, 0.0, 1.0))


def _compute_rhythm_stability(
    punch_segments: list[ActionSegment], duration: float
) -> float | None:
    """Coefficient of variation of punch rate across time windows."""
    n_windows = int(duration // RHYTHM_WINDOW)
    if n_windows < 3:
        return None

    rates = []
    for w in range(n_windows):
        t_start = w * RHYTHM_WINDOW
        t_end = t_start + RHYTHM_WINDOW
        count = sum(1 for s in punch_segments if t_start <= s.t_start < t_end)
        rates.append(count / (RHYTHM_WINDOW / 60.0))  # punches/min

    rates = np.array(rates, dtype=np.float64)
    mean_rate = np.mean(rates)

    if mean_rate == 0:
        return None

    cv = np.std(rates) / mean_rate
    return float(max(0.0, 1.0 - cv / CV_REF))


# ---------------------------------------------------------------------------
# Geometry Utilities
# ---------------------------------------------------------------------------

def _compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute angle at point b (in degrees) formed by points a-b-c."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _get_frames_in_range(
    keypoints: list[KeypointFrame], t_start: float, t_end: float
) -> list[KeypointFrame]:
    """Get keypoint frames within a time range."""
    return [f for f in keypoints if t_start <= f.timestamp <= t_end]


def _get_non_punch_frame_mask(
    segments: list[ActionSegment], keypoints: list[KeypointFrame]
) -> np.ndarray:
    """Boolean mask: True for frames NOT during any punch segment."""
    timestamps = np.array([f.timestamp for f in keypoints])
    mask = np.ones(len(keypoints), dtype=bool)

    punch_segs = [s for s in segments if s.class_id in ALL_PUNCH_CLASSES]
    for seg in punch_segs:
        mask &= ~((timestamps >= seg.t_start) & (timestamps <= seg.t_end))

    return mask
