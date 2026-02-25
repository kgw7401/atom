"""All hyperparameters for the Boxing State Vector.

Reference: spec/state-vector.md §9
"""

from __future__ import annotations

# --- State Vector Dimensions ---
NUM_DIMS = 18
SCHEMA_VERSION = "v1"

# Dimension indices (0-based)
DIM_REPERTOIRE_ENTROPY = 0
DIM_LEVEL_CHANGE_RATIO = 1
DIM_LEAD_REAR_BALANCE = 2
DIM_COMBO_DIVERSITY = 3
DIM_TECH_STRAIGHT = 4
DIM_TECH_HOOK = 5
DIM_TECH_UPPERCUT = 6
DIM_TECH_BODY = 7
DIM_GUARD_CONSISTENCY = 8
DIM_GUARD_RECOVERY = 9
DIM_GUARD_ENDURANCE = 10
DIM_DEFENSIVE_REACTION = 11
DIM_WORK_RATE = 12
DIM_COMBO_FLUENCY = 13
DIM_TRANSITION_SPEED = 14
DIM_VOLUME_ENDURANCE = 15
DIM_TECHNIQUE_ENDURANCE = 16
DIM_RHYTHM_STABILITY = 17

DIM_NAMES = [
    "repertoire_entropy",
    "level_change_ratio",
    "lead_rear_balance",
    "combo_diversity",
    "tech_straight",
    "tech_hook",
    "tech_uppercut",
    "tech_body",
    "guard_consistency",
    "guard_recovery",
    "guard_endurance",
    "defensive_reaction",
    "work_rate",
    "combo_fluency",
    "transition_speed",
    "volume_endurance",
    "technique_endurance",
    "rhythm_stability",
]

DIM_GROUPS = {
    "offensive_profile": [0, 1, 2, 3],
    "technique": [4, 5, 6, 7],
    "defense": [8, 9, 10, 11],
    "rhythm": [12, 13, 14],
    "conditioning": [15, 16, 17],
}

# --- Punch Types (boxing.yaml actions, 0=guard excluded) ---
NUM_PUNCH_TYPES = 8

PUNCH_NAMES = [
    "jab", "cross", "lead_hook", "rear_hook",
    "lead_uppercut", "rear_uppercut", "lead_bodyshot", "rear_bodyshot",
]

# Action class IDs (matching boxing.yaml order)
CLASS_GUARD = 0
CLASS_JAB = 1
CLASS_CROSS = 2
CLASS_LEAD_HOOK = 3
CLASS_REAR_HOOK = 4
CLASS_LEAD_UPPERCUT = 5
CLASS_REAR_UPPERCUT = 6
CLASS_LEAD_BODYSHOT = 7
CLASS_REAR_BODYSHOT = 8

# Punch groupings for technique dimensions
STRAIGHT_CLASSES = {CLASS_JAB, CLASS_CROSS}
HOOK_CLASSES = {CLASS_LEAD_HOOK, CLASS_REAR_HOOK}
UPPERCUT_CLASSES = {CLASS_LEAD_UPPERCUT, CLASS_REAR_UPPERCUT}
BODY_CLASSES = {CLASS_LEAD_BODYSHOT, CLASS_REAR_BODYSHOT}

LEAD_CLASSES = {CLASS_JAB, CLASS_LEAD_HOOK, CLASS_LEAD_UPPERCUT, CLASS_LEAD_BODYSHOT}
REAR_CLASSES = {CLASS_CROSS, CLASS_REAR_HOOK, CLASS_REAR_UPPERCUT, CLASS_REAR_BODYSHOT}

ALL_PUNCH_CLASSES = LEAD_CLASSES | REAR_CLASSES

# --- EMA Update ---
ALPHA = 0.7  # smoothing factor (higher = more weight on history)

# --- Confidence Model ---
N_REF = 5  # sessions for ~63% confidence

# --- Observation: Reference Values ---

# Combo detection
T_COMBO_GAP = 1.0  # max seconds between punches in a combo

# Technique quality sub-score weights (must sum to 1.0)
W_GUARD = 0.35  # non-punching hand protecting chin
W_EXTENSION = 0.25  # arm extension at peak
W_ROTATION = 0.20  # hip/shoulder rotation
W_RETURN = 0.20  # return speed to guard

# Guard reference distance (normalized by shoulder width)
D_REF_GUARD = 0.8  # wrist-to-nose distance considered "guard down"

# Extension angle reference
THETA_REF_EXT = 170.0  # degrees — full arm extension

# Shoulder rotation reference (degrees, punch-type dependent)
PHI_REF_ROT = {
    CLASS_JAB: 15.0,
    CLASS_CROSS: 30.0,
    CLASS_LEAD_HOOK: 25.0,
    CLASS_REAR_HOOK: 25.0,
    CLASS_LEAD_UPPERCUT: 20.0,
    CLASS_REAR_UPPERCUT: 20.0,
    CLASS_LEAD_BODYSHOT: 25.0,
    CLASS_REAR_BODYSHOT: 25.0,
}

# Return time reference
T_REF_RETURN = 0.3  # seconds

# Guard quality threshold
GAMMA_GUARD = 0.7

# Guard recovery time reference
T_REF_RECOVER = 0.5  # seconds

# Defensive reaction time reference
T_REF_REACT = 1.5  # seconds

# Work rate reference
R_REF = 80.0  # punches per minute (high intensity)

# Inter-punch interval reference (combo fluency)
IPI_REF = 0.8  # seconds

# Transition time reference (defense → offense)
T_REF_TRANS = 1.0  # seconds

# Rhythm stability CV reference
CV_REF = 1.0

# Rhythm stability window size
RHYTHM_WINDOW = 30.0  # seconds

# Minimum session duration for conditioning dims
MIN_SESSION_DURATION = 90.0  # seconds

# Minimum instances per technique group
MIN_TECHNIQUE_INSTANCES = 2

# Minimum combo sequences for combo_diversity
MIN_COMBO_SEQUENCES = 3

# Minimum transitions for transition_speed
MIN_TRANSITIONS = 2

# Significant delta threshold
EPSILON = 0.02

# --- Keypoint Indices (within the 11-keypoint subset) ---
KP_NOSE = 0
KP_LEFT_EAR = 1
KP_RIGHT_EAR = 2
KP_LEFT_SHOULDER = 3
KP_RIGHT_SHOULDER = 4
KP_LEFT_ELBOW = 5
KP_RIGHT_ELBOW = 6
KP_LEFT_WRIST = 7
KP_RIGHT_WRIST = 8
KP_LEFT_HIP = 9
KP_RIGHT_HIP = 10
