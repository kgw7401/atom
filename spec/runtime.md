# State Vector Runtime Specification

Version: v0.1
Last Updated: 2026-02-25
Depends on: spec/state-vector.md

---

## 1. Session Lifecycle

A session is the atomic unit of interaction between user and system.
One session = one video analysis = at most one state update.

### 1.1 State Machine

```
CREATED ──▶ UPLOADING ──▶ PROCESSING ──▶ UPDATING ──▶ COMPLETED
                              │              │
                              ▼              ▼
                           FAILED         FAILED
```

| State | Description | Trigger |
|-------|-------------|---------|
| `CREATED` | Session record exists, awaiting video | `POST /sessions` |
| `UPLOADING` | Video transfer in progress | Upload initiated |
| `PROCESSING` | Vision pipeline running | Upload confirmed |
| `UPDATING` | State vector write in progress | Observation computed |
| `COMPLETED` | State updated, report available | Transaction committed |
| `FAILED` | Terminal error state | Unrecoverable error at any stage |

### 1.2 Processing Sub-stages

The `PROCESSING` state contains an ordered pipeline:

```
PROCESSING
  ├─ 1. keypoint_extraction    MediaPipe → keypoint trajectories
  ├─ 2. action_classification  LSTM → action segments
  ├─ 3. observation_compute    f(keypoints, segments) → O_t
  └─ 4. verification           Script ↔ detected actions (AI Session mode only)
```

Each sub-stage is tracked for progress reporting and failure diagnosis:

```json
{
  "session_id": "sess_abc123",
  "status": "PROCESSING",
  "pipeline_stage": "action_classification",
  "pipeline_progress": 0.35
}
```

### 1.3 Terminal States

Only `COMPLETED` and `FAILED` are terminal.

- `COMPLETED`: immutable. No further writes to this session.
- `FAILED`: stores `error_code` and `error_detail`. May be retried (creates a NEW session, not a state change).

---

## 2. State Update Transaction Model

The state update is the most critical write in the system.
It transforms `(S_t, C_t) → (S_{t+1}, C_{t+1})` atomically.

### 2.1 Transaction Boundary

```
BEGIN TRANSACTION

  -- 1. Acquire lock on user's state row
  state = SELECT vector, confidence, obs_counts, version
          FROM user_state
          WHERE user_id = ?
          FOR UPDATE

  -- 2. Verify version (optimistic lock)
  ASSERT state.version == expected_version

  -- 3. Compute new state (pure function, no side effects)
  S_new = ema_update(state.vector, O_t, α, observation_mask)
  C_new = confidence_update(state.obs_counts, observation_mask, n_ref)
  n_new = increment_counts(state.obs_counts, observation_mask)

  -- 4. Validate invariants
  ASSERT all(0 <= S_new[i] <= 1)
  ASSERT all(0 <= C_new[i] <= 1)

  -- 5. Write new state
  UPDATE user_state SET
    vector     = S_new,
    confidence = C_new,
    obs_counts = n_new,
    version    = version + 1,
    updated_at = NOW()
  WHERE user_id = ? AND version = expected_version

  -- 6. Write audit log
  INSERT INTO state_transitions (
    user_id, session_id,
    version_before, version_after,
    vector_before, vector_after,
    observation, observation_mask,
    delta, created_at
  )

  -- 7. Mark session completed
  UPDATE sessions SET status = 'COMPLETED' WHERE session_id = ?

COMMIT
```

### 2.2 Isolation Guarantee

Multiple sessions for the same user MUST serialize at the state update step.

```
Session A:  ──PROCESSING──┐
                          ├── UPDATING (acquires lock) ── COMPLETED
Session B:  ──PROCESSING──┘
                               UPDATING (waits for lock) ── COMPLETED
```

Processing (MediaPipe + LSTM) MAY run in parallel.
State update MUST run sequentially per user.

The `FOR UPDATE` row lock ensures this at the database level.

### 2.3 What Is Inside the Transaction

| Inside (atomic) | Outside (non-atomic) |
|-----------------|---------------------|
| Read current state | Video upload |
| Compute new state | Keypoint extraction |
| Write new state | Action classification |
| Write audit log | Observation computation |
| Mark session completed | Report generation |

Report generation happens AFTER the transaction commits, using the already-persisted `S_{t+1}` and `ΔS_t`.

---

## 3. Idempotency Guarantees

### 3.1 Core Rule

**One session = at most one state update. Ever.**

```
sessions table:
  session_id          UUID PRIMARY KEY
  state_update_applied BOOLEAN DEFAULT FALSE
```

Before executing the state update transaction:

```python
if session.state_update_applied:
    return already_applied_response   # no-op
```

Inside the transaction, after a successful write:

```sql
UPDATE sessions SET state_update_applied = TRUE WHERE session_id = ?
```

### 3.2 Idempotency Scenarios

| Scenario | Behavior |
|----------|----------|
| Same session_id submitted twice | Second call is a no-op |
| Client retries after timeout | Checks `state_update_applied`, returns cached result |
| Server crash mid-transaction | Transaction rolls back, `state_update_applied` stays FALSE, safe to retry |
| Duplicate webhook/callback | Guarded by `state_update_applied` check |

### 3.3 Session ID Contract

- Session IDs are generated server-side (`uuid4`)
- Client CANNOT supply or override session IDs
- Session IDs are immutable after creation

---

## 4. Failure Handling

### 4.1 Failure Classification

| Error Class | Stage | Recovery | State Impact |
|-------------|-------|----------|-------------|
| `UPLOAD_FAILED` | UPLOADING | Client retries upload | None |
| `EXTRACTION_FAILED` | keypoint_extraction | Retry session | None |
| `CLASSIFICATION_FAILED` | action_classification | Retry session | None |
| `OBSERVATION_EMPTY` | observation_compute | No retry — session too short or no punches detected | None |
| `UPDATE_CONFLICT` | UPDATING | Auto-retry with refreshed version | None until retry succeeds |
| `UPDATE_INVARIANT` | UPDATING | Bug — requires investigation | None (rolled back) |
| `INFRA_FAILURE` | Any | Retry session | None |

### 4.2 Partial Observation

A session that produces a partial observation (some dims = ∅) is NOT a failure.

```
Valid observation: O_t = [0.7, ∅, 0.4, ∅, 0.8, ...]
→ Update only observed dims, skip ∅ dims
→ Session status = COMPLETED
```

An observation is rejected only if ALL dimensions are ∅:

```
Empty observation: O_t = [∅, ∅, ∅, ..., ∅]
→ No state update
→ Session status = FAILED
→ Error code = OBSERVATION_EMPTY
```

### 4.3 Retry Policy

```
Max retries: 2
Retry scope: entire pipeline (re-process from keypoint extraction)
Retry trigger: automatic for INFRA_FAILURE, manual for others
Backoff: none (immediate retry — pipeline is deterministic,
         transient infra errors resolve quickly)
```

Important: retries create a NEW pipeline execution within the SAME session.
The session_id does not change. The `state_update_applied` guard prevents double-update.

### 4.4 Invariant Violation

If the computed `S_{t+1}` violates any invariant (values outside [0,1], NaN, etc.):

1. Transaction is rolled back
2. Session is marked `FAILED` with `error_code = UPDATE_INVARIANT`
3. Full diagnostic payload is logged: `O_t`, `S_t`, computed `S_{t+1}`, the offending dimension
4. Alert is raised (this indicates a bug in the observation function)

This MUST NOT be auto-retried — the same inputs will produce the same invalid output.

---

## 5. Versioning Strategy

### 5.1 Two Kinds of Version

| Version | Scope | Purpose |
|---------|-------|---------|
| `row_version` (integer) | Per user_state row | Optimistic concurrency control |
| `schema_version` (string) | Per state vector format | Dimension layout evolution |

### 5.2 Row Version

```sql
CREATE TABLE user_state (
  user_id       UUID PRIMARY KEY,
  vector        FLOAT[18] NOT NULL,
  confidence    FLOAT[18] NOT NULL,
  obs_counts    INT[18]   NOT NULL,
  row_version   INTEGER   NOT NULL DEFAULT 0,
  schema_version VARCHAR  NOT NULL DEFAULT 'v1',
  created_at    TIMESTAMP NOT NULL,
  updated_at    TIMESTAMP NOT NULL
);
```

Every state update increments `row_version` by 1.
Concurrent writes are detected by version mismatch → retry with fresh read.

### 5.3 Schema Version

The schema version tracks the STRUCTURE of the state vector:

```
v1: d=18, dims as defined in spec/state-vector.md v0.1
```

Each user_state row stores its `schema_version`.

### 5.4 Schema Migration Rules

When the state vector schema changes (add, remove, or reorder dimensions):

**Adding dimensions:**
```
v1 (d=18) → v2 (d=20)

Migration:
  S_v2[1..18] = S_v1[1..18]      # preserve existing
  S_v2[19]    = 0.5               # neutral prior for new dim
  S_v2[20]    = 0.5
  C_v2[19]    = 0.0               # no confidence for new dim
  C_v2[20]    = 0.0
  n_v2[19]    = 0
  n_v2[20]    = 0
```

**Removing dimensions:**
```
v2 (d=20) → v3 (d=19)

Migration:
  Drop dim at index k
  S_v3 = S_v2 with element k removed
  Remap all policy thresholds and weights
```

**Redefining a dimension:**
```
If the computation formula for dim i changes:
  Reset: S[i] = 0.5, C[i] = 0.0, n[i] = 0
  Previous values are meaningless under new semantics
```

### 5.5 Migration Execution

Migration is applied **eagerly on deploy**:

```
1. New code deploys with schema_version = 'v2'
2. Background job: SELECT all rows WHERE schema_version = 'v1'
3. For each row: apply migration function, set schema_version = 'v2'
4. Read path asserts schema_version == CURRENT_SCHEMA
```

The migration function is registered in code:

```python
MIGRATIONS = {
    ('v1', 'v2'): migrate_v1_to_v2,
}
```

Migration functions are:
- Pure functions (input → output, no side effects)
- Unit tested with known inputs/outputs
- Reversible (reverse migration also registered for rollback)

---

## 6. State Transition Audit Log

Every state update produces an audit record:

```sql
CREATE TABLE state_transitions (
  id              UUID PRIMARY KEY,
  user_id         UUID NOT NULL,
  session_id      UUID NOT NULL UNIQUE,   -- one transition per session
  version_before  INTEGER NOT NULL,
  version_after   INTEGER NOT NULL,
  vector_before   FLOAT[18] NOT NULL,
  vector_after    FLOAT[18] NOT NULL,
  observation     FLOAT[18],              -- O_t (nulls for ∅ dims)
  observation_mask BOOLEAN[18] NOT NULL,  -- which dims were observed
  delta           FLOAT[18] NOT NULL,     -- ΔS_t = S_{t+1} - S_t
  created_at      TIMESTAMP NOT NULL
);
```

### 6.1 Audit Log Uses

| Consumer | Usage |
|----------|-------|
| Report Generator | ΔS_t for progress visualization |
| Policy Engine | Trend analysis across multiple deltas |
| Debug / Support | Trace exactly how state evolved |
| Rollback | Restore vector_before if bad update discovered |

### 6.2 Retention

- Audit log is append-only
- No auto-deletion in MVP
- Future: archive records older than 1 year

---

## 7. Database Schema Summary

```sql
-- User's current state (single row per user)
CREATE TABLE user_state (
  user_id         UUID PRIMARY KEY,
  vector          FLOAT[18] NOT NULL,
  confidence      FLOAT[18] NOT NULL,
  obs_counts      INT[18]   NOT NULL,
  row_version     INTEGER   NOT NULL DEFAULT 0,
  schema_version  VARCHAR   NOT NULL DEFAULT 'v1',
  created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Session records
CREATE TABLE sessions (
  session_id            UUID PRIMARY KEY,
  user_id               UUID NOT NULL REFERENCES user_state(user_id),
  mode                  VARCHAR NOT NULL,  -- 'shadow', 'heavy_bag', 'ai_session'
  status                VARCHAR NOT NULL DEFAULT 'CREATED',
  pipeline_stage        VARCHAR,
  pipeline_progress     FLOAT,
  state_update_applied  BOOLEAN NOT NULL DEFAULT FALSE,
  error_code            VARCHAR,
  error_detail          TEXT,
  script_id             UUID,              -- if AI session mode
  duration_seconds      FLOAT,
  created_at            TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at            TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Audit log (append-only)
CREATE TABLE state_transitions (
  id                UUID PRIMARY KEY,
  user_id           UUID NOT NULL,
  session_id        UUID NOT NULL UNIQUE,
  version_before    INTEGER NOT NULL,
  version_after     INTEGER NOT NULL,
  vector_before     FLOAT[18] NOT NULL,
  vector_after      FLOAT[18] NOT NULL,
  observation       FLOAT[18],
  observation_mask  BOOLEAN[18] NOT NULL,
  delta             FLOAT[18] NOT NULL,
  created_at        TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_transitions_user ON state_transitions(user_id, created_at);
CREATE INDEX idx_sessions_user ON sessions(user_id, created_at);
```

---

## 8. Runtime Invariants

These MUST hold at all times:

1. **Atomicity.** State update and session completion are in the same transaction.
2. **Idempotency.** A session triggers at most one state update, regardless of retries.
3. **Serialization.** Concurrent state updates for the same user are serialized.
4. **Auditability.** Every state transition has a corresponding audit record.
5. **No silent corruption.** Invariant violations abort the transaction and raise alerts.
6. **Lossless history.** The full state evolution is reconstructable from `state_transitions`.
7. **Schema-aware.** Every state row knows its schema version. No implicit assumptions.

---

# End of Runtime Specification
