# Project Atom – System Overview

Version: v0.1
Last Updated: 2026-02-24
Owner: Project Atom

---

# 1. System Purpose

Project Atom is a state-vector-based adaptive boxing training system.

The system does NOT attempt to be:
- A general AI coach
- A pure video-understanding system
- A motivational statistics dashboard

The system IS:
- A boxing ability state modeling system
- A training control loop built on top of that state
- A mobile-first adaptive training product

Core Idea:

Video → Metrics → State Vector → Policy → Training → Updated State

The system continuously measures, adjusts, and improves a user's boxing ability through repeated state updates.

---

# 2. Core Architecture Philosophy

## 2.1 State-Centric Design

The central abstraction of the system is the **Boxing State Vector (S_t)**.

Everything revolves around:

S_t → A_t → S_t+1

Where:
- S_t = current boxing state
- A_t = generated training session
- S_t+1 = updated state after training

LLM is NOT responsible for computing state.
Vision models are NOT responsible for coaching decisions.

State computation and policy logic must be deterministic and testable.

---

## 2.2 Separation of Responsibilities

### Vision Layer
Responsible for:
- Pose extraction
- Action classification
- Metric calculation

Output:
- Structured metrics only

Vision layer must NOT:
- Generate coaching language
- Decide training policy

---

### State Engine
Responsible for:
- Building fixed-dimension state vectors
- Normalization
- State update rules
- Delta calculation

State engine must be:
- Deterministic
- Purely numerical
- Testable with unit tests

---

### Policy Engine
Responsible for:
- Detecting weaknesses
- Mapping state to training focus
- Generating session structure

Policy must:
- Be rule-based in MVP
- Avoid black-box ML in early phase
- Be explainable

---

### LLM Layer
Responsible for:
- Translating state into human coaching language
- Explaining progress
- Framing training intention

LLM must NOT:
- Calculate metrics
- Override policy decisions
- Modify state values

It is a language adapter only.

---

### Mobile Layer
Responsible for:
- Executing sessions
- Capturing video
- Displaying feedback
- Visualizing progress

Mobile does NOT:
- Compute state
- Run heavy ML
- Contain business logic

---

# 3. System Loop

The system operates as a continuous control loop:

1. User uploads training video
2. Vision pipeline extracts metrics
3. State vector S_t is built
4. Policy engine generates training session A_t
5. User executes session
6. Session results are recorded
7. State updated to S_t+1
8. LLM generates coaching explanation

This loop repeats over time.

Improvement is measured as:

ΔS = S_t+1 − S_t

---

# 4. MVP Scope

Phase 1 includes:

- Shadow boxing analysis
- Heavy bag analysis
- ~15–20 dimensional state vector
- Rule-based policy engine
- Basic AI audio session generation
- LLM coaching summary

Phase 1 excludes:

- Full sparring opponent modeling
- Real-time pose feedback
- Reinforcement learning policies
- Personalized predictive modeling

---

# 5. Non-Goals (Important Constraints)

The system will NOT:

- Send entire videos to LLM
- Use VLM for core analysis
- Rely on end-to-end black-box AI
- Optimize for research novelty
- Over-engineer infrastructure

Primary constraint:
Single-developer maintainability.

---

# 6. Technology Principles

- Deterministic where possible
- Explainable policy logic
- Clear separation between compute layers
- Small, testable modules
- Minimal infrastructure complexity

Backend:
- Python
- FastAPI
- PostgreSQL

Mobile:
- React Native

Vision:
- MediaPipe (MVP)
- ST-GCN (optional Phase 2)

LLM:
- Text-only API
- Strict prompt templates

---

# 7. Long-Term Direction

Future phases may include:

- Adaptive difficulty control
- Learned policy optimization
- User clustering in state space
- Predictive performance modeling

However, the foundation remains:

State Vector → Policy → Training → Updated State

---

# 8. Design Invariants

These must always remain true:

1. State vector has fixed dimension.
2. State update is deterministic.
3. Policy is explainable.
4. LLM is optional and replaceable.
5. System works even if LLM is disabled.

If any future change violates these invariants, redesign is required.

---

# End of Overview