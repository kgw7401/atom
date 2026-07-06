---
argument-hint: [feature-name] [description - can be multiple sentences, as detailed as you want]
description: Spec-driven development workflow. Runs interview → draft → review loop → execute in a single flow.
allowed-tools: Read, Grep, Glob, AskUserQuestion, Bash(find:*), Bash(pytest:*), Bash(ruff:*), Bash(mypy:*), Bash(dbt:*), Bash(git add:*), Bash(git diff:*), Bash(git status:*)
---

# Spec-Driven Development Workflow

Feature name: $1
Context provided by user: $ARGUMENTS

This workflow has 4 phases.

## Decision-Making Principle

**Be autonomous. Make the best decision yourself for anything you are confident about.**
Only use AskUserQuestion for decisions that are:

- **Irreversible** (architecture choices, schema design, external API contracts)
- **Ambiguous** (the codebase and user context give conflicting signals)
- **High-risk** (security, data loss, breaking changes to existing systems)

For everything else, decide on your own and document your reasoning in the spec.
If you got it wrong, the user will correct you during the review loop — that's what it's for.

When you do ask, always use AskUserQuestion with concrete options (not plain text questions).

## ABSOLUTE RULES — never violate under any circumstances

1. **NEVER approve your own work.** Only the user can approve specs, PRDs, changes, and commits.
2. **NEVER run `git commit` on your own.** Stage changes, show a summary, and wait for user approval.
3. **NEVER skip a review round.** Even if you find zero issues, the user must still explicitly approve.
4. **NEVER auto-proceed between phases.** Every phase transition requires a user action via AskUserQuestion.
5. **"No issues found" does NOT mean approved.** Present "no issues found" to the user and let THEM decide to approve.

## Visualization Guidelines

**Use ASCII diagrams to improve readability.**
A diagram replaces paragraphs of text. Default to visual over verbal.
Use plain ASCII art inside code blocks — it renders everywhere (terminal, any editor, GitHub).

Required diagrams (include in every spec unless truly not applicable):

- **Architecture:** Box-and-arrow diagram showing components and data flow
- **Data flow:** Sequence diagram for request/response or event flows
- **Task dependencies:** Graph showing implementation order

Use when helpful:

- **State transitions:** For lifecycle or status changes
- **Data model:** Table or ER-style relationships
- **Decision logic:** Flowchart with branching

ASCII style reference:

```
Boxes:        ┌──────────┐    Arrows:  ───▶  ──▶  ◀──▶
              │ Service  │    Lines:   ─── │ ┌ ┐ └ ┘ ├ ┤
              └──────────┘    Diamond: ◇ (decision)

Sequence:     User          Service        DB
               │──── req ────▶│              │
               │              │── query ────▶│
               │              │◀── result ───│
               │◀── resp ─────│              │

Flow:         [A] ──▶ [B] ──▶ [C]
                       │
                       ▼
                      [D]
```

Formatting rules:

- Keep each diagram under 20 lines — split into multiple if larger
- Add a one-line caption above each diagram: `**Fig: {what this shows}**`
- Use clear labels, not abbreviations
- Always wrap in a code block (```) so monospace alignment is preserved

---

## Phase 1: INTERVIEW (Gather Requirements)

Do NOT write code. Do NOT write a spec yet.

1. Explore the existing codebase for files related to `$1` using Read, Grep, Glob.
2. From the codebase + user context, infer as much as you can on your own:
   - Tech stack, existing patterns, conventions
   - Related modules and their interfaces
   - Likely constraints and dependencies
3. **Only ask about what you genuinely cannot infer.** Use AskUserQuestion.
   - Max 1-2 rounds of questions. Aim for 2-3 critical questions total, not 10.
   - Each question must have concrete options based on your codebase analysis.
   - Example GOOD question (irreversible choice):
     Question: "This feature touches both the Airflow DAG and dbt layer. Which should own the orchestration?"
     Options: ["Airflow triggers dbt", "dbt controls its own scheduling", "New orchestration service in src/like/", "Other"]
   - Do NOT ask about things you can figure out from the code.
4. After questions are answered (or if none are needed), move directly to Phase 2.

---

## Phase 2: DRAFT (Generate Spec)

Create `specs/$1-spec.md` using the template below.
**Do not ask for approval to create the draft. Just create it.**

```markdown
# Spec: {feature-name}

> Status: DRAFT
> Created: {date}
> Last Updated: {date}

## Overview Flow

<!-- This is the FIRST thing the reader sees. -->
<!-- Pick the most appropriate flow type for this feature. -->
<!-- Update this diagram whenever the spec changes. -->

**Fig: How it works (end to end)**
```

┌─────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐
│ Trigger │────▶│ Process A │────▶│ Process B │────▶│ Outcome │
└─────────┘ └─────────────┘ └──────┬──────┘ └──────────┘
│
▼
┌─────────────┐
│ Side effect │
└─────────────┘

Replace with actual flow: user flow, data pipeline, event flow, etc.
The reader should understand the entire feature from this single diagram.

```

## TL;DR
<!-- 5 lines max. Updated every time the spec changes. -->
<!-- This is what the user reads instead of the full spec. -->
- **Goal:** One sentence
- **Approach:** One sentence on the technical strategy
- **Scope:** {N} tasks, estimated complexity S/M/L overall
- **Key risk:** The single biggest risk or open question
- **Decisions needing input:** None / list

## 1. Objective
- **What:** One sentence describing what we are building
- **Why:** Business value and motivation
- **Who:** Target user
- **Success Criteria:**
  - [ ] Measurable completion condition 1
  - [ ] Measurable completion condition 2

## 2. Technical Design

**Fig: System Architecture**
```

┌──────────┐ ┌──────────┐ ┌──────────┐
│Component │──────▶│Component │──────▶│Component │
│ A │ │ B │ │ C │
└──────────┘ └──────────┘ └──────────┘

```

**Fig: Data Flow**
```

User Service DB
│── request ───▶│ │
│ │── query ──────▶│
│ │◀── result ─────│
│◀── response ──│ │

```

- **Data Model:** Schema changes if any (use ASCII ER diagram if relationships exist)
- **API / Interface:** Input and output definitions
- **Dependencies:** External services, libraries
- **Decisions Made:** List choices you made autonomously and why

## 3. Implementation Plan
Each task must be independently implementable, testable, and committable.

**Fig: Task Dependencies**
```

[Task 1] ──▶ [Task 2] ──▶ [Task 4]
│ ▲
└──▶ [Task 3] ──────────┘

Tasks with no arrows between them can run independently.

```

- [ ] **Task 1: {title}**
  - Scope: Files / modules to change
  - Verification: How to confirm completion
  - Complexity: S / M / L

- [ ] **Task 2: {title}**
  - Scope:
  - Verification:
  - Complexity:

## 4. Boundaries
- ✅ **Always:** What must be followed in this feature
- ⚠️ **Ask first:** Decisions that require human approval
- 🚫 **Never:** Hard stops

## 5. Testing Strategy
- **Unit:** Targets and approach
- **Integration:** If applicable
- **Conformance:** Given this input, expect this output
  - Input: ...
  - Expected: ...

## 6. Open Questions
- [ ] Unresolved item 1
- [ ] Unresolved item 2

## Changelog
| Date | Change | Reason |
|------|--------|--------|
| {date} | Initial draft | Phase 2 complete |
```

After creating the draft, perform a silent self-check (do not show the checklist to the user):

- Measurable Success Criteria? All three Boundary tiers? Concrete test cases? Open Questions non-empty?
- Fix any gaps yourself before presenting.

Then **present the Overview Flow diagram + TL;DR** to the user, not the full spec.
Mention where the full spec lives (`specs/$1-spec.md`) so they can read it if they want.

Use AskUserQuestion:
Question: "Spec draft created. Please review before we proceed."
Options: ["I've read it — start the review loop", "Give me a minute to read the full spec"]

- "Start review" → move to Phase 3
- "Give me a minute" → wait, then ask again

---

## Phase 3: REVIEW LOOP (Iterative Spec Refinement)

**This phase is the core of the workflow.**
**Minimum 1 full review round is mandatory.** The "approve" option only appears from the 2nd round onward.

### Each round:

**Step 1 — Proactive Review**
Re-read the spec. Find gaps from these angles:

- Architecture: scalability, failure scenarios, consistency with existing systems
- Task ordering: dependency cycles, missing prerequisites
- Testing: untested scenarios
- Security: auth, input validation, secret management
- Missing edge cases

**Step 2 — Present Findings + Collect Feedback**
Present all issues you found. For each issue, state:

- What the problem is (be specific)
- Your recommended fix

**You MUST always use AskUserQuestion here. Never decide the outcome yourself.**

If this is the **first round** (no review has happened yet):

If issues found:
Question: "I found {N} issues in the spec (listed above). How to proceed?"
Options: ["Apply all recommended fixes", "Let me review each one", "Other"]

If NO issues found:
Question: "I reviewed the spec and found no issues. How to proceed?"
Options: ["I have feedback to add", "Do another review round from a different angle", "Other"]
(No approve option in round 1 — even with zero issues.)

If this is **round 2+**:

If issues found:
Question: "I found {N} issues in the spec (listed above). How to proceed?"
Options: ["Apply all recommended fixes", "Let me review each one", "Spec looks good — approve and implement", "Other"]

If NO issues found:
Question: "I reviewed the spec again and found no new issues. How to proceed?"
Options: ["Approve — move to implementation", "I have feedback to add", "Do another review round", "Other"]

- "Apply all" → apply fixes, update Changelog, do another review round
- "Review each" → use AskUserQuestion per issue only if the user wants to override your recommendation
- "Approve" → exit the loop, go to Phase 4
- "Other" → collect free-form feedback

**Step 3 — Update Spec File**
Apply changes to `specs/$1-spec.md`.
Add Changelog entry.
**Always update the Overview Flow and TL;DR section to reflect the current state of the spec.**
Present the updated Overview Flow + TL;DR to the user, then return to Step 1.

### Exiting the Review Loop

When the user approves:

1. Change Status from `DRAFT` to `APPROVED`
2. Add approval record to Changelog
3. If Open Questions remain, warn once (do not block)
4. **Do NOT start implementation yet.** Move to the Approval Gate.

---

## APPROVAL GATE (Mandatory — cannot be skipped)

**Implementation does not start until the user explicitly approves this gate.**

Present a final summary to the user:

```
════════════════════════════════════════
  SPEC READY FOR IMPLEMENTATION
════════════════════════════════════════

  Feature:    {feature-name}
  Status:     APPROVED
  Tasks:      {N} tasks ({S/M/L} overall)
  Key risk:   {from TL;DR}

  Task Overview:
  ┌─────┬────────────────────┬──────┐
  │  #  │ Task               │ Size │
  ├─────┼────────────────────┼──────┤
  │  1  │ {title}            │  S   │
  │  2  │ {title}            │  M   │
  │  3  │ {title}            │  L   │
  └─────┴────────────────────┴──────┘

  Full spec: specs/{name}-spec.md

════════════════════════════════════════
```

Then use AskUserQuestion:
Question: "Spec is finalized. Ready to start implementation?"
Options: ["Start implementation", "I want to re-read the spec first", "Go back to review loop", "Not now — save for later"]

- "Start implementation" → Phase 4
- "Re-read first" → wait, then ask again
- "Go back to review" → return to Phase 3
- "Not now" → end the session (spec is saved, user can `/spec $1` later)

---

## Phase 4: EXECUTE (Task-by-Task Implementation)

Execute the Implementation Plan one task at a time.

### For each task:

1. **Load only files within the task's scope.**
2. **Implement the task.**
3. **Run verification:** tests, lint, type check as applicable.
4. **Report results briefly:** ✅ passed / ❌ failed with cause.
5. **Stage changes but do NOT commit.** Show the user:
   - Files changed (brief summary)
   - Proposed commit message (conventional commit format)
     Then use AskUserQuestion:
     Question: "Ready to commit?"
     Options: ["Commit", "Let me review the diff first", "Make changes before committing"]
6. **After commit, use AskUserQuestion before proceeding:**
   Question: "Task {N} complete. What next?"
   Options: ["Continue to Task {N+1}", "Review the changes first", "Update the spec before continuing", "Stop here for now"]

### Phase 4 Constraints

All ABSOLUTE RULES from the top of this document apply, plus:

- One task at a time.
- If you need to deviate from the spec, update the spec and use AskUserQuestion to confirm before proceeding.

---

## Quick Re-entry

If the session breaks or context is cleared, just type `/spec $1` again.
The command reads the existing `specs/$1-spec.md` and resumes:

- Status is DRAFT → Phase 3 (Review Loop)
- Status is APPROVED → Phase 4 (Execute), first unchecked task
- No spec file → Phase 1 (Interview)
