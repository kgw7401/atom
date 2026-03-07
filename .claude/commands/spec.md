---
argument-hint: [feature-name] [description - can be multiple sentences, as detailed as you want]
description: Spec-driven development workflow. Runs interview → draft → review loop → execute in a single flow.
allowed-tools: Read, Grep, Glob, AskUserQuestion, Bash(find:*), Bash(pytest:*), Bash(ruff:*), Bash(mypy:*), Bash(dbt:*), Bash(git:*)
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

- **Architecture:** (Mermaid diagram or text description)
- **Data Model:** Schema changes if any
- **API / Interface:** Input and output definitions
- **Dependencies:** External services, libraries
- **Decisions Made:** List choices you made autonomously and why

## 3. Implementation Plan

Each task must be independently implementable, testable, and committable.

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

| Date   | Change        | Reason           |
| ------ | ------------- | ---------------- |
| {date} | Initial draft | Phase 2 complete |
```

After creating the draft, perform a silent self-check (do not show the checklist to the user):

- Measurable Success Criteria? All three Boundary tiers? Concrete test cases? Open Questions non-empty?
- Fix any gaps yourself before presenting.

Then **present only the TL;DR section** to the user, not the full spec.
Mention where the full spec lives (`docs/specs/$1-spec.md`) so they can read it if they want.
Move directly to Phase 3.

---

## Phase 3: REVIEW LOOP (Iterative Spec Refinement)

**This phase is the core of the workflow.**

### Each round:

**Step 1 — Proactive Review**
Re-read the spec. Find gaps from these angles:

- Architecture: scalability, failure scenarios, consistency with existing systems
- Task ordering: dependency cycles, missing prerequisites
- Testing: untested scenarios
- Security: auth, input validation, secret management
- Missing edge cases

**Step 2 — Present Findings + Collect Feedback in One Shot**
Present all issues you found. For each issue, state:

- What the problem is (be specific)
- Your recommended fix

Then use **one** AskUserQuestion to handle everything:
Question: "I found {N} issues in the spec (listed above). How to proceed?"
Options: ["Apply all recommended fixes", "Let me review each one", "Spec looks good — approve and implement", "Other"]

- "Apply all" → apply fixes, update Changelog, do another review round
- "Review each" → use AskUserQuestion per issue only if the user wants to override your recommendation
- "Approve" → exit the loop, go to Phase 4
- "Other" → collect free-form feedback

**Step 3 — Update Spec File**
Apply changes to `docs/specs/$1-spec.md`.
Add Changelog entry.
**Always update the TL;DR section to reflect the current state of the spec.**
Present the updated TL;DR to the user, then return to Step 1.

### Exiting the Review Loop

When the user approves:

1. Change Status from `DRAFT` to `APPROVED`
2. Add approval record to Changelog
3. If Open Questions remain, warn once (do not block)
4. Move to Phase 4 automatically.

---

## Phase 4: EXECUTE (Task-by-Task Implementation)

Execute the Implementation Plan one task at a time.

### For each task:

1. **Load only files within the task's scope.**
2. **Implement the task.**
3. **Run verification:** tests, lint, type check as applicable.
4. **Report results briefly:** ✅ passed / ❌ failed with cause.
5. **Commit** using conventional commit format.
6. **If all checks pass, proceed to the next task automatically.**
   Only use AskUserQuestion if:
   - A test fails and you cannot fix it
   - Implementation requires deviating from the spec
   - You discovered something that changes the spec's assumptions

### Constraints

- One task at a time.
- Never skip a failing test.
- Never violate 🚫 Never boundaries.
- If you need to deviate from the spec, update the spec and use AskUserQuestion to confirm before proceeding.

---

## Quick Re-entry

If the session breaks or context is cleared, just type `/spec $1` again.
The command reads the existing `docs/specs/$1-spec.md` and resumes:

- Status is DRAFT → Phase 3 (Review Loop)
- Status is APPROVED → Phase 4 (Execute), first unchecked task
- No spec file → Phase 1 (Interview)
