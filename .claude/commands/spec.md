---
argument-hint: [feature-name] [description - can be multiple sentences, as detailed as you want]
description: Spec-driven development workflow. Runs interview → draft → review loop → execute in a single flow.
allowed-tools: Read, Grep, Glob, AskUserQuestion, Bash(find:*), Bash(pytest:*), Bash(ruff:*), Bash(mypy:*), Bash(dbt:*), Bash(git:*)
---

# Spec-Driven Development Workflow

Feature name: $1
Context provided by user: $ARGUMENTS

This workflow has 4 phases.
**Ask for user approval before moving to the next phase.**
Never advance without explicit confirmation.

**IMPORTANT: Always use the AskUserQuestion tool when asking the user questions.**
Do not ask questions in plain text. Present structured options so the user can select answers quickly.
The user can always choose "Other" to provide custom input.

---

## Phase 1: INTERVIEW (Gather Requirements)

Do NOT write code. Do NOT write a spec yet.

1. Explore the existing codebase for files related to `$1` using Read, Grep, Glob.
2. Based on your findings, ask the user questions **using AskUserQuestion**:
   - **Max 3 questions per turn.**
   - For each question, provide 2-4 concrete options based on what you found in the codebase. The user can always pick "Other" for a custom answer.
   - No obvious questions. Dig into the hard parts the user might have missed.
   - Topics you must cover:
     - Who is the user of this feature and what does success look like?
     - Dependencies and constraints with existing systems
     - Known edge cases or risks
     - Performance / security requirements
   - Example of a GOOD AskUserQuestion:
     Question: "How should the system handle late-arriving partitions?"
     Options: ["Skip and log warning", "Reprocess affected range", "Block until upstream backfills", "Other"]
   - Example of a BAD question (too vague, no options):
     "What are your requirements?"
3. Continue interviewing until you have covered all critical topics. Then use AskUserQuestion to confirm:
   Question: "I have enough context to draft the spec. Ready to proceed?"
   Options: ["Yes, draft the spec", "I have more context to add"]
4. Proceed to Phase 2 when the user selects "Yes."

---

## Phase 2: DRAFT (Generate Spec)

Create `specs/$1-spec.md` using the template below:

```markdown
# Spec: {feature-name}

> Status: DRAFT
> Created: {date}
> Last Updated: {date}

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

After drafting, perform a self-check:

- Does the Objective include measurable Success Criteria?
- Is each Task independently testable?
- Are all three Boundary tiers (Always / Ask / Never) present?
- Does Testing include concrete input/output examples?
- Are Open Questions non-empty? (Zero open questions is suspicious — think harder.)

Show the self-check results to the user, then use AskUserQuestion:
Question: "Spec draft is ready. Want to start the review loop?"
Options: ["Yes, start reviewing", "I want to check the spec file first", "Redo the draft"]
Proceed to Phase 3 when the user selects "Yes."

---

## Phase 3: REVIEW LOOP (Iterative Spec Refinement)

**This phase is the core of the workflow.**
Repeat until the user gives explicit approval ("LGTM", "approved", "let's build it").

### Each round:

**Step 1 — Proactive Review (Find problems first, don't wait)**
Re-read the spec and look for gaps from these angles:

- Architecture: Scalability, failure scenarios, consistency with existing systems
- Task ordering: Are there dependency cycles or missing prerequisites?
- Testing: Any untested scenarios?
- Security: Auth, input validation, secret management
- Missing edge cases

Present findings **specifically**.
Bad: "Something might be missing."
Good: "Task 2 does not handle the case where Pub/Sub message ordering is not guaranteed."

**Step 2 — Collect User Feedback (using AskUserQuestion)**
After presenting your findings, use AskUserQuestion to collect the user's response.

For each issue found, ask:
Question: "{specific issue description}"
Options: ["Fix it as you suggested", "Fix it differently (I'll explain)", "Not an issue, skip it", "Other"]

If you found multiple issues, you may use multiSelect: true to let the user address several at once.

When all issues are resolved, use AskUserQuestion:
Question: "All identified issues have been addressed. What next?"
Options: ["LGTM — approve and start implementation", "I have additional feedback", "Do another review round"]

- "LGTM" → go to Phase 4
- "additional feedback" → collect it, go to Step 3
- "another round" → go to Step 1

**Step 3 — Update the Spec File Directly**
Apply feedback by editing `docs/specs/$1-spec.md`.
**Always add an entry to the Changelog table:**

```
| {date} | Added Task 3: handle Pub/Sub ordering failure | Review R2 feedback |
```

After editing, summarize the changes for the user.
Then return to Step 1 (proactive review of the updated spec).

### Exiting the Review Loop

When the user explicitly approves:

1. Change Status from `DRAFT` to `APPROVED`
2. Add approval record to Changelog
3. Warn the user if Open Questions remain
4. Say "Ready to start implementation."
5. Proceed to Phase 4 when the user agrees.

---

## Phase 4: EXECUTE (Task-by-Task Implementation)

Execute the Implementation Plan one task at a time.

### For each task:

1. **Load only the files within the task's scope.**
   Do not modify files outside the task scope (report them instead).
2. **Implement the task.**
3. **Run the task's verification criteria:**
   - Tests: `pytest`
   - Lint: `ruff check`
   - Type check: `mypy` (if applicable)
4. **Report results:**
   - ✅ Passed checks
   - ❌ Failed checks with root cause
   - 📝 Discoveries that should be reflected in the spec (add to Changelog)
5. **Commit** using conventional commit format.
6. **Use AskUserQuestion before moving to the next task:**
   Question: "Task {N} complete. What next?"
   Options: ["Continue to Task {N+1}", "Review the changes first", "Update the spec before continuing", "Stop here for now"]

### Constraints

- Never implement more than one task at a time.
- Never proceed to the next task if tests are failing.
- Never violate the spec's Boundaries > 🚫 Never items.
- If implementation requires a different approach than the spec, update the spec first and get user approval before proceeding.

---

## Quick Re-entry

If the session breaks or context is cleared, just type `/spec $1` again.
The command reads the existing `docs/specs/$1-spec.md`, checks the Status and latest Changelog entry, and **resumes from the interrupted phase:**

- Status is DRAFT → Resume at Phase 3 (Review Loop)
- Status is APPROVED → Resume at Phase 4 (Execute), starting from the first unchecked task
- No spec file found → Start fresh from Phase 1 (Interview)
