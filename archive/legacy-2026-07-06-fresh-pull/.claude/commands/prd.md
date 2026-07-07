---
argument-hint: [idea-name] [context - what problem you're thinking about, as detailed as you want]
description: Product discovery workflow. Clarify the problem, explore approaches, and produce a PRD before committing to implementation. Use this before /spec when the direction is not yet clear.
allowed-tools: Read, Grep, Glob, AskUserQuestion, Bash(find:*), Bash(git:*)
---

# Product Discovery Workflow

Idea: $1
Context: $ARGUMENTS

This workflow produces a PRD (Product Requirements Document) through structured discussion.
**This is NOT implementation planning.** No code, no tasks, no technical design yet.
The output feeds into `/spec` when the direction is confirmed.

## Decision-Making Principle

Same as /spec: be autonomous, only ask when irreversible, ambiguous, or high-risk.
But in this workflow, most decisions ARE ambiguous — you're exploring, not executing.
So ask more freely here than in /spec, but still keep it focused.

## Visualization Guidelines

**Use ASCII diagrams to make the PRD scannable.**
At the idea stage, visuals clarify thinking faster than text.
Use plain ASCII art inside code blocks — renders everywhere.

Required diagrams:

- **User journey:** Flow showing the user's path through the solution
- **Approach comparison:** Table comparing options in Phase 2
- **Before/After:** Current vs proposed state

ASCII style reference:

```
Boxes:   ┌──────────┐    Arrows:  ───▶  ◀──▶
         │ Service  │    Lines:   ─── │ ┌ ┐ └ ┘ ├ ┤
         └──────────┘

Flow:    [Start] ──▶ [Step 1] ──▶ [Step 2] ──▶ [Done]
                            │
                            ▼
                      [Alt path]
```

Formatting rules:

- Keep each diagram under 20 lines
- Caption above each: `**Fig: {what this shows}**`
- Plain language labels, no jargon
- Always wrap in code block for monospace alignment

---

## Phase 1: EXPLORE (Understand the Problem Space)

1. If a codebase exists, scan for related modules to understand the current state.
2. From the user's context, identify:
   - What problem is being described
   - Who has this problem
   - What exists today (if anything) and why it's insufficient
3. Use AskUserQuestion to probe the areas the user hasn't articulated yet:
   - **Problem clarity:** "Is the core problem X or Y?" (offer your interpretation as options)
   - **User segment:** "Who feels this pain most?" (with concrete personas if you can infer them)
   - **Scope ambition:** "Are you thinking quick win or foundational investment?"
   - Keep it to 3-5 questions total across 1-2 rounds.
4. When the problem is clear, move to Phase 2.

---

## Phase 2: SHAPE (Explore Approaches)

**Do not converge on one solution yet.** Present 2-3 distinct approaches.

First, show a comparison overview:

**Fig: Approach Comparison**

```
                │ Effort │ Impact │ Risk   │
────────────────┼────────┼────────┼────────┤
 A: Lightweight │ S      │ Medium │ Low    │
 B: Full Build  │ L      │ High   │ Medium │
 C: Integration │ M      │ Medium │ High   │
```

For each approach, write:

- **Name:** A short label (e.g., "Lightweight MVP", "Full Platform", "Integration-first")
- **How it works:** 2-3 sentences
- **Pros:** What it gets right
- **Cons:** What it gives up
- **Effort:** Rough t-shirt size (S/M/L/XL)
- **Risk:** What could go wrong

Then use AskUserQuestion:
Question: "Which direction resonates?"
Options: [approach names + "Combine elements from multiple" + "None of these, let me explain"]

If the user wants to combine or reject all, discuss further and reshape.
When a direction is chosen, move to Phase 3.

---

## Phase 3: DEFINE (Write the PRD)

Create `specs/$1-prd.md` using the template below.
The structure is **dashboard first, details second** — the reader should understand
the entire PRD from the first screen, then drill into sections for depth.

```markdown
# PRD: {idea-name}

> Status: DRAFT
> Created: {date}
> Last Updated: {date}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# PART 1: OVERVIEW (read this first)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Big Picture

<!-- The FIRST thing the reader sees. -->
<!-- Shows the core idea in one visual. Update when the PRD changes. -->

**Fig: How it works (user perspective)**
```

                        ┌─────────────────────────────┐
                        │      PROPOSED SOLUTION       │
                        └─────────────────────────────┘

[User has problem] ──▶ [Action 1] ──▶ [Action 2] ──▶ [Problem solved ✓]
│
▼
[Edge case path]

Replace with actual user/data/event flow.
The reader should grasp the entire idea from this one diagram.

```

## Dashboard

```

┌─ PROBLEM ──────────────────────────────────────────────┐
│ {One sentence: what's broken or missing} │
└────────────────────────────────────────────────────────┘

┌─ SOLUTION ─────────────────────────────────────────────┐
│ {One sentence: what we're doing about it} │
└────────────────────────────────────────────────────────┘

┌─ TARGET USER ──────────┐ ┌─ BET SIZE ────────────────┐
│ {who} │ │ {S / M / L / XL} │
└────────────────────────┘ └───────────────────────────┘

┌─ SUCCESS LOOKS LIKE ───────────────────────────────────┐
│ 1. {metric} │
│ 2. {metric} │
└────────────────────────────────────────────────────────┘

┌─ BIGGEST RISK ─────────────────────────────────────────┐
│ {The riskiest assumption we're betting on} │
└────────────────────────────────────────────────────────┘

```

## Before / After

**Fig: Current State (Before)**
```

[User] ──▶ [Current Step 1] ──▶ [Current Step 2] ──▶ [Pain Point ✗]

```

**Fig: Proposed State (After)**
```

[User] ──▶ [New Step 1] ──▶ [New Step 2] ──▶ [Outcome ✓]

```


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PART 2: DETAILS (drill down)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## 2.1 Problem Deep Dive

**Who is affected?**
```

┌─ Persona ──────────────────────────────────────────────┐
│ Name: {role or persona name} │
│ Context: {when/where they hit this problem} │
│ Pain level: {🔴 high / 🟡 medium / 🟢 low} │
│ Current workaround: {what they do today} │
└────────────────────────────────────────────────────────┘

```
(Repeat for each persona if more than one.)

**What happens if we do nothing?**
- {consequence 1}
- {consequence 2}

## 2.2 Solution Details

- **Direction:** The chosen approach from Phase 2
- **Core concept:** How it works in plain language (no technical jargon)

**Fig: User Journey (detailed)**
```

[Entry point]
│
▼
[Step 1: {action}] ──── success ────▶ [Step 2: {action}]
│ │
▼ (error) ▼
[Error handling] [Step 3: {action}]
│
▼
[✓ Outcome]

```

**What this is NOT** (explicit out-of-scope):
- {thing 1}
- {thing 2}

## 2.3 Success Metrics

```

┌─────┬──────────────────────────┬────────────────┬──────────────┐
│ # │ Metric │ Target │ Measured by │
├─────┼──────────────────────────┼────────────────┼──────────────┤
│ 1 │ {what} │ {number/range} │ {how} │
│ 2 │ {what} │ {number/range} │ {how} │
└─────┴──────────────────────────┴────────────────┴──────────────┘

```

## 2.4 Risks & Assumptions

```

┌─────────────────────────┬─────────────────────┬────────────────────┐
│ Assumption │ Risk if wrong │ How to validate │
├─────────────────────────┼─────────────────────┼────────────────────┤
│ {assumption 1} │ {impact} │ {validation plan} │
│ {assumption 2} │ {impact} │ {validation plan} │
└─────────────────────────┴─────────────────────┴────────────────────┘

```

## 2.5 Open Questions

- [ ] {question 1}
- [ ] {question 2}

## 2.6 Next Steps

```

┌─────┬──────────────────────────────────────┬────────────┐
│ # │ Action │ Owner │
├─────┼──────────────────────────────────────┼────────────┤
│ 1 │ {next step} │ {who} │
│ 2 │ Run /spec to create technical spec │ Engineer │
└─────┴──────────────────────────────────────┴────────────┘

```


## Changelog
| Date | Change | Reason |
|------|--------|--------|
| {date} | Initial draft | Phase 3 complete |
```

Present the **Big Picture + Dashboard** to the user and move to Phase 4.

---

## Phase 4: REVIEW LOOP (Refine the PRD)

### Each round:

**Step 1 — Proactive Review from these angles:**

- Is the problem statement actually validated or just assumed?
- Are we solving the right problem, or a symptom?
- Does the Before/After diagram actually show meaningful change?
- Are the success metrics gameable or misleading?
- Is the scope realistic for the bet size?
- Are there simpler ways to validate the riskiest assumption?
- What would a skeptic say about this PRD?

**Step 2 — Present findings + one AskUserQuestion:**
Question: "I found {N} concerns (listed above). How to proceed?"
Options: ["Apply recommended changes", "Let me review each one", "PRD looks good — finalize", "Other"]

**Step 3 — Update PRD file.**
Apply changes, add Changelog entry.
**Always update the Big Picture and Dashboard to reflect the current state.**
Present the updated Big Picture + Dashboard to the user, then return to Step 1.

### Exiting the Review Loop

When the user finalizes:

1. Change Status to `APPROVED`
2. Add Changelog entry
3. Present the final Big Picture + Dashboard
4. Suggest: "Ready to turn this into a technical spec? Run `/spec $1 Based on the approved PRD at specs/$1-prd.md`"

---

## Quick Re-entry

`/prd $1` resumes from the existing PRD file:

- Status is DRAFT → Phase 4 (Review Loop)
- Status is APPROVED → Suggest running /spec
- No PRD file → Phase 1 (Explore)
