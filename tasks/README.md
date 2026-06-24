# KernelSmith Task Tracking

This directory is the execution tracker for the product roadmap.

## Source of truth

- `ROADMAP.md` defines product direction, milestone goals, and long-term scope.
- `tasks/MILESTONES.md` is the short operational dashboard: milestone status,
  active task IDs, blockers, and next actions.
- `tasks/TASK-XXX.md` files are executable work packets with acceptance
  criteria and verification steps.
- `.cursor/skills/manage-kernelsmith-tasks/SKILL.md` tells Cursor agents how to
  maintain the tracker during implementation work.

When status changes, update the task file and `tasks/MILESTONES.md` in the same
PR. Update `ROADMAP.md` only when product scope or milestone content changes.

## Status values

| Status | Meaning |
|---|---|
| `[ ] Not Started` | Defined but no implementation work has begun. |
| `[~] In Progress` | Work has started or partial implementation exists. |
| `[!] Blocked` | Cannot progress without an external dependency or decision. |
| `[?] Needs Review` | Implementation is ready but needs review/validation. |
| `[x] Complete` | Acceptance criteria are met and verification is recorded. |

## Priority values

| Priority | Meaning |
|---|---|
| `P0` | Required to keep the project buildable or unblock all work. |
| `P1` | Required for the current milestone. |
| `P2` | Important follow-up for a near milestone. |
| `P3` | Nice-to-have or future cleanup. |

## Task file template

```markdown
# TASK-XXX: Task Title

## Status
[ ] Not Started

## Priority
P1

## Milestone
M4 — RISC-V RVV Target

## Owner Agent
Use `.cursor/agents/<agent>.md` or "general".

## Description
What needs to be done and why.

## Acceptance Criteria
- [ ] Observable outcome
- [ ] Tests or validation added

## Dependencies
- TASK-YYY

## Verification
- `ctest --test-dir build --output-on-failure`

## Notes
Design links, constraints, and open questions.

## Log
Progress updates with dates.
```

## Active task index

| ID | Milestone | Title | Status | Priority |
|---|---|---|---|---|
| TASK-001 | M0 | Set up MLIR dialect infrastructure | Complete | P0 |
| TASK-002 | M0 | Implement Kernel dialect core | Complete | P0 |
| TASK-003 | M0/M3 | Implement matmul operation | Complete | P1 |
| TASK-004 | M3/M4 | Implement tiling and vectorization passes | Complete | P1 |
| TASK-005 | M4 | Implement RVV lowering pipeline | Complete | P1 |
| TASK-006 | M4 | Validate RVV correctness and benchmark path | Complete | P0 |
| TASK-009 | M4 | Plan RISC-V simulation verification | Complete | P1 |
| TASK-010 | M4 | Build golden reference verification infrastructure | Complete | P0 |
| TASK-011 | M4 | Add host reference execution comparator | Complete | P1 |
| TASK-012 | M4 | Compare RISC-V RVV outputs against golden references | Complete | P1 |
| TASK-007 | M5 | RISC-V quantization foundation | Complete | P1 |
| TASK-013 | M5 | Define quantized dot/GEMV C APIs and layouts | Complete | P0 |
| TASK-014 | M5 | Add quantized golden validation cases | Complete | P0 |
| TASK-015 | M5 | Add quantized RVV target profile parameters | Complete | P1 |
| TASK-016 | M5 | Implement INT8 dot/GEMV lowering | Complete | P1 |
| TASK-017 | M5 | Implement W4A8 fused dot/GEMV lowering | Complete | P1 |
| TASK-018 | M5 | Integrate generated INT8 RVV objects with the C API | Complete | P1 |
| TASK-008 | M6 | Transformer minimum kernel set | In Progress | P1 |
| TASK-019 | M6 | Add elementwise add/mul ops and linalg lowering | Complete | P1 |
| TASK-020 | M6 | Add transformer helper C APIs | Complete | P1 |
| TASK-021 | M6 | Add transformer helper golden CI coverage | Complete | P1 |
| TASK-022 | M6 | Lower ks.rms_norm to vectorizable linalg/loop form | Not Started | P1 |
| TASK-023 | M6 | Lower ks.softmax with numerically stable form | Not Started | P1 |

## Agent workflow

1. Read `tasks/MILESTONES.md` to identify the active milestone and next task.
2. Read the selected `TASK-XXX.md`, relevant `specs/`, and design docs.
3. Keep task scope small enough for one PR when possible.
4. Update task status, checkboxes, and log entries before handoff.
5. Run the verification listed in the task before marking it complete.
