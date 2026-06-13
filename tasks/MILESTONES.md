# KernelSmith Milestone Dashboard

Use this file as the operational view of `ROADMAP.md`. Keep it short: each
milestone should show status, task links, blockers, and the next concrete work.

## Milestone status legend

| Status | Meaning |
|---|---|
| Done | Acceptance criteria are complete and validated. |
| Active | Current implementation or validation focus. |
| Planned | Scoped in the roadmap but not started. |
| Blocked | Waiting on external tooling, hardware, or a design decision. |

## Dashboard

| Milestone | Status | Task files | Next action |
|---|---|---|---|
| M0 — Dialect Infrastructure | Done | `TASK-001`, `TASK-002`, `TASK-003` | Keep verifier coverage current as ops evolve. |
| M1 — C API + Reference Library | Done | Covered by roadmap history | Add new C APIs through milestone-specific tasks. |
| M2 — Activation Lowering | Partial | Covered by roadmap history | Track generated-object integration when resumed. |
| M3 — Generic MatMul Lowering | Partial | `TASK-004` | Track generated matmul replacement separately if prioritized. |
| M4 — RISC-V RVV Target | Done | `TASK-005`, `TASK-006`, `TASK-009`, `TASK-010`, `TASK-011`, `TASK-012` | Keep RVV validation reports green as new kernels land. |
| M5 — RISC-V Quantization Foundation | Active | `TASK-007`, `TASK-013`, `TASK-014`, `TASK-015`, `TASK-016`, `TASK-017` | Start `TASK-017` W4A8 fused dot/GEMV lowering. |
| M6 — Transformer Minimum Kernel Set | Planned | `TASK-008` | Start after M5 dot/GEMV ABI and validation policy are stable. |
| M7 — Minimal RISC-V Transformer Demo | Planned | Not created | Create tasks after M5/M6 interfaces stabilize. |

## Active focus

1. `TASK-017`: implement W4A8 lowering now that INT8 lowering has established
   the quantized dot/GEMV pattern.
2. `TASK-008`: start M6 transformer helpers after quantized layout decisions are
   stable enough for public C APIs.

## Maintenance rules

- Add one task file for each independent PR-sized work packet.
- Link every active task to a roadmap milestone.
- Move broad roadmap bullets into concrete task acceptance criteria before
  implementation begins.
- Do not keep stale checklist items in completed tasks; move follow-up work into
  a new task and link it here.
