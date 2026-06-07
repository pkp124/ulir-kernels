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
| M4 — RISC-V RVV Target | Active | `TASK-005`, `TASK-006`, `TASK-009`, `TASK-010`, `TASK-011`, `TASK-012` | Use the host golden checks from `TASK-011` to land RVV output comparison. |
| M5 — RISC-V Quantization Foundation | Planned | `TASK-007` | Start after golden verification defines fixed-point comparison policy. |
| M6 — Transformer Minimum Kernel Set | Planned | `TASK-008` | Add elementwise ops and transformer helper C APIs. |
| M7 — Minimal RISC-V Transformer Demo | Planned | Not created | Create tasks after M5/M6 interfaces stabilize. |

## Active focus

1. `TASK-012` (P1): compare RISC-V RVV QEMU output against the same golden
   references at supported VLENs.
2. `TASK-006` remains the parent M4 validation tracker for benchmark and
   tail/stride follow-up after golden correctness is in place.
3. `TASK-007`: start M5 quantization once fixed-point compare policy is
   implemented through golden verification.
4. `TASK-008`: start M6 transformer helpers after quantized layout decisions are
   stable enough for public C APIs.

## Maintenance rules

- Add one task file for each independent PR-sized work packet.
- Link every active task to a roadmap milestone.
- Move broad roadmap bullets into concrete task acceptance criteria before
  implementation begins.
- Do not keep stale checklist items in completed tasks; move follow-up work into
  a new task and link it here.
