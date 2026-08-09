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
| M5 — RISC-V Quantization Foundation | Done | `TASK-007`, `TASK-013`, `TASK-014`, `TASK-015`, `TASK-016`, `TASK-017`, `TASK-018` | Keep quantized RVV validation reports green as transformer kernels land. |
| M6 — Transformer Minimum Kernel Set | Done | `TASK-008`, `TASK-019`, `TASK-020`, `TASK-021`, `TASK-022`, `TASK-023`, `TASK-024` | Keep transformer helper validation green as M7 integration begins. |
| M7 — Known-Runtime Transformer Integration Smoke | Active | `TASK-025`, `TASK-026`, `TASK-027` | Review DES-016, then pin the llama.cpp/model baseline in `TASK-026`. |
| M8 — Quantized RVV Runtime Integration | Planned | `TASK-028`, `TASK-029`, `TASK-030`, `TASK-031`, `TASK-032` | Start after the host runtime seam passes `TASK-027`. |
| M9 — Configurable RISC-V System Simulation | Planned | `TASK-033`, `TASK-034` | Start with one QEMU system node after M8 user-mode validation. |
| M10 — RISC-V Operator Coverage and Hardening | Planned | Not created | Resume broad operator work after the transformer/runtime path is proven. |

## Active focus

1. Review `DES-016` and the runtime-first roadmap in `TASK-025`.
2. Run `TASK-026` as a bounded llama.cpp/model feasibility gate before choosing
   a permanent integration seam.

## Maintenance rules

- Add one task file for each independent PR-sized work packet.
- Link every active task to a roadmap milestone.
- Move broad roadmap bullets into concrete task acceptance criteria before
  implementation begins.
- Do not keep stale checklist items in completed tasks; move follow-up work into
  a new task and link it here.
